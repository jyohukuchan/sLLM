//! Additive `sllm-core` owned-execution adapter for the typed public HIP
//! BF16 copy/add, matmul, and RMSNorm paths. It does not alter the legacy `Backend`
//! control-plane methods.
//!
//! The adapter contains no alternate ABI or kernel path.  It only lowers core
//! owned bindings into the existing `Context`/`Queue`/`Buffer`/
//! typed prepared-operation/submission wrappers.

use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use sllm_hip_sys as sys;

use sllm_core::{
    AdapterResource, BoundSemanticOp, BufferRange, CausalAttentionDescriptor, DispatchEvidence,
    ExecutionAdapterAccess, ExecutionCausalAttentionSubmissionAdapter, ExecutionError,
    ExecutionKvStateSubmissionAdapter, ExecutionLinearAttentionSubmissionAdapter,
    ExecutionMinistral3YarnSubmissionAdapter, ExecutionQueueFenceAdapter, ExecutionReadbackAdapter,
    ExecutionSession, ExecutionSessionAdapter, ExecutionSessionRequest, ExecutionState,
    ExecutionStateImageV1, ExecutionSubmissionAdapter, ExecutionTransferAdapter, KvCacheEncoding,
    OpaqueStatePlane, OwnedTensorBinding, PrepareSupport, PreparedOperation,
    QueueCompletionMode as CoreQueueCompletionMode, ShutdownReport, StateLayerMetadataV1,
    StateOwnerKindV1, StatePlaneKindV1,
};

use crate::argmax::{ArgmaxDispatchInfo, ArgmaxSubmission, PreparedArgmax};
use crate::kv_state::{
    CausalAttentionCompletion, CausalAttentionEvidence, KvAppendCompletion, KvAppendEvidence,
    KvStateResource, native_kv_storage,
};
use crate::linear_attention::{
    LinearAttentionCompletion, LinearAttentionEvidence, LinearAttentionStateResource,
};
use crate::runtime::logical_gcn_arch_name;
use crate::{
    ArgmaxDescriptor, AttentionPreprocessDescriptor, AttentionPreprocessDispatchInfo,
    AttentionPreprocessSubmission, Buffer, Completion, CompletionState, Context,
    DeepSeekV4MoeRouteDescriptor, DeepSeekV4MoeRouteDispatchInfo, DeepSeekV4MoeRouteSubmission,
    ElementwiseDescriptor, ElementwiseDispatchInfo, ElementwiseSubmission, EmbeddingDescriptor,
    EmbeddingDispatchInfo, EmbeddingSubmission, GdnProjectionBundleDescriptor,
    GdnProjectionBundleDispatchInfo, GdnProjectionBundleSubmission, GraphSpan, GraphSpanPlan,
    HipBackend, MatmulDescriptor, MatmulDispatchInfo, MatmulSubmission,
    MiniMaxM3MoeRouteDescriptor, MiniMaxM3MoeRouteDispatchInfo, MiniMaxM3MoeRouteSubmission,
    Ministral3YarnDescriptor, Ministral3YarnDispatchInfo, Ministral3YarnPositionMode,
    Ministral3YarnSubmission as HipMinistral3YarnSubmission, MlpGateUpSiluBundleDescriptor,
    MlpGateUpSiluBundleDispatchInfo, MlpGateUpSiluBundleSubmission, MoeExpertDescriptor,
    MoeExpertDispatchInfo, MoeExpertSubmission, MoeRouteDescriptor, MoeRouteDispatchInfo,
    MoeRouteLayout, MoeRouteSubmission, PreparedAttentionPreprocess, PreparedDeepSeekV4MoeRoute,
    PreparedElementwise, PreparedEmbedding, PreparedGdnProjectionBundle, PreparedMatmul,
    PreparedMiniMaxM3MoeRoute, PreparedMlpGateUpSiluBundle, PreparedMoeExpert, PreparedMoeRoute,
    PreparedQwen38ProjectionPack2, PreparedResidualRmsNorm, PreparedRmsNorm, PreparedRotary,
    PreparedTokenSelector, PreparedWindowedAttention, Queue,
    QueueCompletionMode as HipQueueCompletionMode, Qwen38ProjectionPack2Descriptor,
    Qwen38ProjectionPack2DispatchInfo, Qwen38ProjectionPack2Submission, ResidualRmsNormDescriptor,
    ResidualRmsNormDispatchInfo, ResidualRmsNormSubmission, RmsNormDescriptor, RmsNormDispatchInfo,
    RmsNormSubmission, RotaryDescriptor, RotaryDispatchInfo, RotarySubmission, RuntimeError,
    RuntimeStatus, TokenSelectorDescriptor, TokenSelectorDispatchInfo, TokenSelectorSubmission,
    WindowedAttentionDescriptor, WindowedAttentionDispatchInfo, WindowedAttentionSubmission,
    gemma4_moe_expert_workspace_bytes, moe_expert_workspace_bytes,
};

const HIP_BACKEND_NAME: &str = "hip";
const CLEANUP_ATTEMPT_CAP: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeepSeekV4MoeRouteLowering {
    mode: sllm_core::DeepSeekV4MoeRouteMode,
    active_input_index: usize,
    renormalize: bool,
    routed_scale_bits: u32,
}

impl DeepSeekV4MoeRouteLowering {
    fn from_semantic(descriptor: &sllm_core::SemanticOpDescriptor) -> Result<Self, ExecutionError> {
        if descriptor.kind() != sllm_core::SemanticOpKind::DeepSeekV4MoeRoute {
            return Err(ExecutionError::InvalidRequest {
                reason: "DeepSeek V4 MoE route lowering received the wrong semantic operation"
                    .to_owned(),
            });
        }
        let contract = descriptor.deepseek_v4_moe_route_contract().ok_or_else(|| {
            ExecutionError::InvalidRequest {
                reason: "deepseek_v4_moe_route semantic descriptor is missing its contract"
                    .to_owned(),
            }
        })?;
        let active_input_index = match contract.mode() {
            sllm_core::DeepSeekV4MoeRouteMode::Score => 1,
            sllm_core::DeepSeekV4MoeRouteMode::Hash => 2,
        };
        Ok(Self {
            mode: contract.mode(),
            active_input_index,
            renormalize: contract.renormalize_selected_weights(),
            routed_scale_bits: contract.routed_scale_bits(),
        })
    }

    const fn routed_scale(self) -> f32 {
        f32::from_bits(self.routed_scale_bits)
    }
}

fn image_failure(reason: impl Into<String>) -> ExecutionError {
    ExecutionError::InvalidRequest {
        reason: format!("HIP state image contract violation: {}", reason.into()),
    }
}

fn kv_image_planes(encoding: KvCacheEncoding, count: u32) -> Vec<(u32, StatePlaneKindV1)> {
    let mut planes = vec![
        (sys::SLLM_HIP_KV_STATE_PLANE_KEY, StatePlaneKindV1::KvKey),
        (
            sys::SLLM_HIP_KV_STATE_PLANE_VALUE,
            StatePlaneKindV1::KvValue,
        ),
    ];
    if matches!(
        encoding,
        KvCacheEncoding::Fp8E4M3Fn
            | KvCacheEncoding::Fp8E4M3Block16
            | KvCacheEncoding::Fp8E5M2Block16
            | KvCacheEncoding::Mxfp8E4
            | KvCacheEncoding::Mxfp8E5
    ) {
        planes.extend([
            (
                sys::SLLM_HIP_KV_STATE_PLANE_KEY_SCALE,
                StatePlaneKindV1::KvKeyScale,
            ),
            (
                sys::SLLM_HIP_KV_STATE_PLANE_VALUE_SCALE,
                StatePlaneKindV1::KvValueScale,
            ),
        ]);
    }
    if matches!(encoding, KvCacheEncoding::Nvfp4) {
        planes.extend([
            (
                sys::SLLM_HIP_KV_STATE_PLANE_KEY_SCALE,
                StatePlaneKindV1::KvKeyScale,
            ),
            (
                sys::SLLM_HIP_KV_STATE_PLANE_VALUE_SCALE,
                StatePlaneKindV1::KvValueScale,
            ),
            (
                sys::SLLM_HIP_KV_STATE_PLANE_KEY_OUTER_SCALE,
                StatePlaneKindV1::KvKeyOuterScale,
            ),
            (
                sys::SLLM_HIP_KV_STATE_PLANE_VALUE_OUTER_SCALE,
                StatePlaneKindV1::KvValueOuterScale,
            ),
        ]);
    }
    planes.truncate(count as usize);
    planes
}

fn linear_image_planes(count: u32) -> Vec<(u32, StatePlaneKindV1)> {
    let mut planes = vec![
        (1, StatePlaneKindV1::LinearConvSlot0),
        (2, StatePlaneKindV1::LinearConvSlot1),
        (3, StatePlaneKindV1::LinearRecurrentSlot0),
        (4, StatePlaneKindV1::LinearRecurrentSlot1),
        (5, StatePlaneKindV1::LinearScratch),
    ];
    planes.truncate(count as usize);
    planes
}

fn read_native_plane<F>(size: u64, mut export: F) -> Result<Vec<u8>, ExecutionError>
where
    F: FnMut(u64, &mut [u8]) -> Result<(), RuntimeError>,
{
    let size = usize::try_from(size).map_err(|_| image_failure("plane size exceeds host usize"))?;
    let mut bytes = vec![0_u8; size];
    let chunk_max = usize::try_from(sys::SLLM_HIP_STATE_CHUNK_MAX_BYTES)
        .map_err(|_| image_failure("chunk limit exceeds host usize"))?;
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = (offset + chunk_max).min(bytes.len());
        export(offset as u64, &mut bytes[offset..end]).map_err(map_backend_error)?;
        offset = end;
    }
    Ok(bytes)
}

fn write_native_plane<F>(bytes: &[u8], mut import: F) -> Result<(), ExecutionError>
where
    F: FnMut(u64, &[u8]) -> Result<(), RuntimeError>,
{
    let chunk_max = usize::try_from(sys::SLLM_HIP_STATE_CHUNK_MAX_BYTES)
        .map_err(|_| image_failure("chunk limit exceeds host usize"))?;
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = (offset + chunk_max).min(bytes.len());
        import(offset as u64, &bytes[offset..end]).map_err(map_backend_error)?;
        offset = end;
    }
    Ok(())
}

pub(crate) fn open_execution_session(
    backend: HipBackend,
    request: ExecutionSessionRequest,
) -> Result<Arc<ExecutionSession>, ExecutionError> {
    let device = Context::query_device(request.device_index()).map_err(map_backend_error)?;
    let context = Context::create(request.device_index(), request.expected_target())
        .map_err(map_backend_error)?;
    let adapter = Arc::new(HipExecutionSession {
        state: Arc::new(HipSessionState::new()),
        backend,
        context,
        total_memory_bytes: device.total_memory_bytes,
        available_memory_bytes: device.available_memory_bytes,
    });
    Ok(Arc::new(ExecutionSession::new(HIP_BACKEND_NAME, adapter)))
}

struct HipSessionState {
    activity: ActiveSessionState,
}

impl HipSessionState {
    fn new() -> Self {
        Self {
            activity: ActiveSessionState::new(),
        }
    }

    fn ensure_open(&self) -> Result<(), ExecutionError> {
        self.activity.ensure_open()
    }

    fn acquire_active(self: &Arc<Self>) -> Result<ActiveOperation, ExecutionError> {
        self.activity.acquire_active()?;
        Ok(ActiveOperation {
            state: Arc::clone(self),
            active: true,
        })
    }

    fn begin_shutdown(&self) -> Result<(), ExecutionError> {
        self.activity.begin_shutdown()
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.activity.active_count()
    }
}

struct ActiveSessionState {
    // The high bit is the closing state; the remaining bits are the active
    // operation count.  Admission, closing, and release all update this one
    // word, so shutdown cannot observe zero and then race a stale admission.
    lifecycle: AtomicUsize,
    #[cfg(test)]
    admission_gate: Mutex<Option<AdmissionGate>>,
}

impl ActiveSessionState {
    fn new() -> Self {
        Self {
            lifecycle: AtomicUsize::new(0),
            #[cfg(test)]
            admission_gate: Mutex::new(None),
        }
    }

    const CLOSING_BIT: usize = 1usize << (usize::BITS - 1);
    const ACTIVE_MASK: usize = Self::CLOSING_BIT - 1;

    fn ensure_open(&self) -> Result<(), ExecutionError> {
        if self.lifecycle.load(Ordering::Acquire) & Self::CLOSING_BIT != 0 {
            Err(ExecutionError::Closing)
        } else {
            Ok(())
        }
    }

    fn acquire_active(&self) -> Result<(), ExecutionError> {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            if observed & Self::CLOSING_BIT != 0 {
                return Err(ExecutionError::Closing);
            }
            let active = observed & Self::ACTIVE_MASK;
            if active == Self::ACTIVE_MASK {
                return Err(ExecutionError::Busy);
            }
            #[cfg(test)]
            self.pause_before_admission_cas();
            match self.lifecycle.compare_exchange(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(next) => observed = next,
            }
        }
    }

    fn begin_shutdown(&self) -> Result<(), ExecutionError> {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            if observed & Self::CLOSING_BIT != 0 {
                return if observed & Self::ACTIVE_MASK == 0 {
                    Ok(())
                } else {
                    Err(ExecutionError::Busy)
                };
            }
            let active = observed & Self::ACTIVE_MASK;
            match self.lifecycle.compare_exchange(
                observed,
                observed | Self::CLOSING_BIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return if active == 0 {
                        Ok(())
                    } else {
                        Err(ExecutionError::Busy)
                    };
                }
                Err(next) => observed = next,
            }
        }
    }

    fn release_active(&self) {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            let active = observed & Self::ACTIVE_MASK;
            if active == 0 {
                // A ticket is single-owner and should make this path
                // unreachable.  If an invariant is ever violated, close the
                // state instead of wrapping the counter and admitting work.
                self.lifecycle.fetch_or(Self::CLOSING_BIT, Ordering::AcqRel);
                return;
            }
            match self.lifecycle.compare_exchange(
                observed,
                observed - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(next) => observed = next,
            }
        }
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.lifecycle.load(Ordering::Acquire) & Self::ACTIVE_MASK
    }

    #[cfg(test)]
    fn pause_next_admission(
        &self,
        reached: Arc<std::sync::Barrier>,
        proceed: Arc<std::sync::Barrier>,
    ) {
        *self.admission_gate.lock().expect("admission gate lock") =
            Some(AdmissionGate { reached, proceed });
    }

    #[cfg(test)]
    fn pause_before_admission_cas(&self) {
        let gate = self
            .admission_gate
            .lock()
            .expect("admission gate lock")
            .take();
        if let Some(gate) = gate {
            gate.reached.wait();
            gate.proceed.wait();
        }
    }
}

#[cfg(test)]
struct AdmissionGate {
    reached: Arc<std::sync::Barrier>,
    proceed: Arc<std::sync::Barrier>,
}

struct ActiveOperation {
    state: Arc<HipSessionState>,
    active: bool,
}

impl ActiveOperation {
    fn clone_active(&self) -> Result<Self, ExecutionError> {
        self.state.activity.acquire_active()?;
        Ok(Self {
            state: Arc::clone(&self.state),
            active: true,
        })
    }
}

impl Drop for ActiveOperation {
    fn drop(&mut self) {
        if self.active {
            self.state.activity.release_active();
        }
    }
}

struct HipExecutionSession {
    state: Arc<HipSessionState>,
    backend: HipBackend,
    context: Context,
    total_memory_bytes: u64,
    available_memory_bytes: u64,
}

impl ExecutionSessionAdapter for HipExecutionSession {
    fn expected_target(&self) -> Option<String> {
        self.context.expected_target().map(str::to_owned)
    }

    fn max_transfer_bytes(&self) -> u64 {
        crate::sys::SLLM_HIP_MAX_TRANSFER_BYTES
    }

    fn available_memory_bytes(&self) -> Option<u64> {
        Some(self.available_memory_bytes)
    }

    fn total_memory_bytes(&self) -> Option<u64> {
        Some(self.total_memory_bytes)
    }

    fn supports(&self, descriptor: &sllm_core::SemanticOpDescriptor) -> PrepareSupport {
        if let Err(error) = self.state.ensure_open() {
            return PrepareSupport::Unsupported {
                reason: error.to_string(),
            };
        }
        if let Err(error) = descriptor.validate() {
            return PrepareSupport::Unsupported {
                reason: format!("invalid semantic descriptor: {error}"),
            };
        }
        if !matches!(
            descriptor.kind(),
            sllm_core::SemanticOpKind::Copy
                | sllm_core::SemanticOpKind::Add
                | sllm_core::SemanticOpKind::BroadcastAdd
                | sllm_core::SemanticOpKind::BroadcastMul
                | sllm_core::SemanticOpKind::ScalarMul
                | sllm_core::SemanticOpKind::SiluMul
                | sllm_core::SemanticOpKind::GeluTanhMul
                | sllm_core::SemanticOpKind::SigmoidMul
                | sllm_core::SemanticOpKind::TanhSoftcap
                | sllm_core::SemanticOpKind::Embedding
                | sllm_core::SemanticOpKind::Matmul
                | sllm_core::SemanticOpKind::Qwen38ProjectionPack2
                | sllm_core::SemanticOpKind::GdnProjectionBundle
                | sllm_core::SemanticOpKind::MlpGateUpSiluBundle
                | sllm_core::SemanticOpKind::RmsNorm
                | sllm_core::SemanticOpKind::ResidualRmsNorm
                | sllm_core::SemanticOpKind::Argmax
                | sllm_core::SemanticOpKind::AttentionPreprocess
                | sllm_core::SemanticOpKind::Rotary
                | sllm_core::SemanticOpKind::CausalAttention
                | sllm_core::SemanticOpKind::TokenSelect
                | sllm_core::SemanticOpKind::MoeRoute
                | sllm_core::SemanticOpKind::DeepSeekV4MoeRoute
                | sllm_core::SemanticOpKind::MiniMaxM3MoeRoute
                | sllm_core::SemanticOpKind::MoeExpert
                | sllm_core::SemanticOpKind::SparseMoe
        ) {
            return PrepareSupport::Unsupported {
                reason: "the HIP owned execution bridge does not support this semantic operation"
                    .to_owned(),
            };
        }
        PrepareSupport::Supported
    }

    fn create_queue(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        Queue::create(&self.context)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn set_queue_completion_mode(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        mode: CoreQueueCompletionMode,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?;
        let mode = match mode {
            CoreQueueCompletionMode::Profiled => HipQueueCompletionMode::Profiled,
            CoreQueueCompletionMode::Deferred => HipQueueCompletionMode::Deferred,
        };
        queue.set_completion_mode(mode).map_err(map_backend_error)
    }

    fn create_queue_fence(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
    ) -> Result<Box<dyn ExecutionQueueFenceAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let ticket = self.state.acquire_active()?;
        let completion = access
            .downcast_queue_payload::<Queue>(queue)?
            .fence()
            .map_err(map_backend_error)?;
        Ok(Box::new(HipQueueFence {
            completion,
            _ticket: ticket,
        }))
    }

    fn allocate(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        size_bytes: u64,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        Buffer::allocate(&self.context, size_bytes)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn create_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state_id: sllm_core::KvStateId,
        descriptor: sllm_core::KvStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        KvStateResource::create(&self.context, access.session_id(), state_id, descriptor)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn kv_state_snapshot(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
    ) -> Result<sllm_core::KvStateSnapshot, ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .snapshot()
            .map_err(map_backend_error)
    }

    fn fork_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        source: &sllm_core::KvState,
        destination_id: sllm_core::KvStateId,
        destination_descriptor: sllm_core::KvStateDescriptor,
    ) -> Result<(AdapterResource, sllm_core::StateForkAuditV1), ExecutionError> {
        self.state.ensure_open()?;
        let source_resource = access.downcast_kv_state_payload::<KvStateResource>(source)?;
        let (resource, audit) = source_resource
            .fork(destination_id, destination_descriptor)
            .map_err(map_backend_error)?;
        Ok((AdapterResource::new(resource), audit))
    }

    fn readback_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .readback(plane, byte_offset, destination)
            .map_err(map_backend_error)
    }

    fn kv_state_fork_query(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
    ) -> Result<sllm_core::StateForkAuditV1, ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .fork_query()
            .map_err(map_backend_error)
    }

    fn export_kv_state_image(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
    ) -> Result<ExecutionStateImageV1, ExecutionError> {
        self.state.ensure_open()?;
        let resource = access.downcast_kv_state_payload::<KvStateResource>(state)?;
        let info = resource.image_query().map_err(map_backend_error)?;
        let encoding = state.descriptor().cache_encoding();
        let (dtype, native_encoding, _, _) =
            native_kv_storage(state.descriptor(), self.context.expected_target())
                .map_err(map_backend_error)?;
        if info.session_id != access.session_id().raw()
            || info.layer_id != state.layer_id()
            || info.dtype != dtype
            || info.encoding != native_encoding
            || info.active_slot != 0
            || info.capacity_tokens != state.capacity()
            || info.published_length > info.capacity_tokens
        {
            return Err(image_failure(
                "native KV image metadata does not match descriptor",
            ));
        }
        let retained_start = info.published_length.saturating_sub(
            state
                .descriptor()
                .sliding_window()
                .unwrap_or(info.published_length),
        );
        let image_version_matches = if let Some(window) = state.descriptor().sliding_window() {
            info.info_version == sys::SLLM_HIP_STATE_IMAGE_SLIDING_VERSION
                && (u64::from(info.reserved[0]) | (u64::from(info.reserved[1]) << 32)) == window
                && (u64::from(info.reserved[2]) | (u64::from(info.reserved[3]) << 32))
                    == retained_start
                && info.reserved[4..].iter().all(|value| *value == 0)
        } else {
            info.info_version == sys::SLLM_HIP_STATE_FORK_INFO_VERSION
                && info.reserved.iter().all(|value| *value == 0)
        };
        if !image_version_matches {
            return Err(image_failure(
                "native KV image version or retention metadata is invalid",
            ));
        }
        let planes = kv_image_planes(encoding, info.plane_count);
        if planes.len() != info.plane_count as usize {
            return Err(image_failure("native KV image plane count is invalid"));
        }
        let mut output = Vec::with_capacity(planes.len());
        for (native_plane, semantic_plane) in planes {
            let size = resource
                .image_plane_size(native_plane)
                .map_err(map_backend_error)?;
            let bytes = read_native_plane(size, |offset, destination| {
                resource.export_chunk(native_plane, offset, destination, info.published_length)
            })?;
            output.push(OpaqueStatePlane {
                owner: StateOwnerKindV1::Kv,
                layer_id: state.layer_id(),
                plane: semantic_plane,
                bytes,
            });
        }
        Ok(ExecutionStateImageV1::new(
            StateLayerMetadataV1 {
                owner: StateOwnerKindV1::Kv,
                layer_id: state.layer_id(),
                published_length: info.published_length,
                generation: info.generation,
                active_slot: None,
            },
            output,
        ))
    }

    fn import_kv_state_image(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        image: &ExecutionStateImageV1,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        let resource = access.downcast_kv_state_payload::<KvStateResource>(state)?;
        let info = resource.image_query().map_err(map_backend_error)?;
        if info.published_length != 0 {
            return Err(image_failure("KV image import destination is not empty"));
        }
        let encoding = state.descriptor().cache_encoding();
        let planes = kv_image_planes(encoding, info.plane_count);
        let (expected_dtype, expected_encoding, _, _) =
            native_kv_storage(state.descriptor(), self.context.expected_target())
                .map_err(map_backend_error)?;
        if info.session_id != access.session_id().raw()
            || info.layer_id != state.layer_id()
            || info.capacity_tokens != state.capacity()
            || info.dtype != expected_dtype
            || info.encoding != expected_encoding
        {
            return Err(image_failure("native KV import target metadata mismatch"));
        }
        if image.metadata().owner != StateOwnerKindV1::Kv
            || image.metadata().layer_id != state.layer_id()
            || image.metadata().active_slot.is_some()
            || image.metadata().published_length > state.capacity()
            || image.planes().len() != planes.len()
        {
            return Err(image_failure("KV image metadata or topology mismatch"));
        }
        let mut native_info = info;
        native_info.published_length = image.metadata().published_length;
        native_info.generation = image.metadata().generation;
        let retained_length = state
            .descriptor()
            .sliding_window()
            .map_or(image.metadata().published_length, |window| {
                image.metadata().published_length.min(window)
            });
        if let Some(window) = state.descriptor().sliding_window() {
            let retained_start = image.metadata().published_length - retained_length;
            native_info.reserved[0] = window as u32;
            native_info.reserved[1] = (window >> 32) as u32;
            native_info.reserved[2] = retained_start as u32;
            native_info.reserved[3] = (retained_start >> 32) as u32;
        }
        let sliding_bytes_per_token = state
            .descriptor()
            .sliding_window()
            .map(|_| {
                state
                    .descriptor()
                    .resident_bytes_per_plane()
                    .and_then(|bytes| {
                        bytes.checked_div(state.descriptor().physical_capacity_tokens())
                    })
                    .ok_or_else(|| image_failure("sliding KV plane stride overflow"))
            })
            .transpose()?;
        for (native_plane, semantic_plane) in planes {
            let plane = image
                .planes()
                .iter()
                .find(|plane| plane.plane == semantic_plane)
                .ok_or_else(|| image_failure("KV image is missing a required plane"))?;
            let expected = if let Some(bytes_per_token) = sliding_bytes_per_token {
                retained_length
                    .checked_mul(bytes_per_token)
                    .ok_or_else(|| image_failure("sliding KV image plane size overflow"))?
            } else {
                resource
                    .image_plane_size(native_plane)
                    .map_err(map_backend_error)?
            };
            if plane.bytes.len() as u64 != expected {
                return Err(image_failure("KV image plane byte length mismatch"));
            }
            write_native_plane(&plane.bytes, |offset, bytes| {
                resource.import_chunk(
                    native_plane,
                    offset,
                    bytes,
                    image.metadata().published_length,
                )
            })?;
        }
        resource
            .import_finalize(&native_info)
            .map_err(map_backend_error)
    }

    fn rewind_last_kv_state_transition(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .rewind_last(expected_length, rewind_length)
            .map_err(map_backend_error)
    }

    fn append_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        queue: &sllm_core::ExecutionQueue,
        key: &OwnedTensorBinding,
        value: &OwnedTensorBinding,
        request: &sllm_core::KvStateAppendRequest,
    ) -> Result<(Box<dyn ExecutionKvStateSubmissionAdapter>, DispatchEvidence), ExecutionError>
    {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let key_buffer = access
            .downcast_buffer_payload::<Buffer>(key.buffer())?
            .clone();
        let value_buffer = access
            .downcast_buffer_payload::<Buffer>(value.buffer())?
            .clone();
        let key = key_buffer.binding(key.view().clone());
        let value = value_buffer.binding(value.view().clone());
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.append(&queue, &key, &value, *request) {
            Ok(result) => result,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipKvSubmission {
                completion,
                _ticket: ticket,
            }),
            dispatch_from_kv_append(evidence),
        ))
    }

    fn execute_causal_attention(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        queue: &sllm_core::ExecutionQueue,
        query: &OwnedTensorBinding,
        output: &OwnedTensorBinding,
        descriptor: CausalAttentionDescriptor,
    ) -> Result<
        (
            Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let query_buffer = access
            .downcast_buffer_payload::<Buffer>(query.buffer())?
            .clone();
        let output_buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let query = query_buffer.binding(query.view().clone());
        let output = output_buffer.binding(output.view().clone());
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.causal_attention(
            &queue,
            &query,
            &output,
            descriptor.start_position(),
            descriptor.expected_kv_length(),
            descriptor.sliding_window(),
            descriptor.score_scale(),
        ) {
            Ok(value) => value,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipCausalAttentionSubmission {
                completion,
                _evidence: evidence.clone(),
                _ticket: ticket,
            }),
            dispatch_from_causal_attention(evidence),
        ))
    }

    fn execute_ministral3_yarn(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        query: &OwnedTensorBinding,
        key: &OwnedTensorBinding,
        positions: &OwnedTensorBinding,
        query_output: &OwnedTensorBinding,
        key_output: &OwnedTensorBinding,
        stage: sllm_core::Ministral3YarnQueryScaleStage,
    ) -> Result<
        (
            Box<dyn ExecutionMinistral3YarnSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let binding = |owned: &OwnedTensorBinding| -> Result<crate::TensorBinding, ExecutionError> {
            let buffer = access
                .downcast_buffer_payload::<Buffer>(owned.buffer())?
                .clone();
            Ok(buffer.binding(owned.view().clone()))
        };
        let descriptor = Ministral3YarnDescriptor::new(
            binding(query)?,
            binding(key)?,
            binding(positions)?,
            binding(query_output)?,
            binding(key_output)?,
            stage.start_position(),
            Ministral3YarnPositionMode::Contiguous,
        )
        .map_err(map_backend_error)?;
        let ticket = self.state.acquire_active()?;
        let prepared = match self
            .backend
            .prepare_ministral3_yarn(&self.context, descriptor)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        let (submission, evidence) = match prepared.execute(&queue) {
            Ok(result) => result,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipMinistral3YarnSubmissionAdapter {
                completion: submission,
                _ticket: ticket,
            }),
            dispatch_from_ministral3_yarn(evidence),
        ))
    }

    fn create_linear_attention_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state_id: sllm_core::LinearAttentionStateId,
        descriptor: sllm_core::LinearAttentionStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        LinearAttentionStateResource::create(
            &self.context,
            access.session_id(),
            state_id,
            descriptor,
        )
        .map(AdapterResource::new)
        .map_err(map_backend_error)
    }

    fn linear_attention_state_snapshot(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
    ) -> Result<sllm_core::LinearAttentionStateSnapshot, ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .snapshot()
            .map_err(map_backend_error)
    }

    fn fork_linear_attention_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        source: &sllm_core::LinearAttentionState,
        destination_id: sllm_core::LinearAttentionStateId,
        destination_descriptor: sllm_core::LinearAttentionStateDescriptor,
    ) -> Result<(AdapterResource, sllm_core::StateForkAuditV1), ExecutionError> {
        self.state.ensure_open()?;
        let source_resource = access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(source)?;
        let (resource, audit) = source_resource
            .fork(destination_id, destination_descriptor)
            .map_err(map_backend_error)?;
        Ok((AdapterResource::new(resource), audit))
    }

    fn export_linear_attention_state_image(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
    ) -> Result<ExecutionStateImageV1, ExecutionError> {
        self.state.ensure_open()?;
        let resource = access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?;
        let info = resource.image_query().map_err(map_backend_error)?;
        if info.session_id != access.session_id().raw()
            || info.layer_id != state.layer_id()
            || info.dtype != sys::SLLM_TENSOR_DTYPE_BF16
            || info.encoding != sys::SLLM_TENSOR_ENCODING_UNQUANTIZED
            || info.capacity_tokens != state.capacity()
            || info.active_slot > 1
        {
            return Err(image_failure(
                "native linear image metadata does not match descriptor",
            ));
        }
        let planes = linear_image_planes(info.plane_count);
        if planes.len() != info.plane_count as usize {
            return Err(image_failure("native linear image plane count is invalid"));
        }
        let mut output = Vec::with_capacity(planes.len());
        for (native_plane, semantic_plane) in planes {
            let size = resource
                .image_plane_size(native_plane)
                .map_err(map_backend_error)?;
            let bytes = read_native_plane(size, |offset, destination| {
                resource.export_chunk(native_plane, offset, destination)
            })?;
            output.push(OpaqueStatePlane {
                owner: StateOwnerKindV1::LinearAttention,
                layer_id: state.layer_id(),
                plane: semantic_plane,
                bytes,
            });
        }
        Ok(ExecutionStateImageV1::new(
            StateLayerMetadataV1 {
                owner: StateOwnerKindV1::LinearAttention,
                layer_id: state.layer_id(),
                published_length: info.published_length,
                generation: info.generation,
                active_slot: Some(info.active_slot as u8),
            },
            output,
        ))
    }

    fn import_linear_attention_state_image(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
        image: &ExecutionStateImageV1,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        let resource = access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?;
        let info = resource.image_query().map_err(map_backend_error)?;
        if info.published_length != 0 {
            return Err(image_failure(
                "linear image import destination is not empty",
            ));
        }
        let planes = linear_image_planes(info.plane_count);
        if image.metadata().owner != StateOwnerKindV1::LinearAttention
            || image.metadata().layer_id != state.layer_id()
            || image.metadata().active_slot != Some(0) && image.metadata().active_slot != Some(1)
            || image.metadata().published_length > state.capacity()
            || image.planes().len() != planes.len()
        {
            return Err(image_failure("linear image metadata or topology mismatch"));
        }
        let mut native_info = info;
        native_info.active_slot = image.metadata().active_slot.unwrap() as u32;
        native_info.published_length = image.metadata().published_length;
        native_info.generation = image.metadata().generation;
        for (native_plane, semantic_plane) in planes {
            let plane = image
                .planes()
                .iter()
                .find(|plane| plane.plane == semantic_plane)
                .ok_or_else(|| image_failure("linear image is missing a required plane"))?;
            let expected = resource
                .image_plane_size(native_plane)
                .map_err(map_backend_error)?;
            if plane.bytes.len() as u64 != expected {
                return Err(image_failure("linear image plane byte length mismatch"));
            }
            write_native_plane(&plane.bytes, |offset, bytes| {
                resource.import_chunk(native_plane, offset, bytes)
            })?;
        }
        resource
            .import_finalize(&native_info)
            .map_err(map_backend_error)
    }

    fn rewind_last_linear_attention_transition(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .rewind_last(expected_length, rewind_length)
            .map_err(map_backend_error)
    }

    fn execute_linear_attention(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
        queue: &sllm_core::ExecutionQueue,
        bindings: &sllm_core::LinearAttentionBindings,
        request: sllm_core::LinearAttentionRequest,
    ) -> Result<
        (
            Box<dyn ExecutionLinearAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let owned = [
            bindings.qkv(),
            bindings.z(),
            bindings.b_input(),
            bindings.a_input(),
            bindings.conv_weight(),
            bindings.a_log(),
            bindings.dt_bias(),
            bindings.norm_weight(),
            bindings.output(),
        ];
        let buffers = owned
            .map(|binding| {
                access
                    .downcast_buffer_payload::<Buffer>(binding.buffer())
                    .cloned()
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let native: [crate::TensorBinding; 9] =
            std::array::from_fn(|index| buffers[index].binding(owned[index].view().clone()));
        let references: [&crate::TensorBinding; 9] = std::array::from_fn(|index| &native[index]);
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.execute(&queue, references, request) {
            Ok(result) => result,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipLinearAttentionSubmission {
                completion,
                _evidence: evidence.clone(),
                _ticket: ticket,
            }),
            dispatch_from_linear_attention(evidence),
        ))
    }

    fn prepare(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        operation: &BoundSemanticOp,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        let prepared = match operation.descriptor().kind() {
            sllm_core::SemanticOpKind::Qwen38ProjectionPack2 => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let gate_weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let up_weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let gate_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let up_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[1].buffer())?
                    .clone();
                let descriptor = Qwen38ProjectionPack2Descriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    gate_weight.binding(operation.inputs()[1].view().clone()),
                    up_weight.binding(operation.inputs()[2].view().clone()),
                    gate_output.binding(operation.outputs()[0].view().clone()),
                    up_output.binding(operation.outputs()[1].view().clone()),
                )
                .map_err(map_backend_error)?;
                HipPreparedPlan::Qwen38ProjectionPack2(
                    self.backend
                        .prepare_qwen38_projection_pack2(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::GdnProjectionBundle => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let weights: [Result<Buffer, ExecutionError>; 4] = std::array::from_fn(|index| {
                    access
                        .downcast_buffer_payload::<Buffer>(operation.inputs()[index + 1].buffer())
                        .cloned()
                });
                let weights: [Buffer; 4] = weights
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| ExecutionError::InvalidRequest {
                        reason: "GDN weight binding count mismatch".to_owned(),
                    })?;
                let outputs: [Result<Buffer, ExecutionError>; 4] = std::array::from_fn(|index| {
                    access
                        .downcast_buffer_payload::<Buffer>(operation.outputs()[index].buffer())
                        .cloned()
                });
                let outputs: [Buffer; 4] = outputs
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| ExecutionError::InvalidRequest {
                        reason: "GDN output binding count mismatch".to_owned(),
                    })?;
                let descriptor = GdnProjectionBundleDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    std::array::from_fn(|index| {
                        weights[index].binding(operation.inputs()[index + 1].view().clone())
                    }),
                    std::array::from_fn(|index| {
                        outputs[index].binding(operation.outputs()[index].view().clone())
                    }),
                )
                .map_err(map_backend_error)?;
                HipPreparedPlan::GdnProjectionBundle(
                    self.backend
                        .prepare_gdn_projection_bundle(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::MlpGateUpSiluBundle => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let gate_weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let up_weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let gate_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let up_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[1].buffer())?
                    .clone();
                let silu_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[2].buffer())?
                    .clone();
                let descriptor = MlpGateUpSiluBundleDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    gate_weight.binding(operation.inputs()[1].view().clone()),
                    up_weight.binding(operation.inputs()[2].view().clone()),
                    gate_output.binding(operation.outputs()[0].view().clone()),
                    up_output.binding(operation.outputs()[1].view().clone()),
                    silu_output.binding(operation.outputs()[2].view().clone()),
                )
                .map_err(map_backend_error)?;
                HipPreparedPlan::MlpGateUpSiluBundle(
                    self.backend
                        .prepare_mlp_gate_up_silu_bundle(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::RmsNorm => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = RmsNormDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    raw_scale.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::RmsNorm(
                    self.backend
                        .prepare_rms_norm(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::ResidualRmsNorm => {
                let residual = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let addend = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let residual_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[1].buffer())?
                    .clone();
                let descriptor = ResidualRmsNormDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    residual.binding(operation.inputs()[0].view().clone()),
                    addend.binding(operation.inputs()[1].view().clone()),
                    raw_scale.binding(operation.inputs()[2].view().clone()),
                    residual_output.binding(operation.outputs()[0].view().clone()),
                    output.binding(operation.outputs()[1].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::ResidualRmsNorm(
                    self.backend
                        .prepare_residual_rms_norm(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Copy
            | sllm_core::SemanticOpKind::Add
            | sllm_core::SemanticOpKind::BroadcastAdd
            | sllm_core::SemanticOpKind::BroadcastMul
            | sllm_core::SemanticOpKind::ScalarMul
            | sllm_core::SemanticOpKind::SiluMul
            | sllm_core::SemanticOpKind::GeluTanhMul
            | sllm_core::SemanticOpKind::SigmoidMul
            | sllm_core::SemanticOpKind::TanhSoftcap => {
                let mut inputs = Vec::with_capacity(operation.inputs().len());
                for input in operation.inputs() {
                    let buffer = access
                        .downcast_buffer_payload::<Buffer>(input.buffer())?
                        .clone();
                    inputs.push(buffer.binding(input.view().clone()));
                }
                let output_binding = &operation.outputs()[0];
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_binding.buffer())?
                    .clone();
                let descriptor = ElementwiseDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    inputs,
                    output.binding(output_binding.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Elementwise(
                    self.backend
                        .prepare_elementwise(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Embedding => {
                let weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let token_ids = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = EmbeddingDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    weight.binding(operation.inputs()[0].view().clone()),
                    token_ids.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Embedding(
                    self.backend
                        .prepare_embedding(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Matmul => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = MatmulDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    weight.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Matmul(
                    self.backend
                        .prepare_matmul(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Argmax => {
                let logits = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = ArgmaxDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    logits.binding(operation.inputs()[0].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Argmax(
                    self.backend
                        .prepare_argmax(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::TokenSelect => {
                let logits_owned = &operation.inputs()[0];
                let additive_owned = &operation.inputs()[1];
                let mask_owned = &operation.inputs()[2];
                let output_owned = &operation.outputs()[0];
                let logits = access
                    .downcast_buffer_payload::<Buffer>(logits_owned.buffer())?
                    .clone();
                let additive = access
                    .downcast_buffer_payload::<Buffer>(additive_owned.buffer())?
                    .clone();
                let mask = access
                    .downcast_buffer_payload::<Buffer>(mask_owned.buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_owned.buffer())?
                    .clone();
                let contract = operation
                    .descriptor()
                    .token_selector_contract()
                    .ok_or_else(|| ExecutionError::InvalidRequest {
                        reason: "token_select semantic descriptor is missing its contract"
                            .to_owned(),
                    })?;
                let descriptor = TokenSelectorDescriptor::new(
                    logits.binding(logits_owned.view().clone()),
                    additive.binding(additive_owned.view().clone()),
                    mask.binding(mask_owned.view().clone()),
                    output.binding(output_owned.view().clone()),
                    contract.vocab_size(),
                    contract.temperature(),
                    contract.seed(),
                    contract.counter(),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::TokenSelector(
                    self.backend
                        .prepare_token_selector(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::AttentionPreprocess => {
                let packed_q_gate = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let k = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let q_raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let k_raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[3].buffer())?
                    .clone();
                let positions_binding = &operation.inputs()[4];
                let positions = access
                    .downcast_buffer_payload::<Buffer>(positions_binding.buffer())?
                    .clone();
                let q_output_binding = &operation.outputs()[0];
                let gate_output_binding = &operation.outputs()[1];
                let k_output_binding = &operation.outputs()[2];
                let q_output = access
                    .downcast_buffer_payload::<Buffer>(q_output_binding.buffer())?
                    .clone();
                let gate_output = access
                    .downcast_buffer_payload::<Buffer>(gate_output_binding.buffer())?
                    .clone();
                let k_output = access
                    .downcast_buffer_payload::<Buffer>(k_output_binding.buffer())?
                    .clone();
                let descriptor = AttentionPreprocessDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    packed_q_gate.binding(operation.inputs()[0].view().clone()),
                    k.binding(operation.inputs()[1].view().clone()),
                    q_raw_scale.binding(operation.inputs()[2].view().clone()),
                    k_raw_scale.binding(operation.inputs()[3].view().clone()),
                    positions.binding(positions_binding.view().clone()),
                    q_output.binding(q_output_binding.view().clone()),
                    gate_output.binding(gate_output_binding.view().clone()),
                    k_output.binding(k_output_binding.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::AttentionPreprocess(
                    self.backend
                        .prepare_attention_preprocess(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Rotary => {
                let query = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let key = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let positions = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let query_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let key_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[1].buffer())?
                    .clone();
                let descriptor = RotaryDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    query.binding(operation.inputs()[0].view().clone()),
                    key.binding(operation.inputs()[1].view().clone()),
                    positions.binding(operation.inputs()[2].view().clone()),
                    query_output.binding(operation.outputs()[0].view().clone()),
                    key_output.binding(operation.outputs()[1].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Rotary(
                    self.backend
                        .prepare_rotary(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::CausalAttention => {
                let query = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let key = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let value = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = WindowedAttentionDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    query.binding(operation.inputs()[0].view().clone()),
                    key.binding(operation.inputs()[1].view().clone()),
                    value.binding(operation.inputs()[2].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::WindowedAttention(
                    self.backend
                        .prepare_windowed_attention(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::MoeRoute => {
                let logits_owned = &operation.inputs()[0];
                let metadata_owned = &operation.outputs()[0];
                let logits = access
                    .downcast_buffer_payload::<Buffer>(logits_owned.buffer())?
                    .clone();
                let metadata = access
                    .downcast_buffer_payload::<Buffer>(metadata_owned.buffer())?
                    .clone();
                let descriptor = MoeRouteDescriptor::new(
                    logits.binding(logits_owned.view().clone()),
                    metadata.binding(metadata_owned.view().clone()),
                    8,
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::MoeRoute(
                    self.backend
                        .prepare_moe_route(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::DeepSeekV4MoeRoute => {
                let lowering = DeepSeekV4MoeRouteLowering::from_semantic(operation.descriptor())?;
                let logits_owned = &operation.inputs()[0];
                let active_owned = &operation.inputs()[lowering.active_input_index];
                let metadata_owned = &operation.outputs()[0];
                let logits = access
                    .downcast_buffer_payload::<Buffer>(logits_owned.buffer())?
                    .clone();
                let active = access
                    .downcast_buffer_payload::<Buffer>(active_owned.buffer())?
                    .clone();
                let metadata = access
                    .downcast_buffer_payload::<Buffer>(metadata_owned.buffer())?
                    .clone();
                let logits = logits.binding(logits_owned.view().clone());
                let active = active.binding(active_owned.view().clone());
                let metadata = metadata.binding(metadata_owned.view().clone());
                let descriptor = match lowering.mode {
                    sllm_core::DeepSeekV4MoeRouteMode::Score => {
                        DeepSeekV4MoeRouteDescriptor::new_score(
                            logits,
                            active,
                            metadata,
                            lowering.renormalize,
                            lowering.routed_scale(),
                        )
                    }
                    sllm_core::DeepSeekV4MoeRouteMode::Hash => {
                        DeepSeekV4MoeRouteDescriptor::new_hash(
                            logits,
                            active,
                            metadata,
                            lowering.renormalize,
                            lowering.routed_scale(),
                        )
                    }
                }
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::DeepSeekV4MoeRoute(
                    self.backend
                        .prepare_deepseek_v4_moe_route(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::MiniMaxM3MoeRoute => {
                operation
                    .descriptor()
                    .minimax_m3_moe_route_contract()
                    .ok_or_else(|| ExecutionError::InvalidRequest {
                        reason: "minimax_m3_moe_route semantic descriptor is missing its contract"
                            .to_owned(),
                    })?;
                let logits_owned = &operation.inputs()[0];
                let bias_owned = &operation.inputs()[1];
                let metadata_owned = &operation.outputs()[0];
                let logits = access
                    .downcast_buffer_payload::<Buffer>(logits_owned.buffer())?
                    .clone();
                let bias = access
                    .downcast_buffer_payload::<Buffer>(bias_owned.buffer())?
                    .clone();
                let metadata = access
                    .downcast_buffer_payload::<Buffer>(metadata_owned.buffer())?
                    .clone();
                let descriptor = MiniMaxM3MoeRouteDescriptor::new(
                    logits.binding(logits_owned.view().clone()),
                    bias.binding(bias_owned.view().clone()),
                    metadata.binding(metadata_owned.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::MiniMaxM3MoeRoute(
                    self.backend
                        .prepare_minimax_m3_moe_route(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::MoeExpert => {
                let hidden_owned = &operation.inputs()[0];
                let route_owned = &operation.inputs()[1];
                let blob_owned = &operation.inputs()[2];
                let output_owned = &operation.outputs()[0];
                let hidden = access
                    .downcast_buffer_payload::<Buffer>(hidden_owned.buffer())?
                    .clone();
                let route_metadata = access
                    .downcast_buffer_payload::<Buffer>(route_owned.buffer())?
                    .clone();
                let layer_blob = access
                    .downcast_buffer_payload::<Buffer>(blob_owned.buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_owned.buffer())?
                    .clone();
                let token_count = hidden_owned.view().shape()[0] as u64;
                let workspace_bytes =
                    gemma4_moe_expert_workspace_bytes(token_count).ok_or_else(|| {
                        ExecutionError::ExecutionUnavailable {
                            backend: HIP_BACKEND_NAME,
                            reason: "Gemma 4 MoeExpert workspace size overflow".to_owned(),
                        }
                    })?;
                let workspace_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::U8,
                    &[workspace_bytes as usize],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("Gemma 4 MoeExpert workspace layout: {error}"),
                })?;
                let workspace = Buffer::allocate(&self.context, workspace_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let descriptor = MoeExpertDescriptor::new_gemma4(
                    hidden.binding(hidden_owned.view().clone()),
                    route_metadata.binding(route_owned.view().clone()),
                    layer_blob.binding(blob_owned.view().clone()),
                    workspace.binding(workspace_view),
                    output.binding(output_owned.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::MoeExpert(
                    self.backend
                        .prepare_moe_expert(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::SparseMoe => {
                let hidden_owned = &operation.inputs()[0];
                let router_owned = &operation.inputs()[1];
                let blob_owned = &operation.inputs()[2];
                let output_owned = &operation.outputs()[0];
                let hidden = access
                    .downcast_buffer_payload::<Buffer>(hidden_owned.buffer())?
                    .clone();
                let router_weight = access
                    .downcast_buffer_payload::<Buffer>(router_owned.buffer())?
                    .clone();
                let layer_blob = access
                    .downcast_buffer_payload::<Buffer>(blob_owned.buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_owned.buffer())?
                    .clone();
                let token_count = hidden_owned.view().shape()[0] as u64;
                let logits_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::Bf16,
                    &[token_count as usize, 256],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe logits layout: {error}"),
                })?;
                let route_layout =
                    MoeRouteLayout::new(token_count, 256, 8).map_err(map_backend_error)?;
                let route_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::U8,
                    &[route_layout.metadata_bytes as usize],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe route layout: {error}"),
                })?;
                let workspace_bytes = moe_expert_workspace_bytes(token_count).ok_or_else(|| {
                    ExecutionError::ExecutionUnavailable {
                        backend: HIP_BACKEND_NAME,
                        reason: "SparseMoe workspace size overflow".to_owned(),
                    }
                })?;
                let workspace_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::U8,
                    &[workspace_bytes as usize],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe workspace layout: {error}"),
                })?;
                let logits = Buffer::allocate(&self.context, logits_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let route_metadata = Buffer::allocate(&self.context, route_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let workspace = Buffer::allocate(&self.context, workspace_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let matmul = MatmulDescriptor::new(
                    hidden.binding(hidden_owned.view().clone()),
                    router_weight.binding(router_owned.view().clone()),
                    logits.binding(logits_view.clone()),
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe router matmul: {error}"),
                })?;
                let route = MoeRouteDescriptor::new(
                    logits.binding(logits_view),
                    route_metadata.binding(route_view.clone()),
                    8,
                )
                .map_err(map_backend_error)?;
                let expert = MoeExpertDescriptor::new(
                    hidden.binding(hidden_owned.view().clone()),
                    route_metadata.binding(route_view),
                    layer_blob.binding(blob_owned.view().clone()),
                    workspace.binding(workspace_view),
                    output.binding(output_owned.view().clone()),
                )
                .map_err(map_backend_error)?;
                HipPreparedPlan::SparseMoe(PreparedSparseMoe {
                    router: self
                        .backend
                        .prepare_matmul(&self.context, matmul)
                        .map_err(map_backend_error)?,
                    route: self
                        .backend
                        .prepare_moe_route(&self.context, route)
                        .map_err(map_backend_error)?,
                    expert: self
                        .backend
                        .prepare_moe_expert(&self.context, expert)
                        .map_err(map_backend_error)?,
                })
            }
        };
        Ok(AdapterResource::new(prepared))
    }

    fn submit(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        prepared: &PreparedOperation,
        queue: &sllm_core::ExecutionQueue,
    ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError> {
        self.state.ensure_open()?;
        let plan = access
            .downcast_prepared_payload::<HipPreparedPlan>(prepared)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let ticket = self.state.acquire_active()?;
        let (submission, dispatch) = match plan {
            HipPreparedPlan::RmsNorm(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::RmsNorm(submission),
                    dispatch_from_rmsnorm(dispatch),
                )
            }
            HipPreparedPlan::ResidualRmsNorm(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::ResidualRmsNorm(submission),
                    dispatch_from_residual_rmsnorm(dispatch),
                )
            }
            HipPreparedPlan::Elementwise(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Elementwise(submission),
                    dispatch_from_elementwise(dispatch),
                )
            }
            HipPreparedPlan::Embedding(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Embedding(submission),
                    dispatch_from_embedding(dispatch),
                )
            }
            HipPreparedPlan::Matmul(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Matmul(submission),
                    dispatch_from_matmul(dispatch),
                )
            }
            HipPreparedPlan::Qwen38ProjectionPack2(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Qwen38ProjectionPack2(submission),
                    dispatch_from_qwen38_projection_pack2(dispatch),
                )
            }
            HipPreparedPlan::GdnProjectionBundle(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::GdnProjectionBundle(submission),
                    dispatch_from_gdn_projection_bundle(dispatch),
                )
            }
            HipPreparedPlan::MlpGateUpSiluBundle(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::MlpGateUpSiluBundle(submission),
                    dispatch_from_mlp_gate_up_silu_bundle(dispatch),
                )
            }
            HipPreparedPlan::Argmax(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Argmax(submission),
                    dispatch_from_argmax(dispatch),
                )
            }
            HipPreparedPlan::TokenSelector(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::TokenSelector(submission),
                    dispatch_from_token_selector(dispatch),
                )
            }
            HipPreparedPlan::AttentionPreprocess(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::AttentionPreprocess(submission),
                    dispatch_from_attention_preprocess(dispatch),
                )
            }
            HipPreparedPlan::Rotary(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Rotary(submission),
                    dispatch_from_rotary(dispatch),
                )
            }
            HipPreparedPlan::WindowedAttention(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::WindowedAttention(submission),
                    dispatch_from_windowed_attention(dispatch),
                )
            }
            HipPreparedPlan::MoeRoute(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::MoeRoute(submission),
                    dispatch_from_moe_route(dispatch),
                )
            }
            HipPreparedPlan::DeepSeekV4MoeRoute(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::DeepSeekV4MoeRoute(submission),
                    dispatch_from_deepseek_v4_moe_route(dispatch),
                )
            }
            HipPreparedPlan::MiniMaxM3MoeRoute(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::MiniMaxM3MoeRoute(submission),
                    dispatch_from_minimax_m3_moe_route(dispatch),
                )
            }
            HipPreparedPlan::MoeExpert(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::MoeExpert(submission),
                    dispatch_from_moe_expert(dispatch, 0),
                )
            }
            HipPreparedPlan::SparseMoe(plan) => {
                let (router, _) = plan.router.execute(&queue).map_err(map_backend_error)?;
                let (route, _) = plan.route.execute(&queue).map_err(map_backend_error)?;
                let (expert, dispatch) = plan.expert.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::SparseMoe(SparseMoeSubmission {
                        router,
                        route,
                        expert,
                    }),
                    dispatch_from_moe_expert(dispatch, 3),
                )
            }
        };
        Ok((
            Box::new(HipSubmission {
                submission,
                queue,
                _ticket: ticket,
            }),
            dispatch,
        ))
    }

    fn create_graph_span(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        operations: &[PreparedOperation],
    ) -> Result<(AdapterResource, u64), ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let mut plans = Vec::with_capacity(operations.len());
        for operation in operations {
            let prepared = access.downcast_prepared_payload::<HipPreparedPlan>(operation)?;
            plans.push(graph_span_plan(prepared)?);
        }
        let span = GraphSpan::create(&queue, &plans).map_err(map_backend_error)?;
        let node_count = span.node_count();
        Ok((AdapterResource::new(span), node_count))
    }

    fn submit_graph_span(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        span: &sllm_core::ExecutionGraphSpan,
    ) -> Result<Box<dyn ExecutionSubmissionAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let span = access
            .downcast_graph_span_payload::<GraphSpan>(span)?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = match span.execute() {
            Ok(completion) => completion,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok(Box::new(HipGraphSubmission {
            completion,
            span,
            _ticket: ticket,
        }))
    }

    fn upload(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        destination: &BufferRange,
        bytes: Arc<[u8]>,
    ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let buffer = access
            .downcast_buffer_payload::<Buffer>(destination.buffer())?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = queue
            .copy_to_device(&buffer, bytes.as_ref(), destination.offset_bytes())
            .map_err(map_backend_error)?;
        Ok(Box::new(HipTransfer {
            completion,
            _ticket: ticket,
            _buffers: None,
        }))
    }

    fn readback(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        source: &BufferRange,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let buffer = access
            .downcast_buffer_payload::<Buffer>(source.buffer())?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = queue
            .copy_to_host(&buffer, source.size_bytes(), source.offset_bytes())
            .map_err(map_backend_error)?;
        Ok(Box::new(HipReadback {
            completion,
            _ticket: ticket,
        }))
    }

    fn copy_device_to_device(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        source: &BufferRange,
        destination: &BufferRange,
    ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let source_buffer = access
            .downcast_buffer_payload::<Buffer>(source.buffer())?
            .clone();
        let destination_buffer = access
            .downcast_buffer_payload::<Buffer>(destination.buffer())?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = match queue.copy_device_to_device(
            &source_buffer,
            &destination_buffer,
            source.offset_bytes(),
            destination.offset_bytes(),
            source.size_bytes(),
        ) {
            Ok(completion) => completion,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok(Box::new(HipTransfer {
            completion,
            _ticket: ticket,
            _buffers: Some((source_buffer, destination_buffer)),
        }))
    }

    fn shutdown(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        deadline: Duration,
    ) -> Result<ShutdownReport, ExecutionError> {
        self.state.begin_shutdown()?;
        // The native cleanup API is nonblocking and bounded.  The supplied
        // deadline chooses only its bounded retry budget; it never creates a
        // CPU fallback or releases unresolved native ownership speculatively.
        let attempts = usize::try_from(deadline.as_millis())
            .unwrap_or(CLEANUP_ATTEMPT_CAP)
            .clamp(1, CLEANUP_ATTEMPT_CAP);
        let (retryable_cleanup, durable_quarantine) =
            Context::shutdown_cleanup(attempts).map_err(map_backend_error)?;
        if durable_quarantine != 0 || Context::cleanup_accounting_error_count() != 0 {
            return Err(ExecutionError::CleanupQuarantined);
        }
        Ok(ShutdownReport {
            retryable_cleanup,
            durable_quarantine,
        })
    }
}

fn graph_span_plan(plan: &HipPreparedPlan) -> Result<GraphSpanPlan, ExecutionError> {
    match plan {
        HipPreparedPlan::RmsNorm(plan) => Ok(GraphSpanPlan::RmsNorm(plan.clone())),
        HipPreparedPlan::ResidualRmsNorm(plan) => {
            Ok(GraphSpanPlan::ResidualRmsNorm(plan.clone()))
        }
        HipPreparedPlan::Elementwise(plan) => Ok(GraphSpanPlan::Elementwise(plan.clone())),
        HipPreparedPlan::Matmul(plan) => Ok(GraphSpanPlan::Matmul(plan.clone())),
        HipPreparedPlan::Qwen38ProjectionPack2(plan) => {
            Ok(GraphSpanPlan::Qwen38ProjectionPack2(plan.clone()))
        }
        _ => Err(ExecutionError::Unsupported {
            reason: "HIP graph spans currently support only RMSNorm, elementwise, matmul, and Qwen3.8 projection-pack plans".to_owned(),
        }),
    }
}

#[derive(Clone)]
enum HipPreparedPlan {
    RmsNorm(PreparedRmsNorm),
    ResidualRmsNorm(PreparedResidualRmsNorm),
    Elementwise(PreparedElementwise),
    Embedding(PreparedEmbedding),
    Matmul(PreparedMatmul),
    Qwen38ProjectionPack2(PreparedQwen38ProjectionPack2),
    GdnProjectionBundle(PreparedGdnProjectionBundle),
    MlpGateUpSiluBundle(PreparedMlpGateUpSiluBundle),
    Argmax(PreparedArgmax),
    TokenSelector(PreparedTokenSelector),
    AttentionPreprocess(PreparedAttentionPreprocess),
    Rotary(PreparedRotary),
    WindowedAttention(PreparedWindowedAttention),
    MoeRoute(PreparedMoeRoute),
    DeepSeekV4MoeRoute(PreparedDeepSeekV4MoeRoute),
    MiniMaxM3MoeRoute(PreparedMiniMaxM3MoeRoute),
    MoeExpert(PreparedMoeExpert),
    SparseMoe(PreparedSparseMoe),
}

#[derive(Clone)]
struct PreparedSparseMoe {
    router: PreparedMatmul,
    route: PreparedMoeRoute,
    expert: PreparedMoeExpert,
}

enum HipSemanticSubmission {
    RmsNorm(RmsNormSubmission),
    ResidualRmsNorm(ResidualRmsNormSubmission),
    Elementwise(ElementwiseSubmission),
    Embedding(EmbeddingSubmission),
    Matmul(MatmulSubmission),
    Qwen38ProjectionPack2(Qwen38ProjectionPack2Submission),
    GdnProjectionBundle(GdnProjectionBundleSubmission),
    MlpGateUpSiluBundle(MlpGateUpSiluBundleSubmission),
    Argmax(ArgmaxSubmission),
    TokenSelector(TokenSelectorSubmission),
    AttentionPreprocess(AttentionPreprocessSubmission),
    Rotary(RotarySubmission),
    WindowedAttention(WindowedAttentionSubmission),
    MoeRoute(MoeRouteSubmission),
    DeepSeekV4MoeRoute(DeepSeekV4MoeRouteSubmission),
    MiniMaxM3MoeRoute(MiniMaxM3MoeRouteSubmission),
    MoeExpert(MoeExpertSubmission),
    SparseMoe(SparseMoeSubmission),
}

struct SparseMoeSubmission {
    router: MatmulSubmission,
    route: MoeRouteSubmission,
    expert: MoeExpertSubmission,
}

impl HipSemanticSubmission {
    fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.query(),
            Self::ResidualRmsNorm(submission) => submission.query(),
            Self::Elementwise(submission) => submission.query(),
            Self::Embedding(submission) => submission.query(),
            Self::Matmul(submission) => submission.query(),
            Self::Qwen38ProjectionPack2(submission) => submission.query(),
            Self::GdnProjectionBundle(submission) => submission.query(),
            Self::MlpGateUpSiluBundle(submission) => submission.query(),
            Self::Argmax(submission) => submission.query(),
            Self::TokenSelector(submission) => submission.query(),
            Self::AttentionPreprocess(submission) => submission.query(),
            Self::Rotary(submission) => submission.query(),
            Self::WindowedAttention(submission) => submission.query(),
            Self::MoeRoute(submission) => submission.query(),
            Self::DeepSeekV4MoeRoute(submission) => submission.query(),
            Self::MiniMaxM3MoeRoute(submission) => submission.query(),
            Self::MoeExpert(submission) => submission.query(),
            Self::SparseMoe(submission) => submission.expert.query(),
        }
    }

    fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.wait(timeout),
            Self::ResidualRmsNorm(submission) => submission.wait(timeout),
            Self::Elementwise(submission) => submission.wait(timeout),
            Self::Embedding(submission) => submission.wait(timeout),
            Self::Matmul(submission) => submission.wait(timeout),
            Self::Qwen38ProjectionPack2(submission) => submission.wait(timeout),
            Self::GdnProjectionBundle(submission) => submission.wait(timeout),
            Self::MlpGateUpSiluBundle(submission) => submission.wait(timeout),
            Self::Argmax(submission) => submission.wait(timeout),
            Self::TokenSelector(submission) => submission.wait(timeout),
            Self::AttentionPreprocess(submission) => submission.wait(timeout),
            Self::Rotary(submission) => submission.wait(timeout),
            Self::WindowedAttention(submission) => submission.wait(timeout),
            Self::MoeRoute(submission) => submission.wait(timeout),
            Self::DeepSeekV4MoeRoute(submission) => submission.wait(timeout),
            Self::MiniMaxM3MoeRoute(submission) => submission.wait(timeout),
            Self::MoeExpert(submission) => submission.wait(timeout),
            Self::SparseMoe(submission) => submission.expert.wait(timeout),
        }
    }

    fn finalize_after_token(&mut self, fence_token: u64) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.finalize_after_token(fence_token),
            Self::ResidualRmsNorm(submission) => submission.finalize_after_token(fence_token),
            Self::Elementwise(submission) => submission.finalize_after_token(fence_token),
            Self::Embedding(submission) => submission.finalize_after_token(fence_token),
            Self::Matmul(submission) => submission.finalize_after_token(fence_token),
            Self::Qwen38ProjectionPack2(submission) => submission.finalize_after_token(fence_token),
            Self::GdnProjectionBundle(submission) => submission.finalize_after_token(fence_token),
            Self::MlpGateUpSiluBundle(submission) => submission.finalize_after_token(fence_token),
            Self::Argmax(submission) => submission.finalize_after_token(fence_token),
            Self::TokenSelector(submission) => submission.finalize_after_token(fence_token),
            Self::AttentionPreprocess(submission) => submission.finalize_after_token(fence_token),
            Self::Rotary(submission) => submission.finalize_after_token(fence_token),
            Self::WindowedAttention(submission) => submission.finalize_after_token(fence_token),
            Self::MoeRoute(submission) => submission.finalize_after_token(fence_token),
            Self::DeepSeekV4MoeRoute(submission) => submission.finalize_after_token(fence_token),
            Self::MiniMaxM3MoeRoute(submission) => submission.finalize_after_token(fence_token),
            Self::MoeExpert(submission) => submission.finalize_after_token(fence_token),
            Self::SparseMoe(submission) => {
                submission.router.finalize_after_token(fence_token)?;
                submission.route.finalize_after_token(fence_token)?;
                submission.expert.finalize_after_token(fence_token)
            }
        }
    }

    fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.kernel_elapsed_ns(),
            Self::ResidualRmsNorm(submission) => submission.kernel_elapsed_ns(),
            Self::Elementwise(submission) => submission.kernel_elapsed_ns(),
            Self::Embedding(submission) => submission.kernel_elapsed_ns(),
            Self::Matmul(submission) => submission.kernel_elapsed_ns(),
            Self::Qwen38ProjectionPack2(submission) => submission.kernel_elapsed_ns(),
            Self::GdnProjectionBundle(submission) => submission.kernel_elapsed_ns(),
            Self::MlpGateUpSiluBundle(submission) => submission.kernel_elapsed_ns(),
            Self::Argmax(submission) => submission.kernel_elapsed_ns(),
            Self::TokenSelector(submission) => submission.kernel_elapsed_ns(),
            Self::AttentionPreprocess(submission) => submission.kernel_elapsed_ns(),
            Self::Rotary(submission) => submission.kernel_elapsed_ns(),
            Self::WindowedAttention(submission) => submission.kernel_elapsed_ns(),
            Self::MoeRoute(submission) => submission.kernel_elapsed_ns(),
            Self::DeepSeekV4MoeRoute(submission) => submission.kernel_elapsed_ns(),
            Self::MiniMaxM3MoeRoute(submission) => submission.kernel_elapsed_ns(),
            Self::MoeExpert(submission) => submission.kernel_elapsed_ns(),
            Self::SparseMoe(submission) => {
                let expert = submission.expert.kernel_elapsed_ns()?;
                let route = submission.route.kernel_elapsed_ns()?;
                let router = submission.router.kernel_elapsed_ns()?;
                Ok(router + route + expert)
            }
        }
    }
}

struct HipSubmission {
    submission: HipSemanticSubmission,
    queue: Queue,
    _ticket: ActiveOperation,
}

impl ExecutionSubmissionAdapter for HipSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        self.submission
            .kernel_elapsed_ns()
            .map(Some)
            .map_err(map_async_error)
    }

    fn start_output_readback(
        &mut self,
        access: &ExecutionAdapterAccess<'_>,
        output: &OwnedTensorBinding,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        let buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let ticket = self._ticket.state.acquire_active()?;
        let completion = self
            .queue
            .copy_to_host(
                &buffer,
                output.view().payload_bytes(),
                output.view().byte_offset(),
            )
            .map_err(map_backend_error)?;
        Ok(Box::new(HipReadback {
            completion,
            _ticket: ticket,
        }))
    }
}

struct HipGraphSubmission {
    completion: Completion,
    span: GraphSpan,
    _ticket: ActiveOperation,
}

impl ExecutionSubmissionAdapter for HipGraphSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        Ok(None)
    }

    fn start_output_readback(
        &mut self,
        access: &ExecutionAdapterAccess<'_>,
        output: &OwnedTensorBinding,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        let buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let ticket = self._ticket.state.acquire_active()?;
        let completion = self
            .span
            .queue()
            .copy_to_host(
                &buffer,
                output.view().payload_bytes(),
                output.view().byte_offset(),
            )
            .map_err(map_backend_error)?;
        Ok(Box::new(HipReadback {
            completion,
            _ticket: ticket,
        }))
    }
}

struct HipTransfer {
    completion: Completion,
    _ticket: ActiveOperation,
    _buffers: Option<(Buffer, Buffer)>,
}

struct HipQueueFence {
    completion: Completion,
    _ticket: ActiveOperation,
}

impl ExecutionQueueFenceAdapter for HipQueueFence {
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn token(&self) -> Result<u64, ExecutionError> {
        self.completion.opaque_token().map_err(map_backend_error)
    }
}

struct HipKvSubmission {
    completion: KvAppendCompletion,
    _ticket: ActiveOperation,
}

struct HipCausalAttentionSubmission {
    completion: CausalAttentionCompletion,
    _evidence: CausalAttentionEvidence,
    _ticket: ActiveOperation,
}

struct HipMinistral3YarnSubmissionAdapter {
    completion: HipMinistral3YarnSubmission,
    _ticket: ActiveOperation,
}

struct HipLinearAttentionSubmission {
    completion: LinearAttentionCompletion,
    _evidence: LinearAttentionEvidence,
    _ticket: ActiveOperation,
}

impl ExecutionLinearAttentionSubmissionAdapter for HipLinearAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

impl ExecutionCausalAttentionSubmissionAdapter for HipCausalAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        self.completion
            .kernel_elapsed_ns()
            .map(Some)
            .map_err(map_async_error)
    }
}

impl ExecutionMinistral3YarnSubmissionAdapter for HipMinistral3YarnSubmissionAdapter {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        self.completion
            .kernel_elapsed_ns()
            .map(Some)
            .map_err(map_async_error)
    }
}

impl ExecutionKvStateSubmissionAdapter for HipKvSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn execute_causal_attention_after_kv_append(
        &mut self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        queue: &sllm_core::ExecutionQueue,
        query: &OwnedTensorBinding,
        output: &OwnedTensorBinding,
        descriptor: CausalAttentionDescriptor,
    ) -> Result<
        (
            Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        let state_resource = access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let query_buffer = access
            .downcast_buffer_payload::<Buffer>(query.buffer())?
            .clone();
        let output_buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let query = query_buffer.binding(query.view().clone());
        let output = output_buffer.binding(output.view().clone());
        let ticket = self._ticket.clone_active()?;
        let (completion, evidence) = match state_resource.causal_attention_after_kv_append(
            &queue,
            &self.completion,
            &query,
            &output,
            descriptor.start_position(),
            descriptor.expected_kv_length(),
            descriptor.sliding_window(),
            descriptor.score_scale(),
        ) {
            Ok(value) => value,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipCausalAttentionSubmission {
                completion,
                _evidence: evidence.clone(),
                _ticket: ticket,
            }),
            dispatch_from_causal_attention(evidence),
        ))
    }
}

impl ExecutionTransferAdapter for HipTransfer {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

struct HipReadback {
    completion: Completion,
    _ticket: ActiveOperation,
}

impl ExecutionReadbackAdapter for HipReadback {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
        self.completion
            .read_into(destination)
            .map_err(map_async_error)
    }
}

fn map_completion_state(state: CompletionState) -> ExecutionState {
    match state {
        CompletionState::Pending => ExecutionState::Pending,
        CompletionState::Success => ExecutionState::Success,
        CompletionState::Failure => ExecutionState::Failure,
    }
}

fn map_backend_error(error: RuntimeError) -> ExecutionError {
    match error.status() {
        RuntimeStatus::HipUnavailable => ExecutionError::ExecutionUnavailable {
            backend: HIP_BACKEND_NAME,
            reason: error.message().to_owned(),
        },
        RuntimeStatus::Busy
        | RuntimeStatus::CausalAttentionStateBusy
        | RuntimeStatus::LinearAttentionStateBusy => ExecutionError::Busy,
        RuntimeStatus::NotReady => ExecutionError::NotReady,
        _ => ExecutionError::BackendStatus {
            status: error.status().raw(),
            diagnostic: error.message().to_owned(),
        },
    }
}

fn map_async_error(error: RuntimeError) -> ExecutionError {
    match error.status() {
        RuntimeStatus::Busy => ExecutionError::Busy,
        RuntimeStatus::NotReady => ExecutionError::NotReady,
        _ => ExecutionError::AsyncFailure {
            status: error.status().raw(),
            diagnostic: error.message().to_owned(),
        },
    }
}

fn logical_dispatch_target(target: String) -> String {
    logical_gcn_arch_name(&target).to_owned()
}

fn dispatch_from_rmsnorm(dispatch: RmsNormDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.row_count,
        normalized_size: dispatch.normalized_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_residual_rmsnorm(dispatch: ResidualRmsNormDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.row_count,
        normalized_size: dispatch.normalized_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_elementwise(dispatch: ElementwiseDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: 1,
        normalized_size: dispatch.element_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_embedding(dispatch: EmbeddingDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.hidden_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_matmul(dispatch: MatmulDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: dispatch.output_elements,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_qwen38_projection_pack2(
    dispatch: Qwen38ProjectionPack2DispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: dispatch.output_elements,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_gdn_projection_bundle(
    dispatch: GdnProjectionBundleDispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sllm_hip_sys::SLLM_HIP_ABI_VERSION,
        info_version: sllm_hip_sys::SLLM_HIP_GDN_PROJECTION_BUNDLE_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: 0,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: dispatch.widths.iter().map(|w| *w as u64).sum(),
        backend: 1,
        fallback_allowed: false,
        fallback_used: dispatch.fallback_used != 0,
        kernel_symbol: "sllm_gdn_projection_bundle_bf16_fp32_decode_v1".to_owned(),
        device_symbol: "sllm_gdn_projection_bundle_bf16_fp32_decode_v1".to_owned(),
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_mlp_gate_up_silu_bundle(
    dispatch: MlpGateUpSiluBundleDispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sllm_hip_sys::SLLM_HIP_ABI_VERSION,
        info_version: sllm_hip_sys::SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: dispatch.n * 3,
        backend: 1,
        fallback_allowed: false,
        fallback_used: dispatch.fallback_used != 0,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_argmax(dispatch: ArgmaxDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.row_count,
        normalized_size: dispatch.vocab_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_token_selector(dispatch: TokenSelectorDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: 1,
        normalized_size: dispatch.vocab_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_attention_preprocess(
    dispatch: AttentionPreprocessDispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: 256,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_rotary(dispatch: RotaryDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: u64::from(dispatch.head_dim),
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_windowed_attention(dispatch: WindowedAttentionDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.query_count,
        normalized_size: u64::from(dispatch.head_dim),
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_moe_route(dispatch: MoeRouteDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: sys::SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE,
        grid_size_x: dispatch.token_count as u32,
        row_count: dispatch.token_count,
        normalized_size: dispatch.pair_count,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: "moe_route.bf16.stable_topk_group.v1".to_owned(),
        device_symbol: "sllm_moe_route_stable_topk_group_v1".to_owned(),
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_deepseek_v4_moe_route(
    dispatch: DeepSeekV4MoeRouteDispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.pair_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_minimax_m3_moe_route(dispatch: MiniMaxM3MoeRouteDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.pair_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_ministral3_yarn(dispatch: Ministral3YarnDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: u64::from(dispatch.head_dim),
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_moe_expert(
    dispatch: MoeExpertDispatchInfo,
    prior_dispatch_count: u32,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count + prior_dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.active_pair_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_kv_append(dispatch: KvAppendEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_APPEND_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: 4 * 256,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

fn dispatch_from_causal_attention(dispatch: CausalAttentionEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.query_count,
        normalized_size: dispatch.head_dim as u64,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

fn dispatch_from_linear_attention(dispatch: LinearAttentionEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.recurrent_kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.recurrent_grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: sys::SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM as u64,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: false,
        fallback_used: false,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.recurrent_device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{
        AttentionPreprocessContract, AttentionPreprocessPositionMode, Backend, DType,
        DeepSeekV4MoeRouteContractV1, DeepSeekV4MoeRouteMode as CoreDeepSeekV4MoeRouteMode,
        ExecutionSessionRequest, MiniMaxM3MoeRouteContractV1, SemanticOpDescriptor,
        SplitHalfRotaryContract, TensorView, TokenSelectorContractV1,
        WindowedCausalAttentionContract,
    };

    #[test]
    fn mi300x_feature_tuple_has_one_fail_closed_logical_normalization() {
        assert_eq!(
            logical_dispatch_target("gfx942:sramecc+:xnack-".to_owned()),
            "gfx942"
        );
        assert_eq!(
            logical_dispatch_target("gfx942:sramecc+:xnack+".to_owned()),
            "gfx942:sramecc+:xnack+"
        );
    }
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn generic_backend_bridge_is_additive_and_stub_stays_unavailable() {
        let backend = HipBackend { _private: () };
        let request = ExecutionSessionRequest::new(0, "gfx1201").unwrap();
        assert!(matches!(
            backend.open_execution_session(request),
            Err(ExecutionError::ExecutionUnavailable { .. })
        ));
        assert!(!backend.capabilities().numerical_execution);
    }

    #[test]
    fn owned_bridge_advertises_the_existing_public_transfer_limit_without_gpu() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        assert_eq!(
            adapter.max_transfer_bytes(),
            crate::sys::SLLM_HIP_MAX_TRANSFER_BYTES
        );
        assert_eq!(adapter.max_transfer_bytes(), 1_073_741_824);
    }

    #[test]
    fn closing_rejects_new_active_operations_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        assert_eq!(state.begin_shutdown(), Ok(()));
        assert!(matches!(
            state.acquire_active(),
            Err(ExecutionError::Closing)
        ));
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn active_operation_ticket_changes_count_once_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        assert_eq!(state.active_count(), 0);
        let ticket = state.acquire_active().expect("open state accepts ticket");
        assert_eq!(state.active_count(), 1);
        drop(ticket);
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn shutdown_cas_wins_over_a_paused_admission_without_count_corruption() {
        let state = Arc::new(HipSessionState::new());
        let reached = Arc::new(Barrier::new(2));
        let proceed = Arc::new(Barrier::new(2));
        state
            .activity
            .pause_next_admission(Arc::clone(&reached), Arc::clone(&proceed));

        let admission_state = Arc::clone(&state);
        let admission_thread = thread::spawn(move || admission_state.acquire_active().map(|_| ()));
        reached.wait();

        assert_eq!(state.begin_shutdown(), Ok(()));
        proceed.wait();

        assert_eq!(
            admission_thread.join().expect("admission thread completed"),
            Err(ExecutionError::Closing)
        );
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn closing_with_live_active_operation_is_busy_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        let ticket = state.acquire_active().expect("open state accepts ticket");
        assert_eq!(state.active_count(), 1);
        assert_eq!(state.begin_shutdown(), Err(ExecutionError::Busy));
        assert_eq!(state.active_count(), 1);
        assert!(matches!(
            state.acquire_active(),
            Err(ExecutionError::Closing)
        ));
        drop(ticket);
        assert_eq!(state.active_count(), 0);
        assert_eq!(state.begin_shutdown(), Ok(()));
    }

    fn attention_descriptor() -> SemanticOpDescriptor {
        let contract = AttentionPreprocessContract::new_qwen3_5(
            AttentionPreprocessPositionMode::DecodeContinuation,
            3,
            17,
        )
        .expect("valid attention preprocess contract");
        SemanticOpDescriptor::new_attention_preprocess(
            vec![
                TensorView::contiguous(DType::Bf16, &[17, 16, 512]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 4, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[4, 256]).unwrap(),
                TensorView::contiguous(DType::I32, &[17]).unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[17, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 4, 256]).unwrap(),
            ],
            contract,
        )
        .expect("valid attention preprocess descriptor")
    }

    #[test]
    fn supports_attention_preprocess_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        assert_eq!(
            adapter.supports(&attention_descriptor()),
            PrepareSupport::Supported
        );
    }

    #[test]
    fn supports_split_half_rotary_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        let contract = SplitHalfRotaryContract::new(3, 1, 6, 4, 10_000.0, 255, 3, 262_144)
            .expect("valid rotary contract");
        let descriptor = SemanticOpDescriptor::new_rotary(
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap(),
                TensorView::contiguous(DType::I32, &[3]).unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap(),
            ],
            contract,
        )
        .expect("valid rotary descriptor");
        assert_eq!(adapter.supports(&descriptor), PrepareSupport::Supported);
    }

    #[test]
    fn supports_windowed_attention_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        let contract = WindowedCausalAttentionContract::new(3, 1, 6, 2, 3, 5, Some(4), 1.0)
            .expect("valid windowed attention contract");
        let descriptor = SemanticOpDescriptor::new_causal_attention(
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()],
            contract,
        )
        .expect("valid windowed attention descriptor");
        assert_eq!(adapter.supports(&descriptor), PrepareSupport::Supported);
    }

    #[test]
    fn supports_token_selector_and_maps_dispatch_evidence() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        let contract = TokenSelectorContractV1::new(257, 0.75, 7, 11).unwrap();
        let descriptor = SemanticOpDescriptor::new_token_select(
            vec![
                TensorView::contiguous(DType::Bf16, &[1, 257]).unwrap(),
                TensorView::contiguous(DType::F32, &[1, 257]).unwrap(),
                TensorView::contiguous(DType::U8, &[1, 257]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::U8, &[16]).unwrap()],
            contract,
        )
        .expect("valid token selector descriptor");
        assert_eq!(adapter.supports(&descriptor), PrepareSupport::Supported);

        let dispatch = dispatch_from_token_selector(TokenSelectorDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 9,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 2,
            vocab_size: 257,
            fallback_allowed: false,
            fallback_used: false,
            result_status: 0,
            token_id: 3,
            backend: sys::SLLM_BACKEND_HIP,
            kernel_symbol: "token_selector.bf16_f32_mask.v1".to_owned(),
            device_symbol: "sllm_token_selector_bf16_f32_mask_v1".to_owned(),
            gcn_arch_name: "gfx942".to_owned(),
        });
        assert_eq!(dispatch.normalized_size, 257);
        assert_eq!(dispatch.row_count, 1);
        assert_eq!(dispatch.target, "gfx942");
    }

    #[test]
    fn supports_separate_gemma_route_expert_and_broadcast_mul_semantics() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        let route = SemanticOpDescriptor::new(
            sllm_core::SemanticOpKind::MoeRoute,
            vec![TensorView::contiguous(DType::Bf16, &[3, 128]).unwrap()],
            vec![TensorView::contiguous(DType::U8, &[1_416]).unwrap()],
        )
        .unwrap();
        let expert = SemanticOpDescriptor::new(
            sllm_core::SemanticOpKind::MoeExpert,
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 2_816]).unwrap(),
                TensorView::contiguous(DType::U8, &[1_416]).unwrap(),
                TensorView::contiguous(DType::U8, &[428_215_552]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[3, 2_816]).unwrap()],
        )
        .unwrap();
        let broadcast_mul = SemanticOpDescriptor::new(
            sllm_core::SemanticOpKind::BroadcastMul,
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 2_816]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[2_816]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[3, 2_816]).unwrap()],
        )
        .unwrap();
        assert_eq!(adapter.supports(&route), PrepareSupport::Supported);
        assert_eq!(adapter.supports(&expert), PrepareSupport::Supported);
        assert_eq!(adapter.supports(&broadcast_mul), PrepareSupport::Supported);
    }

    fn deepseek_v4_route_descriptor(
        mode: CoreDeepSeekV4MoeRouteMode,
        renormalize: bool,
        routed_scale: f32,
    ) -> SemanticOpDescriptor {
        let (bias, hash_ids) = match mode {
            CoreDeepSeekV4MoeRouteMode::Score => (
                TensorView::contiguous(DType::F32, &[256]).unwrap(),
                TensorView::contiguous(DType::I32, &[0, 6]).unwrap(),
            ),
            CoreDeepSeekV4MoeRouteMode::Hash => (
                TensorView::contiguous(DType::F32, &[0]).unwrap(),
                TensorView::contiguous(DType::I32, &[3, 6]).unwrap(),
            ),
        };
        SemanticOpDescriptor::new_deepseek_v4_moe_route(
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 256]).unwrap(),
                bias,
                hash_ids,
            ],
            vec![TensorView::contiguous(DType::U8, &[2_344]).unwrap()],
            DeepSeekV4MoeRouteContractV1::new(mode, renormalize, routed_scale).unwrap(),
        )
        .expect("valid DeepSeek V4 route descriptor")
    }

    #[test]
    fn deepseek_v4_score_and_hash_m3_lowering_select_only_the_live_input() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
        };
        let score = deepseek_v4_route_descriptor(CoreDeepSeekV4MoeRouteMode::Score, true, 1.5);
        let hash = deepseek_v4_route_descriptor(CoreDeepSeekV4MoeRouteMode::Hash, false, 1.25);

        let score_lowering = DeepSeekV4MoeRouteLowering::from_semantic(&score).unwrap();
        assert_eq!(score_lowering.mode, CoreDeepSeekV4MoeRouteMode::Score);
        assert_eq!(score_lowering.active_input_index, 1);
        assert!(score_lowering.renormalize);
        assert_eq!(score_lowering.routed_scale().to_bits(), 1.5_f32.to_bits());

        let hash_lowering = DeepSeekV4MoeRouteLowering::from_semantic(&hash).unwrap();
        assert_eq!(hash_lowering.mode, CoreDeepSeekV4MoeRouteMode::Hash);
        assert_eq!(hash_lowering.active_input_index, 2);
        assert!(!hash_lowering.renormalize);
        assert_eq!(hash_lowering.routed_scale().to_bits(), 1.25_f32.to_bits());

        assert_eq!(adapter.supports(&score), PrepareSupport::Supported);
        assert_eq!(adapter.supports(&hash), PrepareSupport::Supported);
    }

    #[test]
    fn deepseek_v4_route_missing_or_mode_inconsistent_contract_is_rejected_by_core() {
        let score_inputs = vec![
            TensorView::contiguous(DType::Bf16, &[3, 256]).unwrap(),
            TensorView::contiguous(DType::F32, &[256]).unwrap(),
            TensorView::contiguous(DType::I32, &[0, 6]).unwrap(),
        ];
        let output = vec![TensorView::contiguous(DType::U8, &[2_344]).unwrap()];
        assert!(
            SemanticOpDescriptor::new(
                sllm_core::SemanticOpKind::DeepSeekV4MoeRoute,
                score_inputs.clone(),
                output.clone(),
            )
            .is_err()
        );

        let hash_contract =
            DeepSeekV4MoeRouteContractV1::new(CoreDeepSeekV4MoeRouteMode::Hash, true, 1.5).unwrap();
        assert!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(score_inputs, output, hash_contract,)
                .is_err()
        );
    }

    #[test]
    fn deepseek_v4_dispatch_evidence_preserves_dedicated_kernel_and_no_fallback() {
        let dispatch = dispatch_from_deepseek_v4_moe_route(DeepSeekV4MoeRouteDispatchInfo {
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION,
            backend: sys::SLLM_BACKEND_HIP,
            dispatch_id: 57,
            dispatch_count: 2,
            kernel_id: sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1,
            workgroup_size_x: sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE,
            grid_size_x: 3,
            token_count: 3,
            expert_count: 256,
            pair_count: 18,
            selected_expert_count: 6,
            mode: crate::DeepSeekV4MoeRouteMode::Hash,
            renormalize: true,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "deepseek_v4_moe_route.bf16_f32.hash.v1".to_owned(),
            device_symbol: "sllm_deepseek_v4_moe_route_score_hash_v1".to_owned(),
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(
            dispatch.kernel_id,
            sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1
        );
        assert_eq!(dispatch.dispatch_count, 2);
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 18);
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
        assert_eq!(dispatch.target, "gfx1201");
        assert_eq!(
            dispatch.kernel_symbol,
            "deepseek_v4_moe_route.bf16_f32.hash.v1"
        );
    }

    #[test]
    fn minimax_m3_fixed_descriptor_and_missing_contract_are_distinct() {
        let inputs = vec![
            TensorView::contiguous(DType::F32, &[3, 128]).unwrap(),
            TensorView::contiguous(DType::F32, &[128]).unwrap(),
        ];
        let outputs = vec![TensorView::contiguous(DType::U8, &[1_224]).unwrap()];
        assert!(
            SemanticOpDescriptor::new(
                sllm_core::SemanticOpKind::MiniMaxM3MoeRoute,
                inputs.clone(),
                outputs.clone(),
            )
            .is_err()
        );
        let descriptor = SemanticOpDescriptor::new_minimax_m3_moe_route(
            inputs,
            outputs,
            MiniMaxM3MoeRouteContractV1::new(),
        )
        .unwrap();
        assert_eq!(
            descriptor.kind(),
            sllm_core::SemanticOpKind::MiniMaxM3MoeRoute
        );
        assert_eq!(
            descriptor.minimax_m3_moe_route_contract(),
            Some(MiniMaxM3MoeRouteContractV1::new())
        );
    }

    #[test]
    fn minimax_m3_dispatch_evidence_preserves_exact_kernel_and_no_fallback() {
        let evidence = dispatch_from_minimax_m3_moe_route(MiniMaxM3MoeRouteDispatchInfo {
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION,
            backend: sys::SLLM_BACKEND_HIP,
            dispatch_id: 58,
            dispatch_count: 2,
            kernel_id: sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_ID_SIGMOID_TOP4_V1,
            workgroup_size_x: sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_WORKGROUP_SIZE,
            grid_size_x: 3,
            token_count: 3,
            expert_count: 128,
            pair_count: 12,
            selected_expert_count: 4,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "sllm.minimax_m3_moe_route.sigmoid_top4.v1".to_owned(),
            device_symbol: "sllm_minimax_m3_moe_route_sigmoid_top4_v1".to_owned(),
            gcn_arch_name: "gfx1030".to_owned(),
        });
        assert_eq!(evidence.dispatch_id, 58);
        assert_eq!(evidence.kernel_id, 1);
        assert_eq!(evidence.row_count, 3);
        assert_eq!(evidence.normalized_size, 12);
        assert_eq!(evidence.target, "gfx1030");
        assert!(!evidence.fallback_allowed);
        assert!(!evidence.fallback_used);
    }

    #[test]
    fn deepseek_v4_semantic_failure_preserves_status_and_diagnostic_at_bridge_boundary() {
        let error = map_async_error(RuntimeError::local(
            RuntimeStatus::InvalidArgument,
            "DeepSeek V4 route rejected duplicate hash expert ids",
        ));
        assert_eq!(
            error,
            ExecutionError::AsyncFailure {
                status: sys::SLLM_STATUS_INVALID_ARGUMENT,
                diagnostic: "DeepSeek V4 route rejected duplicate hash expert ids".to_owned(),
            }
        );
    }

    #[test]
    fn separate_gemma_route_and_expert_dispatches_keep_individual_counts() {
        let route = dispatch_from_moe_route(MoeRouteDispatchInfo {
            dispatch_id: 41,
            dispatch_count: 3,
            kernel_id: sys::SLLM_HIP_MOE_ROUTE_KERNEL_ID_STABLE_TOPK_V1,
            token_count: 3,
            expert_count: 128,
            pair_count: 24,
            selected_expert_count: 8,
            fallback_allowed: false,
            fallback_used: false,
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(route.dispatch_count, 3);
        assert_eq!(route.row_count, 3);
        assert_eq!(route.normalized_size, 24);
        assert_eq!(route.target, "gfx1201");

        let expert_dispatch = MoeExpertDispatchInfo {
            abi_version: 1,
            info_version: 1,
            backend: sys::SLLM_BACKEND_HIP,
            dispatch_id: 42,
            dispatch_count: 4,
            kernel_id: sys::SLLM_HIP_MOE_EXPERT_KERNEL_ID_GEMMA4_PREFILL_V2,
            workgroup_size_x: 256,
            grid_size_x: 24,
            token_count: 3,
            active_pair_count: 24,
            workspace_bytes: 81_312,
            selected_expert_count: 8,
            shared_expert_count: 0,
            fallback_allowed: false,
            fallback_used: false,
            gcn_arch_name: "gfx1201".to_owned(),
            kernel_symbol: "moe_expert.gemma4.nvfp4.prefill.active_pairs.v2".to_owned(),
            device_symbol: "sllm_moe_expert_gemma4_nvfp4_active_pairs_v2".to_owned(),
        };
        let standalone = dispatch_from_moe_expert(expert_dispatch.clone(), 0);
        let qwen_combined = dispatch_from_moe_expert(expert_dispatch, 3);
        assert_eq!(standalone.dispatch_count, 4);
        assert_eq!(qwen_combined.dispatch_count, 7);
        assert_eq!(standalone.row_count, 3);
        assert_eq!(standalone.normalized_size, 24);
    }

    #[test]
    fn rotary_dispatch_mapping_preserves_non_aligned_shape_and_target() {
        let dispatch = dispatch_from_rotary(RotaryDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 12,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 1,
            token_count: 3,
            q_heads: 3,
            kv_heads: 1,
            head_dim: 6,
            rotary_dim: 4,
            start_position: 255,
            max_position: 262_144,
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "rotary.split_half.bf16_fp32.v1".to_owned(),
            device_symbol: "sllm_rotary_split_half_bf16_fp32_v1".to_owned(),
            gcn_arch_name: "gfx1030".to_owned(),
        });
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 6);
        assert_eq!(dispatch.target, "gfx1030");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
    }

    #[test]
    fn ministral3_yarn_dispatch_mapping_preserves_fixed_head_geometry() {
        let dispatch = dispatch_from_ministral3_yarn(Ministral3YarnDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 60,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 4,
            token_count: 3,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            rotary_dim: 128,
            start_position: 16_383,
            max_position: 262_144,
            backend: sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "ministral3_yarn.bf16.split_half.qscale.v1".to_owned(),
            device_symbol: "sllm_ministral3_yarn_bf16_split_half_qscale_v1".to_owned(),
            gcn_arch_name: "gfx1030".to_owned(),
        });
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 128);
        assert_eq!(dispatch.target, "gfx1030");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
    }

    #[test]
    fn windowed_attention_dispatch_mapping_preserves_shape_window_and_target() {
        let dispatch = dispatch_from_windowed_attention(WindowedAttentionDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 13,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 1,
            query_count: 3,
            start_position: 2,
            committed_kv_length: 5,
            sliding_window: 4,
            q_heads: 3,
            kv_heads: 1,
            head_dim: 6,
            scaling_bits: 1.0_f32.to_bits(),
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "attention.windowed.bf16_fp32.v1".to_owned(),
            device_symbol: "sllm_gemma_attention_bf16_fp32_v1".to_owned(),
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 6);
        assert_eq!(dispatch.target, "gfx1201");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
    }

    #[test]
    fn attention_dispatch_mapping_uses_m_rows_and_fixed_256_normalized_size() {
        let dispatch = dispatch_from_attention_preprocess(AttentionPreprocessDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 11,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 1,
            grid_size_x: 340,
            m: 17,
            q_heads: 16,
            k_heads: 4,
            q_head_dim: 256,
            k_head_dim: 256,
            rotary_dim: 64,
            start_position: 255,
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "attention_preprocess.headwise_norm_rope.v1".to_owned(),
            device_symbol: "sllm_attention_preprocess_headwise_norm_rope_v1".to_owned(),
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(dispatch.row_count, 17);
        assert_eq!(dispatch.normalized_size, 256);
        assert_eq!(dispatch.target, "gfx1201");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
        assert_eq!(
            dispatch.kernel_symbol,
            "attention_preprocess.headwise_norm_rope.v1"
        );
    }
}
