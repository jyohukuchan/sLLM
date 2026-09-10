//! Request-owned device sampling storage shared by model execution adapters.

use std::sync::Arc;
use std::time::Duration;

use crate::prepared_execution::require_terminal_success;
use crate::{
    AccessMode, AllocationCategory, DType, DeviceTokenSelectorRequestV1, ExecutionBuffer,
    ExecutionError, ExecutionQueue, ExecutionSession, ExecutionState, OwnedTensorBinding,
    SamplingSelectionV1, SemanticOpDescriptor, TensorView, TokenSelectorContractV1,
};

pub(crate) const FIXED_K20_SUPPORT_BYTES: u64 = 256;
pub(crate) const FIXED_K20_DECISION_BYTES: u64 = 144;
const MAX_FIXED_K20_SUPPORT_SLOTS: usize = 9;

pub(crate) struct DeviceSamplingBuffers {
    pub(crate) additive: ExecutionBuffer,
    pub(crate) mask: ExecutionBuffer,
    pub(crate) output: ExecutionBuffer,
    pub(crate) additive_view: TensorView,
    pub(crate) mask_view: TensorView,
    pub(crate) output_view: TensorView,
    vocab: usize,
    last_additive: Vec<f32>,
    last_mask: Vec<u8>,
    workspace: Option<(ExecutionBuffer, TensorView)>,
    fixed_k20_support_slots: Option<(ExecutionBuffer, usize)>,
}

pub(crate) struct PreparedDeviceSampling {
    pub(crate) descriptor: SemanticOpDescriptor,
    pub(crate) inputs: Vec<OwnedTensorBinding>,
    pub(crate) outputs: Vec<OwnedTensorBinding>,
}

fn invalid(reason: impl ToString) -> ExecutionError {
    ExecutionError::InvalidRequest {
        reason: reason.to_string(),
    }
}

impl DeviceSamplingBuffers {
    pub(crate) fn new(session: &ExecutionSession, vocab: usize) -> Result<Self, ExecutionError> {
        let additive_view = TensorView::contiguous(DType::F32, &[1, vocab]).map_err(invalid)?;
        let mask_view = TensorView::contiguous(DType::U8, &[1, vocab]).map_err(invalid)?;
        let output_view = TensorView::contiguous(DType::U8, &[16]).map_err(invalid)?;
        Ok(Self {
            additive: session.allocate_with_category(
                additive_view.payload_bytes(),
                AllocationCategory::RequestState,
            )?,
            mask: session.allocate_with_category(
                mask_view.payload_bytes(),
                AllocationCategory::RequestState,
            )?,
            output: session.allocate_with_category(
                output_view.payload_bytes(),
                AllocationCategory::RequestState,
            )?,
            additive_view,
            mask_view,
            output_view,
            vocab,
            last_additive: Vec::new(),
            last_mask: Vec::new(),
            workspace: None,
            fixed_k20_support_slots: None,
        })
    }

    pub(crate) fn ensure_fixed_k20_support_slots(
        &mut self,
        session: &ExecutionSession,
        rows: usize,
    ) -> Result<(), ExecutionError> {
        if !(1..=MAX_FIXED_K20_SUPPORT_SLOTS).contains(&rows) {
            return Err(invalid(format!(
                "fixed-K20 support capture requires 1 through {MAX_FIXED_K20_SUPPORT_SLOTS} rows"
            )));
        }
        if self
            .fixed_k20_support_slots
            .as_ref()
            .is_some_and(|(_, capacity)| *capacity >= rows)
        {
            return Ok(());
        }
        let bytes = u64::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(FIXED_K20_SUPPORT_BYTES))
            .ok_or_else(|| invalid("fixed-K20 support slot allocation overflowed"))?;
        let buffer = session.allocate_with_category(bytes, AllocationCategory::RequestState)?;
        self.fixed_k20_support_slots = Some((buffer, rows));
        Ok(())
    }

    pub(crate) fn capture_fixed_k20_support_row(
        &mut self,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        row: usize,
        timeout: Duration,
    ) -> Result<(), ExecutionError> {
        let (workspace, workspace_view) = self
            .workspace
            .as_ref()
            .ok_or_else(|| invalid("fixed-K20 support capture has no selector workspace"))?;
        if workspace_view.payload_bytes() < FIXED_K20_SUPPORT_BYTES {
            return Err(invalid(
                "fixed-K20 support capture workspace is smaller than its record",
            ));
        }
        let (support, capacity) = self
            .fixed_k20_support_slots
            .as_ref()
            .ok_or_else(|| invalid("fixed-K20 support slots were not allocated"))?;
        if row >= *capacity {
            return Err(invalid("fixed-K20 support row is outside request storage"));
        }
        let source = workspace.range(0, FIXED_K20_SUPPORT_BYTES)?;
        let offset = u64::try_from(row)
            .ok()
            .and_then(|row| row.checked_mul(FIXED_K20_SUPPORT_BYTES))
            .ok_or_else(|| invalid("fixed-K20 support row offset overflowed"))?;
        let destination = support.range(offset, FIXED_K20_SUPPORT_BYTES)?;
        let mut copy = session.copy_device_to_device(queue, source, destination)?;
        match copy.wait(timeout)? {
            ExecutionState::Success => Ok(()),
            ExecutionState::Pending => Err(ExecutionError::NotReady),
            ExecutionState::Failure => Err(invalid("fixed-K20 support copy failed")),
        }
    }

    pub(crate) fn fixed_k20_support_range(
        &self,
        rows: usize,
    ) -> Result<crate::BufferRange, ExecutionError> {
        if !(1..=MAX_FIXED_K20_SUPPORT_SLOTS).contains(&rows) {
            return Err(invalid(
                "fixed-K20 support range row count is outside bounds",
            ));
        }
        let (buffer, capacity) = self
            .fixed_k20_support_slots
            .as_ref()
            .ok_or_else(|| invalid("fixed-K20 support slots were not allocated"))?;
        if rows > *capacity {
            return Err(invalid("fixed-K20 support range exceeds request storage"));
        }
        let bytes = u64::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(FIXED_K20_SUPPORT_BYTES))
            .ok_or_else(|| invalid("fixed-K20 support range size overflowed"))?;
        buffer.range(0, bytes)
    }

    pub(crate) fn update(
        &mut self,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        request: &DeviceTokenSelectorRequestV1,
        timeout: Duration,
    ) -> Result<(), ExecutionError> {
        if request.vocab_size() != self.vocab {
            return Err(invalid(
                "device sampler vocabulary changed within one request",
            ));
        }
        // Empty inputs are disabled by the numerical contract and must not be
        // read on device. In particular, do not fill or upload neutral arrays.
        if !request.additive_logits().is_empty() && self.last_additive != request.additive_logits()
        {
            if request.additive_logits().len() != self.vocab {
                return Err(invalid(
                    "device sampler additive length differs from vocabulary",
                ));
            }
            let bytes: Vec<u8> = request
                .additive_logits()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            upload(session, queue, &self.additive, &bytes, timeout)?;
            self.last_additive.clear();
            self.last_additive
                .extend_from_slice(request.additive_logits());
        }
        if !request.valid_mask().is_empty() && self.last_mask != request.valid_mask() {
            if request.valid_mask().len() != self.vocab {
                return Err(invalid(
                    "device sampler mask length differs from vocabulary",
                ));
            }
            upload(session, queue, &self.mask, request.valid_mask(), timeout)?;
            self.last_mask.clear();
            self.last_mask.extend_from_slice(request.valid_mask());
        }
        Ok(())
    }

    pub(crate) fn prepare(
        &mut self,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        logits: OwnedTensorBinding,
        request: &DeviceTokenSelectorRequestV1,
        timeout: Duration,
    ) -> Result<PreparedDeviceSampling, ExecutionError> {
        self.update(session, queue, request, timeout)?;
        let contract = if request.top_k() != 0 || request.top_p() != 1.0 {
            TokenSelectorContractV1::new_fixed(
                self.vocab as u64,
                request.top_k() as u32,
                request.top_p(),
                request.seed(),
                request.counter(),
                !request.additive_logits().is_empty(),
                !request.valid_mask().is_empty(),
            )
            .map_err(invalid)?
        } else {
            TokenSelectorContractV1::new(
                self.vocab as u64,
                request.temperature(),
                request.seed(),
                request.counter(),
            )
            .map_err(invalid)?
        };
        let mut inputs = vec![
            logits,
            session.bind(&self.additive, self.additive_view.clone(), AccessMode::Read)?,
            session.bind(&self.mask, self.mask_view.clone(), AccessMode::Read)?,
        ];
        let scratch = contract.workspace_bytes();
        if scratch != 0 {
            if self
                .workspace
                .as_ref()
                .is_none_or(|(_, view)| view.payload_bytes() != scratch)
            {
                let view =
                    TensorView::contiguous(DType::U8, &[scratch as usize]).map_err(invalid)?;
                let buffer =
                    session.allocate_with_category(scratch, AllocationCategory::RequestState)?;
                self.workspace = Some((buffer, view));
            }
            let (buffer, view) = self
                .workspace
                .as_ref()
                .expect("sampler workspace allocated");
            inputs.push(session.bind(buffer, view.clone(), AccessMode::ReadWrite)?);
        }
        let descriptor = SemanticOpDescriptor::new_token_select(
            inputs
                .iter()
                .map(|binding| binding.view().clone())
                .collect(),
            vec![self.output_view.clone()],
            contract,
        )
        .map_err(invalid)?;
        Ok(PreparedDeviceSampling {
            descriptor,
            inputs,
            outputs: vec![session.bind(
                &self.output,
                self.output_view.clone(),
                AccessMode::Write,
            )?],
        })
    }
}

fn upload(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    bytes: &[u8],
    timeout: Duration,
) -> Result<(), ExecutionError> {
    let maximum = usize::try_from(session.max_transfer_bytes()?).unwrap_or(usize::MAX);
    if maximum == 0 {
        return Err(invalid("device sampler transfer limit is zero"));
    }
    for (index, chunk) in bytes.chunks(maximum).enumerate() {
        let offset = index
            .checked_mul(maximum)
            .ok_or_else(|| invalid("upload offset overflow"))?;
        let range = buffer.range(offset as u64, chunk.len() as u64)?;
        let mut transfer = session.upload(queue, range, Arc::from(chunk))?;
        require_terminal_success("device sampler constraint upload", transfer.wait(timeout)?)
            .map_err(invalid)?;
    }
    Ok(())
}

pub(crate) fn decode_selected_record(
    bytes: &[u8],
    request: &DeviceTokenSelectorRequestV1,
) -> Result<SamplingSelectionV1, ExecutionError> {
    if bytes.len() != 16 {
        return Err(invalid("device sampler returned an invalid record length"));
    }
    let token = i32::from_le_bytes(bytes[0..4].try_into().expect("record token"));
    let status = u32::from_le_bytes(bytes[4..8].try_into().expect("record status"));
    let logprob = f32::from_le_bytes(bytes[8..12].try_into().expect("record logprob"));
    let reserved = u32::from_le_bytes(bytes[12..16].try_into().expect("record reserved"));
    if status != 0 {
        return Err(ExecutionError::BackendStatus {
            status,
            diagnostic: "device sampler record status".to_owned(),
        });
    }
    if reserved != 0 || token < 0 || !request.is_token_valid(token as usize) || !logprob.is_finite()
    {
        return Err(invalid(
            "device sampler returned an invalid or masked token record",
        ));
    }
    Ok(SamplingSelectionV1 {
        token_id: token as u32,
        logprob: f64::from(logprob),
        top_logprobs: Vec::new(),
    })
}
