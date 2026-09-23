//! Whole graph capture for the fixed-width Qwen MTP draft/verify path.
//!
//! This module only assembles device work. It does not read selector records,
//! update tokens on the host, or publish a speculative transition.

use super::decode_capture::CapturedDecode;
use super::*;
use crate::decode_control::{
    DecodeControlModeV1, DecodeControlStatusV1, DecodeControlV1, DecodePhaseKindV1,
};
use crate::{AllocationCategory, QueueCompletionMode};

struct LinearStateCheckpointBindings {
    conv: OwnedTensorBinding,
    recurrent: OwnedTensorBinding,
}

impl QwenExecutionCore {
    fn capture_token_storage(&self) -> Result<(usize, TensorView), QwenExecutionError> {
        let tensor = self.tensor_id("input.token_ids")?;
        let graph_rows = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph(
                "token input row capacity does not fit usize".to_owned(),
            )
        })?;
        let view = self.view(tensor, self.graph.token_count())?;
        if graph_rows == 0
            || view.dtype() != DType::I32
            || view.encoding() != Encoding::Unquantized
            || view.shape() != [graph_rows]
            || view.strides() != [1]
            || view.payload_bytes()
                != u64::try_from(graph_rows)
                    .ok()
                    .and_then(|rows| rows.checked_mul(4))
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(
                            "token input byte size overflowed".to_owned(),
                        )
                    })?
        {
            return Err(QwenExecutionError::InvalidGraph(
                "whole-decode token input must be contiguous unquantized I32 rows".to_owned(),
            ));
        }
        Ok((tensor, view))
    }

    fn capture_token_row_binding(
        &self,
        row: usize,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let (tensor, view) = self.capture_token_storage()?;
        let rows = view.shape()[0];
        if row >= rows {
            return Err(QwenExecutionError::InvalidRequest(
                "token row is outside graph storage".to_owned(),
            ));
        }
        let row_view = row_view(&view, row)?;
        self.bind_view(tensor, row_view, access)
    }

    fn capture_token_span_binding(
        &self,
        first_row: usize,
        rows: usize,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let (tensor, view) = self.capture_token_storage()?;
        let total_rows = view.shape()[0];
        if rows == 0
            || first_row
                .checked_add(rows)
                .is_none_or(|end| end > total_rows)
        {
            return Err(QwenExecutionError::InvalidRequest(
                "token span is outside graph storage".to_owned(),
            ));
        }
        let first_row = u64::try_from(first_row).map_err(|_| {
            QwenExecutionError::InvalidRequest("token span start does not fit u64".to_owned())
        })?;
        let offset = view
            .byte_offset()
            .checked_add(first_row.checked_mul(4).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("token span offset overflowed".to_owned())
            })?)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("token span offset overflowed".to_owned())
            })?;
        let span = TensorView::new(DType::I32, Encoding::Unquantized, &[rows], &[1], offset)?;
        self.bind_view(tensor, span, access)
    }

    fn allocate_mtp_linear_checkpoints(
        &self,
        width: usize,
    ) -> Result<Vec<LinearStateCheckpointBindings>, QwenExecutionError> {
        let mut checkpoints = Vec::new();
        checkpoints
            .try_reserve(self.linear_states.len())
            .map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "linear-state checkpoint owner allocation failed".to_owned(),
                )
            })?;
        for state in self.linear_states.values() {
            let layout = state.descriptor().layout();
            if layout.conv_history() == 0
                || layout.qkv_width() == 0
                || layout.value_heads() == 0
                || layout.head_dim() == 0
            {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "linear-attention layer {} has an unsupported checkpoint layout",
                    state.layer_id()
                )));
            }
            let conv_view = TensorView::contiguous(
                DType::Bf16,
                &[width, layout.conv_history(), layout.qkv_width()],
            )?;
            let recurrent_view = TensorView::contiguous(
                DType::F32,
                &[
                    width,
                    layout.value_heads(),
                    layout.head_dim(),
                    layout.head_dim(),
                ],
            )?;
            let conv_buffer = self.session.allocate_with_category(
                conv_view.payload_bytes(),
                AllocationCategory::RequestState,
            )?;
            let recurrent_buffer = self.session.allocate_with_category(
                recurrent_view.payload_bytes(),
                AllocationCategory::RequestState,
            )?;
            checkpoints.push(LinearStateCheckpointBindings {
                conv: self
                    .session
                    .bind(&conv_buffer, conv_view, AccessMode::ReadWrite)?,
                recurrent: self.session.bind(
                    &recurrent_buffer,
                    recurrent_view,
                    AccessMode::ReadWrite,
                )?,
            });
        }
        Ok(checkpoints)
    }

    pub(super) fn capture_mtp_decode_graph(
        &mut self,
        companion: &mut QwenExecutionCore,
        initial: DecodeControlV1,
        selector: &DeviceTokenSelectorRequestV1,
        stop_ids: &[u32],
    ) -> Result<CapturedDecode, QwenExecutionError> {
        let width = usize::try_from(initial.width).map_err(|_| {
            QwenExecutionError::InvalidRequest("MTP width does not fit usize".to_owned())
        })?;
        let width_u32 = u32::try_from(width).map_err(|_| {
            QwenExecutionError::InvalidRequest("MTP width does not fit u32".to_owned())
        })?;
        let target_rows = width.checked_add(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("MTP target row count overflowed".to_owned())
        })?;
        let target_rows_u32 = u32::try_from(target_rows).map_err(|_| {
            QwenExecutionError::InvalidRequest("MTP target row count does not fit u32".to_owned())
        })?;
        let target_rows_u64 = u64::try_from(target_rows).map_err(|_| {
            QwenExecutionError::InvalidRequest("MTP target row count does not fit u64".to_owned())
        })?;
        if !(1..=8).contains(&width) || initial.mode != DecodeControlModeV1::Mtp {
            return Err(QwenExecutionError::InvalidRequest(
                "MTP capture requires mode Mtp and width 1..=8".to_owned(),
            ));
        }
        let align_position = initial
            .model_position
            .checked_add(u64::from(width_u32))
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("MTP alignment position overflowed".to_owned())
            })?;
        let target_end = initial
            .model_position
            .checked_add(target_rows_u64)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("MTP target position overflowed".to_owned())
            })?;
        if self.graph.is_mtp()
            || !companion.graph.is_mtp()
            || self.session.id() != companion.session.id()
            || self.queue.id() != companion.queue.id()
            || self.committed_length != initial.model_position
            || companion.committed_length != initial.model_position
            || self.pending_speculative.is_some()
            || companion.pending_speculative.is_some()
            || !companion.linear_states.is_empty()
        {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "MTP capture pair is not ready: target_mtp={} companion_mtp={} session_match={} queue_match={} positions={}/{}/{} pending={}/{} companion_gdn={}",
                self.graph.is_mtp(),
                companion.graph.is_mtp(),
                self.session.id() == companion.session.id(),
                self.queue.id() == companion.queue.id(),
                self.committed_length,
                companion.committed_length,
                initial.model_position,
                self.pending_speculative.is_some(),
                companion.pending_speculative.is_some(),
                companion.linear_states.len()
            )));
        }
        if selector.temperature() != 1.0
            || selector.top_k() != 20
            || selector.top_p() != 0.95
            || !selector.additive_logits().is_empty()
            || selector.counter() != initial.sampler_counter
            || selector.seed() != initial.seed
        {
            return Err(QwenExecutionError::InvalidRequest(
                "MTP capture requires the fixed K20 selector".to_owned(),
            ));
        }
        let draft_selector = match companion.draft_vocab_ids.as_ref() {
            Some(map) => selector
                .clone()
                .with_vocab_map(Arc::clone(map))
                .map_err(|error| {
                    QwenExecutionError::InvalidRequest(format!(
                        "MTP draft vocabulary cannot prepare mapped selector: {error}"
                    ))
                })?,
            None => selector.clone(),
        };
        initial.validate(selector.vocab_size() as u32)?;
        if initial.status != DecodeControlStatusV1::Ok
            || initial.halted
            || initial.phase_active
            || initial.output_count >= initial.output_limit
            || initial.hidden_row >= target_rows_u32
        {
            return Err(QwenExecutionError::InvalidRequest(
                "MTP capture requires an active control record with a known target hidden row"
                    .to_owned(),
            ));
        }
        if self.session.fixed_k20_support_scratch_version()? != Some(1)
            || companion.session.fixed_k20_support_scratch_version()? != Some(1)
            || self.session.fixed_k20_mtp_verifier_version()? != Some(1)
            || companion.session.fixed_k20_mtp_verifier_version()? != Some(1)
        {
            return Err(QwenExecutionError::Execution(ExecutionError::Unsupported {
                reason: "whole-graph fixed-K20 MTP support is unavailable on one request backend"
                    .to_owned(),
            }));
        }
        self.ensure_state_lengths(initial.model_position)?;
        companion.ensure_state_lengths(initial.model_position)?;
        let storage =
            self.allocate_decode_capture_storage(&initial, stop_ids, selector.vocab_size())?;
        let target_selected = self.sampler_output_binding()?;
        let companion_selected = companion.sampler_output_binding()?;
        let companion_token = companion.capture_token_row_binding(0, AccessMode::Write)?;
        let target_token = self.capture_token_row_binding(0, AccessMode::Write)?;
        let target_draft_input = self.capture_token_span_binding(1, width, AccessMode::Write)?;

        // Target input rows are ordinary graph workspace and cease being a
        // durable draft-ID owner after embedding. Keep every proposal token
        // in a request-owned plane, copying it to target input only when the
        // verify phase is ready to consume it.
        let draft_storage_words = width.checked_add(4).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("MTP draft storage size overflowed".to_owned())
        })?;
        let draft_ids_storage_view = TensorView::contiguous(DType::I32, &[draft_storage_words])?;
        let draft_ids_buffer = self.session.allocate_with_category(
            draft_ids_storage_view.payload_bytes(),
            AllocationCategory::RequestState,
        )?;
        let draft_storage_bytes =
            usize::try_from(draft_ids_storage_view.payload_bytes()).map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "MTP draft storage byte size does not fit usize".to_owned(),
                )
            })?;
        upload_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &draft_ids_buffer,
            &draft_ids_storage_view,
            &vec![0_u8; draft_storage_bytes],
            self.completion_timeout,
            "whole MTP draft storage initialization",
        )?;
        let draft_ids_view = TensorView::new(DType::I32, Encoding::Unquantized, &[width], &[1], 0)?;
        let draft_ids =
            self.session
                .bind(&draft_ids_buffer, draft_ids_view.clone(), AccessMode::Read)?;
        let mut draft_rows = Vec::new();
        draft_rows.try_reserve(width).map_err(|_| {
            QwenExecutionError::InvalidRequest("MTP draft binding allocation failed".to_owned())
        })?;
        for row in 0..width {
            draft_rows.push(self.session.bind(
                &draft_ids_buffer,
                row_view(&draft_ids_view, row)?,
                AccessMode::Write,
            )?);
        }
        let target_selector_tail_view = TensorView::new(
            DType::U8,
            Encoding::Unquantized,
            &[16],
            &[1],
            u64::from(width_u32) * 4,
        )?;

        let (target_pre_final_hidden, _) = self.final_hidden_tensor_ids()?;
        let target_hidden_view = self.view(target_pre_final_hidden, target_rows_u64)?;
        let target_hidden_width = validate_hidden_row_view(
            &target_hidden_view,
            target_rows,
            "whole MTP target pre-final hidden",
        )?;
        let target_hidden_width_u32 = u32::try_from(target_hidden_width).map_err(|_| {
            QwenExecutionError::InvalidGraph("MTP hidden width does not fit u32".to_owned())
        })?;
        let target_hidden_binding = self.bind_view(
            target_pre_final_hidden,
            target_hidden_view.clone(),
            AccessMode::Read,
        )?;
        let companion_input_hidden = companion.tensor_id("input.target_hidden")?;
        let companion_input_full = companion.view(companion_input_hidden, 1)?;
        let companion_hidden_width = validate_hidden_row_view(
            &companion_input_full,
            1,
            "whole MTP companion target-hidden input",
        )?;
        if companion_hidden_width != target_hidden_width {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "whole MTP hidden widths differ: target {target_hidden_width}, companion {companion_hidden_width}"
            )));
        }
        let companion_input_view = row_view(&companion_input_full, 0)?;
        let companion_input_hidden_binding = companion.bind_view(
            companion_input_hidden,
            companion_input_view.clone(),
            AccessMode::Write,
        )?;
        let (companion_pre_final_hidden, _) = companion.final_hidden_tensor_ids()?;
        let companion_pre_final_full = companion.view(companion_pre_final_hidden, 1)?;
        let companion_output_width = validate_hidden_row_view(
            &companion_pre_final_full,
            1,
            "whole MTP companion pre-final hidden",
        )?;
        if companion_output_width != target_hidden_width {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "whole MTP companion output width is {companion_output_width}, expected {target_hidden_width}"
            )));
        }
        let companion_pre_final_view = row_view(&companion_pre_final_full, 0)?;

        // The target pre-final tensor is workspace reused by the next graph.
        // Seed a durable one-row carry from the warm host-provided row before
        // capture; every replay refreshes this carry after its commit.
        let target_hidden_carry_view =
            TensorView::contiguous(DType::Bf16, &[1, target_hidden_width])?;
        let target_hidden_carry = self.session.allocate_with_category(
            target_hidden_carry_view.payload_bytes(),
            AllocationCategory::RequestState,
        )?;
        let target_hidden_carry_binding = self.session.bind(
            &target_hidden_carry,
            target_hidden_carry_view.clone(),
            AccessMode::ReadWrite,
        )?;
        let warm_hidden_row = row_view(
            &target_hidden_view,
            usize::try_from(initial.hidden_row).map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "initial target hidden row does not fit usize".to_owned(),
                )
            })?,
        )?;
        let mut carry_seed = self.session.copy_device_to_device(
            &self.queue,
            self.tensors[target_pre_final_hidden].buffer.range(
                warm_hidden_row.byte_offset(),
                warm_hidden_row.payload_bytes(),
            )?,
            target_hidden_carry.range(
                target_hidden_carry_view.byte_offset(),
                target_hidden_carry_view.payload_bytes(),
            )?,
        )?;
        require_terminal_success(
            "whole MTP target-hidden carry initialization",
            carry_seed.wait(self.completion_timeout)?,
        )?;
        // Completion observation releases active work; dropping its owner also
        // releases the queue reference required by idle capture admission.
        drop(carry_seed);

        // These graph-owned resources must exist before native capture starts.
        // The capture API clones every binding into the finished graph.
        let checkpoints = self.allocate_mtp_linear_checkpoints(width)?;
        let decision = self.fixed_k20_decision_buffer()?;
        let decision_view = TensorView::contiguous(
            DType::U8,
            &[usize::try_from(FIXED_K20_DECISION_BYTES).map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "fixed-K20 decision size does not fit usize".to_owned(),
                )
            })?],
        )?;
        let decision_binding =
            self.session
                .bind(&decision, decision_view, AccessMode::ReadWrite)?;

        // Allocate all fixed-K20 support slots and selector workspace before
        // capture. The capture itself must contain only device work.
        self.begin_fixed_k20_support_capture(
            target_rows,
            FixedK20SupportCaptureRole::TargetVerify,
        )?;
        if let Err(error) = companion
            .begin_fixed_k20_support_capture(width, FixedK20SupportCaptureRole::CompanionDraft)
        {
            let _ = self.finish_fixed_k20_support_capture();
            return Err(error);
        }
        let assembled = (|| -> Result<CapturedDecode, QwenExecutionError> {
            let target_support = self.fixed_k20_support_binding(target_rows, AccessMode::Read)?;
            let draft_support = companion.fixed_k20_support_binding(width, AccessMode::Read)?;

            self.session
                .set_queue_completion_mode(&self.queue, QueueCompletionMode::Deferred)?;
            let mut capture = self
                .session
                .begin_whole_decode_capture(&self.queue, &storage.control_binding)
                .map_err(|error| QwenExecutionError::NodeExecution {
                    node: "whole_mtp.capture_begin".to_owned(),
                    error: Box::new(error.into()),
                })?;
            if checkpoints.len() != self.linear_states.len() {
                return Err(QwenExecutionError::InvalidGraph(
                    "linear-state checkpoint owner count changed before capture".to_owned(),
                ));
            }
            for (state, checkpoint) in self.linear_states.values().zip(&checkpoints) {
                capture.bind_linear_state(
                    state,
                    Some(&checkpoint.conv),
                    Some(&checkpoint.recurrent),
                    target_rows_u32,
                    width_u32,
                )?;
            }
            // Seed the first target token and restore the durable hidden carry
            // before any draft phase starts.
            self.capture_copy(
                &mut capture,
                storage.control.range(72, 4)?,
                target_token.buffer().range(
                    target_token.view().byte_offset(),
                    target_token.view().payload_bytes(),
                )?,
            )?;
            self.capture_copy(
                &mut capture,
                storage.control.range(72, 4)?,
                companion_token.buffer().range(
                    companion_token.view().byte_offset(),
                    companion_token.view().payload_bytes(),
                )?,
            )?;
            self.capture_copy(
                &mut capture,
                target_hidden_carry.range(
                    target_hidden_carry_view.byte_offset(),
                    target_hidden_carry_view.payload_bytes(),
                )?,
                companion_input_hidden_binding.buffer().range(
                    companion_input_view.byte_offset(),
                    companion_input_view.payload_bytes(),
                )?,
            )?;

            for (index, draft_row) in draft_rows.iter().enumerate() {
                let index_u32 = u32::try_from(index).map_err(|_| {
                    QwenExecutionError::InvalidRequest(
                        "MTP draft phase index does not fit u32".to_owned(),
                    )
                })?;
                let position = initial
                    .model_position
                    .checked_add(u64::from(index_u32))
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "MTP draft position overflowed".to_owned(),
                        )
                    })?;
                capture.command(
                    &crate::ExecutionDecodeCommand::new(
                        1,
                        DecodePhaseKindV1::Draft as u32,
                        index_u32,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ),
                    [None, None, None, None],
                )?;
                companion.lower_graph(
                    1,
                    position,
                    position,
                    position.checked_add(1).ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "MTP draft end position overflowed".to_owned(),
                        )
                    })?,
                    AttentionPreprocessPositionMode::DecodeContinuation,
                    TerminalOutputRows::Last,
                    true,
                    Some(&draft_selector),
                    false,
                    Some(&mut capture),
                )?;
                capture.command(
                    &crate::ExecutionDecodeCommand::new(2, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                    [Some(&companion_selected), None, Some(draft_row), None],
                )?;
                capture.command(
                    &crate::ExecutionDecodeCommand::new(2, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                    [
                        Some(&companion_selected),
                        None,
                        Some(&companion_token),
                        None,
                    ],
                )?;
                // The companion's pre-final hidden row is the next draft's target
                // hidden input. The last copy is overwritten by target alignment.
                let source = companion.tensors[companion_pre_final_hidden].buffer.range(
                    companion_pre_final_view.byte_offset(),
                    companion_pre_final_view.payload_bytes(),
                )?;
                let destination = companion_input_hidden_binding.buffer().range(
                    companion_input_view.byte_offset(),
                    companion_input_view.payload_bytes(),
                )?;
                self.capture_copy(&mut capture, source, destination)?;
            }
            companion.fixed_k20_capture_complete(width)?;

            self.capture_copy(
                &mut capture,
                draft_ids.buffer().range(
                    draft_ids.view().byte_offset(),
                    draft_ids.view().payload_bytes(),
                )?,
                target_draft_input.buffer().range(
                    target_draft_input.view().byte_offset(),
                    target_draft_input.view().payload_bytes(),
                )?,
            )?;

            let mut target_selectors = Vec::with_capacity(target_rows);
            target_selectors.resize(target_rows, selector.clone());
            self.selector_batch = Some(target_selectors);
            let target_result = (|| {
                capture.command(
                    &crate::ExecutionDecodeCommand::new(
                        1,
                        DecodePhaseKindV1::Target as u32,
                        0,
                        target_rows_u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ),
                    [None, None, None, None],
                )?;
                self.lower_graph(
                    target_rows_u64,
                    initial.model_position,
                    initial.model_position,
                    target_end,
                    AttentionPreprocessPositionMode::DecodeContinuation,
                    TerminalOutputRows::All,
                    true,
                    None,
                    false,
                    Some(&mut capture),
                )
            })();
            self.selector_batch = None;
            target_result?;
            self.fixed_k20_capture_complete(target_rows)?;
            self.capture_copy(
                &mut capture,
                target_selected.buffer().range(
                    target_selected.view().byte_offset(),
                    target_selected.view().payload_bytes(),
                )?,
                draft_ids_buffer.range(
                    target_selector_tail_view.byte_offset(),
                    target_selector_tail_view.payload_bytes(),
                )?,
            )?;
            // Per-row target selectors leave the final inactive row with a
            // disabled phase. Restore the generation's target phase so P/Q
            // validates against the root counter and active row count.
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    1,
                    DecodePhaseKindV1::Target as u32,
                    0,
                    target_rows_u32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
                [None, None, None, None],
            )?;
            // P/Q derives its RNG from device control. Its vocabulary field must
            // remain zero; only row_count and token_capacity describe this command.
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    7,
                    0,
                    0,
                    0,
                    0,
                    target_rows_u32,
                    0,
                    width_u32,
                    0,
                    0,
                ),
                [
                    Some(&target_support),
                    Some(&draft_support),
                    Some(&decision_binding),
                    Some(&draft_ids),
                ],
            )?;
            // Alignment follows the replay's device-selected active width.
            // Width zero consumes the pending token and prior hidden carry;
            // positive widths consume draft[active-1] and target hidden
            // row[active-1].
            capture.command(
                &crate::ExecutionDecodeCommand::new(8, 0, 0, 0, 0, 0, 0, width_u32, 0, 0),
                [Some(&draft_ids), None, Some(&companion_token), None],
            )?;
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    9,
                    0,
                    0,
                    0,
                    0,
                    target_rows_u32,
                    target_hidden_width_u32,
                    0,
                    0,
                    0,
                ),
                [
                    Some(&target_hidden_binding),
                    Some(&target_hidden_carry_binding),
                    Some(&companion_input_hidden_binding),
                    None,
                ],
            )?;
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    1,
                    DecodePhaseKindV1::MtpAlign as u32,
                    width_u32,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
                [None, None, None, None],
            )?;
            companion.lower_graph(
                1,
                align_position,
                align_position,
                target_end,
                AttentionPreprocessPositionMode::DecodeContinuation,
                TerminalOutputRows::Last,
                false,
                None,
                true,
                Some(&mut capture),
            )?;
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    6,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    selector.vocab_size() as u32,
                    storage.stop_count,
                ),
                [
                    Some(&decision_binding),
                    None,
                    Some(&storage.ring_binding),
                    storage.stop_binding.as_ref(),
                ],
            )?;
            // Commit selects hidden_row on device. Preserve that target
            // pre-final row before a later replay reuses the tensor arena.
            capture.command(
                &crate::ExecutionDecodeCommand::new(
                    5,
                    0,
                    0,
                    0,
                    0,
                    target_rows_u32,
                    target_hidden_width_u32,
                    0,
                    0,
                    0,
                ),
                [
                    Some(&target_hidden_binding),
                    None,
                    Some(&target_hidden_carry_binding),
                    None,
                ],
            )?;
            for state in self.linear_states.values() {
                capture.select_linear_state(state, target_rows_u32)?;
            }
            Ok(CapturedDecode {
                graph: capture.finish_captured()?,
                ring: storage.ring.clone(),
                initial: initial.clone(),
            })
        })();
        let target_cleanup = self.finish_fixed_k20_support_capture();
        let companion_cleanup = companion.finish_fixed_k20_support_capture();
        let cleanup_error = target_cleanup.err().or_else(|| companion_cleanup.err());
        match (assembled, cleanup_error) {
            (Ok(captured), None) => Ok(captured),
            (Ok(_), Some(error)) => {
                self.lifecycle.cancel();
                companion.lifecycle.cancel();
                Err(error)
            }
            (Err(primary), Some(cleanup)) => {
                self.lifecycle.cancel();
                companion.lifecycle.cancel();
                Err(QwenExecutionError::CleanupFailure {
                    primary: Box::new(primary),
                    cleanup: Box::new(cleanup),
                })
            }
            (Err(error), None) => Err(error),
        }
    }
}
