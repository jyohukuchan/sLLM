//! Assembly of request-owned whole-decode graphs. Capture performs no model
//! execution and never publishes a sampled token or state transition.
use super::*;
use crate::decode_control::{DecodeControlModeV1, DecodeControlV1};
use crate::{AllocationCategory, ExecutionGraphSpan};
use std::time::Instant;

#[derive(Clone, Debug)]
pub(super) struct WholeTargetDecodeConfig {
    pub output_limit: u64,
    pub stop_ids: Vec<u32>,
}

pub(super) struct CaptureStorage {
    pub control: ExecutionBuffer,
    pub control_binding: OwnedTensorBinding,
    pub ring: ExecutionBuffer,
    pub ring_binding: OwnedTensorBinding,
    pub stop_binding: Option<OwnedTensorBinding>,
    pub stop_count: u32,
}

pub(super) struct CapturedDecode {
    pub graph: ExecutionGraphSpan,
    pub ring: ExecutionBuffer,
    pub initial: DecodeControlV1,
}

pub(super) struct RunningTargetDecode {
    replay: crate::DecodeReplayController,
    valid_mask: Vec<u8>,
}

impl QwenExecutionCore {
    pub(super) fn supports_whole_decode(&self) -> bool {
        self.session.backend_name() == "hip"
            && matches!(
                self.session.expected_target().as_deref(),
                Some("gfx1030" | "gfx1201")
            )
            && self.qwen38_artifact.as_ref().is_some_and(|artifact| {
                self.graph.fp8_sidecar_fingerprint() == Some(artifact.recipe_digest())
            })
            && !self.graph.is_multimodal()
            && !self.graph.is_mtp()
            && self.adapters.lora.is_empty()
            && self.adapters.controls.is_empty()
            && self.rope_position_delta == 0
            && self.committed_length == 0
            && !self.kv_states.is_empty()
            && self
                .kv_states
                .values()
                .all(|state| state.descriptor().cache_encoding() == crate::KvCacheEncoding::Mxfp8E4)
    }

    pub(super) fn configure_whole_target_decode(
        &mut self,
        output_limit: u64,
        stop_ids: &[u32],
    ) -> Result<bool, QwenExecutionError> {
        if self.whole_decode_config.is_some()
            || self
                .whole_decode
                .get_mut()
                .map_err(|_| QwenExecutionError::Poisoned)?
                .is_some()
        {
            return Err(QwenExecutionError::Busy);
        }
        self.whole_decode_config = None;
        if !self.supports_whole_decode() || output_limit <= 2 {
            return Ok(false);
        }
        if stop_ids.len() > 16
            || stop_ids
                .iter()
                .any(|id| *id >= crate::QWEN35_VOCAB_SIZE as u32)
        {
            return Err(QwenExecutionError::InvalidRequest(
                "whole-decode stop IDs are invalid".to_owned(),
            ));
        }
        self.qwen38_graph_spans_enabled = false;
        self.whole_decode_config = Some(WholeTargetDecodeConfig {
            output_limit,
            stop_ids: stop_ids.to_vec(),
        });
        Ok(true)
    }

    pub(super) fn maybe_activate_whole_target_decode(
        &mut self,
        output: &QwenExecutionOutput,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<(), QwenExecutionError> {
        let Some(config) = self.whole_decode_config.clone() else {
            return Ok(());
        };
        let fixed_selector = selector.vocab_size() == crate::QWEN35_VOCAB_SIZE
            && selector.counter() == 1
            && selector.temperature() == 1.0
            && selector.top_k() == 20
            && selector.top_p() == 0.95
            && selector.additive_logits().is_empty()
            && selector.valid_mask().is_empty();
        let Some(&token) = output.token_ids.last() else {
            self.whole_decode_config = None;
            return Ok(());
        };
        if !fixed_selector
            || token < 0
            || config.stop_ids.contains(&(token as u32))
            || output.committed_length != self.committed_length
            || output.committed_length == 0
        {
            self.whole_decode_config = None;
            return Ok(());
        }
        let pending_token = u32::try_from(token).map_err(|_| {
            QwenExecutionError::InvalidRequest("whole-decode pending token is invalid".to_owned())
        })?;
        let next_counter = selector.counter().checked_add(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("whole-decode sampler counter overflowed".to_owned())
        })?;
        let next_selector = selector.with_counter(next_counter);
        if let Err(error) = self.start_whole_target_decode(
            pending_token,
            &next_selector,
            2,
            config.output_limit,
            &config.stop_ids,
        ) {
            self.cancel();
            return Err(error);
        }
        self.whole_decode_config = None;
        Ok(())
    }

    pub(super) fn start_whole_target_decode(
        &mut self,
        pending_token: u32,
        selector: &DeviceTokenSelectorRequestV1,
        output_count: u64,
        output_limit: u64,
        stop_ids: &[u32],
    ) -> Result<(), QwenExecutionError> {
        if self
            .whole_decode
            .get_mut()
            .map_err(|_| QwenExecutionError::Poisoned)?
            .is_some()
        {
            return Err(QwenExecutionError::Busy);
        }
        let mut initial = DecodeControlV1::new(
            DecodeControlModeV1::TargetOnly,
            1,
            self.graph.state_capacity(),
            self.committed_length,
            selector.counter(),
            selector.seed(),
            output_limit,
            pending_token,
            selector.vocab_size() as u32,
        )?;
        initial.output_count = output_count;
        let startup = Instant::now();
        let captured = self.capture_target_decode_graph(initial, selector, stop_ids)?;
        let mut replay = crate::DecodeReplayController::new(
            Arc::clone(&self.session),
            captured.graph,
            captured.ring,
            captured.initial,
            selector.vocab_size() as u32,
            self.completion_timeout,
        )?;
        replay.set_startup_ns(u64::try_from(startup.elapsed().as_nanos()).unwrap_or(u64::MAX));
        replay.start()?;
        *self
            .whole_decode
            .get_mut()
            .map_err(|_| QwenExecutionError::Poisoned)? = Some(RunningTargetDecode {
            replay,
            valid_mask: selector.valid_mask().to_vec(),
        });
        self.whole_decode_config = None;
        Ok(())
    }

    pub(super) fn whole_decode_step(
        &mut self,
        token_id: i32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<Option<QwenExecutionOutput>, QwenExecutionError> {
        let Some(mut running) = self
            .whole_decode
            .get_mut()
            .map_err(|_| QwenExecutionError::Poisoned)?
            .take()
        else {
            return Ok(None);
        };
        let result = (|| {
            let control = running.replay.control();
            if token_id < 0
                || token_id as u32 != control.pending_token
                || selector.counter() != control.sampler_counter
                || selector.seed() != control.seed
                || selector.top_k() != 20
                || selector.top_p() != 0.95
                || selector.temperature() != 1.0
                || !selector.additive_logits().is_empty()
                || selector.valid_mask() != running.valid_mask
            {
                return Err(QwenExecutionError::InvalidRequest(
                    "whole-decode inputs changed outside the fixed sampling contract".to_owned(),
                ));
            }
            let result = running.replay.next()?.ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "whole-decode replay has no remaining output".to_owned(),
                )
            })?;
            if result.count != 1 || result.selections.len() != 1 {
                return Err(QwenExecutionError::InvalidRequest(
                    "whole target decode produced an invalid row count".to_owned(),
                ));
            }
            Ok(QwenExecutionOutput {
                token_ids: vec![result.selections[0].token_id as i32],
                last_logits: None,
                selection: Some(result.selections[0].clone()),
                selections: None,
                logits_bf16: None,
                hidden_states_bf16: None,
                embeddings_bf16: None,
                committed_length: result.model_position_after,
            })
        })();
        match result {
            Ok(output) => {
                *self
                    .whole_decode_audit
                    .lock()
                    .map_err(|_| QwenExecutionError::Poisoned)? = Some(running.replay.audit());
                self.committed_length = output.committed_length;
                self.last_output = Some(output.clone());
                if running.replay.finished() {
                    drop(running);
                    self.ensure_state_lengths(self.committed_length)?;
                } else {
                    *self
                        .whole_decode
                        .get_mut()
                        .map_err(|_| QwenExecutionError::Poisoned)? = Some(running);
                }
                Ok(Some(output))
            }
            Err(error) => {
                self.lifecycle.cancel();
                drop(running);
                Err(error)
            }
        }
    }

    pub(super) fn allocate_decode_capture_storage(
        &self,
        initial: &DecodeControlV1,
        stop_ids: &[u32],
        vocabulary_size: usize,
    ) -> Result<CaptureStorage, QwenExecutionError> {
        initial.validate(vocabulary_size as u32)?;
        if stop_ids.len() > 16 || stop_ids.iter().any(|id| *id >= vocabulary_size as u32) {
            return Err(QwenExecutionError::InvalidRequest(
                "whole-decode stop IDs are invalid".to_owned(),
            ));
        }
        let allocate = |dtype, shape: &[usize], bytes: &[u8]| {
            let view = TensorView::contiguous(dtype, shape)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            let buffer = self
                .session
                .allocate_with_category(view.payload_bytes(), AllocationCategory::RequestState)?;
            upload_exact_bytes(
                self.session.as_ref(),
                &self.queue,
                &buffer,
                &view,
                bytes,
                self.completion_timeout,
                "whole-decode control initialization",
            )?;
            let binding = self.session.bind(&buffer, view, AccessMode::ReadWrite)?;
            Ok::<_, QwenExecutionError>((buffer, binding))
        };
        let (control, control_binding) = allocate(DType::U8, &[144], &initial.encode_le()?)?;
        let (ring, ring_binding) = allocate(DType::U8, &[384], &[0; 384])?;
        let stop_binding = if stop_ids.is_empty() {
            None
        } else {
            let bytes = stop_ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>();
            Some(allocate(DType::I32, &[stop_ids.len()], &bytes)?.1)
        };
        Ok(CaptureStorage {
            control,
            control_binding,
            ring,
            ring_binding,
            stop_binding,
            stop_count: stop_ids.len() as u32,
        })
    }

    pub(super) fn capture_copy(
        &self,
        capture: &mut crate::ExecutionWholeDecodeCapture,
        source: crate::BufferRange,
        destination: crate::BufferRange,
    ) -> Result<(), QwenExecutionError> {
        if source.buffer().id() == destination.buffer().id()
            && source.offset_bytes() == destination.offset_bytes()
            && source.size_bytes() == destination.size_bytes()
        {
            // The next input can already alias the previous phase's output.
            return Ok(());
        }
        let copy = self
            .session
            .copy_device_to_device(&self.queue, source, destination)?;
        let mut pending = ExecutionSegment::for_capture(
            self.session.as_ref(),
            &self.queue,
            self.completion_timeout,
            capture,
        );
        pending.retain_device_copy(copy);
        self.close_boundary(&mut pending, ExecutionBoundaryKind::StatePublication)
    }

    pub(super) fn sampler_output_binding(&self) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let storage = self
            .device_sampling
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?;
        let storage = storage.as_ref().ok_or_else(|| {
            QwenExecutionError::InvalidRequest(
                "whole decode requires a warmed device sampler".to_owned(),
            )
        })?;
        Ok(self.session.bind(
            &storage.output,
            storage.output_view.clone(),
            AccessMode::Read,
        )?)
    }

    pub(super) fn fixed_k20_support_binding(
        &self,
        rows: usize,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let range = self.fixed_k20_support_range(rows)?;
        let view = TensorView::new(
            DType::U8,
            crate::Encoding::Unquantized,
            &[range.size_bytes() as usize],
            &[1],
            range.offset_bytes(),
        )
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        Ok(self.session.bind(range.buffer(), view, access)?)
    }

    pub(super) fn capture_target_decode_graph(
        &self,
        initial: DecodeControlV1,
        selector: &DeviceTokenSelectorRequestV1,
        stop_ids: &[u32],
    ) -> Result<CapturedDecode, QwenExecutionError> {
        if self.graph.is_mtp()
            || initial.mode != DecodeControlModeV1::TargetOnly
            || initial.model_position != self.committed_length
            || initial.output_count >= initial.output_limit
            || selector.temperature() != 1.0
            || selector.top_k() != 20
            || selector.top_p() != 0.95
            || !selector.additive_logits().is_empty()
            || self.pending_speculative.is_some()
        {
            return Err(QwenExecutionError::InvalidRequest(
                "whole target decode requires an idle fixed-K20 target request".to_owned(),
            ));
        }
        self.ensure_state_lengths(self.committed_length)?;
        let storage =
            self.allocate_decode_capture_storage(&initial, stop_ids, selector.vocab_size())?;
        let selected = self.sampler_output_binding()?;
        let token_id = *self.tensor_ids.get("input.token_ids").ok_or_else(|| {
            QwenExecutionError::InvalidGraph("decode token input is absent".to_owned())
        })?;
        let input_view = self.view(token_id, 1)?;
        let input = self.tensors[token_id]
            .buffer
            .range(input_view.byte_offset(), 4)?;
        self.session
            .set_queue_completion_mode(&self.queue, crate::QueueCompletionMode::Deferred)?;
        let mut capture = self
            .session
            .begin_whole_decode_capture(&self.queue, &storage.control_binding)?;
        for state in self.linear_states.values() {
            capture.bind_linear_state(state, None, None, 1, 0)?;
        }
        capture.command(
            &crate::ExecutionDecodeCommand::new(1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            [None, None, None, None],
        )?;
        self.capture_copy(&mut capture, storage.control.range(72, 4)?, input)?;
        let expected = self.committed_length.checked_add(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("decode position overflow".to_owned())
        })?;
        self.lower_graph(
            1,
            self.committed_length,
            self.committed_length,
            expected,
            AttentionPreprocessPositionMode::DecodeContinuation,
            TerminalOutputRows::Last,
            true,
            Some(selector),
            false,
            Some(&mut capture),
        )?;
        capture.command(
            &crate::ExecutionDecodeCommand::new(
                6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                selector.vocab_size() as u32,
                storage.stop_count,
            ),
            [
                Some(&selected),
                None,
                Some(&storage.ring_binding),
                storage.stop_binding.as_ref(),
            ],
        )?;
        for state in self.linear_states.values() {
            capture.select_linear_state(state, 1)?;
        }
        Ok(CapturedDecode {
            graph: capture.finish_captured()?,
            ring: storage.ring,
            initial,
        })
    }
}
