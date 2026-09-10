//! Private synchronous fixed-K20 speculative verification bridge.

use sllm_hip_sys as sys;

use crate::runtime::{RuntimeError, ensure_ok, sink};
use crate::{Buffer, Context, Queue};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DraftIdsV1 {
    pub(crate) ids: [u32; 8],
}

// This bridge is internal, like the MTP state checkpoint operations. It is
// deliberately absent from the installed public C header and generated mirror.
unsafe extern "C" {
    fn sllm_token_selector_verify_fixed_k20_mtp_v1(
        context: *const sys::sllm_context_t,
        queue: *const sys::sllm_queue_t,
        target_buffer: *const sys::sllm_buffer_t,
        target_offset: u64,
        target_rows: u32,
        draft_buffer: *const sys::sllm_buffer_t,
        draft_offset: u64,
        draft_rows: u32,
        draft_ids: DraftIdsV1,
        draft_id_count: u32,
        width: u32,
        seed: u64,
        absolute_position: u64,
        decision_buffer: *const sys::sllm_buffer_t,
        decision_offset: u64,
        error_sink: *mut sys::sllm_error_sink_t,
    ) -> sys::sllm_status_t;
}

pub(crate) struct Verification<'a> {
    pub(crate) context: &'a Context,
    pub(crate) queue: &'a Queue,
    pub(crate) target: &'a Buffer,
    pub(crate) target_offset: u64,
    pub(crate) draft: &'a Buffer,
    pub(crate) draft_offset: u64,
    pub(crate) draft_ids: DraftIdsV1,
    pub(crate) width: u32,
    pub(crate) seed: u64,
    pub(crate) absolute_position: u64,
    pub(crate) decision: &'a Buffer,
    pub(crate) decision_offset: u64,
}

impl Verification<'_> {
    pub(crate) fn run(&self) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        // Native validates handles/ranges and retains registry/accounting
        // ownership through the stream fence. No buffer address escapes it.
        let status = unsafe {
            sllm_token_selector_verify_fixed_k20_mtp_v1(
                self.context.raw_handle()?.as_ptr(),
                self.queue.raw_handle()?.as_ptr(),
                self.target.raw_handle()?.as_ptr(),
                self.target_offset,
                self.width + 1,
                self.draft.raw_handle()?.as_ptr(),
                self.draft_offset,
                self.width,
                self.draft_ids,
                self.width,
                self.width,
                self.seed,
                self.absolute_position,
                self.decision.raw_handle()?.as_ptr(),
                self.decision_offset,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }
}
