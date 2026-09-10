//! Private synchronous BF16 row-concatenation bridge.
//!
//! The operation is intentionally absent from the installed C ABI.  Core
//! performs the ownership, range, geometry, and alias checks; this module
//! only lowers checked opaque resources to the native private entry point.

use sllm_hip_sys as sys;

use crate::runtime::{RuntimeError, ensure_ok, sink};
use crate::{Buffer, Context, Queue};

// This bridge is private and versioned independently of the installed public
// runtime header. Offsets are bytes, and native synchronizes the stream before
// returning success.
unsafe extern "C" {
    fn sllm_concat_bf16_rows_v1(
        context: *const sys::sllm_context_t,
        queue: *const sys::sllm_queue_t,
        left_buffer: *const sys::sllm_buffer_t,
        left_offset: u64,
        right_buffer: *const sys::sllm_buffer_t,
        right_offset: u64,
        output_buffer: *const sys::sllm_buffer_t,
        output_offset: u64,
        rows: u64,
        left_columns: u64,
        right_columns: u64,
        error_sink: *mut sys::sllm_error_sink_t,
    ) -> sys::sllm_status_t;
}

pub(crate) struct RowConcat<'a> {
    pub(crate) context: &'a Context,
    pub(crate) queue: &'a Queue,
    pub(crate) left: &'a Buffer,
    pub(crate) left_offset: u64,
    pub(crate) right: &'a Buffer,
    pub(crate) right_offset: u64,
    pub(crate) output: &'a Buffer,
    pub(crate) output_offset: u64,
    pub(crate) rows: u64,
    pub(crate) left_columns: u64,
    pub(crate) right_columns: u64,
}

impl RowConcat<'_> {
    pub(crate) fn run(&self) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sllm_concat_bf16_rows_v1(
                self.context.raw_handle()?.as_ptr(),
                self.queue.raw_handle()?.as_ptr(),
                self.left.raw_handle()?.as_ptr(),
                self.left_offset,
                self.right.raw_handle()?.as_ptr(),
                self.right_offset,
                self.output.raw_handle()?.as_ptr(),
                self.output_offset,
                self.rows,
                self.left_columns,
                self.right_columns,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }
}
