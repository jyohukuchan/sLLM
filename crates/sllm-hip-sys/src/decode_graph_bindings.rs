//! Private whole-decode capture ABI, mirrored from
//! native/hip/src/decode_graph_capture_internal.hpp.
use crate::{
    sllm_completion_t, sllm_context_t, sllm_error_sink_t, sllm_graph_span_t,
    sllm_linear_attention_state_t, sllm_queue_t, sllm_status_t, sllm_tensor_binding_t,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sllm_graph_span_decode_command_desc_t {
    pub struct_size: u32,
    pub abi_version: u32,
    pub info_version: u32,
    pub opcode: u32,
    pub phase_kind: u32,
    pub phase_index: u32,
    pub rows: u32,
    pub input_kind: u32,
    pub row_count: u32,
    pub hidden_width: u32,
    pub token_capacity: u32,
    pub vocabulary_size: u32,
    pub stop_count: u32,
    pub reserved0: u32,
    pub input0: sllm_tensor_binding_t,
    pub input1: sllm_tensor_binding_t,
    pub output: sllm_tensor_binding_t,
    pub stop_ids: sllm_tensor_binding_t,
}

/// Private capture metadata; version 1, layout shared with decode_graph_capture_internal.hpp.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct sllm_graph_span_capture_info_t {
    pub struct_size: u32,
    pub abi_version: u32,
    pub info_version: u32,
    pub reserved0: u32,
    pub actual_node_count: u64,
    pub kernel_node_count: u64,
    pub memcpy_node_count: u64,
    pub logical_plan_count: u64,
}

unsafe extern "C" {
    pub fn sllm_graph_span_bind_linear_state(
        graph: *mut sllm_graph_span_t,
        state: *const sllm_linear_attention_state_t,
        checkpoint_conv: *const sllm_tensor_binding_t,
        checkpoint_recurrent: *const sllm_tensor_binding_t,
        token_count: u32,
        checkpoint_rows: u32,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_select_linear_state(
        graph: *mut sllm_graph_span_t,
        state: *const sllm_linear_attention_state_t,
        token_count: u32,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_prepare_paged_kv(
        graph: *mut sllm_graph_span_t,
        conservative_end: u64,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_publish_state_metadata(
        graph: *mut sllm_graph_span_t,
        expected_initial_position: u64,
        final_position: u64,
        successful_generations: u64,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;

    // Private whole-decode capture ABI (not part of the public C header).
    pub fn sllm_graph_span_begin_capture(
        context: *const sllm_context_t,
        queue: *const sllm_queue_t,
        control: *const sllm_tensor_binding_t,
        graph: *mut *mut sllm_graph_span_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_end_capture(
        graph: *mut sllm_graph_span_t,
        info: *mut sllm_graph_span_capture_info_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_abort_capture(
        graph: *mut *mut sllm_graph_span_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_decode_command(
        graph: *const sllm_graph_span_t,
        command: *const sllm_graph_span_decode_command_desc_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
    pub fn sllm_graph_span_capture_marker(
        graph: *mut sllm_graph_span_t,
        completion: *mut *mut sllm_completion_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> sllm_status_t;
}

// Checked against the native private header on the supported 64-bit ABI.
const _: () = {
    assert!(core::mem::size_of::<sllm_graph_span_capture_info_t>() == 48);
    assert!(core::mem::size_of::<sllm_graph_span_decode_command_desc_t>() == 792);
    assert!(core::mem::offset_of!(sllm_graph_span_decode_command_desc_t, input0) == 56);
    assert!(core::mem::offset_of!(sllm_graph_span_decode_command_desc_t, input1) == 240);
    assert!(core::mem::offset_of!(sllm_graph_span_decode_command_desc_t, output) == 424);
    assert!(core::mem::offset_of!(sllm_graph_span_decode_command_desc_t, stop_ids) == 608);
};
