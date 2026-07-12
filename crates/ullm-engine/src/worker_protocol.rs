// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent names for the stable `ullm.worker.v1` protocol.
//!
//! The implementation remains in `sq8_worker_protocol` during the compatibility
//! migration. New backends must depend on this module so a quantization format
//! does not leak into the API/worker boundary.

pub use crate::sq8_worker_protocol::{
    SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES as WORKER_MAX_ERROR_MESSAGE_BYTES,
    SQ8_WORKER_MAX_JSON_DEPTH as WORKER_MAX_JSON_DEPTH,
    SQ8_WORKER_MAX_RECORD_BYTES as WORKER_MAX_RECORD_BYTES,
    SQ8_WORKER_SCHEMA_VERSION as WORKER_SCHEMA_VERSION,
    Sq8ActiveTerminalFlushAck as ActiveTerminalFlushAck,
    Sq8ActiveTerminalPermit as ActiveTerminalPermit, Sq8BoundedJsonlReader as BoundedJsonlReader,
    Sq8CancelReason as CancelReason, Sq8GenerateCommand as GenerateCommand,
    Sq8JsonlFramingError as JsonlFramingError, Sq8JsonlFramingErrorKind as JsonlFramingErrorKind,
    Sq8JsonlRead as JsonlRead, Sq8OrderedJsonlWriter as OrderedJsonlWriter,
    Sq8PromptProgressTracker as PromptProgressTracker, Sq8ReadyFlushAck as ReadyFlushAck,
    Sq8ReleaseOutcomeEvent as ReleaseOutcomeEvent, Sq8WorkerAdmission as WorkerAdmission,
    Sq8WorkerCancelResult as WorkerCancelResult, Sq8WorkerCommand as WorkerCommand,
    Sq8WorkerCommandInspection as WorkerCommandInspection,
    Sq8WorkerCommandKind as WorkerCommandKind, Sq8WorkerControl as WorkerControl,
    Sq8WorkerControlError as WorkerControlError,
    Sq8WorkerControlErrorKind as WorkerControlErrorKind,
    Sq8WorkerControlSnapshot as WorkerControlSnapshot, Sq8WorkerErrorCode as WorkerErrorCode,
    Sq8WorkerEvent as WorkerEvent, Sq8WorkerLifecycle as WorkerLifecycle,
    Sq8WorkerProfile as WorkerProfile, Sq8WorkerProtocolError as WorkerProtocolError,
    Sq8WorkerProtocolErrorKind as WorkerProtocolErrorKind, Sq8WorkerSampling as WorkerSampling,
    Sq8WorkerShutdownDisposition as WorkerShutdownDisposition, Sq8WorkerTimings as WorkerTimings,
    configured_worker_profile, decode_sq8_worker_command as decode_worker_command,
    inspect_sq8_worker_command as inspect_worker_command, validate_worker_request_id,
};
