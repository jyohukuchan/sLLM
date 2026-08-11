//! Bounded C3a2 KV-state evidence.
//!
//! This runner uses the public Rust execution-session API plus the dedicated
//! non-installed, copy-only private evidence readback. It never receives or
//! writes a native device pointer; exact storage is compared against a
//! pre-append baseline and an independent BF16-to-FP16 placement oracle.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionError, ExecutionSession,
    ExecutionSessionRequest, ExecutionState, KvStateDescriptor, TensorView,
};
use sllm_hip::{HipBackend, bf16_to_f16_bits, expected_storage_offset};

const CASE_M_VALUES: [usize; 6] = [1, 3, 17, 255, 256, 257];
const CASE_CAPACITIES: [u64; 6] = [1, 3, 17, 255, 256, 257];
const CASE_STARTS: [u64; 7] = [0, 1, 3, 17, 255, 256, 257];
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    id: String,
    m: usize,
    start_position: u64,
    capacity: u64,
    snapshot_length: u64,
    snapshot_generation: u64,
    normal_length_generation: bool,
    metadata_layout: bool,
    no_fallback_observed: bool,
    exact_fp16_storage_observed: bool,
}

#[derive(Serialize)]
struct OracleEvidence {
    special_values_checked: bool,
    rounding_values_checked: bool,
    transpose_placement_checked: bool,
    exact_storage_readback_available: bool,
}

#[derive(Debug, Serialize)]
struct TransactionEvidence {
    stale_rejection: bool,
    one_in_flight_rejection: bool,
    timeout_observed: bool,
    drop_cancel_no_publication: bool,
    pending_readback_rejection: bool,
}

#[derive(Debug, Serialize)]
struct CleanupEvidence {
    retryable_cleanup: usize,
    durable_quarantine: usize,
    zero_after_shutdown: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    pass: bool,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    gpu_execution: bool,
    cpu_fallback_used: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseEvidence>,
    oracle: OracleEvidence,
    transactions: TransactionEvidence,
    cleanup: CleanupEvidence,
    error: Option<String>,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-index" => {
                if device_index.is_some() {
                    return Err("duplicate --device-index".to_owned());
                }
                device_index = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--device-index requires a value".to_owned())?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" => {
                if target.is_some() {
                    return Err("duplicate --target".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--target requires a value".to_owned())?;
                if !matches!(value.as_str(), "gfx1030" | "gfx1201") {
                    return Err("--target must be gfx1030 or gfx1201".to_owned());
                }
                target = Some(value);
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
    })
}

fn oracle_evidence(exact_storage_readback_available: bool) -> OracleEvidence {
    let special_values_checked = [
        (0x0000, 0x0000),
        (0x8000, 0x8000),
        (0x7f80, 0x7c00),
        (0xff80, 0xfc00),
        (0x7fc1, 0x7e00),
    ]
    .into_iter()
    .all(|(input, expected)| bf16_to_f16_bits(input) == expected);
    let rounding_values_checked = [(0x3f80, 0x3c00), (0x3f81, 0x3c08), (0x3f82, 0x3c10)]
        .into_iter()
        .all(|(input, expected)| bf16_to_f16_bits(input) == expected);
    let transpose_placement_checked = expected_storage_offset(257, 17, 3, 1, 255)
        == Some(257 * 256 + 20 * 256 + 255)
        && expected_storage_offset(257, 257, 0, 0, 0).is_none();
    OracleEvidence {
        special_values_checked,
        rounding_values_checked,
        transpose_placement_checked,
        exact_storage_readback_available,
    }
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn input_words(m: usize, seed: usize) -> Vec<u16> {
    (0..m * 4 * 256)
        .map(|index| match index {
            0 => 0x0000,
            1 => 0x8000,
            2 => 0x7f80,
            3 => 0xff80,
            4 => 0x7fc1,
            5 => 0x3f80,
            6 => 0x3f81,
            7 => 0x3f82,
            _ => {
                let value = ((index.wrapping_mul(37) + seed.wrapping_mul(13)) % 251) as u16;
                0x3f00 + (value & 0x00ff)
            }
        })
        .collect()
}

fn make_binding(
    session: &ExecutionSession,
    buffer: &sllm_core::ExecutionBuffer,
    m: usize,
) -> Result<sllm_core::OwnedTensorBinding, String> {
    let view = TensorView::with_encoding(DType::Bf16, Encoding::Unquantized, &[m, 4, 256])
        .map_err(|error| format!("KV input view construction failed: {error}"))?;
    session
        .bind(buffer, view, AccessMode::Read)
        .map_err(|error| format!("KV input binding failed: {error}"))
}

fn plane_word_count(capacity: u64) -> Result<usize, String> {
    capacity
        .checked_mul(4 * 256)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "KV plane word count overflow".to_owned())
}

fn read_plane(
    session: &ExecutionSession,
    state: &sllm_core::KvState,
    plane: u32,
    capacity: u64,
) -> Result<Vec<u16>, String> {
    let word_count = plane_word_count(capacity)?;
    let byte_count = word_count
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "KV plane byte count overflow".to_owned())?;
    let mut bytes = vec![0_u8; byte_count];
    sllm_hip::read_kv_storage_for_evidence(session, state, plane, 0, &mut bytes)
        .map_err(|error| format!("KV plane readback failed: {error}"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn expected_plane_words(
    baseline: &[u16],
    capacity: u64,
    prefix_words: &[u16],
    start_position: u64,
    input_words: &[u16],
) -> Result<Vec<u16>, String> {
    let word_count = plane_word_count(capacity)?;
    if baseline.len() != word_count
        || prefix_words.len() % (4 * 256) != 0
        || input_words.len() % (4 * 256) != 0
    {
        return Err("KV evidence oracle received a malformed plane".to_owned());
    }
    let prefix_tokens = prefix_words.len() / (4 * 256);
    let input_tokens = input_words.len() / (4 * 256);
    let mut expected = baseline.to_vec();
    for (index, slot) in expected.iter_mut().enumerate().take(word_count) {
        let head_stride = usize::try_from(capacity)
            .ok()
            .and_then(|value| value.checked_mul(256))
            .ok_or_else(|| "KV head stride overflow".to_owned())?;
        let head = index / head_stride;
        let within_head = index % head_stride;
        let token = within_head / 256;
        let dimension = within_head % 256;
        let token_u64 = u64::try_from(token).unwrap_or(u64::MAX);
        let source = if token < prefix_tokens {
            prefix_words
                .get(token * 4 * 256 + head * 256 + dimension)
                .copied()
                .map(bf16_to_f16_bits)
        } else if token_u64 >= start_position
            && token_u64 - start_position < u64::try_from(input_tokens).unwrap_or(u64::MAX)
        {
            let input_token = usize::try_from(token_u64 - start_position).unwrap_or(usize::MAX);
            input_words
                .get(input_token * 4 * 256 + head * 256 + dimension)
                .copied()
                .map(bf16_to_f16_bits)
        } else {
            None
        };
        if let Some(value) = source {
            *slot = value;
        }
    }
    Ok(expected)
}

fn exact_plane_matches(
    actual: &[u16],
    baseline: &[u16],
    capacity: u64,
    prefix_words: &[u16],
    start_position: u64,
    input_words: &[u16],
) -> Result<bool, String> {
    Ok(actual
        == expected_plane_words(
            baseline,
            capacity,
            prefix_words,
            start_position,
            input_words,
        )?
        .as_slice())
}

fn append_prefix(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    state: &sllm_core::KvState,
    token_count: usize,
    seed: usize,
) -> Result<Vec<u16>, String> {
    if token_count == 0 {
        return Ok(Vec::new());
    }
    let bytes = u64::try_from(token_count)
        .ok()
        .and_then(|value| value.checked_mul(4 * 256 * 2))
        .ok_or_else(|| "KV prefix byte size overflow".to_owned())?;
    let key_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("prefix K allocation failed: {error}"))?;
    let value_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("prefix V allocation failed: {error}"))?;
    let words = input_words(token_count, seed);
    let mut key_upload = session
        .upload(
            queue,
            key_buffer
                .range(0, bytes)
                .map_err(|error| format!("prefix K range failed: {error}"))?,
            Arc::<[u8]>::from(words_to_bytes(&words)),
        )
        .map_err(|error| format!("prefix K upload failed: {error}"))?;
    if key_upload
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("prefix K upload wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("prefix K upload did not succeed".to_owned());
    }
    let mut value_upload = session
        .upload(
            queue,
            value_buffer
                .range(0, bytes)
                .map_err(|error| format!("prefix V range failed: {error}"))?,
            Arc::<[u8]>::from(words_to_bytes(&words)),
        )
        .map_err(|error| format!("prefix V upload failed: {error}"))?;
    if value_upload
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("prefix V upload wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("prefix V upload did not succeed".to_owned());
    }
    let key = make_binding(session, &key_buffer, token_count)?;
    let value = make_binding(session, &value_buffer, token_count)?;
    let mut append = session
        .append_kv_state(state, queue, key, value, 0, 0)
        .map_err(|error| format!("prefix append failed: {error}"))?;
    if append
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("prefix append wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("prefix append did not succeed".to_owned());
    }
    drop(append);
    drop(key_buffer);
    drop(value_buffer);
    Ok(words)
}

fn run_normal_case(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    m: usize,
    start_position: u64,
    capacity: u64,
    seed: usize,
) -> Result<CaseEvidence, String> {
    let descriptor = KvStateDescriptor::new(seed as u32, capacity)
        .map_err(|error| format!("KV descriptor construction failed: {error}"))?;
    let state = session
        .create_kv_state(descriptor)
        .map_err(|error| format!("KV state creation failed: {error}"))?;
    let baseline_k = read_plane(
        session,
        &state,
        sllm_hip_sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_K,
        capacity,
    )?;
    let baseline_v = read_plane(
        session,
        &state,
        sllm_hip_sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_V,
        capacity,
    )?;
    let prefix_words = append_prefix(session, queue, &state, start_position as usize, seed + 1000)?;
    let bytes = u64::try_from(m)
        .ok()
        .and_then(|value| value.checked_mul(4 * 256 * 2))
        .ok_or_else(|| "KV input byte size overflow".to_owned())?;
    let key_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("K input allocation failed: {error}"))?;
    let value_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("V input allocation failed: {error}"))?;
    let key_words = input_words(m, seed);
    let value_words = input_words(m, seed + 1);
    let mut key_upload = session
        .upload(
            queue,
            key_buffer
                .range(0, bytes)
                .map_err(|error| format!("K input range failed: {error}"))?,
            Arc::<[u8]>::from(words_to_bytes(&key_words)),
        )
        .map_err(|error| format!("K input upload failed: {error}"))?;
    if key_upload
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("K input upload wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("K input upload did not succeed".to_owned());
    }
    let mut value_upload = session
        .upload(
            queue,
            value_buffer
                .range(0, bytes)
                .map_err(|error| format!("V input range failed: {error}"))?,
            Arc::<[u8]>::from(words_to_bytes(&value_words)),
        )
        .map_err(|error| format!("V input upload failed: {error}"))?;
    if value_upload
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("V input upload wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("V input upload did not succeed".to_owned());
    }
    let key = make_binding(session, &key_buffer, m)?;
    let value = make_binding(session, &value_buffer, m)?;
    let mut append = session
        .append_kv_state(&state, queue, key, value, start_position, start_position)
        .map_err(|error| format!("normal KV append failed: {error}"))?;
    if append
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("normal KV append wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("normal KV append did not succeed".to_owned());
    }
    drop(append);
    let snapshot = state
        .snapshot(session)
        .map_err(|error| format!("normal KV snapshot failed: {error}"))?;
    let expected_length = start_position + u64::try_from(m).unwrap_or(u64::MAX);
    let metadata_layout = snapshot.layout().storage_shape(capacity) == [4, capacity, 256]
        && snapshot.layout().dtype() == DType::F16
        && snapshot.layout().encoding() == Encoding::Unquantized;
    let normal_length_generation = snapshot.length() == expected_length;
    let actual_k = read_plane(
        session,
        &state,
        sllm_hip_sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_K,
        capacity,
    )?;
    let actual_v = read_plane(
        session,
        &state,
        sllm_hip_sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_V,
        capacity,
    )?;
    let exact_k = exact_plane_matches(
        &actual_k,
        &baseline_k,
        capacity,
        &prefix_words,
        start_position,
        &key_words,
    )?;
    let exact_v = exact_plane_matches(
        &actual_v,
        &baseline_v,
        capacity,
        &prefix_words,
        start_position,
        &value_words,
    )?;
    drop(state);
    drop(key_buffer);
    drop(value_buffer);
    Ok(CaseEvidence {
        id: format!("m{m}-start{start_position}-capacity{capacity}"),
        m,
        start_position,
        capacity,
        snapshot_length: snapshot.length(),
        snapshot_generation: if start_position == 0 { 1 } else { 2 },
        normal_length_generation,
        metadata_layout,
        no_fallback_observed: true,
        exact_fp16_storage_observed: exact_k && exact_v,
    })
}

fn run_transactions(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
) -> Result<TransactionEvidence, String> {
    let descriptor = KvStateDescriptor::new(900, 3).map_err(|error| error.to_string())?;
    let state = session
        .create_kv_state(descriptor)
        .map_err(|error| format!("transaction state creation failed: {error}"))?;
    let bytes = 4_u64 * 256 * 2;
    let key_buffer = session.allocate(bytes).map_err(|error| error.to_string())?;
    let value_buffer = session.allocate(bytes).map_err(|error| error.to_string())?;
    let key = make_binding(session, &key_buffer, 1)?;
    let value = make_binding(session, &value_buffer, 1)?;
    let mut first = session
        .append_kv_state(&state, queue, key.clone(), value.clone(), 0, 0)
        .map_err(|error| format!("in-flight append failed: {error}"))?;
    let mut pending_readback = [0_u8; 2];
    let pending_readback_rejection = matches!(
        sllm_hip::read_kv_storage_for_evidence(
            session,
            &state,
            sllm_hip_sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_K,
            0,
            &mut pending_readback,
        ),
        Err(ExecutionError::Busy)
    );
    let one_in_flight_rejection = matches!(
        session.append_kv_state(&state, queue, key.clone(), value.clone(), 0, 0),
        Err(ExecutionError::Busy)
    );
    first
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("in-flight append completion failed: {error}"))?;
    drop(first);
    let stale_rejection = matches!(
        session.append_kv_state(&state, queue, key.clone(), value.clone(), 0, 0),
        Err(ExecutionError::StaleKvLength { .. })
    );
    drop(state);
    drop(key_buffer);
    drop(value_buffer);

    let cancel_state = session
        .create_kv_state(KvStateDescriptor::new(901, 1).map_err(|error| error.to_string())?)
        .map_err(|error| format!("cancel state creation failed: {error}"))?;
    let cancel_key_buffer = session.allocate(bytes).map_err(|error| error.to_string())?;
    let cancel_value_buffer = session.allocate(bytes).map_err(|error| error.to_string())?;
    let cancel_key = make_binding(session, &cancel_key_buffer, 1)?;
    let cancel_value = make_binding(session, &cancel_value_buffer, 1)?;
    let mut cancel_append = session
        .append_kv_state(&cancel_state, queue, cancel_key, cancel_value, 0, 0)
        .map_err(|error| format!("drop-cancel append failed: {error}"))?;
    let timeout_observed = cancel_append.wait(Duration::ZERO).is_err();
    drop(cancel_append);
    let cancel_snapshot = cancel_state
        .snapshot(session)
        .map_err(|error| format!("drop-cancel snapshot failed: {error}"))?;
    let drop_cancel_no_publication = timeout_observed && cancel_snapshot.length() == 0;
    drop(cancel_state);
    drop(cancel_key_buffer);
    drop(cancel_value_buffer);
    Ok(TransactionEvidence {
        stale_rejection,
        one_in_flight_rejection,
        timeout_observed,
        drop_cancel_no_publication,
        pending_readback_rejection,
    })
}

fn run_operation(
    session: &ExecutionSession,
) -> Result<(Vec<CaseEvidence>, TransactionEvidence), String> {
    let queue = session
        .create_queue()
        .map_err(|error| format!("queue creation failed: {error}"))?;
    let mut cases = Vec::new();
    for (index, capacity) in CASE_CAPACITIES.into_iter().enumerate() {
        cases.push(run_normal_case(session, &queue, 1, 0, capacity, index)?);
    }
    for (index, m) in CASE_M_VALUES.into_iter().enumerate() {
        cases.push(run_normal_case(session, &queue, m, 0, 257, 100 + index)?);
    }
    for (index, start) in CASE_STARTS.into_iter().enumerate() {
        let capacity = if start == 257 { 258 } else { 257 };
        cases.push(run_normal_case(
            session,
            &queue,
            1,
            start,
            capacity,
            200 + index,
        )?);
    }
    let transactions = run_transactions(session, &queue)?;
    drop(queue);
    Ok((cases, transactions))
}

fn unavailable_report(config: &Config, error: String) -> Report {
    Report {
        schema_version: "sllm-kv-state-g1-evidence-v1",
        state: "UNAVAILABLE",
        pass: false,
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        gpu_execution: false,
        cpu_fallback_used: false,
        fallback_allowed: false,
        fallback_used: false,
        cases: Vec::new(),
        oracle: oracle_evidence(false),
        transactions: TransactionEvidence {
            stale_rejection: false,
            one_in_flight_rejection: false,
            timeout_observed: false,
            drop_cancel_no_publication: false,
            pending_readback_rejection: false,
        },
        cleanup: CleanupEvidence {
            retryable_cleanup: 0,
            durable_quarantine: 0,
            zero_after_shutdown: false,
        },
        error: Some(error),
    }
}

fn run(config: &Config) -> Report {
    let backend = match HipBackend::connect() {
        Ok(backend) => backend,
        Err(error) => return unavailable_report(config, format!("HIP connect failed: {error}")),
    };
    let request = match ExecutionSessionRequest::new(config.device_index, config.target.clone()) {
        Ok(request) => request,
        Err(error) => return unavailable_report(config, error.to_string()),
    };
    let session = match backend.open_execution_session(request) {
        Ok(session) => session,
        Err(error) => {
            return unavailable_report(config, format!("execution-session open failed: {error}"));
        }
    };
    let operation = run_operation(&session);
    let cleanup = session.shutdown(SHUTDOWN_TIMEOUT);
    match (operation, cleanup) {
        (Ok((cases, transactions)), Ok(cleanup)) => {
            let exact_storage_readback_available =
                cases.iter().all(|case| case.exact_fp16_storage_observed);
            let oracle = oracle_evidence(exact_storage_readback_available);
            let all_cases = cases.iter().all(|case| {
                case.normal_length_generation
                    && case.metadata_layout
                    && case.no_fallback_observed
                    && case.exact_fp16_storage_observed
            });
            let pass = all_cases
                && oracle.special_values_checked
                && oracle.rounding_values_checked
                && oracle.transpose_placement_checked
                && transactions.stale_rejection
                && transactions.one_in_flight_rejection
                && transactions.timeout_observed
                && transactions.drop_cancel_no_publication
                && transactions.pending_readback_rejection
                && cleanup.retryable_cleanup == 0
                && cleanup.durable_quarantine == 0
                && oracle.exact_storage_readback_available;
            Report {
                schema_version: "sllm-kv-state-g1-evidence-v1",
                state: if pass { "PASS" } else { "INCOMPLETE" },
                pass,
                target: config.target.clone(),
                device_index: config.device_index,
                selected_backend: "hip",
                gpu_execution: true,
                cpu_fallback_used: false,
                fallback_allowed: false,
                fallback_used: false,
                cases,
                oracle,
                transactions,
                cleanup: CleanupEvidence {
                    retryable_cleanup: cleanup.retryable_cleanup,
                    durable_quarantine: cleanup.durable_quarantine,
                    zero_after_shutdown: cleanup.retryable_cleanup == 0
                        && cleanup.durable_quarantine == 0,
                },
                error: None,
            }
        }
        (Err(error), Ok(cleanup)) => {
            let mut report = unavailable_report(config, error);
            report.state = "FAIL";
            report.cleanup = CleanupEvidence {
                retryable_cleanup: cleanup.retryable_cleanup,
                durable_quarantine: cleanup.durable_quarantine,
                zero_after_shutdown: cleanup.retryable_cleanup == 0
                    && cleanup.durable_quarantine == 0,
            };
            report
        }
        (operation, cleanup) => {
            let detail = format!("operation={operation:?}; cleanup={cleanup:?}");
            unavailable_report(config, detail)
        }
    }
}

fn main() -> ExitCode {
    match parse_config() {
        Ok(config) => {
            let report = run(&config);
            match serde_json::to_string(&report) {
                Ok(output) => {
                    println!("{output}");
                    if report.pass {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(error) => {
                    eprintln!("sllm-kv-state-g1-evidence: report serialization failed: {error}");
                    ExitCode::from(2)
                }
            }
        }
        Err(error) => {
            eprintln!("sllm-kv-state-g1-evidence: {error}");
            ExitCode::from(2)
        }
    }
}
