//! Phase 19 A0 full-resident admission and actual text-payload upload probe.

use serde::Serialize;
use sllm_core::{
    AllocationCategory, Backend, ExecutionSessionRequest, ExecutionState,
    QWEN35_MOE_TEXT_RESIDENT_BYTES, Qwen35MoeTensorPlane, verify_qwen35_moe_artifact,
};
use sllm_hip::HipBackend;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

const EXECUTION_GUARD: &str = "SLLM_QWEN35_MOE_GPU_EXECUTION";
const REQUEST_STATE_BYTES: u64 = 148_766_720;
const WORKSPACE_BYTES: u64 = 512 * 1024 * 1024;
const TRANSFER_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Eq, PartialEq)]
struct Config {
    device_index: u32,
    target: String,
    cache: PathBuf,
}

#[derive(Clone, Debug)]
struct SourceSpan {
    file: String,
    begin: u64,
    end: u64,
}

#[derive(Serialize)]
struct Cleanup {
    retryable_cleanup: usize,
    durable_quarantine: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    repository: &'static str,
    revision: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    fallback_used: bool,
    text_tensor_count: usize,
    text_source_bytes: u64,
    model_resident_bytes: u64,
    request_state_bytes: u64,
    workspace_bytes: u64,
    available_before_bytes: u64,
    source_span_count: usize,
    transfer_count: u64,
    uploaded_bytes: u64,
    current_bytes_before_cleanup: u64,
    high_water_bytes: u64,
    cleanup: Cleanup,
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("{name} may be supplied only once"));
    }
    Ok(())
}

fn parse_config(arguments: impl IntoIterator<Item = String>) -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut cache = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--device-index" => set_once(
                &mut device_index,
                value.parse::<u32>().map_err(|_| "invalid device index")?,
                "--device-index",
            )?,
            "--target" => {
                if !matches!(value.as_str(), "gfx1030" | "gfx1201") {
                    return Err("target must be gfx1030 or gfx1201".to_owned());
                }
                set_once(&mut target, value, "--target")?;
            }
            "--cache" => set_once(&mut cache, PathBuf::from(value), "--cache")?,
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "--device-index is required".to_owned())?,
        target: target.ok_or_else(|| "--target is required".to_owned())?,
        cache: cache.ok_or_else(|| "--cache is required".to_owned())?,
    })
}

fn coalesced_spans(planes: &[Qwen35MoeTensorPlane]) -> Result<Vec<SourceSpan>, String> {
    let mut sorted: Vec<&Qwen35MoeTensorPlane> = planes.iter().collect();
    sorted.sort_by(|left, right| {
        (&left.source_file, left.absolute_byte_range[0])
            .cmp(&(&right.source_file, right.absolute_byte_range[0]))
    });
    let mut spans: Vec<SourceSpan> = Vec::new();
    for plane in sorted {
        let [begin, end] = plane.absolute_byte_range;
        if end <= begin {
            return Err(format!("empty source plane: {}", plane.source_name));
        }
        if let Some(previous) = spans.last_mut() {
            if previous.file == plane.source_file && previous.end == begin {
                previous.end = end;
                continue;
            }
        }
        spans.push(SourceSpan {
            file: plane.source_file.clone(),
            begin,
            end,
        });
    }
    let total: u64 = spans.iter().map(|span| span.end - span.begin).sum();
    if total != QWEN35_MOE_TEXT_RESIDENT_BYTES {
        return Err("coalesced text source bytes differ".to_owned());
    }
    Ok(spans)
}

fn run(config: &Config) -> Result<Report, String> {
    if env::var(EXECUTION_GUARD).as_deref() != Ok("1") {
        return Err(format!("{EXECUTION_GUARD}=1 is required"));
    }
    let model = verify_qwen35_moe_artifact(&config.cache).map_err(|error| error.to_string())?;
    let spans = coalesced_spans(model.text_planes())?;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| error.to_string())?;
    let available_before_bytes = session
        .available_memory_bytes()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "HIP available memory is absent".to_owned())?;
    let required = QWEN35_MOE_TEXT_RESIDENT_BYTES + REQUEST_STATE_BYTES + WORKSPACE_BYTES;
    if available_before_bytes < required {
        return Err(format!(
            "device admission failed: requires {required}, available {available_before_bytes}"
        ));
    }
    let mut resident_buffers = Vec::with_capacity(spans.len());
    for span in &spans {
        resident_buffers.push(
            session
                .allocate_with_category(span.end - span.begin, AllocationCategory::ModelResident)
                .map_err(|error| error.to_string())?,
        );
    }
    let request_state = session
        .allocate_with_category(REQUEST_STATE_BYTES, AllocationCategory::RequestState)
        .map_err(|error| error.to_string())?;
    let workspace = session
        .allocate_with_category(WORKSPACE_BYTES, AllocationCategory::Workspace)
        .map_err(|error| error.to_string())?;
    let queue = session.create_queue().map_err(|error| error.to_string())?;
    let chunk_limit = session
        .max_transfer_bytes()
        .map_err(|error| error.to_string())?;
    let mut files: BTreeMap<String, File> = BTreeMap::new();
    let mut destination_offset = 0_u64;
    let mut transfer_count = 0_u64;
    for (span, resident) in spans.iter().zip(&resident_buffers) {
        let file = if let Some(file) = files.get_mut(&span.file) {
            file
        } else {
            files.insert(
                span.file.clone(),
                File::open(config.cache.join(&span.file)).map_err(|error| error.to_string())?,
            );
            files.get_mut(&span.file).unwrap()
        };
        file.seek(SeekFrom::Start(span.begin))
            .map_err(|error| error.to_string())?;
        let mut remaining = span.end - span.begin;
        let mut span_destination_offset = 0_u64;
        while remaining != 0 {
            let count = remaining.min(chunk_limit);
            let mut bytes = vec![0_u8; usize::try_from(count).map_err(|_| "chunk too large")?];
            file.read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            let destination = resident
                .range(span_destination_offset, count)
                .map_err(|error| error.to_string())?;
            let mut transfer = session
                .upload(&queue, destination, Arc::from(bytes))
                .map_err(|error| error.to_string())?;
            if transfer
                .wait(TRANSFER_TIMEOUT)
                .map_err(|error| error.to_string())?
                != ExecutionState::Success
            {
                return Err("text payload upload did not succeed".to_owned());
            }
            destination_offset += count;
            span_destination_offset += count;
            remaining -= count;
            transfer_count += 1;
        }
    }
    if destination_offset != QWEN35_MOE_TEXT_RESIDENT_BYTES {
        return Err("uploaded byte count differs".to_owned());
    }
    let before_cleanup = session.memory_snapshot();
    drop(workspace);
    drop(request_state);
    drop(resident_buffers);
    drop(queue);
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| error.to_string())?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("HIP cleanup is not empty".to_owned());
    }
    Ok(Report {
        schema_version: "sllm-qwen35-moe-admission-v1",
        state: "PASS",
        repository: sllm_core::QWEN35_MOE_REPOSITORY,
        revision: sllm_core::QWEN35_MOE_REVISION,
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        fallback_used: false,
        text_tensor_count: model.text_planes().len(),
        text_source_bytes: QWEN35_MOE_TEXT_RESIDENT_BYTES,
        model_resident_bytes: before_cleanup.model_resident().current_bytes(),
        request_state_bytes: before_cleanup.request_state().current_bytes(),
        workspace_bytes: before_cleanup.workspace().current_bytes(),
        available_before_bytes,
        source_span_count: spans.len(),
        transfer_count,
        uploaded_bytes: destination_offset,
        current_bytes_before_cleanup: before_cleanup.current_bytes(),
        high_water_bytes: before_cleanup.high_water_bytes(),
        cleanup: Cleanup {
            retryable_cleanup: cleanup.retryable_cleanup,
            durable_quarantine: cleanup.durable_quarantine,
        },
    })
}

fn main() -> ExitCode {
    match parse_config(env::args().skip(1)).and_then(|config| run(&config)) {
        Ok(report) => {
            println!("{}", serde_json::to_string(&report).unwrap());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("qwen35-moe-admission: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_missing_duplicate_unknown_and_wrong_target() {
        assert!(parse_config(Vec::<String>::new()).is_err());
        let valid = [
            "--device-index",
            "1",
            "--target",
            "gfx1201",
            "--cache",
            "cache",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        assert_eq!(
            parse_config(valid.clone()).unwrap(),
            Config {
                device_index: 1,
                target: "gfx1201".to_owned(),
                cache: PathBuf::from("cache"),
            }
        );
        let mut duplicate = valid.clone();
        duplicate.extend(["--target".to_owned(), "gfx1030".to_owned()]);
        assert!(parse_config(duplicate).is_err());
        let mut wrong = valid;
        wrong[3] = "gfx942".to_owned();
        assert!(parse_config(wrong).is_err());
    }
}
