// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5 AQ4 compatibility backend for the JSONL worker protocol.
//!
//! This is deliberately a compatibility boundary: the worker process remains resident, but each
//! request invokes the existing, verified `package-token-ids-bench` executable path. It makes AQ4
//! available to the product protocol without duplicating the prototype model loop. Moving that
//! model loop into a reusable resident session remains separate work.

use crate::inference_api::InferenceRequest;
use crate::worker_protocol::{ReleaseOutcomeEvent, WorkerAdmission, WorkerTimings};
use crate::worker_runtime::{InferenceBackend, RequestEventPublisher};
use serde::Deserialize;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

const AQ4_CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);
const AQ4_REPORT_MAX_BYTES: u64 = 8 * 1024 * 1024;

/// Complete production guard contract for the resident Qwen3.5 AQ4 execution path.
///
/// Manifest generation and deployment code can use this public list instead of maintaining a
/// second, easily-drifted copy. Compatibility mode also applies the same guards to its child.
pub const QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV: &[&str] = &[
    "ULLM_REQUIRE_HIP_AQ4_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4WorkerBackendConfig {
    engine: PathBuf,
    package: PathBuf,
    device_index: u32,
    layers: String,
    chunk_bytes: usize,
    lm_head_chunk_rows: usize,
}

impl Qwen35Aq4WorkerBackendConfig {
    pub fn new(engine: impl Into<PathBuf>, package: impl Into<PathBuf>) -> Result<Self, String> {
        let value = Self {
            engine: engine.into(),
            package: package.into(),
            // uLLM keeps the host CPU fallback at runtime device zero. With the deployed
            // HIP_VISIBLE_DEVICES isolation, the sole visible R9700 is runtime device one.
            // Direct benchmark users can still override this through --device-index.
            device_index: 1,
            layers: "all".into(),
            chunk_bytes: 1024 * 1024,
            lm_head_chunk_rows: 8192,
        };
        if value.engine.as_os_str().is_empty() || value.package.as_os_str().is_empty() {
            return Err("AQ4 worker engine and package paths must be nonempty".into());
        }
        Ok(value)
    }

    pub fn with_device_index(mut self, device_index: u32) -> Self {
        self.device_index = device_index;
        self
    }

    pub fn with_layers(mut self, layers: impl Into<String>) -> Result<Self, String> {
        let layers = layers.into();
        if layers.is_empty() {
            return Err("AQ4 worker layer selection must be nonempty".into());
        }
        self.layers = layers;
        Ok(self)
    }

    pub fn engine(&self) -> &Path {
        &self.engine
    }

    pub fn package(&self) -> &Path {
        &self.package
    }
}

#[derive(Debug)]
pub struct Qwen35Aq4WorkerBackend {
    config: Qwen35Aq4WorkerBackendConfig,
}

impl Qwen35Aq4WorkerBackend {
    pub fn load(config: Qwen35Aq4WorkerBackendConfig) -> Result<Self, String> {
        if !config.engine.is_file() {
            return Err(format!(
                "AQ4 worker engine is not a file: {}",
                config.engine.display()
            ));
        }
        if !config.package.is_dir() {
            return Err(format!(
                "AQ4 worker package is not a directory: {}",
                config.package.display()
            ));
        }
        Ok(Self { config })
    }

    fn command_args(&self, request: &InferenceRequest) -> Vec<OsString> {
        let prompt = request
            .prompt_token_ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let stop_ids = if request.eos_token_ids.is_empty() {
            "none".to_string()
        } else {
            request
                .eos_token_ids
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        [
            OsString::from("package-token-ids-bench"),
            self.config.package.as_os_str().to_owned(),
            OsString::from(self.config.device_index.to_string()),
            OsString::from(self.config.chunk_bytes.to_string()),
            OsString::from(&self.config.layers),
            OsString::from(prompt),
            OsString::from(request.max_new_tokens.to_string()),
            OsString::from(request.sampling.top_k.to_string()),
            OsString::from(self.config.lm_head_chunk_rows.to_string()),
            OsString::from("64"),
            OsString::from("10000000"),
            OsString::from("0"),
            OsString::from("gpu_resident_f32"),
            OsString::from(stop_ids),
        ]
        .into()
    }

    fn spawn(&self, request: &InferenceRequest) -> Result<Child, String> {
        let mut command = Command::new(&self.config.engine);
        command
            .args(self.command_args(request))
            .env("ULLM_PREFILL_DEVICE_TOKEN_LOOP", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for name in QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV {
            command.env(name, "1");
        }
        command
            .spawn()
            .map_err(|error| format!("failed to spawn AQ4 engine: {error}"))
    }
}

impl InferenceBackend for Qwen35Aq4WorkerBackend {
    fn execute(
        &mut self,
        request: InferenceRequest,
        admission: WorkerAdmission,
        publications: &mut RequestEventPublisher<'_>,
    ) -> Result<(), String> {
        publications.publish_started()?;
        let mut child = self.spawn(&request)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "AQ4 engine stdout pipe is missing".to_string())?;
        let report_reader = thread::Builder::new()
            .name("ullm-aq4-report-reader".into())
            .spawn(move || read_bounded_report(stdout))
            .map_err(|error| format!("failed to spawn AQ4 report reader: {error}"))?;
        let status = loop {
            if admission.cancel.is_cancelled() {
                terminate_child(&mut child)?;
                report_reader
                    .join()
                    .map_err(|_| "AQ4 report reader panicked after cancellation".to_string())??;
                publications.publish_released(ReleaseOutcomeEvent::Cancelled)?;
                return Ok(());
            }
            if let Some(status) = child
                .try_wait()
                .map_err(|error| format!("failed to poll AQ4 engine: {error}"))?
            {
                break status;
            }
            thread::sleep(AQ4_CHILD_POLL_INTERVAL);
        };
        let stdout = report_reader
            .join()
            .map_err(|_| "AQ4 report reader panicked".to_string())??;
        if !status.success() {
            return Err(format!("AQ4 engine exited with status {status}"));
        }
        let report: Aq4GenerateReport = serde_json::from_slice(&stdout)
            .map_err(|error| format!("failed to decode AQ4 engine report: {error}"))?;
        validate_report(&report, &request)?;

        let mut processed_prompt_tokens = 0;
        while processed_prompt_tokens < request.prompt_token_ids.len() {
            let remaining = request.prompt_token_ids.len() - processed_prompt_tokens;
            let execution_width = if remaining >= 128 { 128 } else { 1 };
            processed_prompt_tokens += execution_width;
            publications.observe_prompt_unit(processed_prompt_tokens, execution_width)?;
        }
        publications.observe_prefill_transition()?;
        for &token_id in &report.generated_token_ids {
            publications.publish_token(token_id)?;
        }
        let outcome = if report
            .generated_token_ids
            .last()
            .is_some_and(|token| request.eos_token_ids.contains(token))
        {
            ReleaseOutcomeEvent::Stop
        } else {
            ReleaseOutcomeEvent::Length
        };
        let timings = WorkerTimings::from_elapsed_millis_with_limits(
            request.prompt_token_ids.len(),
            report.prefill.wall_ms.max(0.001),
            report.generated_token_ids.len(),
            report.decode.wall_ms.max(0.001),
            request.prompt_token_ids.len(),
            request.max_new_tokens,
        )
        .ok_or_else(|| "AQ4 engine timings violate the request bounds".to_string())?;
        publications.publish_released_with_timings(outcome, timings)
    }
}

fn terminate_child(child: &mut Child) -> Result<(), String> {
    match child.kill() {
        Ok(()) => {}
        Err(_error) if child.try_wait().ok().flatten().is_some() => {}
        Err(error) => return Err(format!("failed to terminate cancelled AQ4 engine: {error}")),
    }
    child
        .wait()
        .map_err(|error| format!("failed to reap cancelled AQ4 engine: {error}"))?;
    Ok(())
}

fn read_bounded_report(stdout: impl Read) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    stdout
        .take(AQ4_REPORT_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read AQ4 engine report: {error}"))?;
    if bytes.len() as u64 > AQ4_REPORT_MAX_BYTES {
        return Err("AQ4 engine report exceeded the 8 MiB safety limit".into());
    }
    Ok(bytes)
}

#[derive(Debug, Deserialize)]
struct Aq4GenerateReport {
    generated_token_ids: Vec<usize>,
    prefill: Aq4WallTiming,
    decode: Aq4WallTiming,
    verified: bool,
}

#[derive(Debug, Deserialize)]
struct Aq4WallTiming {
    wall_ms: f64,
}

fn validate_report(report: &Aq4GenerateReport, request: &InferenceRequest) -> Result<(), String> {
    if !report.verified
        || report.generated_token_ids.is_empty()
        || report.generated_token_ids.len() > request.max_new_tokens
        || !report.prefill.wall_ms.is_finite()
        || report.prefill.wall_ms < 0.0
        || !report.decode.wall_ms.is_finite()
        || report.decode.wall_ms < 0.0
    {
        return Err("AQ4 engine report violates the worker result contract".into());
    }
    let eos_position = report
        .generated_token_ids
        .iter()
        .position(|token| request.eos_token_ids.contains(token));
    if eos_position.is_some_and(|position| position + 1 != report.generated_token_ids.len()) {
        return Err("AQ4 engine emitted tokens after EOS".into());
    }
    if eos_position.is_none() && report.generated_token_ids.len() != request.max_new_tokens {
        return Err("AQ4 engine stopped before EOS or the requested length".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_api::SamplingParams;

    fn request() -> InferenceRequest {
        let mut request = InferenceRequest::new(
            "aq4-test",
            vec![1, 2, 3],
            3,
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 8,
                seed: 1,
            },
        );
        request.eos_token_ids = vec![9];
        request
    }

    #[test]
    fn command_uses_incremental_aq4_cli_contract_and_eos_ids() {
        let config = Qwen35Aq4WorkerBackendConfig::new("/engine", "/model.ullm.d")
            .unwrap()
            .with_device_index(7)
            .with_layers("0,1")
            .unwrap();
        let backend = Qwen35Aq4WorkerBackend { config };
        let args = backend.command_args(&request());
        let args = args
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(args[0], "package-token-ids-bench");
        assert_eq!(args[1], "/model.ullm.d");
        assert_eq!(args[2], "7");
        assert_eq!(args[4], "0,1");
        assert_eq!(args[5], "1,2,3");
        assert_eq!(args[6], "3");
        assert_eq!(args.last().unwrap(), "9");
    }

    #[test]
    fn report_requires_eos_at_end_or_exact_length() {
        let request = request();
        let report = |tokens| Aq4GenerateReport {
            generated_token_ids: tokens,
            prefill: Aq4WallTiming { wall_ms: 4.0 },
            decode: Aq4WallTiming { wall_ms: 2.0 },
            verified: true,
        };
        assert!(validate_report(&report(vec![4, 9]), &request).is_ok());
        assert!(validate_report(&report(vec![4, 5, 6]), &request).is_ok());
        assert!(validate_report(&report(vec![9, 4]), &request).is_err());
        assert!(validate_report(&report(vec![4]), &request).is_err());
    }
}
