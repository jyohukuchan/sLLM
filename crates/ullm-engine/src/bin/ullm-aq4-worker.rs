// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::ffi::OsString;
use std::fs::{Metadata, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use ullm_engine::aq4_benchmark_worker_protocol::{
    Aq4BenchmarkTrustedCaseRegistry, decode_aq4_benchmark_case_registry,
};
use ullm_engine::aq4_benchmark_worker_runtime::run_aq4_benchmark_worker_process;
use ullm_engine::aq4_worker_backend::{
    QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV, Qwen35Aq4WorkerBackend, Qwen35Aq4WorkerBackendConfig,
};
use ullm_engine::qwen35_aq4_direct_trace::Qwen35Aq4DirectTraceBinding;
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig,
    activate_p3_production_direct_sequence_output, direct_trace_diagnostic_enabled_value,
};
use ullm_engine::qwen35_aq4_session::{Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig};
use ullm_engine::served_model::{
    ServedModel, ServedModelError, WorkerBackendKind, WorkerStartupConfig, load_served_model,
};
use ullm_engine::session_worker_backend::SessionInferenceBackend;
use ullm_engine::sq8_worker_protocol::Sq8WorkerProfile;
use ullm_engine::worker_runtime::{InferenceBackend, run_worker_process_with_profile};

const PROCESS_IO_BUFFER_BYTES: usize = 64 * 1024;
const RESIDENT_DEVICE_INDEX: u32 = 1;
const RESIDENT_CHUNK_BYTES: usize = 1024 * 1024;
const RESIDENT_LM_HEAD_CHUNK_ROWS: usize = 8192;
const P3_DIRECT_TRACE_BINDING_MANIFEST_SCHEMA: &str =
    "ullm.aq4_p3_candidate_a.direct_trace_binding_manifest.v1";
const P3_DIRECT_TRACE_BINDING_MANIFEST_MAX_ENTRIES: usize = 64;
const P3_PRODUCTION_ACTIVATION_SCHEMA: &str = "ullm.aq4_p3_candidate_a.production_activation.v1";

#[derive(Debug, PartialEq, Eq)]
struct WorkerArgs {
    engine: PathBuf,
    package: PathBuf,
    device_index: u32,
    layers: String,
}

#[derive(Debug, PartialEq, Eq)]
enum WorkerSource {
    Legacy(WorkerArgs),
    ServedModelManifest {
        served_model: PathBuf,
        direct_trace_bindings: Option<DiagnosticBindingSource>,
        production_activation: Option<DiagnosticBindingSource>,
    },
    BenchmarkServedModelManifest {
        served_model: PathBuf,
        case_registry: PathBuf,
        case_registry_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticBindingSource {
    path: PathBuf,
    bytes_sha256: String,
}

enum CliAction {
    Run(WorkerSource),
    Help,
    Version,
}

#[derive(Debug, Clone)]
struct ResidentWorkerConfig {
    model: Qwen35Aq4ModelLoadConfig,
    session: Qwen35Aq4SessionConfig,
    expected_vocab_size: usize,
}

enum LoadedWorker {
    Legacy {
        config: Qwen35Aq4WorkerBackendConfig,
        profile: Sq8WorkerProfile,
    },
    Resident {
        config: ResidentWorkerConfig,
        profile: Sq8WorkerProfile,
        direct_trace_bindings: Option<VecDeque<Qwen35Aq4DirectTraceBinding>>,
    },
    BenchmarkResident {
        config: ResidentWorkerConfig,
        profile: Sq8WorkerProfile,
        registry: Aq4BenchmarkTrustedCaseRegistry,
    },
}

fn main() -> ExitCode {
    match parse_cli(env::args_os().skip(1)) {
        Ok(CliAction::Help) => {
            eprintln!(
                "Usage: ullm-aq4-worker [--engine PATH] --package PATH [--device-index N] [--layers all|CSV]\n\
                 Gateway form: --artifact AQ4_PACKAGE --package COMPAT_PATH [extra options]\n\
                 Manifest mode: ullm-aq4-worker --served-model-manifest PATH\n\
                 P3 diagnostic mode: --served-model-manifest PATH --p3-direct-trace-binding-manifest PATH --p3-direct-trace-binding-manifest-sha256 SHA256\n\
                 P3 production mode: --served-model-manifest PATH --p3-production-activation PATH --p3-production-activation-sha256 SHA256\n\
                 Benchmark mode: ullm-aq4-worker --served-model-manifest PATH --benchmark-wire --benchmark-case-manifest PATH --benchmark-case-manifest-sha256 SHA256\n\
                 Reads ullm.worker.v1/v2 commands from stdin and writes matching events to stdout.\n\
                 Compatibility mode invokes the AQ4 engine CLI once per request.\n\
                 Manifest mode loads one resident AQ4 model and never invokes a sibling engine."
            );
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            eprintln!("ullm-aq4-worker {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(args)) => run_worker(args),
        Err(error) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), Some(&error));
            ExitCode::FAILURE
        }
    }
}

fn run_worker(source: WorkerSource) -> ExitCode {
    let loaded = match load_worker(source) {
        Ok(loaded) => loaded,
        Err(_) => {
            write_process_log("error", "manifest_failed", Some("invalid_manifest"), None);
            return ExitCode::FAILURE;
        }
    };
    let input = BufReader::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdin());
    let output = BufWriter::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdout());
    let result = match loaded {
        LoadedWorker::BenchmarkResident {
            config,
            profile,
            registry,
        } => run_aq4_benchmark_worker_process(input, output, profile, registry, move || {
            load_resident_backend(config)
        })
        .map(|_| ()),
        loaded => run_loaded_worker(
            loaded,
            input,
            output,
            Qwen35Aq4WorkerBackend::load,
            load_resident_backend,
        )
        .map(|_| ()),
    };
    match result {
        Ok(_) => {
            write_process_log("info", "process_stopped", None, None);
            ExitCode::SUCCESS
        }
        Err(error) => {
            write_process_log(
                "error",
                "process_failed",
                Some("process_failed"),
                Some(&error),
            );
            ExitCode::FAILURE
        }
    }
}

fn load_worker(source: WorkerSource) -> Result<LoadedWorker, ServedModelError> {
    let diagnostic_gate = direct_trace_diagnostic_enabled_value(
        env::var("ULLM_AQ4_P3_DIRECT_TRACE_DIAGNOSTIC")
            .ok()
            .as_deref(),
    );
    let direct_requested = direct_trace_diagnostic_enabled_value(
        env::var("ULLM_AQ4_PREFILL_DIRECT_SEQUENCE_OUTPUT")
            .ok()
            .as_deref(),
    );
    match source {
        WorkerSource::Legacy(args) => {
            if diagnostic_gate || direct_requested {
                return Err(ServedModelError(
                    "AQ4 P3 diagnostic gate requires manifest resident mode".into(),
                ));
            }
            let config = Qwen35Aq4WorkerBackendConfig::new(args.engine, args.package)
                .map(|config| config.with_device_index(args.device_index))
                .and_then(|config| config.with_layers(args.layers))
                .map_err(ServedModelError)?;
            Ok(LoadedWorker::Legacy {
                config,
                profile: configured_aq4_worker_profile(),
            })
        }
        WorkerSource::ServedModelManifest {
            served_model,
            direct_trace_bindings,
            production_activation,
        } => {
            validate_direct_trace_ingress(
                diagnostic_gate,
                direct_trace_bindings.is_some(),
                production_activation.is_some(),
                direct_requested,
            )?;
            let activation_build = production_activation
                .map(|source| load_p3_production_activation(&source.path, &source.bytes_sha256))
                .transpose()?;
            let bindings = direct_trace_bindings
                .map(|source| {
                    load_direct_trace_binding_manifest(&source.path, &source.bytes_sha256)
                })
                .transpose()?;
            let model = load_served_model(served_model)?;
            if let Some(build) = activation_build {
                let package_content = model
                    .product
                    .artifact
                    .as_ref()
                    .map(|artifact| artifact.content_sha256.as_str())
                    .ok_or_else(|| {
                        ServedModelError(
                            "AQ4 P3 production activation requires artifact content identity"
                                .into(),
                        )
                    })?;
                if build.binary_sha256 != model.worker.binary_sha256
                    || build.package_content_sha256 != package_content
                {
                    return Err(ServedModelError(
                        "AQ4 P3 activation build differs from served model".into(),
                    ));
                }
                activate_p3_production_direct_sequence_output();
            }
            let current_exe =
                env::current_exe().map_err(|error| ServedModelError(error.to_string()))?;
            let mut loaded = load_resident_worker(&model, &current_exe)?;
            let LoadedWorker::Resident {
                direct_trace_bindings,
                ..
            } = &mut loaded
            else {
                return Err(ServedModelError(
                    "AQ4 P3 diagnostic bindings require a resident worker".into(),
                ));
            };
            *direct_trace_bindings = bindings;
            Ok(loaded)
        }
        WorkerSource::BenchmarkServedModelManifest {
            served_model,
            case_registry,
            case_registry_sha256,
        } => {
            if diagnostic_gate || direct_requested {
                return Err(ServedModelError(
                    "AQ4 P3 diagnostic gate is not admitted on benchmark wire".into(),
                ));
            }
            let model = load_served_model(served_model)?;
            let registry = load_benchmark_case_registry(&case_registry, &case_registry_sha256)?;
            let current_exe =
                env::current_exe().map_err(|error| ServedModelError(error.to_string()))?;
            match load_resident_worker(&model, &current_exe)? {
                LoadedWorker::Resident {
                    config,
                    profile,
                    direct_trace_bindings: None,
                } => Ok(LoadedWorker::BenchmarkResident {
                    config,
                    profile,
                    registry,
                }),
                _ => Err(ServedModelError(
                    "AQ4 benchmark wire requires a resident served model".into(),
                )),
            }
        }
    }
}

fn validate_direct_trace_ingress(
    diagnostic_gate: bool,
    binding_manifest_configured: bool,
    production_activation_configured: bool,
    direct_requested: bool,
) -> Result<(), ServedModelError> {
    if diagnostic_gate != binding_manifest_configured {
        return Err(ServedModelError(
            "AQ4 P3 diagnostic gate and binding manifest must be enabled together".into(),
        ));
    }
    if diagnostic_gate && production_activation_configured {
        return Err(ServedModelError(
            "AQ4 P3 diagnostic and production activation modes are mutually exclusive".into(),
        ));
    }
    if direct_requested != (diagnostic_gate || production_activation_configured) {
        return Err(ServedModelError(
            "AQ4 P3 direct route requires exactly one diagnostic or production activation gate"
                .into(),
        ));
    }
    Ok(())
}

fn load_benchmark_case_registry(
    path: &Path,
    expected_bytes_sha256: &str,
) -> Result<Aq4BenchmarkTrustedCaseRegistry, ServedModelError> {
    load_benchmark_case_registry_with_hook(path, expected_bytes_sha256, |_| {})
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectTraceBindingManifest {
    schema_version: String,
    bindings: Vec<Qwen35Aq4DirectTraceBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ProductionActivation {
    schema_version: String,
    status: String,
    activation_sha256: String,
    candidate: P3ActivationCandidate,
    build: P3ActivationBuild,
    profile: P3ActivationProfile,
    selection: P3ActivationSelection,
    upstream_qualification: P3ActivationQualification,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationCandidate {
    candidate_id: String,
    family: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationBuild {
    identity_sha256: String,
    case_manifest_sha256: String,
    binary_sha256: String,
    package_content_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationProfile {
    selection_file_sha256: String,
    raw_evidence: Vec<P3ActivationRawReference>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationRawReference {
    path: PathBuf,
    sha256: String,
    evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationSelection {
    path: PathBuf,
    sha256: String,
    status: String,
    selected_candidate_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct P3ActivationQualification {
    path: PathBuf,
    sha256: String,
    qualification_sha256: String,
    status: String,
    promotion_eligible: bool,
    reason: String,
}

fn load_bound_p3_bytes(path: &Path, expected_sha256: &str) -> Result<Vec<u8>, ServedModelError> {
    if !path.is_absolute()
        || std::fs::canonicalize(path)
            .map(|canonical| canonical != path)
            .unwrap_or(true)
        || !is_lowercase_sha256(expected_sha256)
    {
        return Err(ServedModelError(
            "AQ4 P3 bound file path or SHA-256 differs".into(),
        ));
    }
    reject_direct_trace_binding_symlink_components(path)?;
    let before = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 P3 bound file metadata failed".into()))?;
    if !before.file_type().is_file() || before.nlink() != 1 {
        return Err(ServedModelError(
            "AQ4 P3 bound file must be single-link regular".into(),
        ));
    }
    const O_NOFOLLOW: i32 = 0o400000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|_| ServedModelError("AQ4 P3 bound file open failed".into()))?;
    let identity = RegistryFileIdentity::from(&before);
    if RegistryFileIdentity::from(
        &file
            .metadata()
            .map_err(|_| ServedModelError("AQ4 P3 bound file metadata failed".into()))?,
    ) != identity
    {
        return Err(ServedModelError(
            "AQ4 P3 bound file changed while opening".into(),
        ));
    }
    let maximum = ullm_engine::worker_protocol::WORKER_MAX_RECORD_BYTES;
    let mut payload = Vec::with_capacity(usize::try_from(identity.size).unwrap_or(maximum));
    file.take(u64::try_from(maximum).unwrap_or(u64::MAX) + 1)
        .read_to_end(&mut payload)
        .map_err(|_| ServedModelError("AQ4 P3 bound file read failed".into()))?;
    if payload.is_empty() || payload.len() > maximum {
        return Err(ServedModelError("AQ4 P3 bound file size differs".into()));
    }
    reject_direct_trace_binding_symlink_components(path)?;
    let after = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 P3 bound file metadata failed".into()))?;
    if RegistryFileIdentity::from(&after) != identity {
        return Err(ServedModelError(
            "AQ4 P3 bound file changed while reading".into(),
        ));
    }
    if format!("{:x}", Sha256::digest(&payload)) != expected_sha256 {
        return Err(ServedModelError("AQ4 P3 bound file SHA-256 differs".into()));
    }
    Ok(payload)
}

fn p3_canonical_self_hash(
    value: &serde_json::Value,
    field: &str,
) -> Result<String, ServedModelError> {
    let mut clone = value.clone();
    clone
        .as_object_mut()
        .ok_or_else(|| ServedModelError("AQ4 P3 object is invalid".into()))?
        .insert(field.to_string(), serde_json::Value::Null);
    let bytes = serde_json::to_vec(&clone)
        .map_err(|_| ServedModelError("AQ4 P3 canonical JSON failed".into()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn p3_exact_keys(value: &serde_json::Value, expected: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == expected.len() && expected.iter().all(|field| object.contains_key(*field))
    })
}

fn p3_load_qualification_ref(
    p2: &serde_json::Value,
    field: &str,
) -> Result<serde_json::Value, ServedModelError> {
    let reference = p2
        .get(field)
        .ok_or_else(|| ServedModelError("AQ4 P3 qualification reference is missing".into()))?;
    if !p3_exact_keys(reference, &["path", "sha256"]) {
        return Err(ServedModelError(
            "AQ4 P3 qualification reference fields differ".into(),
        ));
    }
    let path = reference
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ServedModelError("AQ4 P3 qualification reference path differs".into()))?;
    let sha = reference
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ServedModelError("AQ4 P3 qualification reference SHA differs".into()))?;
    let bytes = load_bound_p3_bytes(Path::new(path), sha)?;
    serde_json::from_slice(&bytes)
        .map_err(|_| ServedModelError("AQ4 P3 qualification reference JSON is invalid".into()))
}

fn validate_p3_qualified_success_chain(
    qualification: &serde_json::Value,
) -> Result<(), ServedModelError> {
    let p2 = qualification
        .get("p2")
        .ok_or_else(|| ServedModelError("AQ4 P3 qualification P2 chain is absent".into()))?;
    let fields = [
        "plan",
        "actual_receipt",
        "policy",
        "calibration_metrics",
        "freeze_receipt",
        "preflight",
        "ledger",
        "holdout_metrics",
        "holdout_receipt",
        "holdout",
    ];
    if !p3_exact_keys(p2, &fields)
        || !p2.get("holdout").is_some_and(|holdout| {
            p3_exact_keys(holdout, &["status", "evaluations_remaining", "executed"])
                && holdout.get("status").and_then(serde_json::Value::as_str) == Some("passed")
                && holdout
                    .get("evaluations_remaining")
                    .and_then(serde_json::Value::as_u64)
                    == Some(0)
                && holdout.get("executed").and_then(serde_json::Value::as_bool) == Some(true)
        })
    {
        return Err(ServedModelError(
            "AQ4 P3 qualification P2 holdout state differs".into(),
        ));
    }
    let plan = p3_load_qualification_ref(p2, "plan")?;
    let _actual = p3_load_qualification_ref(p2, "actual_receipt")?;
    let _policy = p3_load_qualification_ref(p2, "policy")?;
    let _calibration_metrics = p3_load_qualification_ref(p2, "calibration_metrics")?;
    let freeze = p3_load_qualification_ref(p2, "freeze_receipt")?;
    let preflight = p3_load_qualification_ref(p2, "preflight")?;
    let ledger = p3_load_qualification_ref(p2, "ledger")?;
    let _holdout_metrics = p3_load_qualification_ref(p2, "holdout_metrics")?;
    let holdout = p3_load_qualification_ref(p2, "holdout_receipt")?;
    let gates_passed = holdout
        .get("gate_checks")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|checks| {
            !checks.is_empty() && checks.values().all(|value| value.as_bool() == Some(true))
        });
    if plan
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some("ullm.qwen35_aq4_sq8_fidelity_plan.v1")
        || plan.get("status").and_then(serde_json::Value::as_str) != Some("ready_for_calibration")
        || plan
            .get("preflight_only")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || freeze
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("ullm.qwen35_aq4_sq8_fidelity_freeze_receipt.v1")
        || freeze.get("status").and_then(serde_json::Value::as_str)
            != Some("frozen_calibration_envelope")
        || preflight
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("ullm.qwen35_aq4_sq8_fidelity_holdout_preflight.v1")
        || preflight.get("status").and_then(serde_json::Value::as_str)
            != Some("ready_for_one_shot_holdout")
        || ledger
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("ullm.qwen35_aq4_sq8_fidelity_attempt_ledger.v1")
        || ledger.get("status").and_then(serde_json::Value::as_str) != Some("consumed")
        || ledger
            .get("remaining_after")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        || holdout
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("ullm.qwen35_aq4_sq8_fidelity_holdout_receipt.v1")
        || holdout.get("status").and_then(serde_json::Value::as_str) != Some("passed")
        || holdout
            .get("evaluations_remaining")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        || !gates_passed
    {
        return Err(ServedModelError(
            "AQ4 P3 qualification P2 success chain differs".into(),
        ));
    }
    Ok(())
}

fn load_p3_production_activation(
    path: &Path,
    expected_sha256: &str,
) -> Result<P3ActivationBuild, ServedModelError> {
    let payload = load_bound_p3_bytes(path, expected_sha256)?;
    let value: serde_json::Value = serde_json::from_slice(&payload)
        .map_err(|_| ServedModelError("AQ4 P3 activation JSON is invalid".into()))?;
    let manifest: P3ProductionActivation = serde_json::from_value(value.clone())
        .map_err(|_| ServedModelError("AQ4 P3 activation schema is invalid".into()))?;
    if manifest.schema_version != P3_PRODUCTION_ACTIVATION_SCHEMA
        || manifest.status != "production_activated"
        || manifest.candidate.candidate_id != "sequence-output-direct-v1"
        || manifest.candidate.family != "attention_recurrent"
        || manifest.selection.status != "selected"
        || manifest.selection.selected_candidate_id != "sequence-output-direct-v1"
        || manifest.upstream_qualification.status != "qualified_go"
        || !manifest.upstream_qualification.promotion_eligible
        || manifest.upstream_qualification.reason != "upstream_p2_holdout_passed"
        || !is_lowercase_sha256(&manifest.activation_sha256)
        || p3_canonical_self_hash(&value, "activation_sha256")? != manifest.activation_sha256
    {
        return Err(ServedModelError("AQ4 P3 activation state differs".into()));
    }
    for sha in [
        &manifest.build.identity_sha256,
        &manifest.build.case_manifest_sha256,
        &manifest.build.binary_sha256,
        &manifest.build.package_content_sha256,
        &manifest.selection.sha256,
        &manifest.profile.selection_file_sha256,
        &manifest.upstream_qualification.sha256,
        &manifest.upstream_qualification.qualification_sha256,
    ] {
        if !is_lowercase_sha256(sha) {
            return Err(ServedModelError("AQ4 P3 activation digest differs".into()));
        }
    }
    if manifest.selection.sha256 != manifest.profile.selection_file_sha256
        || manifest.profile.raw_evidence.is_empty()
    {
        return Err(ServedModelError(
            "AQ4 P3 activation profile binding differs".into(),
        ));
    }
    let selection = load_bound_p3_bytes(&manifest.selection.path, &manifest.selection.sha256)?;
    let selection_value: serde_json::Value = serde_json::from_slice(&selection)
        .map_err(|_| ServedModelError("AQ4 P3 selection JSON is invalid".into()))?;
    if selection_value
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("selected")
        || selection_value
            .get("selected_candidate_id")
            .and_then(serde_json::Value::as_str)
            != Some("sequence-output-direct-v1")
        || selection_value
            .get("input_binding")
            .and_then(|binding| binding.get("upstream_qualification_sha256"))
            .and_then(serde_json::Value::as_str)
            != Some(
                manifest
                    .upstream_qualification
                    .qualification_sha256
                    .as_str(),
            )
        || !selection_value
            .get("candidates")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|candidates| {
                candidates.iter().any(|candidate| {
                    candidate
                        .get("candidate_id")
                        .and_then(serde_json::Value::as_str)
                        == Some("sequence-output-direct-v1")
                        && candidate
                            .get("eligible")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true)
                })
            })
    {
        return Err(ServedModelError("AQ4 P3 selection state differs".into()));
    }
    let qualification = load_bound_p3_bytes(
        &manifest.upstream_qualification.path,
        &manifest.upstream_qualification.sha256,
    )?;
    let qualification_value: serde_json::Value = serde_json::from_slice(&qualification)
        .map_err(|_| ServedModelError("AQ4 P3 qualification JSON is invalid".into()))?;
    if qualification_value
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("qualified_go")
        || qualification_value
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("ullm.aq4_p3_upstream_p2_qualification.v1")
        || qualification_value
            .get("promotion_eligible")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || qualification_value
            .get("qualification_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(
                manifest
                    .upstream_qualification
                    .qualification_sha256
                    .as_str(),
            )
        || p3_canonical_self_hash(&qualification_value, "qualification_sha256")?
            != manifest.upstream_qualification.qualification_sha256
    {
        return Err(ServedModelError(
            "AQ4 P3 qualification state differs".into(),
        ));
    }
    validate_p3_qualified_success_chain(&qualification_value)?;
    for raw in manifest.profile.raw_evidence {
        if !is_lowercase_sha256(&raw.sha256) || !is_lowercase_sha256(&raw.evidence_sha256) {
            return Err(ServedModelError("AQ4 P3 raw digest differs".into()));
        }
        let bytes = load_bound_p3_bytes(&raw.path, &raw.sha256)?;
        let raw_value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| ServedModelError("AQ4 P3 raw JSON is invalid".into()))?;
        if raw_value.get("status").and_then(serde_json::Value::as_str) != Some("complete")
            || raw_value
                .get("promotion_eligible")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            || p3_canonical_self_hash(&raw_value, "evidence_sha256")? != raw.evidence_sha256
            || raw_value
                .get("upstream_qualification")
                .and_then(|qualification| qualification.get("qualification_sha256"))
                .and_then(serde_json::Value::as_str)
                != Some(
                    manifest
                        .upstream_qualification
                        .qualification_sha256
                        .as_str(),
                )
        {
            return Err(ServedModelError(
                "AQ4 P3 promotion raw state differs".into(),
            ));
        }
    }
    Ok(manifest.build)
}

fn load_direct_trace_binding_manifest(
    path: &Path,
    expected_bytes_sha256: &str,
) -> Result<VecDeque<Qwen35Aq4DirectTraceBinding>, ServedModelError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| part == std::path::Component::ParentDir)
        || std::fs::canonicalize(path)
            .map(|canonical| canonical != path)
            .unwrap_or(true)
    {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest path must be absolute and canonical".into(),
        ));
    }
    if !is_lowercase_sha256(expected_bytes_sha256) {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest expected SHA-256 is invalid".into(),
        ));
    }
    reject_direct_trace_binding_symlink_components(path)?;
    let before = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest metadata failed".into()))?;
    if !before.file_type().is_file() || before.nlink() != 1 {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest must be a single-link regular file".into(),
        ));
    }
    const O_NOFOLLOW: i32 = 0o400000;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest open failed".into()))?;
    let identity = RegistryFileIdentity::from(&before);
    let opened = file
        .metadata()
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest metadata failed".into()))?;
    let opened_path = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest metadata failed".into()))?;
    if !opened.file_type().is_file()
        || opened.nlink() != 1
        || RegistryFileIdentity::from(&opened) != identity
        || RegistryFileIdentity::from(&opened_path) != identity
    {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest changed while opening".into(),
        ));
    }
    let maximum = ullm_engine::worker_protocol::WORKER_MAX_RECORD_BYTES;
    if identity.size > u64::try_from(maximum).unwrap_or(u64::MAX) {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest exceeds the record bound".into(),
        ));
    }
    let mut payload = Vec::with_capacity(usize::try_from(identity.size).unwrap_or(maximum));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| ServedModelError("AQ4 P3 binding manifest read failed".into()))?;
        if count == 0 {
            break;
        }
        if payload
            .len()
            .checked_add(count)
            .is_none_or(|total| total > maximum)
        {
            return Err(ServedModelError(
                "AQ4 P3 binding manifest exceeds the record bound".into(),
            ));
        }
        digest.update(&buffer[..count]);
        payload.extend_from_slice(&buffer[..count]);
    }
    reject_direct_trace_binding_symlink_components(path)?;
    let after_fd = file
        .metadata()
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest metadata failed".into()))?;
    let after_path = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest metadata failed".into()))?;
    if RegistryFileIdentity::from(&after_fd) != identity
        || RegistryFileIdentity::from(&after_path) != identity
    {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest changed while reading".into(),
        ));
    }
    if format!("{:x}", digest.finalize()) != expected_bytes_sha256 {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest bytes SHA-256 differs".into(),
        ));
    }
    let manifest: DirectTraceBindingManifest = serde_json::from_slice(&payload)
        .map_err(|_| ServedModelError("AQ4 P3 binding manifest JSON is invalid".into()))?;
    if manifest.schema_version != P3_DIRECT_TRACE_BINDING_MANIFEST_SCHEMA
        || manifest.bindings.is_empty()
        || manifest.bindings.len() > P3_DIRECT_TRACE_BINDING_MANIFEST_MAX_ENTRIES
    {
        return Err(ServedModelError(
            "AQ4 P3 binding manifest schema or entry count differs".into(),
        ));
    }
    let mut request_ids = BTreeSet::new();
    let mut binding_ids = BTreeSet::new();
    for binding in &manifest.bindings {
        binding.validate().map_err(ServedModelError)?;
        if !request_ids.insert(binding.request_id.clone())
            || !binding_ids.insert(binding.binding_id.clone())
        {
            return Err(ServedModelError(
                "AQ4 P3 binding manifest reuses a request or binding ID".into(),
            ));
        }
    }
    Ok(manifest.bindings.into())
}

fn reject_direct_trace_binding_symlink_components(path: &Path) -> Result<(), ServedModelError> {
    for component in path.ancestors() {
        let metadata = std::fs::symlink_metadata(component)
            .map_err(|_| ServedModelError("AQ4 P3 binding manifest path metadata failed".into()))?;
        if metadata.file_type().is_symlink() {
            return Err(ServedModelError(
                "AQ4 P3 binding manifest path must not contain symlinks".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegistryFileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    size: u64,
    mtime_seconds: i64,
    mtime_nanoseconds: i64,
    ctime_seconds: i64,
    ctime_nanoseconds: i64,
    links: u64,
}

impl From<&Metadata> for RegistryFileIdentity {
    fn from(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            size: metadata.size(),
            mtime_seconds: metadata.mtime(),
            mtime_nanoseconds: metadata.mtime_nsec(),
            ctime_seconds: metadata.ctime(),
            ctime_nanoseconds: metadata.ctime_nsec(),
            links: metadata.nlink(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrySnapshotPoint {
    AfterOpen,
    AfterRead,
}

fn load_benchmark_case_registry_with_hook<F>(
    path: &Path,
    expected_bytes_sha256: &str,
    mut hook: F,
) -> Result<Aq4BenchmarkTrustedCaseRegistry, ServedModelError>
where
    F: FnMut(RegistrySnapshotPoint),
{
    if !path.is_absolute()
        || path
            .components()
            .any(|part| part == std::path::Component::ParentDir)
    {
        return Err(ServedModelError(
            "AQ4 benchmark case registry path must be absolute without parent traversal".into(),
        ));
    }
    if !is_lowercase_sha256(expected_bytes_sha256) {
        return Err(ServedModelError(
            "AQ4 benchmark case registry expected bytes SHA-256 is invalid".into(),
        ));
    }
    reject_registry_symlink_components(path)?;
    let before = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 benchmark case registry metadata failed".into()))?;
    if !before.file_type().is_file() || before.nlink() != 1 {
        return Err(ServedModelError(
            "AQ4 benchmark case registry must be a single-link regular file".into(),
        ));
    }
    const O_NOFOLLOW: i32 = 0o400000;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|_| ServedModelError("AQ4 benchmark case registry open failed".into()))?;
    let opened = file
        .metadata()
        .map_err(|_| ServedModelError("AQ4 benchmark case registry metadata failed".into()))?;
    let opened_path = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 benchmark case registry metadata failed".into()))?;
    let identity = RegistryFileIdentity::from(&before);
    if !opened.file_type().is_file()
        || opened.nlink() != 1
        || RegistryFileIdentity::from(&opened) != identity
        || RegistryFileIdentity::from(&opened_path) != identity
    {
        return Err(ServedModelError(
            "AQ4 benchmark case registry changed while opening".into(),
        ));
    }
    hook(RegistrySnapshotPoint::AfterOpen);
    let maximum = ullm_engine::worker_protocol::WORKER_MAX_RECORD_BYTES;
    if identity.size > u64::try_from(maximum).unwrap_or(u64::MAX) {
        return Err(ServedModelError(
            "AQ4 benchmark case registry exceeds the record bound".into(),
        ));
    }
    let mut payload = Vec::with_capacity(usize::try_from(identity.size).unwrap_or(maximum));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| ServedModelError("AQ4 benchmark case registry read failed".into()))?;
        if count == 0 {
            break;
        }
        if payload
            .len()
            .checked_add(count)
            .is_none_or(|total| total > maximum)
        {
            return Err(ServedModelError(
                "AQ4 benchmark case registry exceeds the record bound".into(),
            ));
        }
        digest.update(&buffer[..count]);
        payload.extend_from_slice(&buffer[..count]);
    }
    hook(RegistrySnapshotPoint::AfterRead);
    reject_registry_symlink_components(path)?;
    let after_fd = file
        .metadata()
        .map_err(|_| ServedModelError("AQ4 benchmark case registry metadata failed".into()))?;
    let after_path = std::fs::symlink_metadata(path)
        .map_err(|_| ServedModelError("AQ4 benchmark case registry metadata failed".into()))?;
    if RegistryFileIdentity::from(&after_fd) != identity
        || RegistryFileIdentity::from(&after_path) != identity
    {
        return Err(ServedModelError(
            "AQ4 benchmark case registry changed while reading".into(),
        ));
    }
    let actual_bytes_sha256 = format!("{:x}", digest.finalize());
    if actual_bytes_sha256 != expected_bytes_sha256 {
        return Err(ServedModelError(
            "AQ4 benchmark case registry bytes SHA-256 differs".into(),
        ));
    }
    decode_aq4_benchmark_case_registry(&payload).map_err(ServedModelError)
}

fn reject_registry_symlink_components(path: &Path) -> Result<(), ServedModelError> {
    for component in path.ancestors() {
        let metadata = std::fs::symlink_metadata(component).map_err(|_| {
            ServedModelError("AQ4 benchmark case registry path metadata failed".into())
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ServedModelError(
                "AQ4 benchmark case registry path must not contain symlinks".into(),
            ));
        }
    }
    Ok(())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn load_resident_worker(
    model: &ServedModel,
    current_exe: &std::path::Path,
) -> Result<LoadedWorker, ServedModelError> {
    validate_resident_model_contract(model)?;
    let startup = model.worker_startup(WorkerBackendKind::Aq4, current_exe)?;
    let (config, profile) = resident_config_from_startup(startup)?;
    Ok(LoadedWorker::Resident {
        config,
        profile,
        direct_trace_bindings: None,
    })
}

fn validate_resident_model_contract(model: &ServedModel) -> Result<(), ServedModelError> {
    if model.format.format_id != "AQ4_0"
        || model.format.implementation_id != "qwen35_aq4_rdna4_v1"
        || model.generation.sampling.temperature
        || model.generation.sampling.top_p
        || model.generation.sampling.top_k != 1
        || model.worker.identity.device != "gfx1201"
        || model.worker.identity.execution_profile != "rdna4_aq4_resident"
    {
        return Err(ServedModelError(
            "AQ4 resident worker format, implementation, identity, or greedy sampling contract is unsupported"
                .into(),
        ));
    }
    let mut actual_environment = model.worker.required_environment.iter().collect::<Vec<_>>();
    actual_environment.sort_unstable();
    let mut required_environment = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.to_vec();
    required_environment.sort_unstable();
    if actual_environment.len() != required_environment.len()
        || actual_environment
            .iter()
            .zip(&required_environment)
            .any(|(actual, required)| actual.as_str() != *required)
    {
        return Err(ServedModelError(
            "AQ4 resident worker required_environment does not exactly match the production HIP guard contract"
                .into(),
        ));
    }
    Ok(())
}

fn resident_config_from_startup(
    startup: WorkerStartupConfig,
) -> Result<(ResidentWorkerConfig, Sq8WorkerProfile), ServedModelError> {
    if startup.artifact_dir.is_some() || startup.profile.top_k != 1 {
        return Err(ServedModelError(
            "AQ4 resident startup contract is inconsistent".into(),
        ));
    }
    let expected_vocab_size = startup.profile.vocab_size;
    let model = Qwen35Aq4ModelLoadConfig {
        package_dir: startup.package_dir,
        device_index: RESIDENT_DEVICE_INDEX,
        expected_architecture: Some(startup.profile.device.clone()),
        chunk_bytes: RESIDENT_CHUNK_BYTES,
        context_length: startup.profile.context_length,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: RESIDENT_LM_HEAD_CHUNK_ROWS,
    };
    let mut session = Qwen35Aq4SessionConfig::greedy(
        startup.profile.max_new_tokens,
        startup.profile.eos_token_ids.clone(),
    );
    session.reasoning_dialect = startup.reasoning;
    Ok((
        ResidentWorkerConfig {
            model,
            session,
            expected_vocab_size,
        },
        startup.profile.into_worker_profile(),
    ))
}

fn load_resident_backend(
    config: ResidentWorkerConfig,
) -> Result<SessionInferenceBackend<Qwen35Aq4InferenceSession>, String> {
    let session = Qwen35Aq4InferenceSession::load(config.model, config.session)?;
    if session.model().geometry().vocab != config.expected_vocab_size {
        return Err(format!(
            "Qwen3.5 AQ4 package vocabulary {} does not match served-model profile {}",
            session.model().geometry().vocab,
            config.expected_vocab_size
        ));
    }
    let operation_traces = session.operation_resolution_traces();
    if operation_traces.len() != session.model().geometry().layers.len() {
        return Err("Qwen3.5 AQ4 operation trace coverage does not match decoder layers".into());
    }
    for (layer_position, traces) in operation_traces.iter().enumerate() {
        for trace in traces {
            eprintln!(
                "{}",
                operation_trace_log_line(layer_position, &trace.audit_json())
            );
        }
    }
    Ok(SessionInferenceBackend::new(session))
}

fn operation_trace_log_line(layer_position: usize, trace_json: &str) -> String {
    format!(
        "{{\"schema_version\":\"ullm.backend_operation.load.v1\",\"layer_position\":{layer_position},\"trace\":{trace_json}}}"
    )
}

enum ResidentBindingBackend<B> {
    Ordinary(B),
    Diagnostic {
        inner: B,
        bindings: VecDeque<Qwen35Aq4DirectTraceBinding>,
    },
}

impl<B: InferenceBackend> InferenceBackend for ResidentBindingBackend<B> {
    fn execute(
        &mut self,
        mut request: ullm_engine::inference_api::InferenceRequest,
        admission: ullm_engine::worker_protocol::WorkerAdmission,
        publications: &mut ullm_engine::worker_runtime::RequestEventPublisher<'_>,
    ) -> Result<(), String> {
        match self {
            Self::Ordinary(inner) => inner.execute(request, admission, publications),
            Self::Diagnostic { inner, bindings } => {
                if request.aq4_p3_direct_trace.is_some() {
                    return Err(
                        "AQ4 P3 worker request already contains a diagnostic binding".into(),
                    );
                }
                let binding = bindings
                    .pop_front()
                    .ok_or_else(|| "AQ4 P3 binding manifest is exhausted".to_string())?;
                if binding.request_id != request.request_id {
                    return Err(
                        "AQ4 P3 binding request ID differs from the parsed worker request".into(),
                    );
                }
                request.aq4_p3_direct_trace = Some(binding);
                inner.execute(request, admission, publications)
            }
        }
    }

    fn shutdown(&mut self) -> Result<(), String> {
        match self {
            Self::Ordinary(inner) => inner.shutdown(),
            Self::Diagnostic { inner, bindings } => {
                let shutdown = inner.shutdown();
                if !bindings.is_empty() {
                    return Err("AQ4 P3 binding manifest has unused entries".into());
                }
                shutdown
            }
        }
    }
}

fn run_loaded_worker<R, W, LB, RB, FL, FR>(
    loaded: LoadedWorker,
    input: R,
    output: W,
    legacy_loader: FL,
    resident_loader: FR,
) -> Result<ullm_engine::worker_runtime::CommandReaderExit, String>
where
    R: BufRead + Send + 'static,
    W: Write + Send + 'static,
    LB: InferenceBackend + 'static,
    RB: InferenceBackend + 'static,
    FL: FnOnce(Qwen35Aq4WorkerBackendConfig) -> Result<LB, String> + Send + 'static,
    FR: FnOnce(ResidentWorkerConfig) -> Result<RB, String> + Send + 'static,
{
    match loaded {
        LoadedWorker::Legacy { config, profile } => {
            run_worker_process_with_profile(input, output, profile, move || legacy_loader(config))
        }
        LoadedWorker::Resident {
            config,
            profile,
            direct_trace_bindings,
        } => run_worker_process_with_profile(input, output, profile, move || {
            let inner = resident_loader(config)?;
            Ok(match direct_trace_bindings {
                Some(bindings) => ResidentBindingBackend::Diagnostic { inner, bindings },
                None => ResidentBindingBackend::Ordinary(inner),
            })
        }),
        LoadedWorker::BenchmarkResident { .. } => {
            Err("AQ4 benchmark resident was routed through the ordinary worker wire".into())
        }
    }
}

fn parse_cli(args: impl IntoIterator<Item = OsString>) -> Result<CliAction, String> {
    let args = args.into_iter().collect::<Vec<_>>();
    if args == [OsString::from("--help")] {
        return Ok(CliAction::Help);
    }
    if args == [OsString::from("--version")] {
        return Ok(CliAction::Version);
    }
    if args.iter().any(|value| value == "--served-model-manifest") {
        let ordinary =
            args.len() == 2 && args[0] == "--served-model-manifest" && !args[1].is_empty();
        let diagnostic = args.len() == 6
            && args[0] == "--served-model-manifest"
            && !args[1].is_empty()
            && args[2] == "--p3-direct-trace-binding-manifest"
            && !args[3].is_empty()
            && args[4] == "--p3-direct-trace-binding-manifest-sha256"
            && !args[5].is_empty();
        let production = args.len() == 6
            && args[0] == "--served-model-manifest"
            && !args[1].is_empty()
            && args[2] == "--p3-production-activation"
            && !args[3].is_empty()
            && args[4] == "--p3-production-activation-sha256"
            && !args[5].is_empty();
        let benchmark = args.len() == 7
            && args[0] == "--served-model-manifest"
            && !args[1].is_empty()
            && args[2] == "--benchmark-wire"
            && args[3] == "--benchmark-case-manifest"
            && !args[4].is_empty()
            && args[5] == "--benchmark-case-manifest-sha256"
            && !args[6].is_empty();
        if !ordinary && !diagnostic && !production && !benchmark {
            return Err("manifest mode and legacy options are mutually exclusive".into());
        }
        return Ok(CliAction::Run(if benchmark {
            WorkerSource::BenchmarkServedModelManifest {
                served_model: PathBuf::from(&args[1]),
                case_registry: PathBuf::from(&args[4]),
                case_registry_sha256: args[6]
                    .clone()
                    .into_string()
                    .map_err(|_| "benchmark case manifest SHA-256 must be UTF-8".to_string())
                    .and_then(|value| {
                        is_lowercase_sha256(&value).then_some(value).ok_or_else(|| {
                            "benchmark case manifest SHA-256 must be lowercase SHA-256".to_string()
                        })
                    })?,
            }
        } else {
            let direct_trace_bindings = if diagnostic {
                let bytes_sha256 = args[5]
                    .clone()
                    .into_string()
                    .map_err(|_| "P3 binding manifest SHA-256 must be UTF-8".to_string())?;
                if !is_lowercase_sha256(&bytes_sha256) {
                    return Err("P3 binding manifest SHA-256 must be lowercase SHA-256".into());
                }
                Some(DiagnosticBindingSource {
                    path: PathBuf::from(&args[3]),
                    bytes_sha256,
                })
            } else {
                None
            };
            let production_activation = if production {
                let bytes_sha256 = args[5]
                    .clone()
                    .into_string()
                    .map_err(|_| "P3 activation SHA-256 must be UTF-8".to_string())?;
                if !is_lowercase_sha256(&bytes_sha256) {
                    return Err("P3 activation SHA-256 must be lowercase SHA-256".into());
                }
                Some(DiagnosticBindingSource {
                    path: PathBuf::from(&args[3]),
                    bytes_sha256,
                })
            } else {
                None
            };
            WorkerSource::ServedModelManifest {
                served_model: PathBuf::from(&args[1]),
                direct_trace_bindings,
                production_activation,
            }
        }));
    }
    let mut engine = None;
    let mut artifact = None;
    let mut package = None;
    let mut device_index = 1_u32;
    let mut layers = "all".to_string();
    let mut index = 0;
    while index < args.len() {
        let option = args[index]
            .to_str()
            .ok_or_else(|| "AQ4 worker option is not valid UTF-8".to_string())?;
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("AQ4 worker option {option} is missing its value"))?;
        match option {
            "--engine" if engine.is_none() => engine = Some(PathBuf::from(value)),
            "--artifact" if artifact.is_none() => artifact = Some(PathBuf::from(value)),
            "--package" if package.is_none() => package = Some(PathBuf::from(value)),
            "--device-index" => {
                device_index = value
                    .to_str()
                    .ok_or_else(|| "AQ4 device index is not valid UTF-8".to_string())?
                    .parse()
                    .map_err(|_| "AQ4 device index must be an unsigned integer".to_string())?;
            }
            "--layers" => {
                layers = value
                    .to_str()
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| "AQ4 layers must be nonempty UTF-8".to_string())?
                    .to_string();
            }
            "--engine" | "--artifact" | "--package" => {
                return Err(format!("AQ4 worker option {option} was provided twice"));
            }
            _ => return Err(format!("AQ4 worker received unknown option {option}")),
        }
        index += 1;
    }
    Ok(CliAction::Run(WorkerSource::Legacy(WorkerArgs {
        engine: engine.map_or_else(default_engine_path, Ok)?,
        package: artifact
            .or(package)
            .ok_or_else(|| "AQ4 worker --artifact or --package is required".to_string())?,
        device_index,
        layers,
    })))
}

fn default_engine_path() -> Result<PathBuf, String> {
    env::current_exe()
        .map(|path| path.with_file_name("ullm-engine"))
        .map_err(|error| format!("failed to resolve AQ4 worker executable: {error}"))
}

fn configured_aq4_worker_profile() -> Sq8WorkerProfile {
    let defaults = Sq8WorkerProfile {
        worker_schema: "ullm.worker.v1".into(),
        model: "ullm-qwen3.5-9b-aq4".into(),
        model_revision: "aq4-cli-compat-v0.1".into(),
        artifact_content_sha256: "0".repeat(64),
        package_manifest_sha256: "0".repeat(64),
        device: "gfx1201".into(),
        execution_profile: "rdna4_aq4_cli_compat".into(),
        context_length: 4096,
        max_new_tokens: 512,
        vocab_size: 248320,
        eos_token_ids: vec![248044, 248046],
        top_k: 1,
        reasoning: None,
    };
    Sq8WorkerProfile::from_environment_with_defaults(&defaults)
}

#[derive(Serialize)]
struct ProcessLog<'a> {
    schema_version: &'static str,
    level: &'static str,
    event: &'static str,
    phase: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<&'a str>,
}

fn write_process_log(
    level: &'static str,
    event: &'static str,
    error_code: Option<&'static str>,
    detail: Option<&str>,
) {
    let record = ProcessLog {
        schema_version: "ullm.worker.log.v1",
        level,
        event,
        phase: "process",
        error_code,
        detail,
    };
    let mut stderr = std::io::stderr().lock();
    let _ = serde_json::to_writer(&mut stderr, &record);
    let _ = stderr.write_all(b"\n");
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::fs::FileTimes;
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};
    use ullm_engine::aq4_benchmark_worker_protocol::{
        AQ4_BENCHMARK_CASE_REGISTRY_SCHEMA_VERSION, Aq4BenchmarkCaseBinding,
        aq4_benchmark_case_registry_sha256, aq4_benchmark_case_sha256,
    };
    use ullm_engine::qwen35_aq4_direct_trace::{
        Qwen35Aq4DirectRuntimeObservation, Qwen35Aq4DirectTraceCollector, Qwen35Aq4DirectTraceRoute,
    };
    use ullm_engine::qwen35_aq4_session::Qwen35Aq4SessionModel;
    use ullm_engine::served_model::WorkerProfileSnapshot;

    static REGISTRY_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn registry_test_root(label: &str) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "ullm-aq4-worker-registry-{}-{label}-{}",
            std::process::id(),
            REGISTRY_TEST_ID.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    fn registry_bytes() -> Vec<u8> {
        let value = serde_json::json!({
            "baseline_mode": "all_m1",
            "cached_prefix_tokens": 0,
            "case_id": "case-1",
            "case_sha256": null,
            "context_tokens": 3,
            "control": {"control_id": "aq4_0_target", "role": "target", "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1", "promotion_eligible": true},
            "control_id": "aq4_0_target",
            "decode_request_count": 0,
            "decode_start_tokens": 0,
            "device": {"device_id": "r9700-rdna4", "runtime_device_index": 1, "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"},
            "fixture_id": "case-1",
            "format_id": "AQ4_0",
            "generated_tokens": 0,
            "implementation_id": "qwen35_aq4_rdna4_v1",
            "mode": "all_m1",
            "path_oracle_case_id": null,
            "path_oracle_result_sha256": null,
            "phase": "cold_prefill",
            "prefill_requested_m": 64,
            "prompt_tokens": 3,
            "request_count": 1,
            "resolved_m": 1,
            "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
            "scope": "full_model",
            "stage_id": "representative",
            "stage_order": 1,
        });
        let mut case: Aq4BenchmarkCaseBinding = serde_json::from_value(value).unwrap();
        case.case_sha256 = Some(aq4_benchmark_case_sha256(&case).unwrap());
        let registry_sha256 =
            aq4_benchmark_case_registry_sha256(std::slice::from_ref(&case)).unwrap();
        serde_json::to_vec(&serde_json::json!({
            "schema_version": AQ4_BENCHMARK_CASE_REGISTRY_SCHEMA_VERSION,
            "registry_sha256": registry_sha256,
            "cases": [case],
        }))
        .unwrap()
    }

    fn bytes_sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn direct_trace_binding_value(request_id: &str) -> serde_json::Value {
        serde_json::json!({
            "side": "baseline",
            "binding_kind": "run",
            "binding_id": format!("binding-{request_id}"),
            "request_id": request_id,
            "implementation_id": "qwen35-aq4-production",
            "source_id": "ullm-aq4-worker",
            "source_sha256": "a".repeat(64),
            "case_id": "case-1",
            "case_sha256": "b".repeat(64),
            "identity_sha256": "c".repeat(64),
            "direct_sequence_output_enabled": false
        })
    }

    fn direct_trace_manifest_bytes(request_ids: &[&str]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": P3_DIRECT_TRACE_BINDING_MANIFEST_SCHEMA,
            "bindings": request_ids
                .iter()
                .map(|request_id| direct_trace_binding_value(request_id))
                .collect::<Vec<_>>()
        }))
        .unwrap()
    }

    fn activation_fixture(root: &Path) -> (PathBuf, String) {
        let write_ref = |name: &str, value: serde_json::Value| {
            let path = root.join(name);
            let bytes = serde_json::to_vec(&value).unwrap();
            std::fs::write(&path, &bytes).unwrap();
            serde_json::json!({"path": path, "sha256": bytes_sha256(&bytes)})
        };
        let plan_ref = write_ref(
            "plan.json",
            serde_json::json!({"schema_version": "ullm.qwen35_aq4_sq8_fidelity_plan.v1", "status": "ready_for_calibration", "preflight_only": false}),
        );
        let actual_ref = write_ref(
            "actual.json",
            serde_json::json!({"status": "actual_verified"}),
        );
        let policy_ref = write_ref("policy.json", serde_json::json!({"status": "frozen"}));
        let calibration_metrics_ref = write_ref(
            "calibration-metrics.json",
            serde_json::json!({"subset": "calibration"}),
        );
        let freeze_ref = write_ref(
            "freeze.json",
            serde_json::json!({"schema_version": "ullm.qwen35_aq4_sq8_fidelity_freeze_receipt.v1", "status": "frozen_calibration_envelope"}),
        );
        let preflight_ref = write_ref(
            "preflight.json",
            serde_json::json!({"schema_version": "ullm.qwen35_aq4_sq8_fidelity_holdout_preflight.v1", "status": "ready_for_one_shot_holdout"}),
        );
        let ledger_ref = write_ref(
            "ledger.json",
            serde_json::json!({"schema_version": "ullm.qwen35_aq4_sq8_fidelity_attempt_ledger.v1", "status": "consumed", "remaining_after": 0}),
        );
        let holdout_metrics_ref = write_ref(
            "holdout-metrics.json",
            serde_json::json!({"subset": "holdout"}),
        );
        let holdout_receipt_ref = write_ref(
            "holdout-receipt.json",
            serde_json::json!({"schema_version": "ullm.qwen35_aq4_sq8_fidelity_holdout_receipt.v1", "status": "passed", "evaluations_remaining": 0, "gate_checks": {"fidelity": true}}),
        );
        let qualification_path = root.join("qualification.json");
        let mut qualification = serde_json::json!({
            "schema_version": "ullm.aq4_p3_upstream_p2_qualification.v1",
            "status": "qualified_go",
            "qualification_sha256": null,
            "promotion_eligible": true,
            "reason": "upstream_p2_holdout_passed",
            "p2": {
                "plan": plan_ref, "actual_receipt": actual_ref, "policy": policy_ref,
                "calibration_metrics": calibration_metrics_ref, "freeze_receipt": freeze_ref,
                "preflight": preflight_ref, "ledger": ledger_ref,
                "holdout_metrics": holdout_metrics_ref, "holdout_receipt": holdout_receipt_ref,
                "holdout": {"status": "passed", "evaluations_remaining": 0, "executed": true}
            }
        });
        let qualification_hash =
            p3_canonical_self_hash(&qualification, "qualification_sha256").unwrap();
        qualification["qualification_sha256"] = qualification_hash.clone().into();
        let qualification_bytes = serde_json::to_vec(&qualification).unwrap();
        std::fs::write(&qualification_path, &qualification_bytes).unwrap();
        let qualification_file_hash = bytes_sha256(&qualification_bytes);

        let raw_path = root.join("raw.json");
        let mut raw = serde_json::json!({
            "status": "complete",
            "promotion_eligible": true,
            "evidence_sha256": null,
            "upstream_qualification": {"qualification_sha256": qualification_hash}
        });
        let evidence_hash = p3_canonical_self_hash(&raw, "evidence_sha256").unwrap();
        raw["evidence_sha256"] = evidence_hash.clone().into();
        let raw_bytes = serde_json::to_vec(&raw).unwrap();
        std::fs::write(&raw_path, &raw_bytes).unwrap();
        let raw_file_hash = bytes_sha256(&raw_bytes);

        let selection_path = root.join("selection.json");
        let selection = serde_json::json!({
            "status": "selected",
            "selected_candidate_id": "sequence-output-direct-v1",
            "input_binding": {"upstream_qualification_sha256": qualification_hash},
            "candidates": [{"candidate_id": "sequence-output-direct-v1", "eligible": true}]
        });
        let selection_bytes = serde_json::to_vec(&selection).unwrap();
        std::fs::write(&selection_path, &selection_bytes).unwrap();
        let selection_hash = bytes_sha256(&selection_bytes);

        let activation_path = root.join("activation.json");
        let mut activation = serde_json::json!({
            "schema_version": P3_PRODUCTION_ACTIVATION_SCHEMA,
            "status": "production_activated",
            "activation_sha256": null,
            "candidate": {"candidate_id": "sequence-output-direct-v1", "family": "attention_recurrent"},
            "build": {
                "identity_sha256": "1".repeat(64), "case_manifest_sha256": "2".repeat(64),
                "binary_sha256": "3".repeat(64), "package_content_sha256": "4".repeat(64)
            },
            "profile": {
                "selection_file_sha256": selection_hash,
                "raw_evidence": [{"path": raw_path, "sha256": raw_file_hash, "evidence_sha256": evidence_hash}]
            },
            "selection": {"path": selection_path, "sha256": selection_hash, "status": "selected", "selected_candidate_id": "sequence-output-direct-v1"},
            "upstream_qualification": {
                "path": qualification_path, "sha256": qualification_file_hash,
                "qualification_sha256": qualification_hash, "status": "qualified_go",
                "promotion_eligible": true, "reason": "upstream_p2_holdout_passed"
            }
        });
        let activation_hash = p3_canonical_self_hash(&activation, "activation_sha256").unwrap();
        activation["activation_sha256"] = activation_hash.into();
        let activation_bytes = serde_json::to_vec(&activation).unwrap();
        std::fs::write(&activation_path, &activation_bytes).unwrap();
        (activation_path, bytes_sha256(&activation_bytes))
    }

    #[test]
    fn cli_accepts_required_paths_and_optional_device_and_layers() {
        let CliAction::Run(parsed) = parse_cli(args(&[
            "--engine",
            "/engine",
            "--package",
            "/package",
            "--device-index",
            "7",
            "--layers",
            "0,1",
        ]))
        .unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(
            parsed,
            WorkerSource::Legacy(WorkerArgs {
                engine: PathBuf::from("/engine"),
                package: PathBuf::from("/package"),
                device_index: 7,
                layers: "0,1".to_string(),
            })
        );
    }

    #[test]
    fn cli_rejects_missing_and_duplicate_required_options() {
        for values in [
            vec![],
            vec!["--engine", "/a", "--engine", "/b", "--package", "/p"],
        ] {
            assert!(parse_cli(args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn cli_accepts_gateway_artifact_form_and_prefers_artifact() {
        let CliAction::Run(parsed) = parse_cli(args(&[
            "--engine",
            "/engine",
            "--artifact",
            "/aq4",
            "--package",
            "/compat",
        ]))
        .unwrap() else {
            panic!("expected run action");
        };
        let WorkerSource::Legacy(parsed) = parsed else {
            panic!("expected legacy source")
        };
        assert_eq!(parsed.package, PathBuf::from("/aq4"));
    }

    #[test]
    fn cli_accepts_manifest_as_an_exclusive_source() {
        let CliAction::Run(parsed) =
            parse_cli(args(&["--served-model-manifest", "/served-model.json"])).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(
            parsed,
            WorkerSource::ServedModelManifest {
                served_model: PathBuf::from("/served-model.json"),
                direct_trace_bindings: None,
                production_activation: None,
            }
        );
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--device-index",
                "7"
            ]))
            .is_err()
        );
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--package",
                "/package"
            ]))
            .is_err()
        );
    }

    #[test]
    fn cli_accepts_p3_binding_sidecar_only_as_exact_manifest_mode() {
        let sha = "a".repeat(64);
        let CliAction::Run(WorkerSource::ServedModelManifest {
            served_model,
            direct_trace_bindings: Some(source),
            production_activation: None,
        }) = parse_cli(args(&[
            "--served-model-manifest",
            "/served-model.json",
            "--p3-direct-trace-binding-manifest",
            "/bindings.json",
            "--p3-direct-trace-binding-manifest-sha256",
            &sha,
        ]))
        .unwrap()
        else {
            panic!("expected diagnostic manifest mode")
        };
        assert_eq!(served_model, PathBuf::from("/served-model.json"));
        assert_eq!(source.path, PathBuf::from("/bindings.json"));
        assert_eq!(source.bytes_sha256, sha);
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--p3-direct-trace-binding-manifest",
                "/bindings.json",
                "--p3-direct-trace-binding-manifest-sha256",
                "BAD",
            ]))
            .is_err()
        );
        assert!(
            parse_cli(args(&[
                "--package",
                "/package",
                "--p3-direct-trace-binding-manifest",
                "/bindings.json"
            ]))
            .is_err()
        );
    }

    #[test]
    fn cli_accepts_p3_production_activation_only_as_exact_manifest_mode() {
        let sha = "b".repeat(64);
        let CliAction::Run(WorkerSource::ServedModelManifest {
            served_model,
            direct_trace_bindings: None,
            production_activation: Some(source),
        }) = parse_cli(args(&[
            "--served-model-manifest",
            "/served-model.json",
            "--p3-production-activation",
            "/activation.json",
            "--p3-production-activation-sha256",
            &sha,
        ]))
        .unwrap()
        else {
            panic!("expected production activation manifest mode")
        };
        assert_eq!(served_model, PathBuf::from("/served-model.json"));
        assert_eq!(source.path, PathBuf::from("/activation.json"));
        assert_eq!(source.bytes_sha256, sha);
    }

    #[test]
    fn p3_binding_ingress_gate_and_sidecar_are_strictly_coupled() {
        assert!(validate_direct_trace_ingress(false, false, false, false).is_ok());
        assert!(validate_direct_trace_ingress(true, true, false, true).is_ok());
        assert!(validate_direct_trace_ingress(false, false, true, true).is_ok());
        assert!(validate_direct_trace_ingress(true, false, false, true).is_err());
        assert!(validate_direct_trace_ingress(false, true, false, false).is_err());
        assert!(validate_direct_trace_ingress(false, false, false, true).is_err());
        assert!(validate_direct_trace_ingress(true, true, true, true).is_err());
    }

    #[test]
    fn p3_production_activation_binds_selected_candidate_and_qualified_upstream() {
        let root = registry_test_root("p3-activation");
        let (path, sha) = activation_fixture(&root);
        assert!(load_p3_production_activation(&path, &sha).is_ok());
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["candidate"]["candidate_id"] = "paged-kv-table-validation-v1".into();
        let bytes = serde_json::to_vec(&value).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_p3_production_activation(&path, &bytes_sha256(&bytes)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cli_accepts_explicit_benchmark_wire_only_with_manifest() {
        let CliAction::Run(WorkerSource::BenchmarkServedModelManifest {
            served_model,
            case_registry,
            case_registry_sha256,
        }) = parse_cli(args(&[
            "--served-model-manifest",
            "/served-model.json",
            "--benchmark-wire",
            "--benchmark-case-manifest",
            "/cases.json",
            "--benchmark-case-manifest-sha256",
            &"a".repeat(64),
        ]))
        .unwrap()
        else {
            panic!("expected benchmark manifest mode");
        };
        assert_eq!(served_model, PathBuf::from("/served-model.json"));
        assert_eq!(case_registry, PathBuf::from("/cases.json"));
        assert_eq!(case_registry_sha256, "a".repeat(64));
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--benchmark-wire",
            ]))
            .is_err()
        );
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--benchmark-wire",
                "--benchmark-case-manifest",
                "/cases.json",
                "--benchmark-case-manifest-sha256",
                "not-a-sha",
            ]))
            .is_err()
        );
        assert!(parse_cli(args(&["--package", "/package", "--benchmark-wire"])).is_err());
    }

    #[test]
    fn benchmark_registry_snapshot_binds_same_fd_bytes_and_self_hash() {
        let root = registry_test_root("valid");
        let path = root.join("registry.json");
        let bytes = registry_bytes();
        std::fs::write(&path, &bytes).unwrap();
        load_benchmark_case_registry(&path, &bytes_sha256(&bytes)).unwrap();
        assert!(load_benchmark_case_registry(&path, &"0".repeat(64)).is_err());

        let mut rebound: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        rebound["registry_sha256"] = "0".repeat(64).into();
        let rebound = serde_json::to_vec(&rebound).unwrap();
        std::fs::write(&path, &rebound).unwrap();
        assert!(load_benchmark_case_registry(&path, &bytes_sha256(&rebound)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn p3_binding_sidecar_binds_canonical_single_link_bytes_and_rejects_reuse() {
        let root = registry_test_root("p3-binding-valid");
        let path = root.join("bindings.json");
        let bytes = direct_trace_manifest_bytes(&["request-m1", "request-m8"]);
        std::fs::write(&path, &bytes).unwrap();
        let bindings = load_direct_trace_binding_manifest(&path, &bytes_sha256(&bytes)).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].request_id, "request-m1");
        assert!(load_direct_trace_binding_manifest(&path, &"0".repeat(64)).is_err());

        let duplicate = direct_trace_manifest_bytes(&["request-m1", "request-m1"]);
        std::fs::write(&path, &duplicate).unwrap();
        assert!(load_direct_trace_binding_manifest(&path, &bytes_sha256(&duplicate)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn p3_binding_sidecar_rejects_tamper_escape_symlink_and_hardlink() {
        let root = registry_test_root("p3-binding-paths");
        let path = root.join("bindings.json");
        let bytes = direct_trace_manifest_bytes(&["request-1"]);
        std::fs::write(&path, &bytes).unwrap();
        let sha = bytes_sha256(&bytes);
        std::fs::write(&path, direct_trace_manifest_bytes(&["request-2"])).unwrap();
        assert!(load_direct_trace_binding_manifest(&path, &sha).is_err());
        std::fs::write(&path, &bytes).unwrap();

        let nested = root.join("nested");
        std::fs::create_dir(&nested).unwrap();
        let escaped = nested.join("..").join("bindings.json");
        assert!(load_direct_trace_binding_manifest(&escaped, &sha).is_err());

        let link = root.join("bindings-link.json");
        symlink(&path, &link).unwrap();
        assert!(load_direct_trace_binding_manifest(&link, &sha).is_err());
        std::fs::remove_file(link).unwrap();

        let hardlink = root.join("bindings-hardlink.json");
        std::fs::hard_link(&path, &hardlink).unwrap();
        assert!(load_direct_trace_binding_manifest(&path, &sha).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn benchmark_registry_snapshot_rejects_symlink_and_hardlink_paths() {
        let root = registry_test_root("links");
        let bytes = registry_bytes();
        let target = root.join("target.json");
        std::fs::write(&target, &bytes).unwrap();
        let digest = bytes_sha256(&bytes);

        let leaf_link = root.join("leaf.json");
        symlink(&target, &leaf_link).unwrap();
        assert!(load_benchmark_case_registry(&leaf_link, &digest).is_err());

        let real_parent = root.join("real");
        std::fs::create_dir(&real_parent).unwrap();
        let ancestor_target = real_parent.join("registry.json");
        std::fs::write(&ancestor_target, &bytes).unwrap();
        let linked_parent = root.join("linked-parent");
        symlink(&real_parent, &linked_parent).unwrap();
        assert!(
            load_benchmark_case_registry(&linked_parent.join("registry.json"), &digest).is_err()
        );

        let hardlink = root.join("hardlink.json");
        std::fs::hard_link(&target, &hardlink).unwrap();
        assert!(load_benchmark_case_registry(&target, &digest).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn benchmark_registry_snapshot_rejects_rename_and_same_size_rewrite() {
        let bytes = registry_bytes();
        let digest = bytes_sha256(&bytes);

        let rename_root = registry_test_root("rename");
        let rename_path = rename_root.join("registry.json");
        let displaced = rename_root.join("displaced.json");
        std::fs::write(&rename_path, &bytes).unwrap();
        let result = load_benchmark_case_registry_with_hook(&rename_path, &digest, |point| {
            if point == RegistrySnapshotPoint::AfterOpen {
                std::fs::rename(&rename_path, &displaced).unwrap();
                std::fs::write(&rename_path, &bytes).unwrap();
            }
        });
        assert!(result.is_err());
        std::fs::remove_dir_all(rename_root).unwrap();

        let rewrite_root = registry_test_root("rewrite");
        let rewrite_path = rewrite_root.join("registry.json");
        std::fs::write(&rewrite_path, &bytes).unwrap();
        let original_mtime = std::fs::metadata(&rewrite_path)
            .unwrap()
            .modified()
            .unwrap();
        let result = load_benchmark_case_registry_with_hook(&rewrite_path, &digest, |point| {
            if point == RegistrySnapshotPoint::AfterRead {
                let mut changed = bytes.clone();
                changed[0] ^= 1;
                std::fs::write(&rewrite_path, changed).unwrap();
                std::fs::File::options()
                    .write(true)
                    .open(&rewrite_path)
                    .unwrap()
                    .set_times(FileTimes::new().set_modified(original_mtime))
                    .unwrap();
            }
        });
        assert!(result.is_err());
        std::fs::remove_dir_all(rewrite_root).unwrap();
    }

    #[test]
    fn manifest_failure_log_omits_sensitive_detail() {
        let value = serde_json::to_value(ProcessLog {
            schema_version: "ullm.worker.log.v1",
            level: "error",
            event: "manifest_failed",
            phase: "process",
            error_code: Some("invalid_manifest"),
            detail: None,
        })
        .unwrap();
        assert_eq!(value["error_code"], "invalid_manifest");
        assert!(value.get("detail").is_none());
    }

    fn profile_snapshot() -> WorkerProfileSnapshot {
        WorkerProfileSnapshot {
            worker_schema: "ullm.worker.v1".into(),
            model: "ullm-qwen3.5-9b-aq4".into(),
            model_revision: "resident-test".into(),
            artifact_content_sha256: "a".repeat(64),
            package_manifest_sha256: "b".repeat(64),
            device: "gfx1201".into(),
            execution_profile: "rdna4_aq4_resident".into(),
            context_length: 4096,
            max_new_tokens: 512,
            vocab_size: 248320,
            eos_token_ids: vec![248044, 248046],
            top_k: 1,
            reasoning: None,
        }
    }

    #[test]
    fn manifest_startup_converts_to_fixed_resident_model_and_greedy_session_config() {
        let startup = WorkerStartupConfig {
            artifact_dir: None,
            package_dir: PathBuf::from("/product/package"),
            profile: profile_snapshot(),
            required_environment: vec!["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL".into()],
            reasoning: None,
        };
        let (config, profile) = resident_config_from_startup(startup).unwrap();

        assert_eq!(config.model.package_dir, PathBuf::from("/product/package"));
        assert_eq!(config.model.device_index, 1);
        assert_eq!(
            config.model.expected_architecture.as_deref(),
            Some("gfx1201")
        );
        assert_eq!(config.model.chunk_bytes, 1024 * 1024);
        assert_eq!(config.model.context_length, 4096);
        assert_eq!(config.model.kv_block_size, 256);
        assert_eq!(config.model.layer_indices, None);
        assert_eq!(config.model.lm_head_mode, PackageLmHeadMode::GpuResidentF32);
        assert_eq!(config.model.lm_head_chunk_rows, 8192);
        assert_eq!(
            config.session,
            Qwen35Aq4SessionConfig::greedy(512, vec![248044, 248046])
        );
        assert_eq!(config.expected_vocab_size, 248320);
        assert_eq!(profile.top_k, 1);
    }

    #[test]
    fn resident_config_rejects_artifact_and_non_greedy_profile() {
        let mut profile = profile_snapshot();
        profile.top_k = 2;
        let error = resident_config_from_startup(WorkerStartupConfig {
            artifact_dir: Some(PathBuf::from("/artifact")),
            package_dir: PathBuf::from("/package"),
            profile,
            required_environment: vec![],
            reasoning: None,
        })
        .unwrap_err();
        assert!(error.0.contains("inconsistent"));
    }

    #[derive(Default)]
    struct ScriptedModel {
        tokens: VecDeque<usize>,
        resets: usize,
        direct_trace_collector: Option<Qwen35Aq4DirectTraceCollector>,
        direct_trace_observations: Arc<Mutex<Vec<Qwen35Aq4DirectRuntimeObservation>>>,
        prefill_widths: Arc<Mutex<Vec<usize>>>,
    }

    impl Qwen35Aq4SessionModel for ScriptedModel {
        fn context_length(&self) -> usize {
            16
        }

        fn vocab_size(&self) -> usize {
            32
        }

        fn direct_trace_diagnostic_enabled(&self) -> bool {
            self.direct_trace_collector.is_some()
        }

        fn begin_direct_trace_request(
            &mut self,
            binding: Qwen35Aq4DirectTraceBinding,
        ) -> Result<bool, String> {
            let Some(collector) = self.direct_trace_collector.as_mut() else {
                return Ok(false);
            };
            collector.begin_request(binding)?;
            Ok(true)
        }

        fn finish_direct_trace_request(
            &mut self,
            terminal_status: &str,
        ) -> Result<Option<Qwen35Aq4DirectRuntimeObservation>, String> {
            let Some(collector) = self.direct_trace_collector.as_mut() else {
                return Ok(None);
            };
            let observation = collector.finish_request(terminal_status)?;
            self.direct_trace_observations
                .lock()
                .unwrap()
                .push(observation.clone());
            Ok(Some(observation))
        }

        fn record_direct_trace_failure_once(&mut self, reason: &str) -> Result<(), String> {
            if let Some(collector) = self.direct_trace_collector.as_mut()
                && collector.request_active()
                && !collector.failure_recorded()
            {
                collector.record_failure(reason)?;
            }
            Ok(())
        }

        fn reset_direct_trace_without_emission(&mut self) {
            if let Some(collector) = self.direct_trace_collector.as_mut() {
                collector.reset_without_emission();
            }
        }

        fn dispatch_token(
            &mut self,
            _: usize,
            _: usize,
            _: f32,
            _: usize,
            _: ullm_engine::execution_batch::ExecutionPhase,
            _: bool,
            _: &str,
        ) -> Result<
            Vec<[ullm_engine::backend_operation_registry::OperationExecutionRecord; 2]>,
            String,
        > {
            Ok(Vec::new())
        }

        fn dispatch_prefill_chunk(
            &mut self,
            token_ids: &[usize],
            _: usize,
            _: f32,
            _: usize,
            _: ullm_engine::execution_batch::ExecutionPhase,
            _: bool,
            _: &str,
        ) -> Result<Vec<ullm_engine::qwen35_aq4_session::Qwen35PrefillExecutionStep>, String>
        {
            self.prefill_widths.lock().unwrap().push(token_ids.len());
            if let Some(collector) = self.direct_trace_collector.as_mut()
                && collector.request_active()
            {
                collector.record_invocation(Qwen35Aq4DirectTraceRoute::Direct, 0, 2, 0)?;
            }
            Ok(Vec::new())
        }

        fn top_token_from_last_layer(&mut self, _: &str) -> Result<usize, String> {
            self.tokens
                .pop_front()
                .ok_or_else(|| "scripted token queue is empty".to_string())
        }

        fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
            self.resets += 1;
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct SharedOutput(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedOutput {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn scripted_profile() -> Sq8WorkerProfile {
        Sq8WorkerProfile {
            worker_schema: "ullm.worker.v1".into(),
            model: "scripted-aq4".into(),
            model_revision: "resident".into(),
            artifact_content_sha256: "a".repeat(64),
            package_manifest_sha256: "b".repeat(64),
            device: "cpu-test".into(),
            execution_profile: "resident-scripted".into(),
            context_length: 16,
            max_new_tokens: 4,
            vocab_size: 32,
            eos_token_ids: vec![2],
            top_k: 1,
            reasoning: None,
        }
    }

    fn scripted_reasoning_dialect() -> ullm_engine::reasoning::ReasoningDialect {
        ullm_engine::reasoning::ReasoningDialect {
            identity: "synthetic.worker-v2.v1".into(),
            start_sequence: vec![10, 11],
            end_sequence: vec![20, 21],
            forced_end_sequence: vec![20, 21],
            max_budget_tokens: 2,
            reserved_answer_tokens: 1,
            enabled_by_default: false,
            effort_budgets: vec![("low".into(), 1), ("medium".into(), 1), ("high".into(), 2)],
            history_reasoning_policy: ullm_engine::reasoning::HistoryReasoningPolicy::Omit,
            initial_phase: ullm_engine::reasoning::InitialReasoningPhase::Reasoning,
            eos_policy: ullm_engine::reasoning::ReasoningEosPolicy::Close,
        }
    }

    fn scripted_reasoning_profile() -> Sq8WorkerProfile {
        let mut profile = scripted_profile();
        profile.worker_schema = "ullm.worker.v2".into();
        profile.reasoning = Some(scripted_reasoning_dialect());
        profile
    }

    fn dummy_resident_config() -> ResidentWorkerConfig {
        ResidentWorkerConfig {
            model: Qwen35Aq4ModelLoadConfig {
                package_dir: PathBuf::from("/never-loaded"),
                device_index: 1,
                expected_architecture: Some("gfx1201".into()),
                chunk_bytes: 1024 * 1024,
                context_length: 16,
                kv_block_size: 256,
                layer_indices: None,
                lm_head_mode: PackageLmHeadMode::GpuResidentF32,
                lm_head_chunk_rows: 8192,
            },
            session: Qwen35Aq4SessionConfig::greedy(4, vec![2]),
            expected_vocab_size: 32,
        }
    }

    fn dummy_resident_reasoning_config() -> ResidentWorkerConfig {
        let mut config = dummy_resident_config();
        config.session.reasoning_dialect = Some(scripted_reasoning_dialect());
        config
    }

    fn run_diagnostic_resident_worker(
        request_id: &str,
        binding_request_id: &str,
        prefill_width: usize,
        second_request_id: Option<&str>,
    ) -> (
        Result<ullm_engine::worker_runtime::CommandReaderExit, String>,
        Vec<serde_json::Value>,
        Vec<serde_json::Value>,
        Vec<Qwen35Aq4DirectRuntimeObservation>,
        Vec<usize>,
    ) {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = SharedOutput::default();
        let captured = output.clone();
        let diagnostic_output = SharedOutput::default();
        let captured_diagnostic = diagnostic_output.clone();
        let observations = Arc::new(Mutex::new(Vec::new()));
        let widths = Arc::new(Mutex::new(Vec::new()));
        let thread_observations = Arc::clone(&observations);
        let thread_widths = Arc::clone(&widths);
        let binding: Qwen35Aq4DirectTraceBinding =
            serde_json::from_value(direct_trace_binding_value(binding_request_id)).unwrap();
        let mut config = dummy_resident_config();
        config.session = config
            .session
            .with_prefill_chunk_tokens(prefill_width)
            .unwrap();
        let process = thread::spawn(move || {
            run_loaded_worker(
                LoadedWorker::Resident {
                    config,
                    profile: scripted_profile(),
                    direct_trace_bindings: Some(VecDeque::from([binding])),
                },
                BufReader::new(input_reader),
                output,
                move |_| {
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel::default(),
                        Qwen35Aq4SessionConfig::greedy(4, vec![2]),
                    )?;
                    Ok(SessionInferenceBackend::new(session))
                },
                move |config| {
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel {
                            tokens: VecDeque::from([2]),
                            direct_trace_collector: Some(Qwen35Aq4DirectTraceCollector::default()),
                            direct_trace_observations: thread_observations,
                            prefill_widths: thread_widths,
                            ..ScriptedModel::default()
                        },
                        config.session,
                    )?;
                    Ok(SessionInferenceBackend::with_diagnostic_writer(
                        session,
                        Box::new(diagnostic_output),
                    ))
                },
            )
        });

        writeln!(
            input_writer,
            "{}",
            serde_json::json!({
                "schema_version": "ullm.worker.v1",
                "type": "generate",
                "request_id": request_id,
                "prompt_token_ids": vec![4; prefill_width],
                "max_new_tokens": 1,
                "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                "eos_token_ids": [2]
            })
        )
        .unwrap();
        input_writer.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let bytes = captured.0.lock().unwrap().clone();
            let text = String::from_utf8_lossy(&bytes);
            if text.contains("\"type\":\"released\"") || text.contains("\"type\":\"error\"") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "diagnostic worker event timed out"
            );
            thread::sleep(Duration::from_millis(5));
        }
        if let Some(second_request_id) = second_request_id {
            writeln!(
                input_writer,
                "{}",
                serde_json::json!({
                    "schema_version": "ullm.worker.v1",
                    "type": "generate",
                    "request_id": second_request_id,
                    "prompt_token_ids": vec![4; prefill_width],
                    "max_new_tokens": 1,
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                    "eos_token_ids": [2]
                })
            )
            .unwrap();
            input_writer.flush().unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let text = String::from_utf8_lossy(&captured.0.lock().unwrap()).into_owned();
                if text.contains("\"type\":\"error\"") {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "diagnostic binding exhaustion timed out"
                );
                thread::sleep(Duration::from_millis(5));
            }
        } else if request_id == binding_request_id {
            writeln!(
                input_writer,
                "{}",
                serde_json::json!({"schema_version": "ullm.worker.v1", "type": "shutdown"})
            )
            .unwrap();
            input_writer.flush().unwrap();
        }
        drop(input_writer);
        let result = process.join().unwrap();
        let lines = captured
            .0
            .lock()
            .unwrap()
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let observed = observations.lock().unwrap().clone();
        let observed_widths = widths.lock().unwrap().clone();
        let diagnostic_lines = captured_diagnostic
            .0
            .lock()
            .unwrap()
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        (result, lines, diagnostic_lines, observed, observed_widths)
    }

    #[test]
    fn p3_binding_ingress_traverses_production_jsonl_to_m1_and_m8_terminal_observation() {
        for width in [1, 8] {
            let request_id = format!("diagnostic-m{width}");
            let (result, lines, diagnostics, observations, widths) =
                run_diagnostic_resident_worker(&request_id, &request_id, width, None);
            assert_eq!(
                result.unwrap(),
                ullm_engine::worker_runtime::CommandReaderExit::IdleShutdown
            );
            assert_eq!(widths, vec![width]);
            assert_eq!(observations.len(), 1);
            assert_eq!(observations[0].request_id, request_id);
            assert_eq!(observations[0].terminal_status, "completed");
            assert_eq!(observations[0].counters.invocation_count, 1);
            assert_eq!(observations[0].counters.launch_count, 2);
            assert_eq!(diagnostics.len(), 1);
            assert_eq!(
                diagnostics[0]["schema_version"],
                ullm_engine::qwen35_aq4_direct_trace::RUNTIME_OBSERVATION_SCHEMA
            );
            assert_eq!(diagnostics[0]["request_id"], request_id);
            assert_eq!(diagnostics[0]["terminal_status"], "completed");
            assert!(lines.iter().all(|line| {
                line["schema_version"] == "ullm.worker.v1"
                    && line.get("counters").is_none()
                    && line.get("diagnostic_gate").is_none()
            }));
            assert_eq!(
                lines
                    .iter()
                    .map(|line| line["type"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["ready", "started", "progress", "token", "released"]
            );
        }
    }

    #[test]
    fn p3_binding_request_mismatch_fails_before_session_dispatch() {
        let (result, lines, diagnostics, observations, widths) =
            run_diagnostic_resident_worker("request-wire", "request-sidecar", 8, None);
        assert!(result.is_err());
        assert!(diagnostics.is_empty());
        assert!(observations.is_empty());
        assert!(widths.is_empty());
        assert!(lines.iter().all(|line| line.get("counters").is_none()));
        assert!(lines.iter().any(|line| line["type"] == "error"));
    }

    #[test]
    fn p3_binding_sidecar_entry_is_consumed_once_without_carryover() {
        let (result, lines, diagnostics, observations, widths) = run_diagnostic_resident_worker(
            "request-first",
            "request-first",
            1,
            Some("request-reuse"),
        );
        assert!(result.is_err());
        assert_eq!(widths, vec![1]);
        assert_eq!(observations.len(), 1);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0]["request_id"], "request-first");
        assert!(lines.iter().any(|line| line["type"] == "error"));
        assert!(lines.iter().all(|line| line.get("counters").is_none()));
    }

    #[test]
    fn manifest_resident_jsonl_route_never_builds_child_backend() {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = SharedOutput::default();
        let captured = output.clone();
        let legacy_builds = Arc::new(AtomicUsize::new(0));
        let resident_builds = Arc::new(AtomicUsize::new(0));
        let thread_legacy_builds = Arc::clone(&legacy_builds);
        let thread_resident_builds = Arc::clone(&resident_builds);
        let process = thread::spawn(move || {
            run_loaded_worker(
                LoadedWorker::Resident {
                    config: dummy_resident_config(),
                    profile: scripted_profile(),
                    direct_trace_bindings: None,
                },
                BufReader::new(input_reader),
                output,
                move |_| {
                    thread_legacy_builds.fetch_add(1, Ordering::SeqCst);
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel::default(),
                        Qwen35Aq4SessionConfig::greedy(4, vec![2]),
                    )?;
                    Ok(SessionInferenceBackend::new(session))
                },
                move |config| {
                    thread_resident_builds.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(config.model.package_dir, PathBuf::from("/never-loaded"));
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel {
                            tokens: VecDeque::from([2]),
                            resets: 0,
                            ..ScriptedModel::default()
                        },
                        config.session,
                    )?;
                    Ok(SessionInferenceBackend::new(session))
                },
            )
        });

        writeln!(
            input_writer,
            "{}",
            serde_json::json!({
                "schema_version": "ullm.worker.v1",
                "type": "generate",
                "request_id": "resident-1",
                "prompt_token_ids": [4],
                "max_new_tokens": 1,
                "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                "eos_token_ids": [2]
            })
        )
        .unwrap();
        input_writer.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let bytes = captured.0.lock().unwrap().clone();
            if String::from_utf8_lossy(&bytes).contains("\"type\":\"released\"") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "resident release event timed out"
            );
            thread::sleep(Duration::from_millis(5));
        }
        writeln!(
            input_writer,
            "{}",
            serde_json::json!({"schema_version": "ullm.worker.v1", "type": "shutdown"})
        )
        .unwrap();
        input_writer.flush().unwrap();
        drop(input_writer);
        assert_eq!(
            process.join().unwrap().unwrap(),
            ullm_engine::worker_runtime::CommandReaderExit::IdleShutdown
        );
        assert_eq!(legacy_builds.load(Ordering::SeqCst), 0);
        assert_eq!(resident_builds.load(Ordering::SeqCst), 1);

        let lines = captured
            .0
            .lock()
            .unwrap()
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let types = lines
            .iter()
            .map(|line| line["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(types, ["ready", "started", "progress", "token", "released"]);
        assert_eq!(lines[3]["token_id"], 2);
        assert_eq!(lines[4]["outcome"], "stop");
    }

    #[test]
    fn resident_v2_reasoning_jsonl_route_forces_close_and_preserves_schema() {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = SharedOutput::default();
        let captured = output.clone();
        let process = thread::spawn(move || {
            run_loaded_worker(
                LoadedWorker::Resident {
                    config: dummy_resident_reasoning_config(),
                    profile: scripted_reasoning_profile(),
                    direct_trace_bindings: None,
                },
                BufReader::new(input_reader),
                output,
                move |_| {
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel::default(),
                        Qwen35Aq4SessionConfig::greedy(4, vec![2]),
                    )?;
                    Ok(SessionInferenceBackend::new(session))
                },
                move |config| {
                    let session = Qwen35Aq4InferenceSession::from_model(
                        ScriptedModel {
                            tokens: VecDeque::from([7, 2]),
                            resets: 0,
                            ..ScriptedModel::default()
                        },
                        config.session,
                    )?;
                    Ok(SessionInferenceBackend::new(session))
                },
            )
        });

        writeln!(
            input_writer,
            "{}",
            serde_json::json!({
                "schema_version": "ullm.worker.v2",
                "type": "generate",
                "request_id": "resident-v2-reasoning",
                "prompt_token_ids": [4],
                "max_new_tokens": 4,
                "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                "eos_token_ids": [2],
                "reasoning": {
                    "enabled": true,
                    "budget_tokens": 1,
                    "dialect_id": "synthetic.worker-v2.v1",
                    "end_token_ids": [20, 21],
                    "forced_end_token_ids": [20, 21],
                    "reserved_answer_tokens": 1
                }
            })
        )
        .unwrap();
        input_writer.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let bytes = captured.0.lock().unwrap().clone();
            if String::from_utf8_lossy(&bytes).contains("\"type\":\"released\"") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "resident v2 release event timed out"
            );
            thread::sleep(Duration::from_millis(5));
        }
        writeln!(
            input_writer,
            "{}",
            serde_json::json!({"schema_version": "ullm.worker.v2", "type": "shutdown"})
        )
        .unwrap();
        input_writer.flush().unwrap();
        drop(input_writer);
        assert_eq!(
            process.join().unwrap().unwrap(),
            ullm_engine::worker_runtime::CommandReaderExit::IdleShutdown
        );

        let lines = captured
            .0
            .lock()
            .unwrap()
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(
            lines
                .iter()
                .all(|line| line["schema_version"] == "ullm.worker.v2")
        );
        let token_ids = lines
            .iter()
            .filter(|line| line["type"] == "token")
            .map(|line| line["token_id"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(token_ids, [7, 20, 21, 2]);
        let released = lines
            .iter()
            .find(|line| line["type"] == "released")
            .unwrap();
        assert_eq!(released["outcome"], "stop");
        assert_eq!(released["completion_tokens"], 4);
        assert_eq!(released["reasoning_tokens"], 1);
        assert_eq!(released["forced_end_tokens"], 2);
    }

    #[test]
    fn resident_contract_rechecks_format_and_sampling() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../services/openai-gateway/tests/fixtures/served-model/aq4/served-model.json",
        );
        let mut model = load_served_model(fixture).unwrap();
        model.worker.required_environment = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        validate_resident_model_contract(&model).unwrap();
        model.generation.sampling.temperature = true;
        assert!(validate_resident_model_contract(&model).is_err());
        model.generation.sampling.temperature = false;
        model.generation.sampling.top_p = true;
        assert!(validate_resident_model_contract(&model).is_err());
        model.generation.sampling.top_p = false;
        model.generation.sampling.top_k = 2;
        assert!(validate_resident_model_contract(&model).is_err());
        model.generation.sampling.top_k = 1;
        model.format.format_id = "SQ8_0".into();
        assert!(validate_resident_model_contract(&model).is_err());
        model.format.format_id = "AQ4_0".into();
        model.format.implementation_id = "wrong".into();
        assert!(validate_resident_model_contract(&model).is_err());
        model.format.implementation_id = "qwen35_aq4_rdna4_v1".into();
        model.worker.identity.device = "cpu".into();
        assert!(validate_resident_model_contract(&model).is_err());
        model.worker.identity.device = "gfx1201".into();
        model.worker.identity.execution_profile = "compat".into();
        assert!(validate_resident_model_contract(&model).is_err());
        model.worker.identity.execution_profile = "rdna4_aq4_resident".into();
        model.worker.required_environment.pop();
        assert!(validate_resident_model_contract(&model).is_err());
        model.worker.required_environment = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        model
            .worker
            .required_environment
            .push("ULLM_REQUIRE_HIP_UNKNOWN_KERNEL".into());
        assert!(validate_resident_model_contract(&model).is_err());
    }

    #[test]
    fn deployment_profile_matches_resident_worker_contract() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/served-models/qwen35-9b-aq4.profile.json");
        let profile: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(profile["format"]["format_id"], "AQ4_0");
        assert_eq!(
            profile["format"]["implementation_id"],
            "qwen35_aq4_rdna4_v1"
        );
        assert_eq!(profile["worker"]["identity"]["device"], "gfx1201");
        assert_eq!(
            profile["worker"]["identity"]["execution_profile"],
            "rdna4_aq4_resident"
        );
        let mut actual = profile["worker"]["required_environment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(actual.contains(&"ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL"));
        assert!(actual.contains(&"ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL"));
        assert!(actual.contains(&"ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL"));
        actual.sort_unstable();
        let mut expected = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.to_vec();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn operation_trace_worker_audit_wrapper_is_exact_json() {
        let line = operation_trace_log_line(7, "{\"implementation_id\":\"impl-1\"}");
        assert_eq!(
            line,
            "{\"schema_version\":\"ullm.backend_operation.load.v1\",\"layer_position\":7,\"trace\":{\"implementation_id\":\"impl-1\"}}"
        );
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["trace"]["implementation_id"], "impl-1");
    }
}
