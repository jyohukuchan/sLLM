// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use ullm_engine::aq4_worker_backend::{
    QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV, Qwen35Aq4WorkerBackend, Qwen35Aq4WorkerBackendConfig,
};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig};
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
    ServedModelManifest(PathBuf),
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
    },
}

fn main() -> ExitCode {
    match parse_cli(env::args_os().skip(1)) {
        Ok(CliAction::Help) => {
            eprintln!(
                "Usage: ullm-aq4-worker [--engine PATH] --package PATH [--device-index N] [--layers all|CSV]\n\
                 Gateway form: --artifact AQ4_PACKAGE --package COMPAT_PATH [extra options]\n\
                 Manifest mode: ullm-aq4-worker --served-model-manifest PATH\n\
                 Reads ullm.worker.v1 commands from stdin and writes events to stdout.\n\
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
    match run_loaded_worker(
        loaded,
        input,
        output,
        Qwen35Aq4WorkerBackend::load,
        load_resident_backend,
    ) {
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
    match source {
        WorkerSource::Legacy(args) => {
            let config = Qwen35Aq4WorkerBackendConfig::new(args.engine, args.package)
                .map(|config| config.with_device_index(args.device_index))
                .and_then(|config| config.with_layers(args.layers))
                .map_err(ServedModelError)?;
            Ok(LoadedWorker::Legacy {
                config,
                profile: configured_aq4_worker_profile(),
            })
        }
        WorkerSource::ServedModelManifest(path) => {
            let model = load_served_model(path)?;
            let current_exe =
                env::current_exe().map_err(|error| ServedModelError(error.to_string()))?;
            load_resident_worker(&model, &current_exe)
        }
    }
}

fn load_resident_worker(
    model: &ServedModel,
    current_exe: &std::path::Path,
) -> Result<LoadedWorker, ServedModelError> {
    validate_resident_model_contract(model)?;
    let startup = model.worker_startup(WorkerBackendKind::Aq4, current_exe)?;
    let (config, profile) = resident_config_from_startup(startup)?;
    Ok(LoadedWorker::Resident { config, profile })
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
        chunk_bytes: RESIDENT_CHUNK_BYTES,
        context_length: startup.profile.context_length,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: RESIDENT_LM_HEAD_CHUNK_ROWS,
    };
    let session = Qwen35Aq4SessionConfig::greedy(
        startup.profile.max_new_tokens,
        startup.profile.eos_token_ids.clone(),
    );
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
    Ok(SessionInferenceBackend::new(session))
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
        LoadedWorker::Resident { config, profile } => {
            run_worker_process_with_profile(input, output, profile, move || resident_loader(config))
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
        if args.len() != 2 || args[0] != "--served-model-manifest" || args[1].is_empty() {
            return Err("manifest mode and legacy options are mutually exclusive".into());
        }
        return Ok(CliAction::Run(WorkerSource::ServedModelManifest(
            PathBuf::from(&args[1]),
        )));
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
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};
    use ullm_engine::qwen35_aq4_session::Qwen35Aq4SessionModel;
    use ullm_engine::served_model::WorkerProfileSnapshot;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
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
            WorkerSource::ServedModelManifest(PathBuf::from("/served-model.json"))
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
        }
    }

    #[test]
    fn manifest_startup_converts_to_fixed_resident_model_and_greedy_session_config() {
        let startup = WorkerStartupConfig {
            artifact_dir: None,
            package_dir: PathBuf::from("/product/package"),
            profile: profile_snapshot(),
            required_environment: vec!["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL".into()],
        };
        let (config, profile) = resident_config_from_startup(startup).unwrap();

        assert_eq!(config.model.package_dir, PathBuf::from("/product/package"));
        assert_eq!(config.model.device_index, 1);
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
        })
        .unwrap_err();
        assert!(error.0.contains("inconsistent"));
    }

    #[derive(Default)]
    struct ScriptedModel {
        tokens: VecDeque<usize>,
        resets: usize,
    }

    impl Qwen35Aq4SessionModel for ScriptedModel {
        fn context_length(&self) -> usize {
            16
        }

        fn vocab_size(&self) -> usize {
            32
        }

        fn dispatch_token(
            &mut self,
            _: usize,
            _: usize,
            _: f32,
            _: usize,
            _: bool,
            _: &str,
        ) -> Result<(), String> {
            Ok(())
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
        }
    }

    fn dummy_resident_config() -> ResidentWorkerConfig {
        ResidentWorkerConfig {
            model: Qwen35Aq4ModelLoadConfig {
                package_dir: PathBuf::from("/never-loaded"),
                device_index: 1,
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
}
