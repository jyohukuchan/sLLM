//! Same-process BF16 versus NVFP4 MTP companion whole-graph comparison.

use super::*;

#[derive(Serialize)]
pub(super) struct Report {
    schema_version: &'static str,
    pub(super) state: &'static str,
    target: String,
    device_index: u32,
    model: ModelReport,
    binary: Stage9BinaryIdentity,
    fixture: FixtureReport,
    draft_width: usize,
    warmups_per_variant: usize,
    measured_per_variant: usize,
    draft_vocabulary_sha256: String,
    candidate_companion_digest: String,
    baseline_companion_resident_bytes: u64,
    candidate_companion_resident_bytes: u64,
    rounds: Vec<Round>,
    all_rounds_candidate_faster: bool,
    all_rounds_meet_one_percent: bool,
    all_rounds_hip_only: bool,
    all_rounds_request_cleanup_zero: bool,
    cleanup: CleanupReport,
}

#[derive(Serialize)]
struct Round {
    round: usize,
    order: &'static str,
    bf16: Stage9VariantReport,
    nvfp4: Stage9VariantReport,
    nvfp4_tpot_shortening_percent: f64,
    nvfp4_faster: bool,
    nvfp4_meets_one_percent: bool,
    generated_tokens_equal: bool,
}

pub(super) fn run(config: Config) -> Result<Report, String> {
    let required_row = RowSpec {
        prompt_tokens: PHASE83_PROMPT_CAPACITY,
        output_tokens: 128,
    };
    if !config.phase83
        || !config.mtp.enabled
        || config.mtp.draft_width != 2
        || config.sampling.mode != SamplingMode::GpuFixed
        || config.sampling.replay_inputs
        || config.rows.as_slice() != [required_row]
        || config.warmups != 1
        || config.measured != 3
    {
        return Err(format!(
            "{PHASE87_STAGE3_ABBA_ENV}=1 requires Phase83 coding8192, MTP width2, gpu-fixed sampling, row 8192/128, warmup1, measured3"
        ));
    }
    let companion_path = config.mtp_companion_path.as_ref().ok_or_else(|| {
        format!("{PHASE87_STAGE3_ABBA_ENV}=1 requires {PHASE84_MTP_COMPANION_PATH_ENV}")
    })?;
    let artifact = Arc::new(
        verify_unsloth_qwen38_nvfp4(&config.model_root).map_err(|error| error.to_string())?,
    );
    let fixture = build_prompt_fixture(&artifact, PromptFixtureKind::Coding8192)?;
    let tokenizer = load_locked_tokenizer(artifact.root())?;
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let candidate = Arc::new(
        verify_qwen38_mtp_quantized_sidecar(
            &lock,
            &artifact,
            &companion_path.join("manifest.json"),
            &companion_path.join("payload.safetensors"),
        )
        .map_err(|error| format!("Stage3 NVFP4 sidecar verification failed: {error}"))?,
    );
    if candidate.encoding() != sllm_core::MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 {
        return Err("Stage3 AB/BA candidate must be the calibrated NVFP4 MTP sidecar".to_owned());
    }
    let companion_digest = candidate.combined_recipe_digest(artifact.recipe_digest());
    let target_plan =
        build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|error| error.to_string())?;
    let target_graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &target_plan,
        &artifact,
        config.chunk_capacity,
        config.state_capacity,
        config.kv_cache,
    )
    .map_err(|error| format!("Stage3 target graph failed: {error}"))?;
    let mtp_plan = build_qwen38_nvfp4_mtp_weight_load_plan(&lock, &artifact)
        .map_err(|error| format!("Stage3 MTP plan failed: {error}"))?;
    let priming_capacity =
        parse_env_or("SLLM_PHASE85_MTP_PRIMING_CHUNK_CAPACITY", Some(1_024_u64))?;
    if priming_capacity == 0 || priming_capacity > config.state_capacity {
        return Err("Stage3 MTP priming capacity is outside request state".to_owned());
    }
    let vocab = load_qwen38_mtp_draft_vocabulary(&config.model_root)
        .map_err(|error| format!("Stage3 draft vocabulary failed: {error}"))?
        .ok_or("Stage3 AB/BA requires the reviewed reduced draft vocabulary")?;
    let bf16_graph = build_qwen38_nvfp4_mtp_graph_with_companion_and_vocabulary_ids(
        &lock,
        &mtp_plan,
        &artifact,
        config.state_capacity,
        config.kv_cache,
        priming_capacity,
        None,
        vocab.ids(),
    )
    .map_err(|error| format!("Stage3 BF16 graph failed: {error}"))?;
    let nvfp4_graph = build_qwen38_nvfp4_mtp_graph_with_companion_and_vocabulary_ids(
        &lock,
        &mtp_plan,
        &artifact,
        config.state_capacity,
        config.kv_cache,
        priming_capacity,
        Some(&candidate),
        vocab.ids(),
    )
    .map_err(|error| format!("Stage3 NVFP4 graph failed: {error}"))?;
    let binary = stage9_binary_identity(&config.target)?;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(config.device_index, config.target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let stop_token_ids = lock.generation_stop_policy().stop_token_ids.clone();
    let operation = (|| -> Result<_, String> {
        let target = QwenResidentModel::new_unsloth_qwen38_nvfp4(
            Arc::clone(&session),
            target_graph.clone(),
            target_plan,
            Arc::clone(&artifact),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("Stage3 target provisioning failed: {error}"))?;
        let target_bytes = allocation_report(session.memory_snapshot())
            .model_resident
            .current_bytes;
        let bf16 = provision_qwen38_mtp_resident(
            &target,
            bf16_graph.clone(),
            mtp_plan.clone(),
            Arc::clone(&artifact),
            None,
        )?;
        let with_bf16 = allocation_report(session.memory_snapshot())
            .model_resident
            .current_bytes;
        let nvfp4 = provision_qwen38_mtp_resident(
            &target,
            nvfp4_graph.clone(),
            mtp_plan,
            Arc::clone(&artifact),
            Some(candidate.clone()),
        )?;
        let with_both = allocation_report(session.memory_snapshot())
            .model_resident
            .current_bytes;
        let bf16_bytes = with_bf16
            .checked_sub(target_bytes)
            .ok_or("Stage3 BF16 resident memory decreased after load")?;
        let nvfp4_bytes = with_both
            .checked_sub(with_bf16)
            .ok_or("Stage3 NVFP4 resident memory decreased after load")?;
        let mut rounds = Vec::with_capacity(2);
        for (round, order) in [(1, "AB"), (2, "BA")] {
            let run_bf16 = || {
                run_stage9_variant(
                    &session,
                    &target,
                    &target_graph,
                    &bf16,
                    &bf16_graph,
                    &fixture.tokens,
                    &config.target,
                    &tokenizer.tokenizer,
                    config.sampling,
                    Some(stop_token_ids.as_slice()),
                    "A-bf16",
                )
            };
            let run_nvfp4 = || {
                run_stage9_variant(
                    &session,
                    &target,
                    &target_graph,
                    &nvfp4,
                    &nvfp4_graph,
                    &fixture.tokens,
                    &config.target,
                    &tokenizer.tokenizer,
                    config.sampling,
                    Some(stop_token_ids.as_slice()),
                    "B-nvfp4",
                )
            };
            let (baseline, changed) = if order == "AB" {
                (run_bf16()?, run_nvfp4()?)
            } else {
                let changed = run_nvfp4()?;
                (run_bf16()?, changed)
            };
            let control_tpot = baseline.measured_tpot_ms.median;
            let changed_tpot = changed.measured_tpot_ms.median;
            if control_tpot <= 0.0 || changed_tpot <= 0.0 {
                return Err("Stage3 AB/BA measured TPOT is non-positive".to_owned());
            }
            let shortening = (control_tpot - changed_tpot) * 100.0 / control_tpot;
            let generated_tokens_equal = baseline
                .measured
                .iter()
                .zip(&changed.measured)
                .all(|(left, right)| left.generated_tokens == right.generated_tokens);
            rounds.push(Round {
                round,
                order,
                bf16: baseline,
                nvfp4: changed,
                nvfp4_tpot_shortening_percent: shortening,
                nvfp4_faster: shortening > 0.0,
                nvfp4_meets_one_percent: shortening >= 1.0,
                generated_tokens_equal,
            });
        }
        drop(nvfp4);
        drop(bf16);
        drop(target);
        Ok((rounds, bf16_bytes, nvfp4_bytes))
    })();
    let before_shutdown = allocation_report(session.memory_snapshot());
    let shutdown = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("Stage3 AB/BA session shutdown failed: {error}"));
    let (rounds, bf16_bytes, nvfp4_bytes) = match operation {
        Ok(result) => result,
        Err(error) => {
            let cleanup = shutdown
                .map(|report| {
                    format!(
                        "current_bytes={} poisoned={} retryable_cleanup={} durable_quarantine={}",
                        before_shutdown.current_bytes,
                        before_shutdown.poisoned,
                        report.retryable_cleanup,
                        report.durable_quarantine
                    )
                })
                .unwrap_or_else(|cleanup_error| cleanup_error);
            return Err(format!("{error}; post-error cleanup: {cleanup}"));
        }
    };
    let shutdown = shutdown?;
    let cleanup_zero = before_shutdown.current_bytes == 0
        && !before_shutdown.poisoned
        && shutdown.retryable_cleanup == 0
        && shutdown.durable_quarantine == 0;
    let all_rounds_hip_only = rounds.iter().all(|round| {
        round.bf16.all_dispatches_hip
            && round.nvfp4.all_dispatches_hip
            && !round.bf16.fallback_used
            && !round.nvfp4.fallback_used
    });
    let all_rounds_request_cleanup_zero = rounds
        .iter()
        .all(|round| round.bf16.request_cleanup_zero && round.nvfp4.request_cleanup_zero);
    Ok(Report {
        schema_version: "phase87-stage3-companion-abba-v1",
        state: if cleanup_zero && all_rounds_hip_only && all_rounds_request_cleanup_zero {
            "PASS"
        } else {
            "FAIL"
        },
        target: config.target.clone(),
        device_index: config.device_index,
        model: ModelReport {
            root: config.model_root.display().to_string(),
            path_environment: config.model_env,
            repository: UNSLOTH_QWEN38_NVFP4_REPOSITORY,
            revision: UNSLOTH_QWEN38_NVFP4_REVISION,
            model_bytes: UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
            model_sha256: UNSLOTH_QWEN38_NVFP4_MODEL_SHA256,
        },
        binary,
        fixture: fixture.report,
        draft_width: 2,
        warmups_per_variant: 1,
        measured_per_variant: 3,
        draft_vocabulary_sha256: vocab.vocab_sha256().to_owned(),
        candidate_companion_digest: companion_digest,
        baseline_companion_resident_bytes: bf16_bytes,
        candidate_companion_resident_bytes: nvfp4_bytes,
        all_rounds_candidate_faster: rounds.iter().all(|round| round.nvfp4_faster),
        all_rounds_meet_one_percent: rounds.iter().all(|round| round.nvfp4_meets_one_percent),
        all_rounds_hip_only,
        all_rounds_request_cleanup_zero,
        rounds,
        cleanup: CleanupReport {
            allocation_after_resident_drop_before_shutdown: before_shutdown,
            retryable_cleanup: shutdown.retryable_cleanup,
            durable_quarantine: shutdown.durable_quarantine,
            zero: cleanup_zero,
        },
    })
}
