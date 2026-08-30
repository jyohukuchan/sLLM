//! Phase 53 Qwen3.5 FP16-to-block16 quality candidate runner.

mod sequential {
    use std::env;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::sync::Arc;
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use sllm_core::{
        Backend, ExecutionSession, ExecutionSessionRequest, KvCacheEncoding,
        KvCacheSelectionRequest, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID, QWEN35_4B_REVISION,
        QwenExecutionAudit, QwenResidentModel, build_qwen35_graph_with_kv_cache_selection,
        build_verified_weight_load_plan, read_model_lock, resolve_kv_cache_selection,
    };
    use sllm_hip::HipBackend;

    const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
    const DATASET_SHA256: &str = "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
    const VOCAB_SIZE: usize = 248_320;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Dataset {
        schema_version: String,
        dataset_id: String,
        license: String,
        provenance: String,
        seed: u64,
        token_generator: String,
        sample_order: String,
        cases: Vec<DatasetCase>,
        coverage: serde_json::Value,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DatasetCase {
        id: String,
        length: usize,
        start: u64,
        step: u64,
        expected_next: i32,
        band: String,
        block_tail: bool,
    }

    struct PreparedCase {
        id: String,
        tokens: Vec<i32>,
        expected_next: i32,
    }

    struct LogitPair {
        prefill: Vec<f32>,
        decode: Vec<f32>,
    }

    struct EncodingRun {
        rows: Vec<LogitPair>,
        dispatches: u64,
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn load_dataset(path: &Path) -> Result<(String, Vec<PreparedCase>), String> {
        let bytes = fs::read(path).map_err(|error| format!("read dataset: {error}"))?;
        let actual = sha256(&bytes);
        if actual != DATASET_SHA256 {
            return Err(format!(
                "dataset digest differs: expected {DATASET_SHA256}, got {actual}"
            ));
        }
        let dataset: Dataset =
            serde_json::from_slice(&bytes).map_err(|error| format!("parse dataset: {error}"))?;
        if dataset.schema_version != "sllm-phase46-kv-quality-dataset-v1"
            || dataset.dataset_id != "phase46-kv-quality-baseline-v1"
            || dataset.license != "CC0-1.0"
            || dataset.provenance.is_empty()
            || dataset.token_generator != "token[i] = 1 + ((start + i * step + seed) mod 200000)"
            || dataset.sample_order != "listed"
            || dataset.seed != 1729
            || dataset.cases.len() != 10
            || !dataset.coverage.is_object()
        {
            return Err("dataset identity differs".to_owned());
        }
        let mut prepared = Vec::with_capacity(dataset.cases.len());
        for case in dataset.cases {
            if case.id.is_empty()
                || case.length == 0
                || case.length > 513
                || case.band.is_empty()
                || !(0..VOCAB_SIZE as i32).contains(&case.expected_next)
            {
                return Err("dataset case is invalid".to_owned());
            }
            let mut tokens = Vec::with_capacity(case.length);
            for index in 0..case.length {
                let value = case
                    .start
                    .checked_add(
                        (index as u64)
                            .checked_mul(case.step)
                            .ok_or("token product overflow")?,
                    )
                    .and_then(|value| value.checked_add(dataset.seed))
                    .ok_or("token generator overflow")?
                    % 200_000
                    + 1;
                tokens.push(i32::try_from(value).map_err(|_| "token does not fit i32")?);
            }
            let _ = case.block_tail;
            prepared.push(PreparedCase {
                id: case.id,
                tokens,
                expected_next: case.expected_next,
            });
        }
        Ok((format!("sha256:{actual}"), prepared))
    }

    fn validate_audit(audit: &QwenExecutionAudit, target: &str) -> Result<(), String> {
        if audit.selected_backend() != "hip"
            || audit.target() != target
            || audit.fallback_used()
            || !audit.all_dispatches_hip()
            || audit.kernel_dispatch_count() == 0
        {
            return Err(format!(
                "execution was not exact HIP/no-fallback: {audit:?}"
            ));
        }
        Ok(())
    }

    fn validate_logits(values: &[f32], label: &str) -> Result<(), String> {
        if values.len() != VOCAB_SIZE || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{label} logits are non-finite or truncated"));
        }
        Ok(())
    }

    fn execute_encoding(
        session: &Arc<ExecutionSession>,
        lock: &sllm_core::ModelLock,
        cache: &Arc<sllm_core::VerifiedCache>,
        plan: &sllm_core::WeightLoadPlan,
        cases: &[PreparedCase],
        encoding: KvCacheEncoding,
        target: &str,
    ) -> Result<EncodingRun, String> {
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("session was not empty before resident creation".to_owned());
        }
        let maximum = cases
            .iter()
            .map(|case| case.tokens.len())
            .max()
            .unwrap_or(0);
        let selection = resolve_kv_cache_selection(KvCacheSelectionRequest::new(
            Some(encoding),
            target,
            QWEN35_4B_FINGERPRINT,
            true,
            true,
            true,
            256,
        ))
        .map_err(|error| format!("resolve target-aware KV selection: {error}"))?;
        let seed_graph = build_qwen35_graph_with_kv_cache_selection(
            lock,
            plan,
            maximum as u64,
            maximum as u64 + 1,
            selection,
        )
        .map_err(|error| format!("build seed graph: {error}"))?;
        let resident = QwenResidentModel::new(
            Arc::clone(session),
            seed_graph,
            plan.clone(),
            Arc::clone(cache),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("create resident: {error}"))?;
        let ready = session.memory_snapshot();
        if ready.poisoned()
            || ready.model_resident().current_bytes() == 0
            || ready.request_state().current_bytes() != 0
            || ready.workspace().current_bytes() != 0
        {
            return Err("resident baseline is invalid".to_owned());
        }
        let mut rows = Vec::with_capacity(cases.len());
        let mut dispatches = 0_u64;
        for case in cases {
            let graph = build_qwen35_graph_with_kv_cache_selection(
                lock,
                plan,
                case.tokens.len() as u64,
                case.tokens.len() as u64 + 1,
                selection,
            )
            .map_err(|error| format!("build {}: {error}", case.id))?;
            let mut request = resident
                .new_request(graph)
                .map_err(|error| format!("request {}: {error}", case.id))?;
            let prefill = request
                .prefill_with_last_logits(&case.tokens)
                .map_err(|error| format!("prefill {}: {error}", case.id))?
                .last_logits()
                .ok_or_else(|| "prefill omitted logits".to_owned())?
                .to_vec();
            validate_logits(&prefill, "prefill")?;
            let decode = request
                .decode_with_last_logits(case.expected_next)
                .map_err(|error| format!("decode {}: {error}", case.id))?
                .last_logits()
                .ok_or_else(|| "decode omitted logits".to_owned())?
                .to_vec();
            validate_logits(&decode, "decode")?;
            let audit = request
                .audit_snapshot()
                .map_err(|error| error.to_string())?;
            validate_audit(&audit, target)?;
            dispatches = dispatches
                .checked_add(audit.kernel_dispatch_count())
                .ok_or("dispatch overflow")?;
            rows.push(LogitPair { prefill, decode });
            drop(request);
            let restored = session.memory_snapshot();
            if restored.poisoned()
                || restored.model_resident().current_bytes()
                    != ready.model_resident().current_bytes()
                || restored.request_state().current_bytes() != 0
                || restored.workspace().current_bytes() != 0
                || restored.current_bytes() != ready.current_bytes()
            {
                return Err("request cleanup did not restore resident baseline".to_owned());
            }
        }
        drop(resident);
        if session.memory_snapshot().poisoned() || session.memory_snapshot().current_bytes() != 0 {
            return Err("resident release was incomplete".to_owned());
        }
        Ok(EncodingRun { rows, dispatches })
    }

    fn top1(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map_or(0, |(index, _)| index)
    }

    fn logsumexp(values: &[f32]) -> f64 {
        let maximum = f64::from(values.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        maximum
            + values
                .iter()
                .map(|value| (f64::from(*value) - maximum).exp())
                .sum::<f64>()
                .ln()
    }

    fn nll(values: &[f32], target: i32) -> f64 {
        logsumexp(values) - f64::from(values[target as usize])
    }

    fn kld(reference: &[f32], candidate: &[f32]) -> f64 {
        let reference_lse = logsumexp(reference);
        let candidate_lse = logsumexp(candidate);
        reference
            .iter()
            .zip(candidate)
            .map(|(reference, candidate)| {
                let log_p = f64::from(*reference) - reference_lse;
                let log_q = f64::from(*candidate) - candidate_lse;
                log_p.exp() * (log_p - log_q)
            })
            .sum::<f64>()
            .max(0.0)
    }

    fn percentile(values: &[f64], quantile: f64) -> f64 {
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    struct FileSync;

    impl FileSync {
        fn sync(path: &Path) -> Result<(), String> {
            OpenOptions::new()
                .read(true)
                .open(path)
                .and_then(|file| file.sync_all())
                .map_err(|error| format!("sync output directory: {error}"))
        }
    }

    #[derive(Serialize)]
    struct CandidateIdentity {
        policy_sha256: String,
        dataset_sha256: String,
        model_lock_fingerprint: &'static str,
        model_lock_sha256: String,
        derived_lock_fingerprint: String,
        derived_lock_sha256: String,
        binary_sha256: String,
        descriptor_id: &'static str,
        scale_recipe: &'static str,
    }

    #[derive(Serialize)]
    struct MetricSampleCounts {
        perplexity: usize,
        kld: usize,
        top1: usize,
        task: usize,
        #[serde(rename = "long-context")]
        long_context: usize,
    }

    #[derive(Serialize)]
    struct ComparisonMetrics {
        selected_count: usize,
        metric_sample_counts: MetricSampleCounts,
        perplexity_relative_delta: f64,
        kld_p99: f64,
        top1_agreement: f64,
        task_score_delta: f64,
        long_context_score_delta: f64,
        hip_dispatches: u64,
        fallback_used: bool,
        all_dispatches_hip: bool,
    }

    #[derive(Serialize)]
    struct CandidateRepeat {
        repeat: u32,
        fp16_released_before_block16: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        block16_released_after_repeat: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        block16_released_before_mxfp8: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mxfp8_released_after_repeat: Option<bool>,
        block16: ComparisonMetrics,
        #[serde(skip_serializing_if = "Option::is_none")]
        mxfp8: Option<ComparisonMetrics>,
    }

    #[derive(Serialize)]
    struct Mxfp8Comparison {
        status: &'static str,
        encoding: Option<&'static str>,
        reference_only: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<&'static str>,
    }

    #[derive(Serialize)]
    struct CandidateCleanup {
        retryable: usize,
        durable: usize,
        terminal_zero: bool,
    }

    #[derive(Serialize)]
    struct CandidateReport {
        #[serde(rename = "$schema")]
        schema: &'static str,
        schema_version: &'static str,
        state: &'static str,
        identity: CandidateIdentity,
        target: String,
        device_index: u32,
        encoding: &'static str,
        mxfp8_comparison: Mxfp8Comparison,
        reference_encoding: &'static str,
        sequential_residents: bool,
        completely_sequential_order: Vec<&'static str>,
        selected_count: usize,
        repeats: Vec<CandidateRepeat>,
        cleanup: CandidateCleanup,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DerivedLock {
        schema_version: String,
        fingerprint: String,
        semantic_model_id: String,
        source_lock_fingerprints: Vec<String>,
        converter: serde_json::Value,
        output: serde_json::Value,
    }

    fn exact_target(value: &str) -> bool {
        matches!(value, "gfx942:sramecc+:xnack-" | "gfx1201" | "gfx1030")
    }

    type CandidateEncoding = (
        KvCacheEncoding,
        &'static str,
        &'static str,
        Option<(KvCacheEncoding, &'static str)>,
    );

    fn candidate_encoding(target: &str, value: &str) -> Result<CandidateEncoding, String> {
        match (target, value) {
            ("gfx942:sramecc+:xnack-", "kv-fp8-e4-block16") => Ok((
                KvCacheEncoding::Fp8E4M3Block16,
                "kv-fp8-e4-block16",
                "kv-fp8-e4-block16-v2",
                None,
            )),
            ("gfx1201", "kv-fp8-e4-block16") => Ok((
                KvCacheEncoding::Fp8E4M3Block16,
                "kv-fp8-e4-block16",
                "kv-fp8-e4-block16-v2",
                Some((KvCacheEncoding::Mxfp8E4, "kv-mxfp8-e4")),
            )),
            ("gfx1030", "kv-fp8-e5-block16") => Ok((
                KvCacheEncoding::Fp8E5M2Block16,
                "kv-fp8-e5-block16",
                "kv-fp8-e5-block16-v2",
                Some((KvCacheEncoding::Mxfp8E5, "kv-mxfp8-e5")),
            )),
            _ => Err("candidate encoding does not match the exact target policy".to_owned()),
        }
    }

    fn compare_metrics(
        cases: &[PreparedCase],
        reference: &EncodingRun,
        candidate: &EncodingRun,
    ) -> Result<ComparisonMetrics, String> {
        if reference.rows.len() != cases.len()
            || candidate.rows.len() != cases.len()
            || cases.is_empty()
        {
            return Err("quality run row count differs from the non-empty dataset".to_owned());
        }
        let mut reference_loss = 0.0_f64;
        let mut candidate_loss = 0.0_f64;
        let mut klds = Vec::with_capacity(cases.len() * 2);
        let mut top1_matches = 0_usize;
        let mut baseline_task = 0_usize;
        let mut candidate_task = 0_usize;
        let mut long_matches = 0_usize;
        let mut long_count = 0_usize;
        for ((case, baseline), observed) in cases.iter().zip(&reference.rows).zip(&candidate.rows) {
            reference_loss += nll(&baseline.prefill, case.expected_next);
            candidate_loss += nll(&observed.prefill, case.expected_next);
            baseline_task += usize::from(top1(&baseline.prefill) == case.expected_next as usize);
            candidate_task += usize::from(top1(&observed.prefill) == case.expected_next as usize);
            for (left, right) in [
                (&baseline.prefill, &observed.prefill),
                (&baseline.decode, &observed.decode),
            ] {
                let matched = top1(left) == top1(right);
                top1_matches += usize::from(matched);
                klds.push(kld(left, right));
                if case.tokens.len() >= 255 {
                    long_count += 1;
                    long_matches += usize::from(matched);
                }
            }
        }
        klds.sort_by(f64::total_cmp);
        let selected_count = klds.len();
        if selected_count == 0 || long_count == 0 || candidate.dispatches == 0 {
            return Err("quality metric selection or HIP dispatch count is zero".to_owned());
        }
        let reference_perplexity = (reference_loss / cases.len() as f64).exp();
        let candidate_perplexity = (candidate_loss / cases.len() as f64).exp();
        let perplexity_relative_delta =
            (candidate_perplexity - reference_perplexity) / reference_perplexity;
        let top1_agreement = top1_matches as f64 / selected_count as f64;
        let baseline_task_score = baseline_task as f64 / cases.len() as f64;
        let candidate_task_score = candidate_task as f64 / cases.len() as f64;
        let task_score_delta = (baseline_task_score - candidate_task_score).max(0.0);
        let long_context_score_delta = 1.0 - long_matches as f64 / long_count as f64;
        for (label, value) in [
            ("perplexity_relative_delta", perplexity_relative_delta),
            ("kld_p99", percentile(&klds, 0.99)),
            ("top1_agreement", top1_agreement),
            ("task_score_delta", task_score_delta),
            ("long_context_score_delta", long_context_score_delta),
        ] {
            if !value.is_finite() {
                return Err(format!("{label} is non-finite"));
            }
        }
        Ok(ComparisonMetrics {
            selected_count,
            metric_sample_counts: MetricSampleCounts {
                perplexity: cases.len(),
                kld: selected_count,
                top1: selected_count,
                task: cases.len(),
                long_context: long_count,
            },
            perplexity_relative_delta,
            kld_p99: percentile(&klds, 0.99),
            top1_agreement,
            task_score_delta,
            long_context_score_delta,
            hip_dispatches: candidate.dispatches,
            fallback_used: false,
            all_dispatches_hip: true,
        })
    }

    fn publish_candidate(report: &CandidateReport, output: &Path) -> Result<String, String> {
        if output.exists() {
            return Err("output already exists".to_owned());
        }
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| format!("create output parent: {error}"))?;
        let name = output
            .file_name()
            .ok_or("output must name a file")?
            .to_string_lossy();
        let partial = parent.join(format!(".{name}.partial.{}", std::process::id()));
        let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
        let report_digest = format!("sha256:{}", sha256(&bytes));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&partial)
                .map_err(|error| error.to_string())?;
            file.write_all(&bytes).map_err(|error| error.to_string())?;
            file.sync_all().map_err(|error| error.to_string())?;
            fs::hard_link(&partial, output).map_err(|error| error.to_string())?;
            fs::remove_file(&partial).map_err(|error| error.to_string())?;
            FileSync::sync(parent)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&partial);
            let _ = fs::remove_file(output);
        }
        result?;
        Ok(report_digest)
    }

    fn run_candidate(arguments: &[String]) -> Result<(CandidateReport, PathBuf), String> {
        if arguments.len() != 9 {
            return Err("usage: LOCK CACHE DERIVED_LOCK DATASET_JSON POLICY_JSON DEVICE_INDEX TARGET ENCODING OUTPUT_JSON".to_owned());
        }
        let lock_path = PathBuf::from(&arguments[0]);
        let cache_path = PathBuf::from(&arguments[1]);
        let derived_path = PathBuf::from(&arguments[2]);
        let dataset_path = PathBuf::from(&arguments[3]);
        let policy_path = PathBuf::from(&arguments[4]);
        let device_index = arguments[5]
            .parse::<u32>()
            .map_err(|_| "device index must be u32")?;
        let target = arguments[6].clone();
        if !exact_target(&target) {
            return Err("target must be an exact Phase 53 target".to_owned());
        }
        let (encoding, encoding_name, descriptor_id, mxfp8) =
            candidate_encoding(&target, &arguments[7])?;
        let output_path = PathBuf::from(&arguments[8]);
        let policy_bytes =
            fs::read(policy_path).map_err(|error| format!("read policy: {error}"))?;
        let policy: serde_json::Value = serde_json::from_slice(&policy_bytes)
            .map_err(|error| format!("parse policy: {error}"))?;
        if policy
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            != Some("kv-cache-default-v2")
        {
            return Err("policy is not kv-cache-default-v2".to_owned());
        }
        let policy_sha256 = format!("sha256:{}", sha256(&policy_bytes));
        let (dataset_sha256, cases) = load_dataset(&dataset_path)?;
        let lock_bytes =
            fs::read(&lock_path).map_err(|error| format!("read model lock bytes: {error}"))?;
        let lock =
            read_model_lock(&lock_path).map_err(|error| format!("read model lock: {error}"))?;
        if lock.model.repo_id != QWEN35_4B_REPO_ID
            || lock.model.resolved_revision != QWEN35_4B_REVISION
            || lock.fingerprint() != QWEN35_4B_FINGERPRINT
        {
            return Err("candidate requires the reviewed Qwen3.5-4B lock".to_owned());
        }
        let derived_bytes =
            fs::read(&derived_path).map_err(|error| format!("read derived lock: {error}"))?;
        let derived: DerivedLock = serde_json::from_slice(&derived_bytes)
            .map_err(|error| format!("parse derived lock: {error}"))?;
        if derived.schema_version != "derived-gguf-lock-v1"
            || derived.semantic_model_id != format!("qwen35:{QWEN35_4B_FINGERPRINT}")
            || derived.source_lock_fingerprints != [QWEN35_4B_FINGERPRINT]
            || derived.fingerprint.len() != 71
            || !derived.fingerprint.starts_with("sha256:")
            || !derived.converter.is_object()
            || !derived.output.is_object()
        {
            return Err("derived lock identity differs".to_owned());
        }
        let cache = Arc::new(
            lock.verify_cache(cache_path)
                .map_err(|error| format!("verify cache: {error}"))?,
        );
        let plan = build_verified_weight_load_plan(&lock, &cache)
            .map_err(|error| format!("build weight plan: {error}"))?;
        let backend = HipBackend::connect().map_err(|error| format!("connect HIP: {error}"))?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("open HIP session: {error}"))?;
        let measurement = (|| {
            let mut repeats = Vec::with_capacity(3);
            for repeat in 1..=3 {
                let baseline = execute_encoding(
                    &session,
                    &lock,
                    &cache,
                    &plan,
                    &cases,
                    KvCacheEncoding::Fp16,
                    &target,
                )?;
                if session.memory_snapshot().current_bytes() != 0 {
                    return Err(format!(
                        "FP16 resident remained before candidate repeat {repeat}"
                    ));
                }
                let block16 =
                    execute_encoding(&session, &lock, &cache, &plan, &cases, encoding, &target)?;
                if session.memory_snapshot().current_bytes() != 0 {
                    return Err(format!(
                        "block16 resident remained before MXFP8 repeat {repeat}"
                    ));
                }
                let mxfp8_run = if let Some((mxfp8_encoding, _)) = mxfp8 {
                    let run = execute_encoding(
                        &session,
                        &lock,
                        &cache,
                        &plan,
                        &cases,
                        mxfp8_encoding,
                        &target,
                    )?;
                    if session.memory_snapshot().current_bytes() != 0 {
                        return Err(format!("MXFP8 resident remained after repeat {repeat}"));
                    }
                    Some(run)
                } else {
                    None
                };
                repeats.push(CandidateRepeat {
                    repeat,
                    fp16_released_before_block16: true,
                    block16_released_after_repeat: mxfp8.is_none().then_some(true),
                    block16_released_before_mxfp8: mxfp8.is_some().then_some(true),
                    mxfp8_released_after_repeat: mxfp8.is_some().then_some(true),
                    block16: compare_metrics(&cases, &baseline, &block16)?,
                    mxfp8: mxfp8_run
                        .as_ref()
                        .map(|candidate| compare_metrics(&cases, &baseline, candidate))
                        .transpose()?,
                });
            }
            Ok::<_, String>(repeats)
        })();
        let repeats = match measurement {
            Ok(repeats) => repeats,
            Err(error) => {
                let _ = session.shutdown(SHUTDOWN_TIMEOUT);
                return Err(error);
            }
        };
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("shutdown: {error}"))?;
        if cleanup.retryable_cleanup != 0
            || cleanup.durable_quarantine != 0
            || session.memory_snapshot().current_bytes() != 0
        {
            return Err(format!("final cleanup was nonzero: {cleanup:?}"));
        }
        let executable = env::current_exe().map_err(|error| error.to_string())?;
        let binary_sha256 = format!(
            "sha256:{}",
            sha256(&fs::read(executable).map_err(|error| error.to_string())?)
        );
        Ok((
            CandidateReport {
                schema: "https://sllm.dev/schema/phase53-qwen35-kv-quality-candidate-v2.schema.json",
                schema_version: "sllm-phase53-qwen35-kv-quality-candidate-v2",
                state: "PASS",
                identity: CandidateIdentity {
                    policy_sha256,
                    dataset_sha256,
                    model_lock_fingerprint: QWEN35_4B_FINGERPRINT,
                    model_lock_sha256: format!("sha256:{}", sha256(&lock_bytes)),
                    derived_lock_fingerprint: derived.fingerprint,
                    derived_lock_sha256: format!("sha256:{}", sha256(&derived_bytes)),
                    binary_sha256,
                    descriptor_id,
                    scale_recipe: "standard-mx-floor-power-v1",
                },
                target,
                device_index,
                encoding: encoding_name,
                mxfp8_comparison: match mxfp8 {
                    Some((_, name)) => Mxfp8Comparison {
                        status: "complete",
                        encoding: Some(name),
                        reference_only: true,
                        reason: None,
                    },
                    None => Mxfp8Comparison {
                        status: "unsupported",
                        encoding: None,
                        reference_only: true,
                        reason: Some(
                            "gfx942 OCP MXFP8 is intentionally unsupported because CDNA3 FNUZ element bytes differ",
                        ),
                    },
                },
                reference_encoding: "fp16",
                sequential_residents: true,
                completely_sequential_order: if mxfp8.is_some() {
                    vec!["fp16", "block16", "mxfp8"]
                } else {
                    vec!["fp16", "block16"]
                },
                selected_count: cases.len() * 2,
                repeats,
                cleanup: CandidateCleanup {
                    retryable: 0,
                    durable: 0,
                    terminal_zero: true,
                },
            },
            output_path,
        ))
    }

    pub(super) fn entry() -> ExitCode {
        match run_candidate(&env::args().skip(1).collect::<Vec<_>>()) {
            Ok((report, output)) => match publish_candidate(&report, &output) {
                Ok(digest) => {
                    println!("{} {digest}", output.display());
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("candidate publication failed: {error}");
                    ExitCode::from(2)
                }
            },
            Err(error) => {
                eprintln!("candidate failed: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn main() -> std::process::ExitCode {
    sequential::entry()
}
