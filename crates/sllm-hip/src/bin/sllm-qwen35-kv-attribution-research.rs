//! Phase 54 Qwen3.5-4B KV attribution research harness.
//!
//! This runner compares an unmodified FP16-state request with exactly one
//! research-only K/V plane intervention.  The intervention is request-local
//! and is deliberately not a persistent block16 state or production
//! descriptor.  Residents are created and released one at a time, while the
//! same HIP execution session is reused for the complete comparison.

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
        KvCacheSelectionRequest, PHASE54_KV_ATTRIBUTION_LAYER_ENV,
        PHASE54_KV_ATTRIBUTION_SEMANTICS, Phase54KvAttributionMode, QWEN35_4B_FINGERPRINT,
        QWEN35_4B_REPO_ID, QWEN35_4B_REVISION, QwenExecutionAudit, QwenResidentModel,
        build_qwen35_graph_with_kv_cache_selection, build_verified_weight_load_plan,
        is_allowed_layer, parse_layer, read_model_lock, resolve_kv_cache_selection,
    };
    use sllm_hip::HipBackend;

    const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
    const DATASET_SHA256: &str = "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
    const VOCAB_SIZE: usize = 248_320;
    const ATTRIBUTION_ENV: &str = "SLLM_PHASE54_KV_ATTRIBUTION";

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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RunMode {
        Off,
        Intervention(Phase54KvAttributionMode),
    }

    impl RunMode {
        fn parse_intervention(value: &str) -> Result<Self, String> {
            let mode = Phase54KvAttributionMode::parse(value)
                .map_err(|error| format!("invalid attribution mode: {error}"))?;
            if !mode.is_enabled() {
                return Err(
                    "attribution mode must be key-only, value-only, or key-and-value".to_owned(),
                );
            }
            Ok(Self::Intervention(mode))
        }

        const fn selector(self) -> Phase54KvAttributionMode {
            match self {
                Self::Off => Phase54KvAttributionMode::Off,
                Self::Intervention(mode) => mode,
            }
        }

        const fn id(self) -> &'static str {
            self.selector().identity_tag()
        }
    }

    /// The selector is process-global, but every mode change happens between
    /// fully released residents.  Drop is fail-safe: even an early return
    /// leaves the process in explicit `off` mode.
    struct AttributionEnvGuard;

    impl AttributionEnvGuard {
        fn install(mode: RunMode, layer: Option<u32>) -> Result<Self, String> {
            let selector = mode.selector().identity_tag();
            // SAFETY: this research binary performs all submissions serially;
            // no worker thread observes the process environment.
            unsafe { env::set_var(ATTRIBUTION_ENV, selector) };
            match layer {
                Some(layer) => unsafe {
                    env::set_var(PHASE54_KV_ATTRIBUTION_LAYER_ENV, layer.to_string())
                },
                None => unsafe { env::remove_var(PHASE54_KV_ATTRIBUTION_LAYER_ENV) },
            }
            let observed = match Phase54KvAttributionMode::from_env() {
                Ok(observed) => observed,
                Err(error) => {
                    Self::restore_off();
                    return Err(format!("read attribution selector: {error}"));
                }
            };
            let observed_layer = match env::var_os(PHASE54_KV_ATTRIBUTION_LAYER_ENV)
                .map(|value| {
                    value
                        .to_str()
                        .ok_or_else(|| "attribution layer is not valid UTF-8".to_owned())
                        .and_then(|value| parse_layer(value).map_err(|error| error.to_string()))
                })
                .transpose()
            {
                Ok(observed_layer) => observed_layer,
                Err(error) => {
                    Self::restore_off();
                    return Err(format!("read attribution layer: {error}"));
                }
            };
            if observed != mode.selector() || observed_layer != layer {
                Self::restore_off();
                return Err(format!(
                    "attribution selector/layer mismatch: expected ({selector}, {layer:?}), got ({}, {observed_layer:?})",
                    observed.identity_tag(),
                ));
            }
            Ok(Self)
        }

        fn restore_off() {
            // SAFETY: see the setter above.  This serial runner has no worker
            // thread observing its process environment.
            unsafe { env::set_var(ATTRIBUTION_ENV, "off") };
            unsafe { env::remove_var(PHASE54_KV_ATTRIBUTION_LAYER_ENV) };
        }
    }

    impl Drop for AttributionEnvGuard {
        fn drop(&mut self) {
            Self::restore_off();
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn parse_repeats(value: &str) -> Result<u32, String> {
        match value {
            "1" => Ok(1),
            "3" => Ok(3),
            _ => Err("REPEATS must be exactly 1 or 3".to_owned()),
        }
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

    fn validate_audit(
        audit: &QwenExecutionAudit,
        target: &str,
        mode: RunMode,
        layer: u32,
    ) -> Result<(), String> {
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
        let expected_semantics = match mode {
            RunMode::Off => None,
            RunMode::Intervention(_) => Some(PHASE54_KV_ATTRIBUTION_SEMANTICS),
        };
        if audit.phase54_kv_attribution_semantics() != expected_semantics {
            return Err(format!(
                "attribution audit semantics mismatch: expected {expected_semantics:?}, got {:?}",
                audit.phase54_kv_attribution_semantics()
            ));
        }
        let expected_layer = mode.selector().is_enabled().then_some(layer);
        if audit.phase54_kv_attribution_layer() != expected_layer {
            return Err(format!(
                "attribution audit layer mismatch: expected {expected_layer:?}, got {:?}",
                audit.phase54_kv_attribution_layer()
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

    #[allow(clippy::too_many_arguments)]
    fn execute_encoding(
        session: &Arc<ExecutionSession>,
        lock: &sllm_core::ModelLock,
        cache: &Arc<sllm_core::VerifiedCache>,
        plan: &sllm_core::WeightLoadPlan,
        cases: &[PreparedCase],
        target: &str,
        mode: RunMode,
        layer: u32,
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
            Some(KvCacheEncoding::Fp16),
            target,
            QWEN35_4B_FINGERPRINT,
            true,
            true,
            true,
            256,
        ))
        .map_err(|error| format!("resolve FP16 KV selection: {error}"))?;
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
            if request.phase54_kv_attribution_mode() != mode.selector()
                || request.phase54_kv_attribution_layer()
                    != mode.selector().is_enabled().then_some(layer)
            {
                return Err(format!(
                    "request {} captured the wrong attribution mode/layer",
                    case.id
                ));
            }
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
            validate_audit(&audit, target, mode, layer)?;
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
        let released = session.memory_snapshot();
        if released.poisoned() || released.current_bytes() != 0 {
            return Err("resident release was incomplete".to_owned());
        }
        Ok(EncodingRun { rows, dispatches })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mode(
        session: &Arc<ExecutionSession>,
        lock: &sllm_core::ModelLock,
        cache: &Arc<sllm_core::VerifiedCache>,
        plan: &sllm_core::WeightLoadPlan,
        cases: &[PreparedCase],
        target: &str,
        mode: RunMode,
        layer: u32,
    ) -> Result<EncodingRun, String> {
        // The guard is installed before a resident exists and is dropped only
        // after execute_encoding has released it and all request owners.
        let guard =
            AttributionEnvGuard::install(mode, mode.selector().is_enabled().then_some(layer))?;
        let result = execute_encoding(session, lock, cache, plan, cases, target, mode, layer);
        drop(guard);
        result
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

    fn kld(reference: &[f32], observed: &[f32]) -> f64 {
        let reference_lse = logsumexp(reference);
        let observed_lse = logsumexp(observed);
        reference
            .iter()
            .zip(observed)
            .map(|(reference, observed)| {
                let log_p = f64::from(*reference) - reference_lse;
                let log_q = f64::from(*observed) - observed_lse;
                log_p.exp() * (log_p - log_q)
            })
            .sum::<f64>()
            .max(0.0)
    }

    fn percentile(values: &[f64], quantile: f64) -> f64 {
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    #[derive(Clone, Debug, Serialize)]
    struct LogitDeltaLocator {
        case_id: String,
        row: &'static str,
        measured_row_index: usize,
        logit_index: usize,
        max_abs_logit_delta: f64,
    }

    #[derive(Clone, Debug, Serialize)]
    struct Top1Divergence {
        case_id: String,
        row: &'static str,
        measured_row_index: usize,
        reference_top1: usize,
        observed_top1: usize,
    }

    #[derive(Serialize)]
    struct NllMetrics {
        reference: f64,
        observed: f64,
        delta: f64,
    }

    #[derive(Serialize)]
    struct RowMetrics {
        reference_top1: usize,
        observed_top1: usize,
        top1_match: bool,
        kld: f64,
        max_abs_logit_delta: f64,
        max_abs_logit_index: usize,
        reference_finite: bool,
        observed_finite: bool,
    }

    #[derive(Serialize)]
    struct PerCaseMetrics {
        case_id: String,
        length: usize,
        long: bool,
        prefill_nll: NllMetrics,
        prefill: RowMetrics,
        decode: RowMetrics,
    }

    #[derive(Serialize)]
    struct MetricSampleCounts {
        kld: usize,
        top1: usize,
        nll: usize,
        task: usize,
        #[serde(rename = "long-context")]
        long_context: usize,
    }

    #[derive(Serialize)]
    struct AggregateMetrics {
        selected_count: usize,
        metric_sample_counts: MetricSampleCounts,
        kld_p99: f64,
        top1_agreement: f64,
        reference_nll: f64,
        observed_nll: f64,
        nll_delta: f64,
        task_score_delta: f64,
        long_context_score_delta: f64,
        first_top1_divergence: Option<Top1Divergence>,
        maximum_logit_delta: LogitDeltaLocator,
        finite: bool,
        hip_dispatches: u64,
        fallback_used: bool,
        all_dispatches_hip: bool,
    }

    #[derive(Serialize)]
    struct ComparisonMetrics {
        reference_encoding: &'static str,
        observed_encoding: &'static str,
        aggregate: AggregateMetrics,
        cases: Vec<PerCaseMetrics>,
    }

    #[derive(Serialize)]
    struct ResearchRepeat {
        repeat: u32,
        order: [&'static str; 2],
        reference_released_before_intervention: bool,
        intervention_released_after_repeat: bool,
        comparison: ComparisonMetrics,
    }

    #[derive(Serialize)]
    struct ResearchIdentity {
        dataset_sha256: String,
        model_lock_fingerprint: &'static str,
        model_lock_sha256: String,
        derived_lock_fingerprint: String,
        derived_lock_sha256: String,
        binary_sha256: String,
    }

    #[derive(Serialize)]
    struct ResearchCleanup {
        retryable: usize,
        durable: usize,
        poisoned: bool,
        terminal_zero: bool,
    }

    #[derive(Serialize)]
    struct ResearchReport {
        #[serde(rename = "$schema")]
        schema: &'static str,
        schema_version: &'static str,
        state: &'static str,
        research_only: bool,
        identity: ResearchIdentity,
        target: &'static str,
        device_index: u32,
        layer: u32,
        semantics: &'static str,
        audit_semantics_verified: bool,
        reference_mode: &'static str,
        intervention_mode: &'static str,
        kv_state: &'static str,
        session_scope: &'static str,
        sequential_residents: bool,
        repeats: Vec<ResearchRepeat>,
        cleanup: ResearchCleanup,
    }

    fn row_metrics(reference: &[f32], observed: &[f32]) -> (RowMetrics, usize, f64) {
        let (max_abs_logit_index, max_abs_logit_delta) = reference
            .iter()
            .zip(observed)
            .enumerate()
            .map(|(index, (left, right))| (index, (f64::from(*left) - f64::from(*right)).abs()))
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .unwrap_or((0, 0.0));
        let reference_top1 = top1(reference);
        let observed_top1 = top1(observed);
        (
            RowMetrics {
                reference_top1,
                observed_top1,
                top1_match: reference_top1 == observed_top1,
                kld: kld(reference, observed),
                max_abs_logit_delta,
                max_abs_logit_index,
                reference_finite: reference.iter().all(|value| value.is_finite()),
                observed_finite: observed.iter().all(|value| value.is_finite()),
            },
            max_abs_logit_index,
            max_abs_logit_delta,
        )
    }

    fn update_maximum(current: &mut Option<LogitDeltaLocator>, candidate: LogitDeltaLocator) {
        let replace = current
            .as_ref()
            .is_none_or(|existing| candidate.max_abs_logit_delta > existing.max_abs_logit_delta);
        if replace {
            *current = Some(candidate);
        }
    }

    fn compare_metrics(
        cases: &[PreparedCase],
        reference: &EncodingRun,
        observed: &EncodingRun,
        intervention_mode: &'static str,
    ) -> Result<ComparisonMetrics, String> {
        if reference.rows.len() != cases.len()
            || observed.rows.len() != cases.len()
            || cases.is_empty()
        {
            return Err("attribution row count differs from the non-empty dataset".to_owned());
        }
        let mut reference_loss = 0.0_f64;
        let mut observed_loss = 0.0_f64;
        let mut klds = Vec::with_capacity(cases.len() * 2);
        let mut top1_matches = 0_usize;
        let mut reference_task = 0_usize;
        let mut observed_task = 0_usize;
        let mut long_matches = 0_usize;
        let mut long_count = 0_usize;
        let mut first_top1_divergence = None;
        let mut maximum_logit_delta = None;
        let mut per_case = Vec::with_capacity(cases.len());
        for (case_index, ((case, baseline), candidate)) in cases
            .iter()
            .zip(&reference.rows)
            .zip(&observed.rows)
            .enumerate()
        {
            let reference_nll = nll(&baseline.prefill, case.expected_next);
            let observed_nll = nll(&candidate.prefill, case.expected_next);
            reference_loss += reference_nll;
            observed_loss += observed_nll;
            reference_task += usize::from(top1(&baseline.prefill) == case.expected_next as usize);
            observed_task += usize::from(top1(&candidate.prefill) == case.expected_next as usize);
            let (prefill, prefill_index, prefill_delta) =
                row_metrics(&baseline.prefill, &candidate.prefill);
            let (decode, decode_index, decode_delta) =
                row_metrics(&baseline.decode, &candidate.decode);
            for (row_offset, row, metrics, index, delta) in [
                (0, "prefill", &prefill, prefill_index, prefill_delta),
                (1, "decode", &decode, decode_index, decode_delta),
            ] {
                let measured_row_index = case_index * 2 + row_offset;
                top1_matches += usize::from(metrics.top1_match);
                klds.push(metrics.kld);
                if case.tokens.len() >= 255 {
                    long_count += 1;
                    long_matches += usize::from(metrics.top1_match);
                }
                if !metrics.top1_match && first_top1_divergence.is_none() {
                    first_top1_divergence = Some(Top1Divergence {
                        case_id: case.id.clone(),
                        row,
                        measured_row_index,
                        reference_top1: metrics.reference_top1,
                        observed_top1: metrics.observed_top1,
                    });
                }
                update_maximum(
                    &mut maximum_logit_delta,
                    LogitDeltaLocator {
                        case_id: case.id.clone(),
                        row,
                        measured_row_index,
                        logit_index: index,
                        max_abs_logit_delta: delta,
                    },
                );
            }
            per_case.push(PerCaseMetrics {
                case_id: case.id.clone(),
                length: case.tokens.len(),
                long: case.tokens.len() >= 255,
                prefill_nll: NllMetrics {
                    reference: reference_nll,
                    observed: observed_nll,
                    delta: observed_nll - reference_nll,
                },
                prefill,
                decode,
            });
        }
        klds.sort_by(f64::total_cmp);
        let selected_count = klds.len();
        if selected_count == 0 || long_count == 0 || observed.dispatches == 0 {
            return Err("attribution metric selection or HIP dispatch count is zero".to_owned());
        }
        let kld_p99 = percentile(&klds, 0.99);
        let top1_agreement = top1_matches as f64 / selected_count as f64;
        let reference_nll = reference_loss / cases.len() as f64;
        let observed_nll = observed_loss / cases.len() as f64;
        let nll_delta = observed_nll - reference_nll;
        let reference_task_score = reference_task as f64 / cases.len() as f64;
        let observed_task_score = observed_task as f64 / cases.len() as f64;
        // Report quality loss in the same non-negative direction as the
        // Phase 46/54 quality harness: positive means the intervention is
        // worse than the FP16-state reference.
        let task_score_delta = (reference_task_score - observed_task_score).max(0.0);
        let long_context_score_delta = 1.0 - long_matches as f64 / long_count as f64;
        for (label, value) in [
            ("kld_p99", kld_p99),
            ("top1_agreement", top1_agreement),
            ("reference_nll", reference_nll),
            ("observed_nll", observed_nll),
            ("nll_delta", nll_delta),
            ("task_score_delta", task_score_delta),
            ("long_context_score_delta", long_context_score_delta),
        ] {
            if !value.is_finite() {
                return Err(format!("{label} is non-finite"));
            }
        }
        let maximum_logit_delta = maximum_logit_delta
            .ok_or_else(|| "maximum logit delta locator was not selected".to_owned())?;
        Ok(ComparisonMetrics {
            reference_encoding: "fp16-state/off",
            observed_encoding: match intervention_mode {
                "key-only" => "fp16-state/key-only",
                "value-only" => "fp16-state/value-only",
                "key-and-value" => "fp16-state/key-and-value",
                _ => return Err("unknown intervention mode".to_owned()),
            },
            aggregate: AggregateMetrics {
                selected_count,
                metric_sample_counts: MetricSampleCounts {
                    kld: selected_count,
                    top1: selected_count,
                    nll: cases.len(),
                    task: cases.len(),
                    long_context: long_count,
                },
                kld_p99,
                top1_agreement,
                reference_nll,
                observed_nll,
                nll_delta,
                task_score_delta,
                long_context_score_delta,
                first_top1_divergence,
                maximum_logit_delta,
                finite: per_case.iter().all(|case| {
                    case.prefill.reference_finite
                        && case.prefill.observed_finite
                        && case.decode.reference_finite
                        && case.decode.observed_finite
                        && case.prefill_nll.reference.is_finite()
                        && case.prefill_nll.observed.is_finite()
                        && case.prefill_nll.delta.is_finite()
                }),
                hip_dispatches: observed.dispatches,
                fallback_used: false,
                all_dispatches_hip: true,
            },
            cases: per_case,
        })
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

    fn publish_report(report: &ResearchReport, output: &Path) -> Result<String, String> {
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
        // Compact JSON is intentional: reports carry metrics and identities,
        // never raw logits or traces.
        let bytes = serde_json::to_vec(report).map_err(|error| error.to_string())?;
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

    fn run_research(arguments: &[String]) -> Result<(ResearchReport, PathBuf), String> {
        if arguments.len() != 10 {
            return Err("usage: LOCK CACHE DERIVED_LOCK DATASET_JSON DEVICE_INDEX TARGET LAYER MODE REPEATS OUTPUT_JSON".to_owned());
        }
        let lock_path = PathBuf::from(&arguments[0]);
        let cache_path = PathBuf::from(&arguments[1]);
        let derived_path = PathBuf::from(&arguments[2]);
        let dataset_path = PathBuf::from(&arguments[3]);
        let device_index = arguments[4]
            .parse::<u32>()
            .map_err(|_| "DEVICE_INDEX must be u32")?;
        let target = arguments[5].as_str();
        if target != "gfx1030" {
            return Err("TARGET must be exact gfx1030 for Phase 54 attribution".to_owned());
        }
        let layer = arguments[6]
            .parse::<u32>()
            .map_err(|_| "LAYER must be an unsigned integer")?;
        if !is_allowed_layer(layer) {
            return Err(format!(
                "LAYER {layer} is not in the reviewed Phase 54 layer set"
            ));
        }
        let intervention = RunMode::parse_intervention(&arguments[7])?;
        let repeat_count = parse_repeats(&arguments[8])?;
        let output_path = PathBuf::from(&arguments[9]);

        // Clear an inherited selector before any model/session work and keep
        // the off guard alive through every error path.
        let off_guard = AttributionEnvGuard::install(RunMode::Off, None)?;
        let (dataset_sha256, cases) = load_dataset(&dataset_path)?;
        let lock_bytes =
            fs::read(&lock_path).map_err(|error| format!("read model lock bytes: {error}"))?;
        let lock =
            read_model_lock(&lock_path).map_err(|error| format!("read model lock: {error}"))?;
        if lock.model.repo_id != QWEN35_4B_REPO_ID
            || lock.model.resolved_revision != QWEN35_4B_REVISION
            || lock.fingerprint() != QWEN35_4B_FINGERPRINT
        {
            return Err("research requires the reviewed Qwen3.5-4B lock".to_owned());
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
                ExecutionSessionRequest::new(device_index, target.to_owned())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("open HIP session: {error}"))?;

        let measurement = (|| {
            let mut repeats = Vec::with_capacity(repeat_count as usize);
            for repeat in 1..=repeat_count {
                let reference = execute_mode(
                    &session,
                    &lock,
                    &cache,
                    &plan,
                    &cases,
                    target,
                    RunMode::Off,
                    layer,
                )?;
                // execute_mode has dropped its resident and restored `off`
                // before this mode transition.
                let observed = execute_mode(
                    &session,
                    &lock,
                    &cache,
                    &plan,
                    &cases,
                    target,
                    intervention,
                    layer,
                )?;
                let comparison = compare_metrics(&cases, &reference, &observed, intervention.id())?;
                repeats.push(ResearchRepeat {
                    repeat,
                    order: ["fp16-state/off", "fp16-state/intervention"],
                    reference_released_before_intervention: true,
                    intervention_released_after_repeat: true,
                    comparison,
                });
            }
            Ok::<_, String>(repeats)
        })();
        let repeats = match measurement {
            Ok(repeats) => repeats,
            Err(error) => {
                let _ = session.shutdown(SHUTDOWN_TIMEOUT);
                drop(off_guard);
                return Err(error);
            }
        };
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("shutdown: {error}"))?;
        let final_snapshot = session.memory_snapshot();
        if cleanup.retryable_cleanup != 0
            || cleanup.durable_quarantine != 0
            || final_snapshot.poisoned()
            || final_snapshot.current_bytes() != 0
        {
            return Err(format!(
                "final cleanup was nonzero or poisoned: {cleanup:?}"
            ));
        }
        let executable = env::current_exe().map_err(|error| error.to_string())?;
        let binary_sha256 = format!(
            "sha256:{}",
            sha256(&fs::read(executable).map_err(|error| error.to_string())?)
        );
        let report = ResearchReport {
            schema: "https://sllm.dev/schema/phase54-qwen35-kv-attribution-research-v1.schema.json",
            schema_version: "sllm-phase54-qwen35-kv-attribution-research-v1",
            state: "PASS",
            research_only: true,
            identity: ResearchIdentity {
                dataset_sha256,
                model_lock_fingerprint: QWEN35_4B_FINGERPRINT,
                model_lock_sha256: format!("sha256:{}", sha256(&lock_bytes)),
                derived_lock_fingerprint: derived.fingerprint,
                derived_lock_sha256: format!("sha256:{}", sha256(&derived_bytes)),
                binary_sha256,
            },
            target: "gfx1030",
            device_index,
            layer,
            semantics: PHASE54_KV_ATTRIBUTION_SEMANTICS,
            audit_semantics_verified: true,
            reference_mode: "off",
            intervention_mode: intervention.id(),
            kv_state: "fp16-state",
            session_scope: "single-process-single-hip-execution-session",
            sequential_residents: true,
            repeats,
            cleanup: ResearchCleanup {
                retryable: 0,
                durable: 0,
                poisoned: false,
                terminal_zero: true,
            },
        };
        // Keep the explicit off restore until report construction and hashing
        // have completed; no resident or request remains at this point.
        drop(off_guard);
        Ok((report, output_path))
    }

    pub(super) fn entry() -> ExitCode {
        match run_research(&env::args().skip(1).collect::<Vec<_>>()) {
            Ok((report, output)) => match publish_report(&report, &output) {
                Ok(digest) => {
                    println!("{} {digest}", output.display());
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("research publication failed: {error}");
                    ExitCode::from(2)
                }
            },
            Err(error) => {
                eprintln!("research failed: {error}");
                ExitCode::FAILURE
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn intervention_mode_is_closed_and_enabled_only() {
            assert_eq!(
                RunMode::parse_intervention("key-only").unwrap().id(),
                "key-only"
            );
            assert_eq!(
                RunMode::parse_intervention("value-only").unwrap().id(),
                "value-only"
            );
            assert_eq!(
                RunMode::parse_intervention("key-and-value").unwrap().id(),
                "key-and-value"
            );
            for rejected in ["off", "bogus", "key", "value"] {
                assert!(
                    RunMode::parse_intervention(rejected).is_err(),
                    "accepted {rejected}"
                );
            }
        }

        #[test]
        fn attribution_layer_is_the_reviewed_closed_set() {
            for layer in [3_u32, 7, 11, 15, 19, 23, 27, 31] {
                assert!(is_allowed_layer(layer));
                assert_eq!(parse_layer(&layer.to_string()).unwrap(), layer);
            }
            for rejected in [0_u32, 1, 2, 4, 8, 12, 32, 99] {
                assert!(!is_allowed_layer(rejected));
                assert!(parse_layer(&rejected.to_string()).is_err());
            }
        }

        #[test]
        fn repeats_are_exactly_one_or_three() {
            assert_eq!(parse_repeats("1").unwrap(), 1);
            assert_eq!(parse_repeats("3").unwrap(), 3);
            for rejected in ["0", "2", "03", "4", "one"] {
                assert!(parse_repeats(rejected).is_err(), "accepted {rejected}");
            }
        }

        #[test]
        fn row_metrics_uses_first_maximum_index() {
            let reference = [0.0_f32, 3.0, -2.0];
            let observed = [0.0_f32, 1.0, 0.0];
            let (metrics, index, delta) = row_metrics(&reference, &observed);
            assert_eq!(index, 1);
            assert_eq!(delta, 2.0);
            assert_eq!(metrics.max_abs_logit_index, 1);
            assert!(metrics.top1_match);
            assert!(metrics.kld.is_finite());
        }

        #[test]
        fn percentile_uses_ceil_index() {
            assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.99), 4.0);
            assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 3.0);
        }
    }
}

fn main() -> std::process::ExitCode {
    sequential::entry()
}
