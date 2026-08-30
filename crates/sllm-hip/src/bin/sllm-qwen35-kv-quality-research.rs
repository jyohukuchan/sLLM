//! Phase 54 Qwen3.5 KV FP8 block16 quality research harness.
//!
//! Candidate recipes are a closed K/V pair consumed by the research-only HIP
//! runtime API. The production comparator and all cleanup paths restore the
//! production Floor/Floor pair before continuing.

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
    #[cfg(feature = "phase54-research")]
    use sllm_core::{
        PHASE54_KQ_TRANSFORM_DIGEST, PHASE54_KQ_TRANSFORM_ENV, PHASE54_KQ_TRANSFORM_SEMANTICS,
        PHASE54_VO_TRANSFORM_BACKEND, PHASE54_VO_TRANSFORM_DIGEST, PHASE54_VO_TRANSFORM_ENV,
        PHASE54_VO_TRANSFORM_SEMANTICS, Phase54KqTransformConfig, Phase54KqTransformMode,
        Phase54VoTransformConfig, Phase54VoTransformMode,
    };
    use sllm_hip::HipBackend;

    const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
    const DATASET_SHA256: &str = "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
    const VOCAB_SIZE: usize = 248_320;
    #[cfg(not(feature = "phase54-research"))]
    const PHASE54_KQ_TRANSFORM_ENV: &str = "SLLM_PHASE54_KQ_TRANSFORM";
    #[cfg(not(feature = "phase54-research"))]
    const PHASE54_VO_TRANSFORM_ENV: &str = "SLLM_PHASE54_VO_TRANSFORM";
    const PHASE54_VO_LAYERS19_31_SELECTOR: &str = "transpose16x16-v-layers19-31-output-inverse";
    #[cfg(feature = "phase54-research")]
    const PHASE54_VO_LAYERS19_31_SEMANTICS: &str =
        "vo-fixed-permutation/transpose16x16-layers19-31-v1";
    #[cfg(feature = "phase54-research")]
    const PHASE54_VO_LAYERS19_31_DIGEST: &str =
        "sha256:5439e11e91b4c2acfd060fb1ec4d8f5fee2f1244e28c3e6588f2202fbe8e9a74";

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

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "kebab-case")]
    enum Recipe {
        Floor,
        Ceil,
        NearestEven,
        Parent32Duplicate,
    }

    impl Recipe {
        const fn runtime_value(self) -> u32 {
            match self {
                Self::Floor => 0,
                Self::Ceil => 1,
                Self::NearestEven => 2,
                Self::Parent32Duplicate => 3,
            }
        }

        const fn id(self) -> &'static str {
            match self {
                Self::Floor => "floor",
                Self::Ceil => "ceil",
                Self::NearestEven => "nearest-even",
                Self::Parent32Duplicate => "parent32-duplicate",
            }
        }
    }

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(deny_unknown_fields)]
    struct CandidateSpec {
        schema_version: String,
        candidate_id: String,
        scale_selector: String,
        rounding: String,
        k_recipe: Recipe,
        v_recipe: Recipe,
        transform: String,
        calibration_digest: Option<String>,
        descriptor_compatibility: String,
    }

    impl CandidateSpec {
        fn is_parent32_duplicate_candidate(&self) -> bool {
            self.k_recipe == Recipe::Parent32Duplicate
                && self.v_recipe == Recipe::Parent32Duplicate
                && self.transform == "none"
        }

        fn is_production_control(&self) -> bool {
            self.k_recipe == Recipe::Floor
                && self.v_recipe == Recipe::Floor
                && self.transform == "none"
        }

        fn is_transform_candidate(&self) -> bool {
            self.k_recipe == Recipe::Floor
                && self.v_recipe == Recipe::Floor
                && self.transform == "transpose16x16-all-full"
        }

        fn is_vo_transform_candidate(&self) -> bool {
            self.k_recipe == Recipe::Floor
                && self.v_recipe == Recipe::Floor
                && self.transform == "transpose16x16-v-layer19-output-inverse"
        }

        fn is_vo_layers19_31_transform_candidate(&self) -> bool {
            self.k_recipe == Recipe::Floor
                && self.v_recipe == Recipe::Floor
                && self.transform == PHASE54_VO_LAYERS19_31_SELECTOR
        }

        fn expected_candidate_id(&self) -> String {
            if self.is_production_control() {
                "production-control-v2".to_owned()
            } else if self.is_transform_candidate() {
                "phase54-kq-transpose16x16-all-full-v1".to_owned()
            } else if self.is_vo_transform_candidate() {
                "phase54-vo-transpose16x16-layer19-v1".to_owned()
            } else if self.is_vo_layers19_31_transform_candidate() {
                "phase54-vo-transpose16x16-layers19-31-v1".to_owned()
            } else {
                format!(
                    "phase54-k-{}-v-{}-v1",
                    self.k_recipe.id(),
                    self.v_recipe.id()
                )
            }
        }

        fn expected_descriptor_compatibility(&self) -> &'static str {
            if self.is_production_control() {
                "exact-production-v2"
            } else {
                "research-build-semantic-override-not-v2-compatible"
            }
        }

        fn validate(&self) -> Result<(), String> {
            if self.schema_version != "sllm-phase54-kv-candidate-spec-v1"
                || self.candidate_id != self.expected_candidate_id()
                || self.scale_selector != "independent-k-v-closed-enum-v1"
                || self.rounding != "nearest-even"
                || self.calibration_digest.is_some()
                || self.descriptor_compatibility != self.expected_descriptor_compatibility()
                || ((self.k_recipe == Recipe::Parent32Duplicate
                    || self.v_recipe == Recipe::Parent32Duplicate)
                    && !self.is_parent32_duplicate_candidate())
                || (!self.is_transform_candidate()
                    && !self.is_vo_transform_candidate()
                    && !self.is_vo_layers19_31_transform_candidate()
                    && self.transform != "none")
            {
                return Err(
                    "candidate spec does not match the runtime-consumed closed recipe identity"
                        .to_owned(),
                );
            }
            Ok(())
        }

        fn candidate_descriptor_id(&self, production_descriptor: &str) -> String {
            if self.is_production_control() {
                production_descriptor.to_owned()
            } else {
                format!("{}-{}", production_descriptor, self.candidate_id)
            }
        }
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

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn parse_candidate_spec(bytes: &[u8]) -> Result<(CandidateSpec, String), String> {
        let spec: CandidateSpec = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse candidate spec: {error}"))?;
        spec.validate()?;
        let canonical = serde_json::to_vec(&spec)
            .map_err(|error| format!("canonicalize candidate spec: {error}"))?;
        Ok((spec, format!("sha256:{}", sha256(&canonical))))
    }

    fn parse_repeats(value: &str) -> Result<u32, String> {
        match value {
            "1" => Ok(1),
            "3" => Ok(3),
            _ => Err("REPEATS must be exactly 1 or 3".to_owned()),
        }
    }

    #[cfg(not(test))]
    unsafe extern "C" {
        fn sllm_phase54_kv_research_set_recipe_pair_v1(key: u32, value: u32) -> i32;
        fn sllm_phase54_kv_research_get_recipe_pair_v1(key: *mut u32, value: *mut u32) -> i32;
    }

    #[cfg(not(test))]
    fn set_and_verify_recipe_pair(key: Recipe, value: Recipe) -> Result<(), String> {
        // SAFETY: both functions are process-global research controls. The
        // arguments are closed enum values, and the getter receives valid
        // writable u32 pointers for the duration of the call.
        let set_status = unsafe {
            sllm_phase54_kv_research_set_recipe_pair_v1(key.runtime_value(), value.runtime_value())
        };
        if set_status != 0 {
            return Err(format!(
                "set Phase 54 recipe pair failed with status {set_status}"
            ));
        }
        let mut observed_key = u32::MAX;
        let mut observed_value = u32::MAX;
        // SAFETY: the output pointers refer to initialized stack values and do
        // not escape the getter call.
        let get_status = unsafe {
            sllm_phase54_kv_research_get_recipe_pair_v1(&mut observed_key, &mut observed_value)
        };
        if get_status != 0 {
            return Err(format!(
                "get Phase 54 recipe pair failed with status {get_status}"
            ));
        }
        if (observed_key, observed_value) != (key.runtime_value(), value.runtime_value()) {
            return Err(format!(
                "Phase 54 recipe getter mismatch: expected ({}, {}), got ({observed_key}, {observed_value})",
                key.runtime_value(),
                value.runtime_value()
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn set_and_verify_recipe_pair(_key: Recipe, _value: Recipe) -> Result<(), String> {
        Ok(())
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TransformRunMode {
        Off,
        Candidate,
    }

    impl TransformRunMode {
        const fn selector(self) -> &'static str {
            match self {
                Self::Off => "off",
                Self::Candidate => "transpose16x16-all-full",
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum VoTransformRunMode {
        Off,
        Candidate,
        Layers19And31Candidate,
    }

    impl VoTransformRunMode {
        const fn selector(self) -> &'static str {
            match self {
                Self::Off => "off",
                Self::Candidate => "transpose16x16-v-layer19-output-inverse",
                Self::Layers19And31Candidate => PHASE54_VO_LAYERS19_31_SELECTOR,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AuditTransformMode {
        Off,
        KqCandidate,
        VoCandidate,
        VoLayers19And31Candidate,
    }

    /// The K/Q transform selector is process-global, so changes are scoped to
    /// one fully released resident.  Every drop path restores the explicit
    /// `off` selector, including failed candidate setup or execution.
    struct TransformEnvGuard;

    impl TransformEnvGuard {
        fn install(mode: TransformRunMode, target: &str) -> Result<Self, String> {
            let selector = mode.selector();
            #[cfg(not(feature = "phase54-research"))]
            let _ = target;
            // SAFETY: this research binary performs submissions serially; no
            // worker thread observes the process environment.
            unsafe { env::set_var(PHASE54_KQ_TRANSFORM_ENV, selector) };
            #[cfg(feature = "phase54-research")]
            {
                let observed = Phase54KqTransformConfig::from_env(Some(target))
                    .map_err(|error| format!("read K/Q transform selector: {error}"));
                let observed = match observed {
                    Ok(observed) => observed,
                    Err(error) => {
                        Self::restore_off();
                        return Err(error);
                    }
                };
                let expected = match mode {
                    TransformRunMode::Off => Phase54KqTransformMode::Off,
                    TransformRunMode::Candidate => Phase54KqTransformMode::Transpose16x16AllFull,
                };
                if observed.mode() != expected {
                    Self::restore_off();
                    return Err(format!(
                        "K/Q transform selector mismatch: expected {}, got {}",
                        selector,
                        observed.mode().identity_tag()
                    ));
                }
            }
            #[cfg(not(feature = "phase54-research"))]
            if mode != TransformRunMode::Off {
                Self::restore_off();
                return Err(
                    "transpose16x16-all-full requires the phase54-research feature".to_owned(),
                );
            }
            Ok(Self)
        }

        fn restore_off() {
            // SAFETY: see the setter above.  This serial runner has no worker
            // thread observing the process environment.
            unsafe { env::set_var(PHASE54_KQ_TRANSFORM_ENV, "off") };
        }
    }

    impl Drop for TransformEnvGuard {
        fn drop(&mut self) {
            Self::restore_off();
        }
    }

    /// The V/O transform selector is process-global and is always scoped
    /// independently from K/Q.  Keeping a separate guard prevents a K/Q
    /// candidate from inheriting the V/O selector (and vice versa).
    struct VoTransformEnvGuard;

    impl VoTransformEnvGuard {
        fn install(mode: VoTransformRunMode, target: &str) -> Result<Self, String> {
            let selector = mode.selector();
            #[cfg(not(feature = "phase54-research"))]
            let _ = target;
            // SAFETY: this research binary performs submissions serially; no
            // worker thread observes the process environment.
            unsafe { env::set_var(PHASE54_VO_TRANSFORM_ENV, selector) };
            #[cfg(feature = "phase54-research")]
            {
                let observed = Phase54VoTransformConfig::from_env(Some(target))
                    .map_err(|error| format!("read V/O transform selector: {error}"));
                let observed = match observed {
                    Ok(observed) => observed,
                    Err(error) => {
                        Self::restore_off();
                        return Err(error);
                    }
                };
                let expected = match mode {
                    VoTransformRunMode::Off => Phase54VoTransformMode::Off,
                    VoTransformRunMode::Candidate => {
                        Phase54VoTransformMode::Transpose16x16VLayer19OutputInverse
                    }
                    VoTransformRunMode::Layers19And31Candidate => {
                        Phase54VoTransformMode::Transpose16x16VLayers19And31OutputInverse
                    }
                };
                if observed.mode() != expected {
                    Self::restore_off();
                    return Err(format!(
                        "V/O transform selector mismatch: expected {selector}, got {}",
                        observed.mode().identity_tag()
                    ));
                }
            }
            #[cfg(not(feature = "phase54-research"))]
            if mode != VoTransformRunMode::Off {
                Self::restore_off();
                return Err(format!("{selector} requires the phase54-research feature"));
            }
            Ok(Self)
        }

        fn restore_off() {
            // SAFETY: see the setter above.  This serial runner has no worker
            // thread observing the process environment.
            unsafe { env::set_var(PHASE54_VO_TRANSFORM_ENV, "off") };
        }
    }

    impl Drop for VoTransformEnvGuard {
        fn drop(&mut self) {
            Self::restore_off();
        }
    }

    struct RecipeResetGuard;

    impl RecipeResetGuard {
        fn install() -> Result<Self, String> {
            set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor)?;
            Ok(Self)
        }
    }

    impl Drop for RecipeResetGuard {
        fn drop(&mut self) {
            let _ = set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor);
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
        transform: AuditTransformMode,
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
        #[cfg(feature = "phase54-research")]
        {
            let (expected_kq_semantics, expected_kq_digest) = match transform {
                AuditTransformMode::KqCandidate => (
                    Some(PHASE54_KQ_TRANSFORM_SEMANTICS),
                    Some(PHASE54_KQ_TRANSFORM_DIGEST),
                ),
                AuditTransformMode::Off
                | AuditTransformMode::VoCandidate
                | AuditTransformMode::VoLayers19And31Candidate => (None, None),
            };
            let (expected_vo_semantics, expected_vo_digest, expected_vo_backend) = match transform {
                AuditTransformMode::VoCandidate => (
                    Some(PHASE54_VO_TRANSFORM_SEMANTICS),
                    Some(PHASE54_VO_TRANSFORM_DIGEST),
                    Some(PHASE54_VO_TRANSFORM_BACKEND),
                ),
                AuditTransformMode::VoLayers19And31Candidate => (
                    Some(PHASE54_VO_LAYERS19_31_SEMANTICS),
                    Some(PHASE54_VO_LAYERS19_31_DIGEST),
                    Some(PHASE54_VO_TRANSFORM_BACKEND),
                ),
                AuditTransformMode::Off | AuditTransformMode::KqCandidate => (None, None, None),
            };
            if audit.phase54_kq_transform_semantics() != expected_kq_semantics
                || audit.phase54_kq_transform_digest() != expected_kq_digest
                || audit.phase54_vo_transform_semantics() != expected_vo_semantics
                || audit.phase54_vo_transform_digest() != expected_vo_digest
                || audit.phase54_vo_transform_backend() != expected_vo_backend
            {
                return Err(format!(
                    "Qwen audit transform mismatch: expected K/Q ({expected_kq_semantics:?}, {expected_kq_digest:?}), V/O ({expected_vo_semantics:?}, {expected_vo_digest:?}, {expected_vo_backend:?}), got K/Q ({:?}, {:?}), V/O ({:?}, {:?}, {:?})",
                    audit.phase54_kq_transform_semantics(),
                    audit.phase54_kq_transform_digest(),
                    audit.phase54_vo_transform_semantics(),
                    audit.phase54_vo_transform_digest(),
                    audit.phase54_vo_transform_backend(),
                ));
            }
        }
        #[cfg(not(feature = "phase54-research"))]
        if transform != AuditTransformMode::Off {
            return Err("transform candidate requires the phase54-research feature".to_owned());
        }
        Ok(())
    }

    fn validate_logits(values: &[f32], label: &str) -> Result<(), String> {
        if values.len() != VOCAB_SIZE || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{label} logits are non-finite or truncated"));
        }
        Ok(())
    }

    fn encoding_runs_are_identical(left: &EncodingRun, right: &EncodingRun) -> bool {
        left.rows.len() == right.rows.len()
            && left.rows.iter().zip(&right.rows).all(|(left, right)| {
                left.prefill.len() == right.prefill.len()
                    && left.decode.len() == right.decode.len()
                    && left
                        .prefill
                        .iter()
                        .zip(&right.prefill)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
                    && left
                        .decode
                        .iter()
                        .zip(&right.decode)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_encoding(
        session: &Arc<ExecutionSession>,
        lock: &sllm_core::ModelLock,
        cache: &Arc<sllm_core::VerifiedCache>,
        plan: &sllm_core::WeightLoadPlan,
        cases: &[PreparedCase],
        encoding: KvCacheEncoding,
        target: &str,
        transform: AuditTransformMode,
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
            validate_audit(&audit, target, transform)?;
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
        perplexity: usize,
        kld: usize,
        top1: usize,
        task: usize,
        #[serde(rename = "long-context")]
        long_context: usize,
    }

    #[derive(Serialize)]
    struct AggregateMetrics {
        selected_count: usize,
        metric_sample_counts: MetricSampleCounts,
        perplexity_relative_delta: f64,
        kld_p99: f64,
        top1_agreement: f64,
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
        encoding: &'static str,
        descriptor_id: String,
        aggregate: AggregateMetrics,
        cases: Vec<PerCaseMetrics>,
    }

    #[derive(Serialize)]
    struct ResearchRepeat {
        repeat: u32,
        completely_sequential_order: [&'static str; 4],
        fp16_released_before_production_control: bool,
        production_control_released_before_candidate: bool,
        candidate_released_before_mxfp8: bool,
        mxfp8_released_after_repeat: bool,
        production_control: ComparisonMetrics,
        candidate: ComparisonMetrics,
        mxfp8: ComparisonMetrics,
    }

    #[derive(Serialize)]
    struct ResearchIdentity {
        policy_sha256: String,
        dataset_sha256: String,
        model_lock_fingerprint: &'static str,
        model_lock_sha256: String,
        derived_lock_fingerprint: String,
        derived_lock_sha256: String,
        binary_sha256: String,
        candidate_spec_sha256: String,
        production_descriptor_id: &'static str,
        candidate_descriptor_id: String,
        descriptor_compatibility: String,
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
        identity: ResearchIdentity,
        target: String,
        device_index: u32,
        encoding: &'static str,
        candidate_spec: CandidateSpec,
        repeat_count: u32,
        reference_encoding: &'static str,
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
        encoding: &'static str,
        descriptor_id: &str,
    ) -> Result<ComparisonMetrics, String> {
        if reference.rows.len() != cases.len()
            || observed.rows.len() != cases.len()
            || cases.is_empty()
        {
            return Err("quality run row count differs from the non-empty dataset".to_owned());
        }
        let mut reference_loss = 0.0_f64;
        let mut observed_loss = 0.0_f64;
        let mut klds = Vec::with_capacity(cases.len() * 2);
        let mut top1_matches = 0_usize;
        let mut baseline_task = 0_usize;
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
            baseline_task += usize::from(top1(&baseline.prefill) == case.expected_next as usize);
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
            return Err("quality metric selection or HIP dispatch count is zero".to_owned());
        }
        let reference_perplexity = (reference_loss / cases.len() as f64).exp();
        let observed_perplexity = (observed_loss / cases.len() as f64).exp();
        let perplexity_relative_delta =
            (observed_perplexity - reference_perplexity) / reference_perplexity;
        let top1_agreement = top1_matches as f64 / selected_count as f64;
        let baseline_task_score = baseline_task as f64 / cases.len() as f64;
        let observed_task_score = observed_task as f64 / cases.len() as f64;
        let task_score_delta = (baseline_task_score - observed_task_score).max(0.0);
        let long_context_score_delta = 1.0 - long_matches as f64 / long_count as f64;
        let kld_p99 = percentile(&klds, 0.99);
        for (label, value) in [
            ("perplexity_relative_delta", perplexity_relative_delta),
            ("kld_p99", kld_p99),
            ("top1_agreement", top1_agreement),
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
            encoding,
            descriptor_id: descriptor_id.to_owned(),
            aggregate: AggregateMetrics {
                selected_count,
                metric_sample_counts: MetricSampleCounts {
                    perplexity: cases.len(),
                    kld: selected_count,
                    top1: selected_count,
                    task: cases.len(),
                    long_context: long_count,
                },
                perplexity_relative_delta,
                kld_p99,
                top1_agreement,
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

    struct TargetEncodings {
        block16: KvCacheEncoding,
        block16_name: &'static str,
        block16_descriptor: &'static str,
        mxfp8: KvCacheEncoding,
        mxfp8_name: &'static str,
        mxfp8_descriptor: &'static str,
    }

    fn target_encodings(target: &str, value: &str) -> Result<TargetEncodings, String> {
        match (target, value) {
            ("gfx1201", "kv-fp8-e4-block16") => Ok(TargetEncodings {
                block16: KvCacheEncoding::Fp8E4M3Block16,
                block16_name: "kv-fp8-e4-block16",
                block16_descriptor: "kv-fp8-e4-block16-v2",
                mxfp8: KvCacheEncoding::Mxfp8E4,
                mxfp8_name: "kv-mxfp8-e4",
                mxfp8_descriptor: "kv-mxfp8-e4-v1",
            }),
            ("gfx1030", "kv-fp8-e5-block16") => Ok(TargetEncodings {
                block16: KvCacheEncoding::Fp8E5M2Block16,
                block16_name: "kv-fp8-e5-block16",
                block16_descriptor: "kv-fp8-e5-block16-v2",
                mxfp8: KvCacheEncoding::Mxfp8E5,
                mxfp8_name: "kv-mxfp8-e5",
                mxfp8_descriptor: "kv-mxfp8-e5-v1",
            }),
            _ => {
                Err("ENCODING must match exact gfx1030 E5 or gfx1201 E4 Phase 54 target".to_owned())
            }
        }
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

    fn run_research(arguments: &[String]) -> Result<(ResearchReport, PathBuf), String> {
        if arguments.len() != 11 {
            return Err("usage: LOCK CACHE DERIVED_LOCK DATASET_JSON POLICY_JSON DEVICE_INDEX TARGET ENCODING CANDIDATE_SPEC_JSON REPEATS OUTPUT_JSON".to_owned());
        }
        let target = arguments[6].clone();
        // Clear inherited transform selectors before any argument/model/session
        // work and keep both off guards alive through every error path.
        let _transform_reset_guard = TransformEnvGuard::install(TransformRunMode::Off, &target)?;
        let _vo_transform_reset_guard =
            VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
        let lock_path = PathBuf::from(&arguments[0]);
        let cache_path = PathBuf::from(&arguments[1]);
        let derived_path = PathBuf::from(&arguments[2]);
        let dataset_path = PathBuf::from(&arguments[3]);
        let policy_path = PathBuf::from(&arguments[4]);
        let device_index = arguments[5]
            .parse::<u32>()
            .map_err(|_| "DEVICE_INDEX must be u32")?;
        let encodings = target_encodings(&target, &arguments[7])?;
        let candidate_spec_bytes =
            fs::read(&arguments[8]).map_err(|error| format!("read candidate spec: {error}"))?;
        let (candidate_spec, candidate_spec_sha256) = parse_candidate_spec(&candidate_spec_bytes)?;
        #[cfg(not(feature = "phase54-research"))]
        if candidate_spec.is_transform_candidate() {
            return Err("transpose16x16-all-full requires the phase54-research feature".to_owned());
        }
        #[cfg(not(feature = "phase54-research"))]
        if candidate_spec.is_vo_transform_candidate() {
            return Err(
                "transpose16x16-v-layer19-output-inverse requires the phase54-research feature"
                    .to_owned(),
            );
        }
        #[cfg(not(feature = "phase54-research"))]
        if candidate_spec.is_vo_layers19_31_transform_candidate() {
            return Err(format!(
                "{PHASE54_VO_LAYERS19_31_SELECTOR} requires the phase54-research feature"
            ));
        }
        let repeat_count = parse_repeats(&arguments[9])?;
        let output_path = PathBuf::from(&arguments[10]);

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
        let _recipe_reset_guard = RecipeResetGuard::install()?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("open HIP session: {error}"))?;
        let candidate_descriptor_id =
            candidate_spec.candidate_descriptor_id(encodings.block16_descriptor);
        let candidate_transform = if candidate_spec.is_transform_candidate() {
            AuditTransformMode::KqCandidate
        } else if candidate_spec.is_vo_transform_candidate() {
            AuditTransformMode::VoCandidate
        } else if candidate_spec.is_vo_layers19_31_transform_candidate() {
            AuditTransformMode::VoLayers19And31Candidate
        } else {
            AuditTransformMode::Off
        };

        let measurement = (|| {
            let mut repeats = Vec::with_capacity(repeat_count as usize);
            for repeat in 1..=repeat_count {
                set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor)?;
                let fp16 = {
                    let _transform_guard =
                        TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                    let _vo_transform_guard =
                        VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
                    execute_encoding(
                        &session,
                        &lock,
                        &cache,
                        &plan,
                        &cases,
                        KvCacheEncoding::Fp16,
                        &target,
                        AuditTransformMode::Off,
                    )?
                };
                let production_control = {
                    let _transform_guard =
                        TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                    let _vo_transform_guard =
                        VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
                    execute_encoding(
                        &session,
                        &lock,
                        &cache,
                        &plan,
                        &cases,
                        encodings.block16,
                        &target,
                        AuditTransformMode::Off,
                    )?
                };
                set_and_verify_recipe_pair(candidate_spec.k_recipe, candidate_spec.v_recipe)?;
                let candidate = {
                    match candidate_transform {
                        AuditTransformMode::KqCandidate => {
                            let _vo_transform_guard =
                                VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
                            let _transform_guard =
                                TransformEnvGuard::install(TransformRunMode::Candidate, &target)?;
                            execute_encoding(
                                &session,
                                &lock,
                                &cache,
                                &plan,
                                &cases,
                                encodings.block16,
                                &target,
                                candidate_transform,
                            )?
                        }
                        AuditTransformMode::VoCandidate => {
                            let _transform_guard =
                                TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                            let _vo_transform_guard = VoTransformEnvGuard::install(
                                VoTransformRunMode::Candidate,
                                &target,
                            )?;
                            execute_encoding(
                                &session,
                                &lock,
                                &cache,
                                &plan,
                                &cases,
                                encodings.block16,
                                &target,
                                candidate_transform,
                            )?
                        }
                        AuditTransformMode::VoLayers19And31Candidate => {
                            let _transform_guard =
                                TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                            let _vo_transform_guard = VoTransformEnvGuard::install(
                                VoTransformRunMode::Layers19And31Candidate,
                                &target,
                            )?;
                            execute_encoding(
                                &session,
                                &lock,
                                &cache,
                                &plan,
                                &cases,
                                encodings.block16,
                                &target,
                                candidate_transform,
                            )?
                        }
                        AuditTransformMode::Off => {
                            let _transform_guard =
                                TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                            let _vo_transform_guard =
                                VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
                            execute_encoding(
                                &session,
                                &lock,
                                &cache,
                                &plan,
                                &cases,
                                encodings.block16,
                                &target,
                                candidate_transform,
                            )?
                        }
                    }
                };
                if candidate_spec.is_production_control()
                    && !encoding_runs_are_identical(&production_control, &candidate)
                {
                    return Err(
                        "production-control candidate did not reproduce production block16 logits"
                            .to_owned(),
                    );
                }
                set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor)?;
                let mxfp8 = {
                    let _transform_guard =
                        TransformEnvGuard::install(TransformRunMode::Off, &target)?;
                    let _vo_transform_guard =
                        VoTransformEnvGuard::install(VoTransformRunMode::Off, &target)?;
                    execute_encoding(
                        &session,
                        &lock,
                        &cache,
                        &plan,
                        &cases,
                        encodings.mxfp8,
                        &target,
                        AuditTransformMode::Off,
                    )?
                };
                if candidate_spec.is_parent32_duplicate_candidate()
                    && !encoding_runs_are_identical(&candidate, &mxfp8)
                {
                    return Err(
                        "parent32-duplicate candidate did not reproduce MXFP8 logits exactly"
                            .to_owned(),
                    );
                }
                repeats.push(ResearchRepeat {
                    repeat,
                    completely_sequential_order: [
                        "fp16",
                        "production-control-block16",
                        "candidate-block16",
                        "mxfp8",
                    ],
                    fp16_released_before_production_control: true,
                    production_control_released_before_candidate: true,
                    candidate_released_before_mxfp8: true,
                    mxfp8_released_after_repeat: true,
                    production_control: compare_metrics(
                        &cases,
                        &fp16,
                        &production_control,
                        encodings.block16_name,
                        encodings.block16_descriptor,
                    )?,
                    candidate: compare_metrics(
                        &cases,
                        &fp16,
                        &candidate,
                        encodings.block16_name,
                        &candidate_descriptor_id,
                    )?,
                    mxfp8: compare_metrics(
                        &cases,
                        &fp16,
                        &mxfp8,
                        encodings.mxfp8_name,
                        encodings.mxfp8_descriptor,
                    )?,
                });
            }
            Ok::<_, String>(repeats)
        })();
        let repeats = match measurement {
            Ok(repeats) => repeats,
            Err(error) => {
                let reset = set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor);
                let _ = session.shutdown(SHUTDOWN_TIMEOUT);
                return match reset {
                    Ok(()) => Err(error),
                    Err(reset_error) => Err(format!(
                        "{error}; additionally failed to restore production recipe: {reset_error}"
                    )),
                };
            }
        };
        let recipe_reset = set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor);
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("shutdown: {error}"))?;
        recipe_reset?;
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
        Ok((
            ResearchReport {
                schema: "https://sllm.dev/schema/phase54-qwen35-kv-quality-research-v1.schema.json",
                schema_version: "sllm-phase54-qwen35-kv-quality-research-v1",
                state: "PASS",
                identity: ResearchIdentity {
                    policy_sha256,
                    dataset_sha256,
                    model_lock_fingerprint: QWEN35_4B_FINGERPRINT,
                    model_lock_sha256: format!("sha256:{}", sha256(&lock_bytes)),
                    derived_lock_fingerprint: derived.fingerprint,
                    derived_lock_sha256: format!("sha256:{}", sha256(&derived_bytes)),
                    binary_sha256,
                    candidate_spec_sha256,
                    production_descriptor_id: encodings.block16_descriptor,
                    candidate_descriptor_id,
                    descriptor_compatibility: candidate_spec.descriptor_compatibility.clone(),
                },
                target,
                device_index,
                encoding: encodings.block16_name,
                candidate_spec,
                repeat_count,
                reference_encoding: "fp16",
                session_scope: "single-process-single-hip-execution-session",
                sequential_residents: true,
                repeats,
                cleanup: ResearchCleanup {
                    retryable: 0,
                    durable: 0,
                    poisoned: false,
                    terminal_zero: true,
                },
            },
            output_path,
        ))
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

        const SPEC: &str = r#"{
            "schema_version":"sllm-phase54-kv-candidate-spec-v1",
            "candidate_id":"production-control-v2",
            "scale_selector":"independent-k-v-closed-enum-v1",
            "rounding":"nearest-even",
            "k_recipe":"floor",
            "v_recipe":"floor",
            "transform":"none",
            "calibration_digest":null,
            "descriptor_compatibility":"exact-production-v2"
        }"#;

        #[test]
        fn candidate_spec_is_canonical_across_key_order_and_whitespace() {
            let (_, digest) = parse_candidate_spec(SPEC.as_bytes()).unwrap();
            let reordered = r#"{"descriptor_compatibility":"exact-production-v2","calibration_digest":null,"transform":"none","v_recipe":"floor","k_recipe":"floor","rounding":"nearest-even","scale_selector":"independent-k-v-closed-enum-v1","candidate_id":"production-control-v2","schema_version":"sllm-phase54-kv-candidate-spec-v1"}"#;
            assert_eq!(
                parse_candidate_spec(reordered.as_bytes()).unwrap().1,
                digest
            );
        }

        #[test]
        fn non_control_identity_is_derived_from_runtime_pair() {
            let candidate = SPEC
                .replace("production-control-v2", "phase54-k-ceil-v-nearest-even-v1")
                .replace("\"k_recipe\":\"floor\"", "\"k_recipe\":\"ceil\"")
                .replace("\"v_recipe\":\"floor\"", "\"v_recipe\":\"nearest-even\"")
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            let (spec, _) = parse_candidate_spec(candidate.as_bytes()).unwrap();
            assert_eq!(spec.k_recipe.runtime_value(), 1);
            assert_eq!(spec.v_recipe.runtime_value(), 2);
            assert_eq!(
                spec.candidate_descriptor_id("kv-fp8-e5-block16-v2"),
                "kv-fp8-e5-block16-v2-phase54-k-ceil-v-nearest-even-v1"
            );
        }

        #[test]
        fn transform_candidate_identity_is_floor_floor_only() {
            let candidate = SPEC
                .replace(
                    "production-control-v2",
                    "phase54-kq-transpose16x16-all-full-v1",
                )
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-all-full\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            let (spec, _) = parse_candidate_spec(candidate.as_bytes()).unwrap();
            assert!(spec.is_transform_candidate());
            assert!(!spec.is_production_control());
            assert_eq!(
                spec.candidate_descriptor_id("kv-fp8-e5-block16-v2"),
                "kv-fp8-e5-block16-v2-phase54-kq-transpose16x16-all-full-v1"
            );
        }

        #[test]
        fn transform_candidate_rejects_nonfloor_recipe_pair() {
            let candidate = SPEC
                .replace("production-control-v2", "phase54-k-ceil-v-floor-v1")
                .replace("\"k_recipe\":\"floor\"", "\"k_recipe\":\"ceil\"")
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-all-full\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            assert!(parse_candidate_spec(candidate.as_bytes()).is_err());
        }

        #[test]
        fn vo_transform_candidate_identity_is_floor_floor_only() {
            let candidate = SPEC
                .replace(
                    "production-control-v2",
                    "phase54-vo-transpose16x16-layer19-v1",
                )
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-v-layer19-output-inverse\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            let (spec, _) = parse_candidate_spec(candidate.as_bytes()).unwrap();
            assert!(spec.is_vo_transform_candidate());
            assert!(!spec.is_production_control());
            assert_eq!(
                spec.candidate_descriptor_id("kv-fp8-e5-block16-v2"),
                "kv-fp8-e5-block16-v2-phase54-vo-transpose16x16-layer19-v1"
            );
        }

        #[test]
        fn vo_transform_candidate_rejects_nonfloor_recipe_pair() {
            let candidate = SPEC
                .replace("production-control-v2", "phase54-k-ceil-v-floor-v1")
                .replace("\"k_recipe\":\"floor\"", "\"k_recipe\":\"ceil\"")
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-v-layer19-output-inverse\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            assert!(parse_candidate_spec(candidate.as_bytes()).is_err());
        }

        #[test]
        fn vo_layers19_31_transform_candidate_identity_is_floor_floor_only() {
            let candidate = SPEC
                .replace(
                    "production-control-v2",
                    "phase54-vo-transpose16x16-layers19-31-v1",
                )
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-v-layers19-31-output-inverse\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            let (spec, _) = parse_candidate_spec(candidate.as_bytes()).unwrap();
            assert!(spec.is_vo_layers19_31_transform_candidate());
            assert!(!spec.is_production_control());
            assert_eq!(
                spec.candidate_descriptor_id("kv-fp8-e5-block16-v2"),
                "kv-fp8-e5-block16-v2-phase54-vo-transpose16x16-layers19-31-v1"
            );
        }

        #[test]
        fn vo_layers19_31_transform_candidate_rejects_nonfloor_recipe_pair() {
            let candidate = SPEC
                .replace("production-control-v2", "phase54-k-ceil-v-floor-v1")
                .replace("\"k_recipe\":\"floor\"", "\"k_recipe\":\"ceil\"")
                .replace(
                    "\"transform\":\"none\"",
                    "\"transform\":\"transpose16x16-v-layers19-31-output-inverse\"",
                )
                .replace(
                    "exact-production-v2",
                    "research-build-semantic-override-not-v2-compatible",
                );
            assert!(parse_candidate_spec(candidate.as_bytes()).is_err());
        }

        #[test]
        fn transform_env_guard_is_fail_closed_and_restores_off() {
            TransformEnvGuard::restore_off();
            VoTransformEnvGuard::restore_off();
            #[cfg(feature = "phase54-research")]
            {
                let guard =
                    TransformEnvGuard::install(TransformRunMode::Candidate, "gfx1030").unwrap();
                assert_eq!(
                    env::var(PHASE54_KQ_TRANSFORM_ENV).as_deref(),
                    Ok("transpose16x16-all-full")
                );
                drop(guard);
                let vo_guard =
                    VoTransformEnvGuard::install(VoTransformRunMode::Candidate, "gfx1030").unwrap();
                assert_eq!(
                    env::var(PHASE54_VO_TRANSFORM_ENV).as_deref(),
                    Ok("transpose16x16-v-layer19-output-inverse")
                );
                drop(vo_guard);
                let vo_layers_guard = VoTransformEnvGuard::install(
                    VoTransformRunMode::Layers19And31Candidate,
                    "gfx1030",
                )
                .unwrap();
                assert_eq!(
                    env::var(PHASE54_VO_TRANSFORM_ENV).as_deref(),
                    Ok(PHASE54_VO_LAYERS19_31_SELECTOR)
                );
                drop(vo_layers_guard);
            }
            #[cfg(not(feature = "phase54-research"))]
            {
                assert!(
                    TransformEnvGuard::install(TransformRunMode::Candidate, "gfx1030").is_err()
                );
                assert!(
                    VoTransformEnvGuard::install(VoTransformRunMode::Candidate, "gfx1030").is_err()
                );
                assert!(
                    VoTransformEnvGuard::install(
                        VoTransformRunMode::Layers19And31Candidate,
                        "gfx1030"
                    )
                    .is_err()
                );
            }
            assert_eq!(env::var(PHASE54_KQ_TRANSFORM_ENV).as_deref(), Ok("off"));
            assert_eq!(env::var(PHASE54_VO_TRANSFORM_ENV).as_deref(), Ok("off"));
        }

        #[test]
        fn candidate_identity_and_compatibility_cover_all_recipe_pairs() {
            for key in [Recipe::Floor, Recipe::Ceil, Recipe::NearestEven] {
                for value in [Recipe::Floor, Recipe::Ceil, Recipe::NearestEven] {
                    let control = key == Recipe::Floor && value == Recipe::Floor;
                    let candidate_id = if control {
                        "production-control-v2".to_owned()
                    } else {
                        format!("phase54-k-{}-v-{}-v1", key.id(), value.id())
                    };
                    let descriptor_compatibility = if control {
                        "exact-production-v2"
                    } else {
                        "research-build-semantic-override-not-v2-compatible"
                    };
                    let spec = CandidateSpec {
                        schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
                        candidate_id,
                        scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
                        rounding: "nearest-even".to_owned(),
                        k_recipe: key,
                        v_recipe: value,
                        transform: "none".to_owned(),
                        calibration_digest: None,
                        descriptor_compatibility: descriptor_compatibility.to_owned(),
                    };
                    spec.validate().unwrap();
                    assert_eq!(
                        spec.expected_descriptor_compatibility(),
                        descriptor_compatibility
                    );
                }
            }

            let parent32 = CandidateSpec {
                schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
                candidate_id: "phase54-k-parent32-duplicate-v-parent32-duplicate-v1".to_owned(),
                scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
                rounding: "nearest-even".to_owned(),
                k_recipe: Recipe::Parent32Duplicate,
                v_recipe: Recipe::Parent32Duplicate,
                transform: "none".to_owned(),
                calibration_digest: None,
                descriptor_compatibility: "research-build-semantic-override-not-v2-compatible"
                    .to_owned(),
            };
            parent32.validate().unwrap();

            let mut mixed = parent32.clone();
            mixed.k_recipe = Recipe::Floor;
            mixed.candidate_id = "phase54-k-floor-v-parent32-duplicate-v1".to_owned();
            assert!(mixed.validate().is_err());
        }

        #[test]
        fn candidate_spec_rejects_unconsumed_recipe() {
            let changed = SPEC.replace("nearest-even", "stochastic");
            assert!(parse_candidate_spec(changed.as_bytes()).is_err());
            let unknown = SPEC.replace(
                "\"descriptor_compatibility\"",
                "\"unconsumed\":true,\"descriptor_compatibility\"",
            );
            assert!(parse_candidate_spec(unknown.as_bytes()).is_err());
        }

        #[test]
        fn repeat_count_is_exactly_one_or_three() {
            assert_eq!(parse_repeats("1").unwrap(), 1);
            assert_eq!(parse_repeats("3").unwrap(), 3);
            for rejected in ["0", "2", "03", "4", "one"] {
                assert!(parse_repeats(rejected).is_err(), "accepted {rejected}");
            }
        }

        #[test]
        fn maximum_locator_keeps_first_measured_row_on_tie() {
            let mut maximum = None;
            update_maximum(
                &mut maximum,
                LogitDeltaLocator {
                    case_id: "first".to_owned(),
                    row: "prefill",
                    measured_row_index: 0,
                    logit_index: 7,
                    max_abs_logit_delta: 2.0,
                },
            );
            update_maximum(
                &mut maximum,
                LogitDeltaLocator {
                    case_id: "second".to_owned(),
                    row: "decode",
                    measured_row_index: 1,
                    logit_index: 9,
                    max_abs_logit_delta: 2.0,
                },
            );
            let selected = maximum.unwrap();
            assert_eq!(selected.case_id, "first");
            assert_eq!(selected.row, "prefill");
            assert_eq!(selected.logit_index, 7);
        }
    }
}

fn main() -> std::process::ExitCode {
    sequential::entry()
}
