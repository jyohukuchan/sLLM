// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded, fail-closed loader for `ullm.served_model.v1` and `.v2`.

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

pub const SERVED_MODEL_SCHEMA_VERSION: &str = "ullm.served_model.v1";
pub const SERVED_MODEL_SCHEMA_VERSION_V2: &str = "ullm.served_model.v2";
pub const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_JSON_DEPTH: usize = 16;
const MAX_JSON_NODES: usize = 16_384;
const MAX_STRING_BYTES: usize = 65_536;
const MAX_TOKENIZER_FILES: usize = 128;
const MAX_ARGUMENTS: usize = 128;
const MAX_REQUIRED_ENVIRONMENT: usize = 128;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;

pub const LEGACY_MODEL_ENVIRONMENT: &[&str] = &[
    "ULLM_MODEL_ID",
    "ULLM_MODEL_REVISION",
    "ULLM_ARTIFACT_CONTENT_SHA256",
    "ULLM_PACKAGE_MANIFEST_SHA256",
    "ULLM_DEVICE",
    "ULLM_EXECUTION_PROFILE",
    "ULLM_MODEL_CONTEXT_LENGTH",
    "ULLM_MAX_NEW_TOKENS",
    "ULLM_VOCAB_SIZE",
    "ULLM_EOS_TOKEN_IDS",
    "ULLM_TOP_K",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedModelError(pub String);

impl fmt::Display for ServedModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ServedModelError {}

type Result<T> = std::result::Result<T, ServedModelError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub upstream_id: String,
    pub revision: String,
    pub context_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingContract {
    pub top_k: usize,
    pub temperature: bool,
    pub top_p: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationContract {
    pub max_completion_tokens: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<usize>,
    pub sampling: SamplingContract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatContract {
    pub format_id: String,
    pub implementation_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerContract {
    pub root: PathBuf,
    pub transformers_version: String,
    pub class_name: String,
    pub chat_template_sha256: String,
    pub files: Vec<TokenizerFile>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerIdentity {
    pub device: String,
    pub execution_profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerContract {
    pub protocol: String,
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub arguments: Vec<String>,
    pub required_environment: Vec<String>,
    pub identity: WorkerIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIdentity {
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub content_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageIdentity {
    pub manifest_path: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductContract {
    pub root: PathBuf,
    pub artifact: Option<ArtifactIdentity>,
    pub package: PackageIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationAuditIdentity {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationLineageIdentity {
    pub input_path: PathBuf,
    pub runtime_path: PathBuf,
    pub sha256: String,
    pub entries_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessIdentity {
    pub container_name: String,
    pub container_id: String,
    pub image_id: String,
    pub config_image: String,
    pub network_name: String,
    pub network_id: String,
    pub network_driver: String,
    pub bridge_interface: String,
    pub url: String,
    pub path: String,
    pub expected_status: usize,
    pub expected_body: String,
    pub expected_body_sha256: String,
    pub timeout_seconds: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionContract {
    pub source_commit: String,
    pub receipt: PathBuf,
    pub receipt_sha256: String,
    pub authorization_audit: Option<AuthorizationAuditIdentity>,
    pub authorization_lineage: Option<AuthorizationLineageIdentity>,
    pub readiness: Option<ReadinessIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProfileSnapshot {
    pub worker_schema: String,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_sha256: String,
    pub package_manifest_sha256: String,
    pub device: String,
    pub execution_profile: String,
    pub context_length: usize,
    pub max_new_tokens: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<usize>,
    pub top_k: usize,
    pub reasoning: Option<crate::reasoning::ReasoningDialect>,
}

impl WorkerProfileSnapshot {
    pub fn into_worker_profile(self) -> crate::sq8_worker_protocol::Sq8WorkerProfile {
        crate::sq8_worker_protocol::Sq8WorkerProfile {
            worker_schema: self.worker_schema,
            model: self.model,
            model_revision: self.model_revision,
            artifact_content_sha256: self.artifact_content_sha256,
            package_manifest_sha256: self.package_manifest_sha256,
            device: self.device,
            execution_profile: self.execution_profile,
            context_length: self.context_length,
            max_new_tokens: self.max_new_tokens,
            vocab_size: self.vocab_size,
            eos_token_ids: self.eos_token_ids,
            top_k: self.top_k,
            reasoning: self.reasoning,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedModel {
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub public: PublicModel,
    pub generation: GenerationContract,
    pub format: FormatContract,
    pub tokenizer: TokenizerContract,
    pub worker: WorkerContract,
    pub product: ProductContract,
    pub promotion: PromotionContract,
    pub reasoning: Option<crate::reasoning::ReasoningDialect>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerBackendKind {
    Sq8,
    Aq4,
    Aq4Sq8Overlay,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerStartupConfig {
    pub artifact_dir: Option<PathBuf>,
    pub package_dir: PathBuf,
    pub profile: WorkerProfileSnapshot,
    pub required_environment: Vec<String>,
    pub reasoning: Option<crate::reasoning::ReasoningDialect>,
}

impl ServedModel {
    pub fn profile_snapshot(&self) -> WorkerProfileSnapshot {
        WorkerProfileSnapshot {
            worker_schema: self.worker.protocol.clone(),
            model: self.public.id.clone(),
            model_revision: self.public.revision.clone(),
            artifact_content_sha256: self
                .product
                .artifact
                .as_ref()
                .map(|artifact| artifact.content_sha256.clone())
                .unwrap_or_else(|| self.product.package.manifest_sha256.clone()),
            package_manifest_sha256: self.product.package.manifest_sha256.clone(),
            device: self.worker.identity.device.clone(),
            execution_profile: self.worker.identity.execution_profile.clone(),
            context_length: self.public.context_length,
            max_new_tokens: self.generation.max_completion_tokens,
            vocab_size: self.generation.vocab_size,
            eos_token_ids: self.generation.eos_token_ids.clone(),
            top_k: self.generation.sampling.top_k,
            reasoning: self.reasoning.clone(),
        }
    }

    pub fn worker_startup(
        &self,
        kind: WorkerBackendKind,
        current_exe: &Path,
    ) -> Result<WorkerStartupConfig> {
        let mixed = LEGACY_MODEL_ENVIRONMENT
            .iter()
            .copied()
            .filter(|name| std::env::var_os(name).is_some())
            .collect::<Vec<_>>();
        if !mixed.is_empty() {
            return Err(ServedModelError(format!(
                "served-model manifest mode cannot be mixed with legacy model environment: {}",
                mixed.join(",")
            )));
        }
        if self.worker.protocol != "ullm.worker.v1" && self.worker.protocol != "ullm.worker.v2" {
            return Err(ServedModelError("worker protocol is unsupported".into()));
        }
        let current_exe = safe_regular_file(current_exe, "current worker binary")?;
        if current_exe != self.worker.binary {
            return Err(ServedModelError(
                "manifest worker.binary does not identify the running worker".into(),
            ));
        }
        let (format_id, requires_artifact) = match kind {
            WorkerBackendKind::Sq8 => ("SQ8_0", true),
            WorkerBackendKind::Aq4 => ("AQ4_0", false),
            WorkerBackendKind::Aq4Sq8Overlay => ("AQ4_0", true),
        };
        if self.format.format_id != format_id
            || self.product.artifact.is_some() != requires_artifact
        {
            return Err(ServedModelError(
                "manifest format/product shape does not match worker backend".into(),
            ));
        }
        for name in &self.worker.required_environment {
            if std::env::var(name).ok().as_deref() != Some("1") {
                return Err(ServedModelError(format!(
                    "required worker environment {name} must equal 1"
                )));
            }
        }
        let artifact_dir = self.product.artifact.as_ref().map(|artifact| {
            self.product
                .root
                .join(&artifact.manifest_path)
                .parent()
                .expect("validated artifact manifest has a parent")
                .to_path_buf()
        });
        let package_dir = self
            .product
            .root
            .join(&self.product.package.manifest_path)
            .parent()
            .expect("validated package manifest has a parent")
            .to_path_buf();
        Ok(WorkerStartupConfig {
            artifact_dir,
            package_dir,
            profile: self.profile_snapshot(),
            required_environment: self.worker.required_environment.clone(),
            reasoning: self.reasoning.clone(),
        })
    }
}

pub fn load_served_model(path: impl AsRef<Path>) -> Result<ServedModel> {
    let manifest_path = safe_regular_file(path.as_ref(), "served-model manifest")?;
    let raw = bounded_read(&manifest_path, MAX_MANIFEST_BYTES, "served-model manifest")?;
    load_served_model_bytes(manifest_path, &raw)
}

/// Parses one already-pinned served-model manifest snapshot.
///
/// `manifest_path` is deliberately used only as the logical path recorded in the model and as
/// the base for resolving manifest-relative resources. This entry point never opens the manifest
/// path; callers that establish an inherited-FD trust boundary can therefore parse and hash the
/// exact bytes read from that descriptor without a path fallback.
pub fn load_served_model_bytes(manifest_path: impl AsRef<Path>, raw: &[u8]) -> Result<ServedModel> {
    if raw.len() > MAX_MANIFEST_BYTES {
        return Err(ServedModelError(
            "served-model manifest exceeds its size limit".into(),
        ));
    }
    let manifest_path = manifest_path.as_ref().to_path_buf();
    let value = decode_strict_json(raw)?;
    validate_exact_shape(&value)?;
    let raw_manifest: RawManifest = serde_json::from_value(value)
        .map_err(|_| ServedModelError("manifest typed schema is invalid".into()))?;
    if raw_manifest.schema_version != SERVED_MODEL_SCHEMA_VERSION
        && raw_manifest.schema_version != SERVED_MODEL_SCHEMA_VERSION_V2
    {
        return Err(ServedModelError(
            "manifest schema_version is unsupported".into(),
        ));
    }
    let base = manifest_path
        .parent()
        .ok_or_else(|| ServedModelError("manifest path has no parent".into()))?;
    let public = parse_public(raw_manifest.public)?;
    let generation = parse_generation(raw_manifest.generation, &public)?;
    let reasoning = raw_manifest
        .reasoning
        .map(|raw| parse_reasoning(raw, generation.vocab_size))
        .transpose()?;
    if let Some(dialect) = reasoning.as_ref() {
        let reserved_for_max_budget = dialect
            .max_budget_tokens
            .checked_add(dialect.forced_end_sequence.len())
            .and_then(|value| value.checked_add(dialect.reserved_answer_tokens))
            .ok_or_else(|| ServedModelError("reasoning maximum reservation overflows".into()))?;
        if reserved_for_max_budget > generation.max_completion_tokens {
            return Err(ServedModelError(
                "reasoning maximum budget exceeds the generation reservation".into(),
            ));
        }
    }
    let format = FormatContract {
        format_id: bounded_text(raw_manifest.format.format_id, "format.format_id", 128)?,
        implementation_id: bounded_text(
            raw_manifest.format.implementation_id,
            "format.implementation_id",
            256,
        )?,
    };
    let tokenizer = parse_tokenizer(raw_manifest.tokenizer, base)?;
    let worker = parse_worker(raw_manifest.worker, base)?;
    let expected_worker_schema = if raw_manifest.schema_version == SERVED_MODEL_SCHEMA_VERSION_V2 {
        "ullm.worker.v2"
    } else {
        "ullm.worker.v1"
    };
    if worker.protocol != expected_worker_schema {
        return Err(ServedModelError(
            "manifest schema_version and worker.protocol must be version aligned".into(),
        ));
    }
    let product = parse_product(raw_manifest.product, base)?;
    let promotion = parse_promotion(raw_manifest.promotion, base)?;
    Ok(ServedModel {
        manifest_path,
        manifest_sha256: sha256_bytes(raw),
        public,
        generation,
        format,
        tokenizer,
        worker,
        product,
        promotion,
        reasoning,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    schema_version: String,
    public: RawPublic,
    generation: RawGeneration,
    format: RawFormat,
    tokenizer: RawTokenizer,
    worker: RawWorker,
    product: RawProduct,
    promotion: RawPromotion,
    reasoning: Option<RawReasoning>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReasoning {
    enabled_by_default: bool,
    dialect_id: String,
    start_token_ids: Vec<usize>,
    end_token_ids: Vec<usize>,
    forced_end_token_ids: Vec<usize>,
    initial_phase: String,
    eos_policy: String,
    effort_budgets: BTreeMap<String, usize>,
    max_budget_tokens: usize,
    reserved_answer_tokens: usize,
    history_reasoning_policy: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPublic {
    id: String,
    name: String,
    description: String,
    upstream_id: String,
    revision: String,
    context_length: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGeneration {
    max_completion_tokens: usize,
    vocab_size: usize,
    eos_token_ids: Vec<usize>,
    sampling: RawSampling,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSampling {
    top_k: usize,
    temperature: bool,
    top_p: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFormat {
    format_id: String,
    implementation_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTokenizer {
    root: String,
    transformers_version: String,
    #[serde(rename = "class")]
    class_name: String,
    chat_template_sha256: String,
    files: BTreeMap<String, String>,
    template_options: RawTemplateOptions,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTemplateOptions {
    add_generation_prompt: bool,
    enable_thinking: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorker {
    protocol: String,
    binary: String,
    binary_sha256: String,
    arguments: Vec<String>,
    required_environment: Vec<String>,
    identity: RawWorkerIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerIdentity {
    device: String,
    execution_profile: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProduct {
    root: String,
    artifact: Option<RawArtifact>,
    package: RawPackage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    manifest_path: String,
    manifest_sha256: String,
    content_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPackage {
    manifest_path: String,
    manifest_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPromotion {
    source_commit: String,
    receipt: String,
    receipt_sha256: String,
    authorization_audit: Option<RawAuthorizationAuditIdentity>,
    authorization_lineage: Option<RawAuthorizationLineageIdentity>,
    readiness: Option<RawReadinessIdentity>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationAuditIdentity {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageIdentity {
    schema_version: String,
    input_path: String,
    runtime_path: String,
    sha256: String,
    entries_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReadinessIdentity {
    schema: String,
    container: RawReadinessContainer,
    network: RawReadinessNetwork,
    endpoint: RawReadinessEndpoint,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReadinessContainer {
    name: String,
    id: String,
    image_id: String,
    config_image: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReadinessNetwork {
    name: String,
    id: String,
    driver: String,
    bridge_interface: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReadinessEndpoint {
    url: String,
    path: String,
    expected_status: usize,
    expected_body: String,
    expected_body_sha256: String,
    timeout_seconds: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndependentAuditReceipt {
    schema_version: String,
    auditor_task_id: String,
    audited_at_utc: String,
    audited_source: RawAuditSource,
    runtime: RawAuditRuntime,
    fixed_request_id: String,
    gate_state: RawAuditGateState,
    topology: RawAuditTopology,
    verdict: String,
    actual: String,
    tests: RawAuditTests,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditSource {
    commit: String,
    tree_sha256: String,
    archive_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditRuntime {
    path: String,
    gate: RawAuditReference,
    worker: RawAuditReference,
    profile: RawAuditReference,
    served_model: RawAuditReference,
    prepared_receipt: RawAuditReference,
    binding: RawAuditBindingReference,
    package: RawAuditReference,
    authorization_lineage_manifest: RawAuditReference,
    sha256sums: RawAuditReference,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditReference {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditBindingReference {
    path: String,
    sha256: String,
    content_sha256: String,
    tensor_set_sha256: String,
    tensor_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditGateState {
    status: String,
    actual_run_allowed: bool,
    prepared_receipt_status: String,
    prepared_receipt_actual: RawPreparedReceiptActual,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPreparedReceiptActual {
    status: String,
    required: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditTopology {
    artifact_directory_count: usize,
    artifact_payload_and_scale_files_hashed: usize,
    artifact_regular_file_bytes: u64,
    artifact_regular_file_count: usize,
    current_runtime_reference_count: usize,
    executable_file_mode: String,
    historical_runtime_reference_count: usize,
    package_directory_count: usize,
    package_regular_file_count: usize,
    regular_file_mode: String,
    regular_file_nlink: u64,
    runtime_directory_mode: String,
    runtime_directory_nlink: u64,
    runtime_member_count: usize,
    special_file_count: usize,
    symlink_count: usize,
    worker_source_and_immutable_are_runtime_self: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditTests {
    actual_output: String,
    artifact_live_content: String,
    authorization_boundary: String,
    bridge_readiness_binding: String,
    candidate_wrapper_dry_run: String,
    fixed_request_id_recomputation: String,
    formal_lineage_manifest: String,
    gpu_or_service_execution: bool,
    historical_runtime_references: String,
    lineage_external_runtime_copy: String,
    package_live_identity: String,
    runtime_modes_links_and_symlinks: String,
    runtime_sha256sums: String,
    source_commit_tree_archive: String,
    source_worktree: String,
    sudo_execution: bool,
    worker_live_identity: String,
    worker_runtime_self_identity: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageManifest {
    schema_version: String,
    disposition: String,
    source: RawAuthorizationLineageSource,
    entries: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageSource {
    archive_sha256: String,
    commit: String,
    tree_oid: String,
}

fn parse_public(raw: RawPublic) -> Result<PublicModel> {
    if raw.context_length == 0 {
        return Err(ServedModelError(
            "public.context_length must be positive".into(),
        ));
    }
    Ok(PublicModel {
        id: bounded_text(raw.id, "public.id", 256)?,
        name: bounded_text(raw.name, "public.name", 512)?,
        description: bounded_text(raw.description, "public.description", 4096)?,
        upstream_id: bounded_text(raw.upstream_id, "public.upstream_id", 512)?,
        revision: bounded_text(raw.revision, "public.revision", 256)?,
        context_length: raw.context_length,
    })
}

fn parse_generation(raw: RawGeneration, public: &PublicModel) -> Result<GenerationContract> {
    if raw.max_completion_tokens == 0
        || raw.max_completion_tokens > public.context_length
        || raw.vocab_size == 0
        || raw.eos_token_ids.is_empty()
        || raw.sampling.top_k == 0
        || raw.sampling.top_k > raw.vocab_size
    {
        return Err(ServedModelError("generation limits are invalid".into()));
    }
    let mut eos = HashSet::new();
    if raw
        .eos_token_ids
        .iter()
        .any(|token| *token >= raw.vocab_size || !eos.insert(*token))
    {
        return Err(ServedModelError(
            "generation EOS contract is invalid".into(),
        ));
    }
    if (!raw.sampling.temperature || !raw.sampling.top_p) && raw.sampling.top_k != 1 {
        return Err(ServedModelError(
            "disabled sampling requires deterministic top_k=1".into(),
        ));
    }
    Ok(GenerationContract {
        max_completion_tokens: raw.max_completion_tokens,
        vocab_size: raw.vocab_size,
        eos_token_ids: raw.eos_token_ids,
        sampling: SamplingContract {
            top_k: raw.sampling.top_k,
            temperature: raw.sampling.temperature,
            top_p: raw.sampling.top_p,
        },
    })
}

fn parse_reasoning(
    raw: RawReasoning,
    vocab_size: usize,
) -> Result<crate::reasoning::ReasoningDialect> {
    let dialect = crate::reasoning::ReasoningDialect {
        identity: bounded_text(raw.dialect_id, "reasoning.dialect_id", 256)?,
        start_sequence: raw.start_token_ids,
        end_sequence: raw.end_token_ids,
        forced_end_sequence: raw.forced_end_token_ids,
        max_budget_tokens: raw.max_budget_tokens,
        reserved_answer_tokens: raw.reserved_answer_tokens,
        enabled_by_default: raw.enabled_by_default,
        effort_budgets: ["low", "medium", "high"]
            .into_iter()
            .map(|name| {
                raw.effort_budgets
                    .get(name)
                    .copied()
                    .map(|budget| (name.to_string(), budget))
                    .ok_or_else(|| {
                        ServedModelError("reasoning effort budgets are incomplete".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?,
        history_reasoning_policy: match raw.history_reasoning_policy.as_str() {
            "omit" => crate::reasoning::HistoryReasoningPolicy::Omit,
            "preserve" => crate::reasoning::HistoryReasoningPolicy::Preserve,
            _ => {
                return Err(ServedModelError(
                    "reasoning history policy is invalid".into(),
                ));
            }
        },
        initial_phase: match raw.initial_phase.as_str() {
            "reasoning" => crate::reasoning::InitialReasoningPhase::Reasoning,
            "answer" => crate::reasoning::InitialReasoningPhase::Answer,
            _ => {
                return Err(ServedModelError(
                    "reasoning initial phase is invalid".into(),
                ));
            }
        },
        eos_policy: match raw.eos_policy.as_str() {
            "close" => crate::reasoning::ReasoningEosPolicy::Close,
            "finish" => crate::reasoning::ReasoningEosPolicy::Finish,
            "continue" => crate::reasoning::ReasoningEosPolicy::Continue,
            _ => return Err(ServedModelError("reasoning EOS policy is invalid".into())),
        },
    };
    dialect
        .validate(vocab_size)
        .map_err(|_| ServedModelError("reasoning dialect is invalid".into()))?;
    Ok(dialect)
}

fn parse_tokenizer(raw: RawTokenizer, base: &Path) -> Result<TokenizerContract> {
    let root = safe_directory(
        &resolve_root(base, &raw.root, "tokenizer.root")?,
        "tokenizer.root",
    )?;
    if raw.files.is_empty() || raw.files.len() > MAX_TOKENIZER_FILES {
        return Err(ServedModelError("tokenizer.files size is invalid".into()));
    }
    let mut files = Vec::with_capacity(raw.files.len());
    for (path, digest) in raw.files {
        let path = relative_path(&path, "tokenizer file path")?;
        let digest = validate_sha256(digest, "tokenizer file SHA-256")?;
        let target = contained_regular_file(&root, &path, "tokenizer file")?;
        verify_file_sha256(&target, &digest, "tokenizer file")?;
        files.push(TokenizerFile {
            path,
            sha256: digest,
        });
    }
    Ok(TokenizerContract {
        root,
        transformers_version: bounded_text(
            raw.transformers_version,
            "tokenizer.transformers_version",
            64,
        )?,
        class_name: bounded_text(raw.class_name, "tokenizer.class", 128)?,
        chat_template_sha256: validate_sha256(
            raw.chat_template_sha256,
            "tokenizer.chat_template_sha256",
        )?,
        files,
        add_generation_prompt: raw.template_options.add_generation_prompt,
        enable_thinking: raw.template_options.enable_thinking,
    })
}

fn parse_worker(raw: RawWorker, base: &Path) -> Result<WorkerContract> {
    let binary = safe_regular_file(
        &resolve_root(base, &raw.binary, "worker.binary")?,
        "worker.binary",
    )?;
    if binary.metadata().map_err(io_error)?.permissions().mode() & 0o111 == 0 {
        return Err(ServedModelError("worker.binary is not executable".into()));
    }
    let binary_sha256 = validate_sha256(raw.binary_sha256, "worker.binary_sha256")?;
    verify_file_sha256(&binary, &binary_sha256, "worker.binary")?;
    if raw.arguments.len() > MAX_ARGUMENTS
        || raw
            .arguments
            .iter()
            .filter(|value| value.as_str() == "{manifest}")
            .count()
            != 1
    {
        return Err(ServedModelError("worker.arguments is invalid".into()));
    }
    let arguments = raw
        .arguments
        .into_iter()
        .enumerate()
        .map(|(index, value)| bounded_text(value, &format!("worker.arguments[{index}]"), 4096))
        .collect::<Result<Vec<_>>>()?;
    if raw.required_environment.len() > MAX_REQUIRED_ENVIRONMENT {
        return Err(ServedModelError(
            "worker.required_environment is invalid".into(),
        ));
    }
    let mut seen = HashSet::new();
    for name in &raw.required_environment {
        if !valid_environment_name(name) || !seen.insert(name.as_str()) {
            return Err(ServedModelError(
                "worker.required_environment is invalid".into(),
            ));
        }
    }
    Ok(WorkerContract {
        protocol: bounded_text(raw.protocol, "worker.protocol", 128)?,
        binary,
        binary_sha256,
        arguments,
        required_environment: raw.required_environment,
        identity: WorkerIdentity {
            device: bounded_text(raw.identity.device, "worker.identity.device", 128)?,
            execution_profile: bounded_text(
                raw.identity.execution_profile,
                "worker.identity.execution_profile",
                256,
            )?,
        },
    })
}

fn parse_product(raw: RawProduct, base: &Path) -> Result<ProductContract> {
    let root = safe_directory(
        &resolve_root(base, &raw.root, "product.root")?,
        "product.root",
    )?;
    let artifact = raw
        .artifact
        .map(|raw| {
            let manifest_path =
                relative_path(&raw.manifest_path, "product.artifact.manifest_path")?;
            let manifest_sha256 =
                validate_sha256(raw.manifest_sha256, "artifact manifest SHA-256")?;
            let target = contained_regular_file(&root, &manifest_path, "artifact manifest")?;
            verify_file_sha256(&target, &manifest_sha256, "artifact manifest")?;
            Ok(ArtifactIdentity {
                manifest_path,
                manifest_sha256,
                content_sha256: validate_sha256(raw.content_sha256, "artifact content SHA-256")?,
            })
        })
        .transpose()?;
    let package_path = relative_path(&raw.package.manifest_path, "product.package.manifest_path")?;
    let package_sha = validate_sha256(raw.package.manifest_sha256, "package manifest SHA-256")?;
    let package_target = contained_regular_file(&root, &package_path, "package manifest")?;
    verify_file_sha256(&package_target, &package_sha, "package manifest")?;
    Ok(ProductContract {
        root,
        artifact,
        package: PackageIdentity {
            manifest_path: package_path,
            manifest_sha256: package_sha,
        },
    })
}

fn canonical_absolute_regular_file(raw: String, label: &str, immutable: bool) -> Result<PathBuf> {
    let raw = bounded_text(raw, label, 4096)?;
    let path = PathBuf::from(&raw);
    if !path.is_absolute() {
        return Err(ServedModelError(format!(
            "{label} must be a canonical absolute path"
        )));
    }
    let resolved = safe_regular_file(&path, label)?;
    if resolved != path {
        return Err(ServedModelError(format!(
            "{label} must be a canonical absolute path"
        )));
    }
    if immutable {
        let metadata = fs::symlink_metadata(&resolved).map_err(io_error)?;
        if metadata.permissions().mode() & 0o777 != 0o444 || metadata.nlink() != 1 {
            return Err(ServedModelError(format!(
                "{label} must be immutable single-link"
            )));
        }
    }
    Ok(resolved)
}

fn validate_canonical_absolute_text(raw: String, label: &str) -> Result<String> {
    let raw = bounded_text(raw, label, 4096)?;
    let path = Path::new(&raw);
    if !path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::CurDir | Component::ParentDir | Component::Prefix(_)
            )
        })
    {
        return Err(ServedModelError(format!(
            "{label} must be a canonical absolute path"
        )));
    }
    Ok(raw)
}

fn validate_hex40(value: String, label: &str) -> Result<String> {
    if value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(value)
    } else {
        Err(ServedModelError(format!(
            "{label} must be lowercase hexadecimal"
        )))
    }
}

fn validate_request_id(value: String, label: &str) -> Result<String> {
    let Some(digest) = value.strip_prefix("sq8-promotion-") else {
        return Err(ServedModelError(format!("{label} is invalid")));
    };
    validate_sha256(digest.to_string(), label)?;
    Ok(value)
}

fn validate_audit_reference(raw: RawAuditReference, label: &str) -> Result<()> {
    validate_canonical_absolute_text(raw.path, &format!("{label}.path"))?;
    validate_sha256(raw.sha256, &format!("{label}.sha256"))?;
    Ok(())
}

fn parse_authorization_audit(
    raw: RawAuthorizationAuditIdentity,
    source_commit: &str,
) -> Result<AuthorizationAuditIdentity> {
    let path =
        canonical_absolute_regular_file(raw.path, "promotion.authorization_audit.path", true)?;
    let sha256 = validate_sha256(raw.sha256, "promotion.authorization_audit.sha256")?;
    verify_file_sha256(&path, &sha256, "promotion.authorization_audit")?;
    let bytes = bounded_read(&path, MAX_MANIFEST_BYTES, "promotion.authorization_audit")?;
    let value = decode_strict_json(&bytes)?;
    exact_keys(
        &value,
        &[
            "schema_version",
            "auditor_task_id",
            "audited_at_utc",
            "audited_source",
            "runtime",
            "fixed_request_id",
            "gate_state",
            "topology",
            "verdict",
            "actual",
            "tests",
        ],
        "promotion.authorization_audit receipt",
    )?;
    let audit: RawIndependentAuditReceipt = serde_json::from_value(value).map_err(|_| {
        ServedModelError("promotion.authorization_audit typed schema is invalid".into())
    })?;
    if audit.schema_version != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
        || audit.verdict != "implementation_ready"
        || audit.actual != "not_executed"
        || audit.audited_source.commit != source_commit
        || audit.gate_state.status != "ready_for_independent_audit"
        || audit.gate_state.actual_run_allowed
        || audit.gate_state.prepared_receipt_status != "prepared_not_executed"
        || audit.gate_state.prepared_receipt_actual.status != "pending"
        || !audit.gate_state.prepared_receipt_actual.required
        || audit.tests.gpu_or_service_execution
        || audit.tests.sudo_execution
    {
        return Err(ServedModelError(
            "promotion.authorization_audit verdict differs".into(),
        ));
    }
    bounded_text(
        audit.auditor_task_id,
        "promotion.authorization_audit.auditor_task_id",
        256,
    )?;
    bounded_text(
        audit.audited_at_utc,
        "promotion.authorization_audit.audited_at_utc",
        64,
    )?;
    validate_hex40(
        audit.audited_source.commit,
        "promotion.authorization_audit.audited_source.commit",
    )?;
    validate_hex40(
        audit.audited_source.tree_sha256,
        "promotion.authorization_audit.audited_source.tree_sha256",
    )?;
    validate_sha256(
        audit.audited_source.archive_sha256,
        "promotion.authorization_audit.audited_source.archive_sha256",
    )?;
    validate_request_id(
        audit.fixed_request_id,
        "promotion.authorization_audit.fixed_request_id",
    )?;
    validate_canonical_absolute_text(
        audit.runtime.path,
        "promotion.authorization_audit.runtime.path",
    )?;
    for (reference, label) in [
        (audit.runtime.gate, "runtime.gate"),
        (audit.runtime.worker, "runtime.worker"),
        (audit.runtime.profile, "runtime.profile"),
        (audit.runtime.served_model, "runtime.served_model"),
        (audit.runtime.prepared_receipt, "runtime.prepared_receipt"),
        (audit.runtime.package, "runtime.package"),
        (
            audit.runtime.authorization_lineage_manifest,
            "runtime.authorization_lineage_manifest",
        ),
        (audit.runtime.sha256sums, "runtime.sha256sums"),
    ] {
        validate_audit_reference(reference, &format!("promotion.authorization_audit.{label}"))?;
    }
    validate_canonical_absolute_text(
        audit.runtime.binding.path,
        "promotion.authorization_audit.runtime.binding.path",
    )?;
    validate_sha256(
        audit.runtime.binding.sha256,
        "promotion.authorization_audit.runtime.binding.sha256",
    )?;
    validate_sha256(
        audit.runtime.binding.content_sha256,
        "promotion.authorization_audit.runtime.binding.content_sha256",
    )?;
    validate_sha256(
        audit.runtime.binding.tensor_set_sha256,
        "promotion.authorization_audit.runtime.binding.tensor_set_sha256",
    )?;
    if audit.runtime.binding.tensor_count != 48 {
        return Err(ServedModelError(
            "promotion.authorization_audit runtime binding differs".into(),
        ));
    }
    let topology = audit.topology;
    if topology.artifact_directory_count != 3
        || topology.artifact_payload_and_scale_files_hashed != 96
        || topology.artifact_regular_file_bytes == 0
        || topology.artifact_regular_file_count != 98
        || topology.current_runtime_reference_count == 0
        || topology.executable_file_mode != "0555"
        || topology.historical_runtime_reference_count != 0
        || topology.package_directory_count == 0
        || topology.package_regular_file_count == 0
        || topology.regular_file_mode != "0444"
        || topology.regular_file_nlink != 1
        || topology.runtime_directory_mode != "0555"
        || topology.runtime_directory_nlink != 2
        || topology.runtime_member_count != 8
        || topology.special_file_count != 0
        || topology.symlink_count != 0
        || !topology.worker_source_and_immutable_are_runtime_self
    {
        return Err(ServedModelError(
            "promotion.authorization_audit topology differs".into(),
        ));
    }
    for (text, label) in [
        (audit.tests.actual_output, "actual_output"),
        (audit.tests.artifact_live_content, "artifact_live_content"),
        (audit.tests.authorization_boundary, "authorization_boundary"),
        (
            audit.tests.bridge_readiness_binding,
            "bridge_readiness_binding",
        ),
        (
            audit.tests.candidate_wrapper_dry_run,
            "candidate_wrapper_dry_run",
        ),
        (
            audit.tests.fixed_request_id_recomputation,
            "fixed_request_id_recomputation",
        ),
        (
            audit.tests.formal_lineage_manifest,
            "formal_lineage_manifest",
        ),
        (
            audit.tests.historical_runtime_references,
            "historical_runtime_references",
        ),
        (
            audit.tests.lineage_external_runtime_copy,
            "lineage_external_runtime_copy",
        ),
        (audit.tests.package_live_identity, "package_live_identity"),
        (
            audit.tests.runtime_modes_links_and_symlinks,
            "runtime_modes_links_and_symlinks",
        ),
        (audit.tests.runtime_sha256sums, "runtime_sha256sums"),
        (
            audit.tests.source_commit_tree_archive,
            "source_commit_tree_archive",
        ),
        (audit.tests.source_worktree, "source_worktree"),
        (audit.tests.worker_live_identity, "worker_live_identity"),
        (
            audit.tests.worker_runtime_self_identity,
            "worker_runtime_self_identity",
        ),
    ] {
        bounded_text(
            text,
            &format!("promotion.authorization_audit.tests.{label}"),
            4096,
        )?;
    }
    Ok(AuthorizationAuditIdentity { path, sha256 })
}

fn validate_lineage_document(
    bytes: &[u8],
    source_commit: &str,
    entries_sha256: &str,
) -> Result<()> {
    let value = decode_strict_json(bytes)?;
    exact_keys(
        &value,
        &["schema_version", "disposition", "source", "entries"],
        "promotion.authorization_lineage manifest",
    )?;
    let document: RawAuthorizationLineageManifest =
        serde_json::from_value(value).map_err(|_| {
            ServedModelError(
                "promotion.authorization_lineage manifest typed schema is invalid".into(),
            )
        })?;
    if document.schema_version != "ullm.sq8_authorization_lineage_input.v1"
        || document.disposition != "authorization_input_not_yet_runtime_bound"
        || document.source.commit != source_commit
        || document.entries.len() != 6
    {
        return Err(ServedModelError(
            "promotion.authorization_lineage manifest differs".into(),
        ));
    }
    validate_sha256(
        document.source.archive_sha256,
        "promotion.authorization_lineage.source.archive_sha256",
    )?;
    validate_hex40(
        document.source.commit,
        "promotion.authorization_lineage.source.commit",
    )?;
    validate_hex40(
        document.source.tree_oid,
        "promotion.authorization_lineage.source.tree_oid",
    )?;
    let encoded = serde_json::to_vec(&document.entries).map_err(|_| {
        ServedModelError("promotion.authorization_lineage entries are not canonical JSON".into())
    })?;
    if sha256_bytes(&encoded) != entries_sha256 {
        return Err(ServedModelError(
            "promotion.authorization_lineage entries SHA-256 differs".into(),
        ));
    }
    Ok(())
}

fn parse_authorization_lineage(
    raw: RawAuthorizationLineageIdentity,
    source_commit: &str,
) -> Result<AuthorizationLineageIdentity> {
    if raw.schema_version != "ullm.sq8_authorization_lineage_ref.v1" {
        return Err(ServedModelError(
            "promotion.authorization_lineage schema differs".into(),
        ));
    }
    let input_path = canonical_absolute_regular_file(
        raw.input_path,
        "promotion.authorization_lineage.input_path",
        true,
    )?;
    let runtime_path = canonical_absolute_regular_file(
        raw.runtime_path,
        "promotion.authorization_lineage.runtime_path",
        true,
    )?;
    let sha256 = validate_sha256(raw.sha256, "promotion.authorization_lineage.sha256")?;
    let entries_sha256 = validate_sha256(
        raw.entries_sha256,
        "promotion.authorization_lineage.entries_sha256",
    )?;
    for path in [&input_path, &runtime_path] {
        verify_file_sha256(path, &sha256, "promotion.authorization_lineage")?;
        let bytes = bounded_read(path, MAX_MANIFEST_BYTES, "promotion.authorization_lineage")?;
        validate_lineage_document(&bytes, source_commit, &entries_sha256)?;
    }
    Ok(AuthorizationLineageIdentity {
        input_path,
        runtime_path,
        sha256,
        entries_sha256,
    })
}

fn parse_readiness(raw: RawReadinessIdentity) -> Result<ReadinessIdentity> {
    if raw.schema != "ullm.bridge_container_readiness.v1" {
        return Err(ServedModelError(
            "promotion.readiness schema differs".into(),
        ));
    }
    let container_name = bounded_text(
        raw.container.name,
        "promotion.readiness.container.name",
        256,
    )?;
    let container_id = validate_sha256(raw.container.id, "promotion.readiness.container.id")?;
    let image_id = bounded_text(
        raw.container.image_id,
        "promotion.readiness.container.image_id",
        71,
    )?;
    let Some(image_digest) = image_id.strip_prefix("sha256:") else {
        return Err(ServedModelError(
            "promotion.readiness identity differs".into(),
        ));
    };
    validate_sha256(
        image_digest.to_string(),
        "promotion.readiness.container.image_id",
    )?;
    let config_image = bounded_text(
        raw.container.config_image,
        "promotion.readiness.container.config_image",
        512,
    )?;
    let network_name = bounded_text(raw.network.name, "promotion.readiness.network.name", 256)?;
    let network_id = validate_sha256(raw.network.id, "promotion.readiness.network.id")?;
    let network_driver =
        bounded_text(raw.network.driver, "promotion.readiness.network.driver", 64)?;
    let bridge_interface = bounded_text(
        raw.network.bridge_interface,
        "promotion.readiness.network.bridge_interface",
        64,
    )?;
    let url = bounded_text(raw.endpoint.url, "promotion.readiness.endpoint.url", 512)?;
    let path = bounded_text(raw.endpoint.path, "promotion.readiness.endpoint.path", 256)?;
    let expected_body = bounded_text(
        raw.endpoint.expected_body,
        "promotion.readiness.endpoint.expected_body",
        256,
    )?;
    let expected_body_sha256 = validate_sha256(
        raw.endpoint.expected_body_sha256,
        "promotion.readiness.endpoint.expected_body_sha256",
    )?;
    if container_name != "open-webui"
        || network_driver != "bridge"
        || bridge_interface != format!("br-{}", &network_id[..12])
        || url != "http://172.20.0.1:8000/readyz"
        || path != "/readyz"
        || raw.endpoint.expected_status != 200
        || expected_body != r#"{"status":"ready"}"#
        || sha256_bytes(expected_body.as_bytes()) != expected_body_sha256
        || raw.endpoint.timeout_seconds != 5
    {
        return Err(ServedModelError(
            "promotion.readiness identity differs".into(),
        ));
    }
    Ok(ReadinessIdentity {
        container_name,
        container_id,
        image_id,
        config_image,
        network_name,
        network_id,
        network_driver,
        bridge_interface,
        url,
        path,
        expected_status: raw.endpoint.expected_status,
        expected_body,
        expected_body_sha256,
        timeout_seconds: raw.endpoint.timeout_seconds,
    })
}

fn parse_promotion(raw: RawPromotion, base: &Path) -> Result<PromotionContract> {
    let source_commit = bounded_text(raw.source_commit, "promotion.source_commit", 256)?;
    let receipt = safe_regular_file(
        &resolve_root(base, &raw.receipt, "promotion.receipt")?,
        "promotion.receipt",
    )?;
    let digest = validate_sha256(raw.receipt_sha256, "promotion.receipt_sha256")?;
    verify_file_sha256(&receipt, &digest, "promotion.receipt")?;
    let authorization_audit = raw
        .authorization_audit
        .map(|value| parse_authorization_audit(value, &source_commit))
        .transpose()?;
    let authorization_lineage = raw
        .authorization_lineage
        .map(|value| parse_authorization_lineage(value, &source_commit))
        .transpose()?;
    let readiness = raw.readiness.map(parse_readiness).transpose()?;
    if authorization_audit.is_some() && (authorization_lineage.is_none() || readiness.is_none()) {
        return Err(ServedModelError(
            "authorized promotion requires audit, lineage, and readiness".into(),
        ));
    }
    Ok(PromotionContract {
        source_commit,
        receipt,
        receipt_sha256: digest,
        authorization_audit,
        authorization_lineage,
        readiness,
    })
}

fn validate_exact_shape(value: &Value) -> Result<()> {
    let schema = value
        .get("schema_version")
        .and_then(Value::as_str)
        .ok_or_else(|| ServedModelError("manifest schema_version is invalid".into()))?;
    let mut manifest_keys = vec![
        "schema_version",
        "public",
        "generation",
        "format",
        "tokenizer",
        "worker",
        "product",
        "promotion",
    ];
    if schema == SERVED_MODEL_SCHEMA_VERSION_V2 {
        manifest_keys.push("reasoning");
    }
    exact_keys(value, &manifest_keys, "manifest")?;
    if schema != SERVED_MODEL_SCHEMA_VERSION && schema != SERVED_MODEL_SCHEMA_VERSION_V2 {
        return Err(ServedModelError(
            "manifest schema_version is unsupported".into(),
        ));
    }
    exact_keys(
        &value["public"],
        &[
            "id",
            "name",
            "description",
            "upstream_id",
            "revision",
            "context_length",
        ],
        "public",
    )?;
    exact_keys(
        &value["generation"],
        &[
            "max_completion_tokens",
            "vocab_size",
            "eos_token_ids",
            "sampling",
        ],
        "generation",
    )?;
    exact_keys(
        &value["generation"]["sampling"],
        &["top_k", "temperature", "top_p"],
        "generation.sampling",
    )?;
    exact_keys(
        &value["format"],
        &["format_id", "implementation_id"],
        "format",
    )?;
    exact_keys(
        &value["tokenizer"],
        &[
            "root",
            "transformers_version",
            "class",
            "chat_template_sha256",
            "files",
            "template_options",
        ],
        "tokenizer",
    )?;
    exact_keys(
        &value["tokenizer"]["template_options"],
        &["add_generation_prompt", "enable_thinking"],
        "tokenizer.template_options",
    )?;
    exact_keys(
        &value["worker"],
        &[
            "protocol",
            "binary",
            "binary_sha256",
            "arguments",
            "required_environment",
            "identity",
        ],
        "worker",
    )?;
    exact_keys(
        &value["worker"]["identity"],
        &["device", "execution_profile"],
        "worker.identity",
    )?;
    exact_keys(
        &value["product"],
        &["root", "artifact", "package"],
        "product",
    )?;
    if !value["product"]["artifact"].is_null() {
        exact_keys(
            &value["product"]["artifact"],
            &["manifest_path", "manifest_sha256", "content_sha256"],
            "product.artifact",
        )?;
    }
    exact_keys(
        &value["product"]["package"],
        &["manifest_path", "manifest_sha256"],
        "product.package",
    )?;
    required_optional_keys(
        &value["promotion"],
        &["source_commit", "receipt", "receipt_sha256"],
        &["authorization_audit", "authorization_lineage", "readiness"],
        "promotion",
    )?;
    if schema == SERVED_MODEL_SCHEMA_VERSION_V2 {
        exact_keys(
            &value["reasoning"],
            &[
                "enabled_by_default",
                "dialect_id",
                "start_token_ids",
                "end_token_ids",
                "forced_end_token_ids",
                "initial_phase",
                "eos_policy",
                "effort_budgets",
                "max_budget_tokens",
                "reserved_answer_tokens",
                "history_reasoning_policy",
            ],
            "reasoning",
        )?;
        exact_keys(
            &value["reasoning"]["effort_budgets"],
            &["low", "medium", "high"],
            "reasoning.effort_budgets",
        )?;
    }
    Ok(())
}

fn exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| ServedModelError(format!("{label} must be an object")))?;
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err(ServedModelError(format!("{label} field set differs")));
    }
    Ok(())
}

fn required_optional_keys(
    value: &Value,
    required: &[&str],
    optional: &[&str],
    label: &str,
) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| ServedModelError(format!("{label} must be an object")))?;
    if required.iter().any(|key| !object.contains_key(*key))
        || object
            .keys()
            .any(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(ServedModelError(format!("{label} field set differs")));
    }
    Ok(())
}

fn decode_strict_json(raw: &[u8]) -> Result<Value> {
    std::str::from_utf8(raw).map_err(|_| ServedModelError("manifest is not valid UTF-8".into()))?;
    let nodes = Rc::new(Cell::new(0));
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let value = StrictValueSeed { depth: 1, nodes }
        .deserialize(&mut deserializer)
        .map_err(|_| ServedModelError("manifest is not strict JSON".into()))?;
    deserializer
        .end()
        .map_err(|_| ServedModelError("manifest is not strict JSON".into()))?;
    Ok(value)
}

#[derive(Clone)]
struct StrictValueSeed {
    depth: usize,
    nodes: Rc<Cell<usize>>,
}

impl StrictValueSeed {
    fn count<E: de::Error>(&self) -> std::result::Result<(), E> {
        let count = self
            .nodes
            .get()
            .checked_add(1)
            .ok_or_else(|| E::custom("node overflow"))?;
        if count > MAX_JSON_NODES || self.depth > MAX_JSON_DEPTH {
            return Err(E::custom("JSON bounds"));
        }
        self.nodes.set(count);
        Ok(())
    }
    fn child(&self) -> Self {
        Self {
            depth: self.depth + 1,
            nodes: Rc::clone(&self.nodes),
        }
    }
}

impl<'de> DeserializeSeed<'de> for StrictValueSeed {
    type Value = Value;
    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        self.count()?;
        deserializer.deserialize_any(StrictValueVisitor(self))
    }
}

struct StrictValueVisitor(StrictValueSeed);
impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded JSON")
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Value, E> {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite"))
    }
    fn visit_none<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Value, E> {
        self.visit_string(v.to_owned())
    }
    fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Value, E> {
        if v.len() > MAX_STRING_BYTES {
            Err(E::custom("string bounds"))
        } else {
            Ok(Value::String(v))
        }
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element_seed(self.0.child())? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        let mut out = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if key.len() > MAX_STRING_BYTES || out.contains_key(&key) {
                return Err(de::Error::custom("duplicate or long key"));
            }
            let value = map.next_value_seed(self.0.child())?;
            out.insert(key, value);
        }
        Ok(Value::Object(out))
    }
}

fn bounded_text(value: String, label: &str, maximum: usize) -> Result<String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(|ch| (ch as u32) < 0x20) {
        Err(ServedModelError(format!(
            "{label} must be bounded nonempty text"
        )))
    } else {
        Ok(value)
    }
}

fn validate_sha256(value: String, label: &str) -> Result<String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(value)
    } else {
        Err(ServedModelError(format!(
            "{label} must be lowercase SHA-256"
        )))
    }
}

fn valid_environment_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'_'))
        && bytes.all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

fn relative_path(value: &str, label: &str) -> Result<String> {
    if value.is_empty()
        || value.starts_with('/')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        Err(ServedModelError(format!(
            "{label} must be a contained relative path"
        )))
    } else {
        Ok(value.to_string())
    }
}

fn resolve_root(base: &Path, raw: &str, label: &str) -> Result<PathBuf> {
    let raw = bounded_text(raw.to_string(), label, 4096)?;
    let path = PathBuf::from(&raw);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(base.join(relative_path(&raw, label)?))
    }
}

fn safe_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let path = safe_path(path, label)?;
    let meta = path.metadata().map_err(io_error)?;
    if !meta.is_dir() || meta.permissions().mode() & 0o002 != 0 {
        Err(ServedModelError(format!("{label} is not a safe directory")))
    } else {
        Ok(path)
    }
}
fn safe_regular_file(path: &Path, label: &str) -> Result<PathBuf> {
    let path = safe_path(path, label)?;
    let meta = path.metadata().map_err(io_error)?;
    if !meta.is_file() || meta.permissions().mode() & 0o002 != 0 {
        Err(ServedModelError(format!(
            "{label} is not a safe regular file"
        )))
    } else {
        Ok(path)
    }
}
fn safe_path(path: &Path, label: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(io_error)?.join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) => {
                return Err(ServedModelError(format!("{label} has unsupported prefix")));
            }
            _ => current.push(component.as_os_str()),
        }
        let meta = fs::symlink_metadata(&current)
            .map_err(|_| ServedModelError(format!("{label} is absent or unreadable")))?;
        if meta.file_type().is_symlink() {
            return Err(ServedModelError(format!("{label} traverses a symlink")));
        }
    }
    absolute.canonicalize().map_err(io_error)
}
fn contained_regular_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf> {
    let target = safe_regular_file(&root.join(relative), label)?;
    if !target.starts_with(root) {
        Err(ServedModelError(format!("{label} escapes its root")))
    } else {
        Ok(target)
    }
}
fn bounded_read(path: &Path, maximum: usize, label: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(io_error)?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() > maximum {
        Err(ServedModelError(format!("{label} exceeds its size limit")))
    } else {
        Ok(bytes)
    }
}
fn verify_file_sha256(path: &Path, expected: &str, label: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ServedModelError(format!("{label} SHA-256 differs")))
    }
}
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(io_error)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(io_error)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn sha256_bytes(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}
fn io_error(error: std::io::Error) -> ServedModelError {
    ServedModelError(format!("resource I/O failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../services/openai-gateway/tests/fixtures/served-model")
            .join(name)
            .join("served-model.json")
    }

    struct AuthorizationFixture {
        root: PathBuf,
        manifest_path: PathBuf,
        value: Value,
    }

    impl Drop for AuthorizationFixture {
        fn drop(&mut self) {
            for path in [
                self.root.join("audit.json"),
                self.root.join("lineage-input.json"),
                self.root.join("lineage-runtime.json"),
            ] {
                if path.exists() {
                    let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
                }
            }
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn write_immutable(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
    }

    fn authorized_fixture(audit_schema: &str, audit_verdict: &str) -> AuthorizationFixture {
        let root = std::env::temp_dir().join(format!(
            "ullm-served-model-authorization-{}-{}",
            std::process::id(),
            TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let source_commit = "a".repeat(40);
        let sha_b = "b".repeat(64);
        let sha_c = "c".repeat(40);
        let sha_d = "d".repeat(64);
        let entries = json!([{}, {}, {}, {}, {}, {}]);
        let entries_sha256 = sha256_bytes(&serde_json::to_vec(&entries).unwrap());
        let lineage = json!({
            "schema_version": "ullm.sq8_authorization_lineage_input.v1",
            "disposition": "authorization_input_not_yet_runtime_bound",
            "source": {
                "archive_sha256": sha_b,
                "commit": source_commit,
                "tree_oid": sha_c,
            },
            "entries": entries,
        });
        let lineage_bytes = serde_json::to_vec(&lineage).unwrap();
        let lineage_sha256 = sha256_bytes(&lineage_bytes);
        let lineage_input = root.join("lineage-input.json");
        let lineage_runtime = root.join("lineage-runtime.json");
        write_immutable(&lineage_input, &lineage_bytes);
        write_immutable(&lineage_runtime, &lineage_bytes);
        let reference = |name: &str| json!({"path": format!("/tmp/{name}"), "sha256": sha_d});
        let audit = json!({
            "schema_version": audit_schema,
            "auditor_task_id": "fixture-auditor",
            "audited_at_utc": "2026-07-16T00:00:00Z",
            "audited_source": {
                "commit": source_commit,
                "tree_sha256": sha_c,
                "archive_sha256": sha_b,
            },
            "runtime": {
                "path": "/tmp/unauthorized-runtime",
                "gate": reference("gate.json"),
                "worker": reference("worker"),
                "profile": reference("profile.json"),
                "served_model": reference("served-model.json"),
                "prepared_receipt": reference("promotion-receipt.json"),
                "binding": {
                    "path": "/tmp/binding.json",
                    "sha256": sha_d,
                    "content_sha256": sha_d,
                    "tensor_set_sha256": sha_d,
                    "tensor_count": 48,
                },
                "package": reference("package.json"),
                "authorization_lineage_manifest": reference("lineage.json"),
                "sha256sums": reference("SHA256SUMS"),
            },
            "fixed_request_id": format!("sq8-promotion-{}", sha_d),
            "gate_state": {
                "status": "ready_for_independent_audit",
                "actual_run_allowed": false,
                "prepared_receipt_status": "prepared_not_executed",
                "prepared_receipt_actual": {"status": "pending", "required": true},
            },
            "topology": {
                "artifact_directory_count": 3,
                "artifact_payload_and_scale_files_hashed": 96,
                "artifact_regular_file_bytes": 1,
                "artifact_regular_file_count": 98,
                "current_runtime_reference_count": 1,
                "executable_file_mode": "0555",
                "historical_runtime_reference_count": 0,
                "package_directory_count": 1,
                "package_regular_file_count": 1,
                "regular_file_mode": "0444",
                "regular_file_nlink": 1,
                "runtime_directory_mode": "0555",
                "runtime_directory_nlink": 2,
                "runtime_member_count": 8,
                "special_file_count": 0,
                "symlink_count": 0,
                "worker_source_and_immutable_are_runtime_self": true,
            },
            "verdict": audit_verdict,
            "actual": "not_executed",
            "tests": {
                "actual_output": "absent",
                "artifact_live_content": "passed",
                "authorization_boundary": "passed",
                "bridge_readiness_binding": "passed",
                "candidate_wrapper_dry_run": "passed",
                "fixed_request_id_recomputation": "passed",
                "formal_lineage_manifest": "passed",
                "gpu_or_service_execution": false,
                "historical_runtime_references": "zero",
                "lineage_external_runtime_copy": "passed",
                "package_live_identity": "passed",
                "runtime_modes_links_and_symlinks": "passed",
                "runtime_sha256sums": "passed",
                "source_commit_tree_archive": "passed",
                "source_worktree": "clean",
                "sudo_execution": false,
                "worker_live_identity": "passed",
                "worker_runtime_self_identity": "passed",
            },
        });
        let audit_path = root.join("audit.json");
        let audit_bytes = serde_json::to_vec(&audit).unwrap();
        write_immutable(&audit_path, &audit_bytes);
        let mut value = serde_json::from_slice::<Value>(
            &bounded_read(&fixture("aq4"), MAX_MANIFEST_BYTES, "fixture").unwrap(),
        )
        .unwrap();
        let promotion = value["promotion"].as_object_mut().unwrap();
        promotion.insert("source_commit".into(), Value::String(source_commit));
        promotion.insert(
            "authorization_audit".into(),
            json!({"path": audit_path, "sha256": sha256_bytes(&audit_bytes)}),
        );
        promotion.insert(
            "authorization_lineage".into(),
            json!({
                "schema_version": "ullm.sq8_authorization_lineage_ref.v1",
                "input_path": lineage_input,
                "runtime_path": lineage_runtime,
                "sha256": lineage_sha256,
                "entries_sha256": entries_sha256,
            }),
        );
        let body = r#"{"status":"ready"}"#;
        promotion.insert(
            "readiness".into(),
            json!({
                "schema": "ullm.bridge_container_readiness.v1",
                "container": {
                    "name": "open-webui",
                    "id": "1".repeat(64),
                    "image_id": format!("sha256:{}", "2".repeat(64)),
                    "config_image": "ullm/open-webui:test",
                },
                "network": {
                    "name": "open-webui-network",
                    "id": "3".repeat(64),
                    "driver": "bridge",
                    "bridge_interface": format!("br-{}", "3".repeat(12)),
                },
                "endpoint": {
                    "url": "http://172.20.0.1:8000/readyz",
                    "path": "/readyz",
                    "expected_status": 200,
                    "expected_body": body,
                    "expected_body_sha256": sha256_bytes(body.as_bytes()),
                    "timeout_seconds": 5,
                },
            }),
        );
        AuthorizationFixture {
            root,
            manifest_path: fixture("aq4"),
            value,
        }
    }

    #[test]
    fn sq8_and_aq4_gateway_fixtures_use_the_same_loader() {
        let sq8 = load_served_model(fixture("sq8")).unwrap();
        let aq4 = load_served_model(fixture("aq4")).unwrap();
        assert_eq!(sq8.format.format_id, "SQ8_0");
        assert!(sq8.product.artifact.is_some());
        assert_eq!(sq8.generation.vocab_size, 151_936);
        assert_eq!(aq4.format.format_id, "AQ4_0");
        assert!(aq4.product.artifact.is_none());
        assert_eq!(aq4.generation.vocab_size, 248_320);
        assert_eq!(
            aq4.profile_snapshot().artifact_content_sha256,
            aq4.product.package.manifest_sha256
        );
    }

    #[test]
    fn authorized_promotion_contract_is_typed_and_fail_closed() {
        let authorized = authorized_fixture(
            "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "implementation_ready",
        );
        let raw = serde_json::to_vec(&authorized.value).unwrap();
        let model = load_served_model_bytes(&authorized.manifest_path, &raw).unwrap();
        assert!(model.promotion.authorization_audit.is_some());
        assert!(model.promotion.authorization_lineage.is_some());
        assert!(model.promotion.readiness.is_some());

        let mut cases = Vec::new();
        let mut value = authorized.value.clone();
        value["promotion"]["authorization_audit"]
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), Value::Bool(true));
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]
            .as_object_mut()
            .unwrap()
            .remove("readiness");
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["readiness"]["endpoint"]["expected_status"] =
            Value::String("200".into());
        cases.push(value);

        let mut value = authorized.value.clone();
        let audit_path = value["promotion"]["authorization_audit"]["path"]
            .as_str()
            .unwrap()
            .to_string();
        let directory = authorized.root.file_name().unwrap().to_string_lossy();
        value["promotion"]["authorization_audit"]["path"] = Value::String(format!(
            "{}/../{directory}/audit.json",
            authorized.root.display()
        ));
        assert_ne!(
            value["promotion"]["authorization_audit"]["path"],
            Value::String(audit_path)
        );
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["authorization_audit"]["sha256"] = Value::String("0".repeat(64));
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["authorization_lineage"]["entries_sha256"] =
            Value::String("0".repeat(64));
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["readiness"]["endpoint"]["url"] =
            Value::String("http://127.0.0.1:8000/readyz".into());
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["readiness"]["endpoint"]["expected_body_sha256"] =
            Value::String("0".repeat(64));
        cases.push(value);

        let mut value = authorized.value.clone();
        value["promotion"]["authorization_lineage"]
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), Value::Bool(true));
        cases.push(value);

        for value in cases {
            let raw = serde_json::to_vec(&value).unwrap();
            assert!(load_served_model_bytes(&authorized.manifest_path, &raw).is_err());
        }

        let raw = String::from_utf8(serde_json::to_vec(&authorized.value).unwrap()).unwrap();
        let duplicate = raw.replacen(
            r#""authorization_audit":{"#,
            r#""authorization_audit":{"path":"/tmp/duplicate","#,
            1,
        );
        assert!(decode_strict_json(duplicate.as_bytes()).is_err());

        let bad_status = authorized_fixture(
            "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "implementation_no_go",
        );
        let raw = serde_json::to_vec(&bad_status.value).unwrap();
        assert!(load_served_model_bytes(&bad_status.manifest_path, &raw).is_err());

        let bad_schema = authorized_fixture(
            "ullm.qwen35_aq4_sq8_overlay_independent_audit.v2",
            "implementation_ready",
        );
        let raw = serde_json::to_vec(&bad_schema.value).unwrap();
        assert!(load_served_model_bytes(&bad_schema.manifest_path, &raw).is_err());
    }

    #[test]
    fn actual_failed_authorized_sq8_manifest_is_accepted_when_available() {
        let path = PathBuf::from(
            "/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-6fef8baafda003b5/served-model.json",
        );
        if !path.exists() {
            return;
        }
        assert_eq!(
            sha256_file(&path).unwrap(),
            "a4d541a8c44edd73e505f223b15cf92933b4e0bf2a257e8e9d08dbad94192542"
        );
        let model = load_served_model(path).unwrap();
        assert!(model.promotion.authorization_audit.is_some());
        assert!(model.promotion.authorization_lineage.is_some());
        assert!(model.promotion.readiness.is_some());
    }

    #[test]
    fn strict_json_rejects_duplicate_unknown_and_bounds() {
        assert!(decode_strict_json(br#"{"a":1,"a":2}"#).is_err());
        let mut value = serde_json::from_slice::<Value>(
            &bounded_read(&fixture("sq8"), MAX_MANIFEST_BYTES, "fixture").unwrap(),
        )
        .unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), Value::Null);
        assert!(validate_exact_shape(&value).is_err());
        assert!(
            decode_strict_json(format!("{}0{}", "[".repeat(17), "]".repeat(17)).as_bytes())
                .is_err()
        );
    }

    #[test]
    fn generation_cross_contract_is_fail_closed() {
        let public = PublicModel {
            id: "m".into(),
            name: "m".into(),
            description: "m".into(),
            upstream_id: "m".into(),
            revision: "r".into(),
            context_length: 8,
        };
        let invalid = RawGeneration {
            max_completion_tokens: 9,
            vocab_size: 4,
            eos_token_ids: vec![4],
            sampling: RawSampling {
                top_k: 2,
                temperature: false,
                top_p: true,
            },
        };
        assert!(parse_generation(invalid, &public).is_err());
    }
}
