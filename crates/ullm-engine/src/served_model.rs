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
    pub schema_version: String,
    pub entry_count: Option<usize>,
    pub current_implementation_audit: Option<AuthorizationAuditIdentity>,
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
#[serde(tag = "schema_version")]
enum RawAuthorizationLineageIdentity {
    #[serde(rename = "ullm.sq8_authorization_lineage_ref.v1")]
    V1(RawAuthorizationLineageReferenceV1),
    #[serde(rename = "ullm.sq8_authorization_lineage_ref.v2")]
    V2(RawAuthorizationLineageReferenceV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageReferenceV1 {
    input_path: String,
    runtime_path: String,
    sha256: String,
    entries_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageReferenceV2 {
    input_path: String,
    runtime_path: String,
    sha256: String,
    entries_sha256: String,
    entry_count: usize,
    current_implementation_audit: RawAuthorizationAuditIdentity,
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
#[serde(untagged)]
enum RawAuditTopology {
    Legacy(RawAuditTopologyLegacy),
    MigratedV2(RawAuditTopologyMigratedV2),
    CurrentV2(RawAuditTopologyCurrentV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditTopologyLegacy {
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
struct RawAuditTopologyMigratedV2 {
    artifact_directory_count: usize,
    artifact_payload_and_scale_bytes_hashed: u64,
    artifact_payload_and_scale_files_hashed: usize,
    artifact_regular_file_bytes: u64,
    artifact_regular_file_count: usize,
    current_runtime_reference_count: usize,
    executable_file_mode: String,
    historical_runtime_reference_count: usize,
    package_directory_count: usize,
    package_regular_file_bytes: u64,
    package_regular_file_count: usize,
    regular_file_mode: String,
    regular_file_nlink: u64,
    runtime_directory_mode: String,
    runtime_directory_nlink: u64,
    runtime_member_count: usize,
    special_file_count: usize,
    symlink_count: usize,
    worker_source_and_immutable_are_runtime_self: bool,
    authorization_lineage_entries_sha256: String,
    authorization_lineage_entry_count: usize,
    authorization_lineage_migrated_prefix_count: usize,
    authorization_lineage_migrated_prefix_sha256: String,
    authorization_lineage_propagation_target_count: usize,
    authorization_lineage_schema: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditTopologyCurrentV2 {
    artifact_directory_count: usize,
    artifact_payload_and_scale_bytes_hashed: u64,
    artifact_payload_and_scale_files_hashed: usize,
    artifact_regular_file_bytes: u64,
    artifact_regular_file_count: usize,
    authorization_lineage_entries_sha256: String,
    authorization_lineage_entry_count: usize,
    authorization_lineage_predecessor_entries_sha256: String,
    authorization_lineage_predecessor_entry_count: usize,
    authorization_lineage_propagation_target_count: usize,
    authorization_lineage_schema: String,
    current_runtime_reference_count: usize,
    executable_file_mode: String,
    historical_direct_authorization_reference_count: usize,
    historical_runtime_reference_count: usize,
    package_directory_count: usize,
    package_regular_file_bytes: u64,
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
#[serde(untagged)]
enum RawAuditTests {
    Legacy(RawAuditTestsLegacy),
    MigratedV2(RawAuditTestsMigratedV2),
    CurrentV2(RawAuditTestsCurrentV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuditTestsLegacy {
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
struct RawAuditTestsMigratedV2 {
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
    lineage_v1_authorization_rejection: String,
    lineage_v1_migration: String,
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
struct RawAuditTestsCurrentV2 {
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
    lineage_old_v2_authorization_rejection: String,
    lineage_v1_authorization_rejection: String,
    lineage_v2_successor: String,
    package_live_identity: String,
    runtime_modes_links_and_symlinks: String,
    runtime_sha256sums: String,
    served_model_cpu_validation: String,
    source_commit_tree_archive: String,
    source_worktree: String,
    sudo_execution: bool,
    worker_runtime_self_identity: String,
}

impl RawAuditTests {
    fn recorded_forbidden_execution(&self) -> bool {
        match self {
            Self::Legacy(tests) => tests.gpu_or_service_execution || tests.sudo_execution,
            Self::MigratedV2(tests) => tests.gpu_or_service_execution || tests.sudo_execution,
            Self::CurrentV2(tests) => tests.gpu_or_service_execution || tests.sudo_execution,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "schema_version")]
enum RawAuthorizationLineageManifest {
    #[serde(rename = "ullm.sq8_authorization_lineage_input.v1")]
    V1(RawAuthorizationLineageManifestV1),
    #[serde(rename = "ullm.sq8_authorization_lineage_input.v2")]
    V2(RawAuthorizationLineageManifestV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageManifestV1 {
    disposition: String,
    source: RawAuthorizationLineageSource,
    entries: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageManifestV2 {
    disposition: String,
    source: RawAuthorizationLineageSource,
    predecessor: RawAuthorizationLineagePredecessor,
    entries: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(tag = "schema_version")]
enum RawAuthorizationLineagePredecessor {
    #[serde(rename = "ullm.sq8_authorization_lineage_input.v1")]
    V1(RawAuthorizationLineagePredecessorV1),
    #[serde(rename = "ullm.sq8_authorization_lineage_input.v2")]
    V2(RawAuthorizationLineagePredecessorV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineagePredecessorV1 {
    path: String,
    sha256: String,
    migrated_prefix_sha256: String,
    migrated_prefix_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineagePredecessorV2 {
    path: String,
    sha256: String,
    entries_sha256: String,
    entry_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuthorizationLineageEntryV2 {
    sequence: usize,
    relation: String,
    path: String,
    sha256: String,
    schema_version: String,
    status: String,
    request_id: Option<String>,
    source_commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AuditLineageBinding {
    MigratedV2 {
        manifest_sha256: String,
        entries_sha256: String,
        entry_count: usize,
        migrated_prefix_count: usize,
        migrated_prefix_sha256: String,
    },
    CurrentV2 {
        manifest_sha256: String,
        entries_sha256: String,
        entry_count: usize,
        predecessor_entry_count: usize,
        predecessor_entries_sha256: String,
    },
}

struct ParsedAuthorizationAudit {
    identity: AuthorizationAuditIdentity,
    lineage: Option<AuditLineageBinding>,
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
) -> Result<ParsedAuthorizationAudit> {
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
        || audit.tests.recorded_forbidden_execution()
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
    let runtime_lineage_sha256 = audit.runtime.authorization_lineage_manifest.sha256.clone();
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
    macro_rules! validate_common_topology {
        ($topology:expr) => {
            if $topology.artifact_directory_count != 3
                || $topology.artifact_payload_and_scale_files_hashed != 96
                || $topology.artifact_regular_file_bytes == 0
                || $topology.artifact_regular_file_count != 98
                || $topology.current_runtime_reference_count == 0
                || $topology.executable_file_mode != "0555"
                || $topology.historical_runtime_reference_count != 0
                || $topology.package_directory_count == 0
                || $topology.package_regular_file_count == 0
                || $topology.regular_file_mode != "0444"
                || $topology.regular_file_nlink != 1
                || $topology.runtime_directory_mode != "0555"
                || $topology.runtime_directory_nlink != 2
                || $topology.runtime_member_count != 8
                || $topology.special_file_count != 0
                || $topology.symlink_count != 0
                || !$topology.worker_source_and_immutable_are_runtime_self
            {
                return Err(ServedModelError(
                    "promotion.authorization_audit topology differs".into(),
                ));
            }
        };
    }
    macro_rules! validate_common_tests {
        ($tests:expr) => {
            for (text, label) in [
                ($tests.actual_output, "actual_output"),
                ($tests.artifact_live_content, "artifact_live_content"),
                ($tests.authorization_boundary, "authorization_boundary"),
                ($tests.bridge_readiness_binding, "bridge_readiness_binding"),
                (
                    $tests.candidate_wrapper_dry_run,
                    "candidate_wrapper_dry_run",
                ),
                (
                    $tests.fixed_request_id_recomputation,
                    "fixed_request_id_recomputation",
                ),
                ($tests.formal_lineage_manifest, "formal_lineage_manifest"),
                (
                    $tests.historical_runtime_references,
                    "historical_runtime_references",
                ),
                (
                    $tests.lineage_external_runtime_copy,
                    "lineage_external_runtime_copy",
                ),
                ($tests.package_live_identity, "package_live_identity"),
                (
                    $tests.runtime_modes_links_and_symlinks,
                    "runtime_modes_links_and_symlinks",
                ),
                ($tests.runtime_sha256sums, "runtime_sha256sums"),
                (
                    $tests.source_commit_tree_archive,
                    "source_commit_tree_archive",
                ),
                ($tests.source_worktree, "source_worktree"),
                (
                    $tests.worker_runtime_self_identity,
                    "worker_runtime_self_identity",
                ),
            ] {
                bounded_text(
                    text,
                    &format!("promotion.authorization_audit.tests.{label}"),
                    4096,
                )?;
            }
        };
    }
    let lineage = match (audit.topology, audit.tests) {
        (RawAuditTopology::Legacy(topology), RawAuditTests::Legacy(tests)) => {
            validate_common_topology!(topology);
            validate_common_tests!(tests);
            bounded_text(
                tests.worker_live_identity,
                "promotion.authorization_audit.tests.worker_live_identity",
                4096,
            )?;
            None
        }
        (RawAuditTopology::MigratedV2(topology), RawAuditTests::MigratedV2(tests)) => {
            validate_common_topology!(topology);
            validate_common_tests!(tests);
            for (text, label) in [
                (tests.worker_live_identity, "worker_live_identity"),
                (
                    tests.lineage_v1_authorization_rejection,
                    "lineage_v1_authorization_rejection",
                ),
                (tests.lineage_v1_migration, "lineage_v1_migration"),
            ] {
                bounded_text(
                    text,
                    &format!("promotion.authorization_audit.tests.{label}"),
                    4096,
                )?;
            }
            let entries_sha256 = validate_sha256(
                topology.authorization_lineage_entries_sha256,
                "promotion.authorization_audit.topology.authorization_lineage_entries_sha256",
            )?;
            let migrated_prefix_sha256 = validate_sha256(
                topology.authorization_lineage_migrated_prefix_sha256,
                "promotion.authorization_audit.topology.authorization_lineage_migrated_prefix_sha256",
            )?;
            if topology.artifact_payload_and_scale_bytes_hashed == 0
                || topology.package_regular_file_bytes == 0
                || topology.authorization_lineage_propagation_target_count != 5
                || topology.authorization_lineage_schema
                    != "ullm.sq8_authorization_lineage_input.v2"
                || topology.authorization_lineage_entry_count < 8
                || topology.authorization_lineage_migrated_prefix_count != 6
            {
                return Err(ServedModelError(
                    "promotion.authorization_audit lineage topology differs".into(),
                ));
            }
            Some(AuditLineageBinding::MigratedV2 {
                manifest_sha256: validate_sha256(
                    runtime_lineage_sha256,
                    "promotion.authorization_audit.runtime.authorization_lineage_manifest.sha256",
                )?,
                entries_sha256,
                entry_count: topology.authorization_lineage_entry_count,
                migrated_prefix_count: topology.authorization_lineage_migrated_prefix_count,
                migrated_prefix_sha256,
            })
        }
        (RawAuditTopology::CurrentV2(topology), RawAuditTests::CurrentV2(tests)) => {
            validate_common_topology!(topology);
            validate_common_tests!(tests);
            for (text, label) in [
                (tests.lineage_v2_successor, "lineage_v2_successor"),
                (
                    tests.lineage_old_v2_authorization_rejection,
                    "lineage_old_v2_authorization_rejection",
                ),
                (
                    tests.lineage_v1_authorization_rejection,
                    "lineage_v1_authorization_rejection",
                ),
                (
                    tests.served_model_cpu_validation,
                    "served_model_cpu_validation",
                ),
            ] {
                bounded_text(
                    text,
                    &format!("promotion.authorization_audit.tests.{label}"),
                    4096,
                )?;
            }
            let entries_sha256 = validate_sha256(
                topology.authorization_lineage_entries_sha256,
                "promotion.authorization_audit.topology.authorization_lineage_entries_sha256",
            )?;
            let predecessor_entries_sha256 = validate_sha256(
                topology.authorization_lineage_predecessor_entries_sha256,
                "promotion.authorization_audit.topology.authorization_lineage_predecessor_entries_sha256",
            )?;
            if topology.artifact_payload_and_scale_bytes_hashed == 0
                || topology.package_regular_file_bytes == 0
                || topology.authorization_lineage_propagation_target_count != 5
                || topology.authorization_lineage_schema
                    != "ullm.sq8_authorization_lineage_input.v2"
                || topology.historical_direct_authorization_reference_count != 0
                || topology.authorization_lineage_predecessor_entry_count < 8
                || topology
                    .authorization_lineage_predecessor_entry_count
                    .checked_add(2)
                    != Some(topology.authorization_lineage_entry_count)
            {
                return Err(ServedModelError(
                    "promotion.authorization_audit lineage topology differs".into(),
                ));
            }
            Some(AuditLineageBinding::CurrentV2 {
                manifest_sha256: validate_sha256(
                    runtime_lineage_sha256,
                    "promotion.authorization_audit.runtime.authorization_lineage_manifest.sha256",
                )?,
                entries_sha256,
                entry_count: topology.authorization_lineage_entry_count,
                predecessor_entry_count: topology.authorization_lineage_predecessor_entry_count,
                predecessor_entries_sha256,
            })
        }
        _ => {
            return Err(ServedModelError(
                "promotion.authorization_audit topology/tests variant differs".into(),
            ));
        }
    };
    Ok(ParsedAuthorizationAudit {
        identity: AuthorizationAuditIdentity { path, sha256 },
        lineage,
    })
}

#[derive(Clone)]
struct ValidatedLineageDocument {
    schema_version: &'static str,
    source_commit: String,
    source_tree: String,
    source_archive: String,
    entries: Vec<Value>,
    entries_sha256: String,
    current_implementation_audit: Option<AuthorizationAuditIdentity>,
    migrated_prefix: Option<(usize, String)>,
    v2_predecessor: Option<(usize, String)>,
}

struct ParsedAuthorizationLineage {
    identity: AuthorizationLineageIdentity,
    migrated_prefix: Option<(usize, String)>,
    v2_predecessor: Option<(usize, String)>,
}

fn canonical_json_sha256(value: &Value, label: &str) -> Result<String> {
    let encoded = serde_json::to_vec(value)
        .map_err(|_| ServedModelError(format!("{label} is not canonical JSON")))?;
    Ok(sha256_bytes(&encoded))
}

fn lineage_source(
    raw: RawAuthorizationLineageSource,
    expected_commit: Option<&str>,
) -> Result<(String, String, String)> {
    let commit = validate_hex40(raw.commit, "promotion.authorization_lineage.source.commit")?;
    let tree = validate_hex40(
        raw.tree_oid,
        "promotion.authorization_lineage.source.tree_oid",
    )?;
    let archive = validate_sha256(
        raw.archive_sha256,
        "promotion.authorization_lineage.source.archive_sha256",
    )?;
    if expected_commit.is_some_and(|expected| expected != commit) {
        return Err(ServedModelError(
            "promotion.authorization_lineage source differs".into(),
        ));
    }
    Ok((commit, tree, archive))
}

fn lineage_live_document(path: String, digest: String, label: &str) -> Result<(PathBuf, Value)> {
    let path = canonical_absolute_regular_file(path, &format!("{label}.path"), true)?;
    let digest = validate_sha256(digest, &format!("{label}.sha256"))?;
    verify_file_sha256(&path, &digest, label)?;
    let bytes = bounded_read(&path, MAX_MANIFEST_BYTES, label)?;
    Ok((path, decode_strict_json(&bytes)?))
}

fn lineage_entry_source(entry: &RawAuthorizationLineageEntryV2, index: usize) -> Result<Value> {
    let (_, source) = lineage_live_document(
        entry.path.clone(),
        entry.sha256.clone(),
        &format!("promotion.authorization_lineage entry {index}"),
    )?;
    if source.get("schema_version").and_then(Value::as_str) != Some(entry.schema_version.as_str()) {
        return Err(ServedModelError(
            "promotion.authorization_lineage entry schema differs".into(),
        ));
    }
    Ok(source)
}

fn source_commit_from_receipt(source: &Value, schema: &str) -> Option<String> {
    let value = if schema == "ullm.qwen35_aq4_sq8_overlay_promotion.v1" {
        source.get("source_commit")
    } else {
        source
            .get("audited_source")
            .and_then(|value| value.get("commit"))
    };
    value.and_then(Value::as_str).map(str::to_owned)
}

fn parse_v2_entry(value: &Value, index: usize) -> Result<(RawAuthorizationLineageEntryV2, Value)> {
    let entry: RawAuthorizationLineageEntryV2 =
        serde_json::from_value(value.clone()).map_err(|_| {
            ServedModelError(
                "promotion.authorization_lineage v2 entry typed schema is invalid".into(),
            )
        })?;
    if entry.sequence != index {
        return Err(ServedModelError(
            "promotion.authorization_lineage v2 sequence differs".into(),
        ));
    }
    validate_sha256(
        entry.sha256.clone(),
        "promotion.authorization_lineage entry SHA-256",
    )?;
    validate_hex40(
        entry.source_commit.clone(),
        "promotion.authorization_lineage entry source commit",
    )?;
    if let Some(request_id) = entry.request_id.clone() {
        validate_request_id(
            request_id,
            "promotion.authorization_lineage entry request ID",
        )?;
    }
    let source = lineage_entry_source(&entry, index)?;
    if source_commit_from_receipt(&source, &entry.schema_version).as_deref()
        != Some(entry.source_commit.as_str())
    {
        return Err(ServedModelError(
            "promotion.authorization_lineage entry source differs".into(),
        ));
    }
    let status = source.get("status").and_then(Value::as_str);
    let verdict = source.get("verdict").and_then(Value::as_str);
    let actual = source.get("actual");
    match entry.relation.as_str() {
        "actual_failure" => {
            if entry.schema_version != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
                || entry.status != "actual_failed"
                || entry.request_id.is_none()
                || status != Some(entry.status.as_str())
                || source.get("request_id").and_then(Value::as_str) != entry.request_id.as_deref()
                || actual
                    .and_then(|value| value.get("status"))
                    .and_then(Value::as_str)
                    != Some("failed")
                || actual
                    .and_then(|value| value.get("request_id"))
                    .and_then(Value::as_str)
                    != entry.request_id.as_deref()
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage actual failure differs".into(),
                ));
            }
        }
        "implementation_ready_current"
        | "capture_implementation_no_go"
        | "restore_implementation_no_go"
        | "historical_implementation_audit"
        | "historical_runtime_audit" => {
            let capture_schema = "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1";
            let runtime_schema = "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1";
            if ![capture_schema, runtime_schema].contains(&entry.schema_version.as_str())
                || verdict != Some(entry.status.as_str())
                || actual.and_then(Value::as_str) != Some("not_executed")
                || entry.request_id.as_ref().is_some_and(|request_id| {
                    source.get("fixed_request_id").and_then(Value::as_str)
                        != Some(request_id.as_str())
                })
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage audit entry differs".into(),
                ));
            }
            let relation_ok = match entry.relation.as_str() {
                "implementation_ready_current" => {
                    entry.status == "implementation_ready"
                        && (entry.schema_version != capture_schema
                            || source
                                .get("authorization")
                                .and_then(|value| {
                                    value.get("eligible_for_fresh_authorization_builder")
                                })
                                .and_then(Value::as_bool)
                                == Some(true))
                }
                "capture_implementation_no_go" => {
                    entry.schema_version == capture_schema && entry.status == "implementation_no_go"
                }
                "restore_implementation_no_go" => {
                    entry.schema_version == runtime_schema
                        && entry.status == "implementation_no_go"
                        && source.get("reason_code").and_then(Value::as_str)
                            == Some("restore_retry_terminal_identity_not_fail_closed")
                }
                "historical_implementation_audit" => {
                    entry.schema_version == capture_schema
                        && ["implementation_ready", "implementation_no_go"]
                            .contains(&entry.status.as_str())
                }
                "historical_runtime_audit" => {
                    entry.schema_version == runtime_schema
                        && ["implementation_ready", "implementation_no_go"]
                            .contains(&entry.status.as_str())
                }
                _ => false,
            };
            if !relation_ok {
                return Err(ServedModelError(
                    "promotion.authorization_lineage entry relation differs".into(),
                ));
            }
        }
        _ => {
            return Err(ServedModelError(
                "promotion.authorization_lineage entry relation differs".into(),
            ));
        }
    }
    Ok((entry, source))
}

fn validate_v1_entry_semantics(
    object: &serde_json::Map<String, Value>,
    source: &Value,
    sequence: usize,
    schema: &str,
) -> Result<()> {
    match sequence {
        0 => {
            if schema != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
                || object.get("verdict").and_then(Value::as_str) != Some("implementation_ready")
                || object.get("actual").and_then(Value::as_str) != Some("not_executed")
                || source.get("verdict") != object.get("verdict")
                || source.get("actual") != object.get("actual")
                || source
                    .get("authorization")
                    .and_then(Value::as_object)
                    .and_then(|authorization| {
                        authorization
                            .get("eligible_for_fresh_authorization_builder")
                            .and_then(Value::as_bool)
                    })
                    != Some(true)
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 implementation GO differs".into(),
                ));
            }
        }
        1 | 2 => {
            let reason_codes = object
                .get("reason_codes")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    ServedModelError(
                        "promotion.authorization_lineage v1 capture No-Go differs".into(),
                    )
                })?;
            let mut seen = HashSet::new();
            let reason_codes_are_valid = !reason_codes.is_empty()
                && reason_codes.iter().all(|reason_code| {
                    reason_code.as_str().is_some_and(|reason_code| {
                        !reason_code.is_empty() && seen.insert(reason_code)
                    })
                });
            if schema != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
                || object.get("verdict").and_then(Value::as_str) != Some("implementation_no_go")
                || object.get("actual").and_then(Value::as_str) != Some("not_executed")
                || !reason_codes_are_valid
                || source.get("verdict") != object.get("verdict")
                || source.get("actual") != object.get("actual")
                || source.get("reason_codes") != object.get("reason_codes")
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 capture No-Go differs".into(),
                ));
            }
        }
        3 | 4 => {
            let actual = source.get("actual").and_then(Value::as_object);
            if schema != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
                || object.get("status").and_then(Value::as_str) != Some("actual_failed")
                || object.get("actual_status").and_then(Value::as_str) != Some("failed")
                || source.get("status") != object.get("status")
                || source.get("request_id") != object.get("request_id")
                || actual.and_then(|actual| actual.get("status")) != object.get("actual_status")
                || actual.and_then(|actual| actual.get("request_id")) != object.get("request_id")
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 actual failure differs".into(),
                ));
            }
        }
        5 => {
            if schema != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
                || object.get("verdict").and_then(Value::as_str) != Some("implementation_no_go")
                || object.get("actual").and_then(Value::as_str) != Some("not_executed")
                || object.get("reason_code").and_then(Value::as_str)
                    != Some("restore_retry_terminal_identity_not_fail_closed")
                || source.get("verdict") != object.get("verdict")
                || source.get("actual") != object.get("actual")
                || source.get("reason_code") != object.get("reason_code")
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 restore No-Go differs".into(),
                ));
            }
        }
        _ => unreachable!("v1 lineage entry count is validated before migration"),
    }
    Ok(())
}

fn migrate_v1_entries(entries: &[Value]) -> Result<Vec<Value>> {
    const OLD_RELATIONS: [&str; 6] = [
        "implementation_go_eligible_for_fresh_runtime_audit",
        "superseded_capture_implementation_no_go",
        "superseded_capture_implementation_no_go",
        "consumed_actual_failure_latest",
        "consumed_actual_failure_predecessor",
        "superseded_restore_implementation_no_go",
    ];
    const NEW_RELATIONS: [&str; 6] = [
        "historical_implementation_audit",
        "capture_implementation_no_go",
        "capture_implementation_no_go",
        "actual_failure",
        "actual_failure",
        "restore_implementation_no_go",
    ];
    if entries.len() != 6 {
        return Err(ServedModelError(
            "promotion.authorization_lineage v1 entry count differs".into(),
        ));
    }
    entries
        .iter()
        .enumerate()
        .map(|(sequence, value)| {
            let object = value.as_object().ok_or_else(|| {
                ServedModelError("promotion.authorization_lineage v1 entry differs".into())
            })?;
            let mut expected = vec![
                "relation",
                "path",
                "sha256",
                "schema_version",
                "consumed",
                "reusable_as_runtime_authorization",
            ];
            match sequence {
                0 => expected.extend(["verdict", "actual"]),
                1 | 2 => expected.extend(["verdict", "actual", "reason_codes"]),
                3 | 4 => expected.extend(["status", "actual_status", "request_id"]),
                _ => expected.extend(["verdict", "actual", "reason_code"]),
            }
            exact_keys(value, &expected, "promotion.authorization_lineage v1 entry")?;
            if object.get("relation").and_then(Value::as_str) != Some(OLD_RELATIONS[sequence]) {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 relation differs".into(),
                ));
            }
            if object
                .get("reusable_as_runtime_authorization")
                .and_then(Value::as_bool)
                != Some(false)
                || object.get("consumed").and_then(Value::as_bool) != Some(sequence != 0)
            {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 disposition differs".into(),
                ));
            }
            let path = object.get("path").and_then(Value::as_str).ok_or_else(|| {
                ServedModelError("promotion.authorization_lineage v1 path differs".into())
            })?;
            let digest = object
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ServedModelError("promotion.authorization_lineage v1 SHA differs".into())
                })?;
            let schema = object
                .get("schema_version")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ServedModelError("promotion.authorization_lineage v1 schema differs".into())
                })?;
            let (_, source) = lineage_live_document(
                path.to_owned(),
                digest.to_owned(),
                "promotion.authorization_lineage v1 entry",
            )?;
            if source.get("schema_version").and_then(Value::as_str) != Some(schema) {
                return Err(ServedModelError(
                    "promotion.authorization_lineage v1 entry schema differs".into(),
                ));
            }
            validate_v1_entry_semantics(object, &source, sequence, schema)?;
            let status = object
                .get("status")
                .or_else(|| object.get("verdict"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ServedModelError("promotion.authorization_lineage v1 status differs".into())
                })?;
            let request_id = object
                .get("request_id")
                .and_then(Value::as_str)
                .or_else(|| source.get("fixed_request_id").and_then(Value::as_str));
            if let Some(request_id) = request_id {
                validate_request_id(
                    request_id.to_owned(),
                    "promotion.authorization_lineage migrated request ID",
                )?;
            }
            let source_commit = source_commit_from_receipt(&source, schema).ok_or_else(|| {
                ServedModelError("promotion.authorization_lineage v1 source differs".into())
            })?;
            validate_hex40(
                source_commit.clone(),
                "promotion.authorization_lineage migrated source commit",
            )?;
            Ok(serde_json::json!({
                "sequence": sequence,
                "relation": NEW_RELATIONS[sequence],
                "path": path,
                "sha256": digest,
                "schema_version": schema,
                "status": status,
                "request_id": request_id,
                "source_commit": source_commit,
            }))
        })
        .collect()
}

fn validate_lineage_file(
    path: &Path,
    digest: &str,
    expected_commit: Option<&str>,
    seen: &mut HashSet<PathBuf>,
) -> Result<ValidatedLineageDocument> {
    let canonical = canonical_absolute_regular_file(
        path.to_string_lossy().into_owned(),
        "promotion.authorization_lineage manifest",
        true,
    )?;
    if !seen.insert(canonical.clone()) {
        return Err(ServedModelError(
            "promotion.authorization_lineage predecessor cycle differs".into(),
        ));
    }
    let result = (|| {
        verify_file_sha256(&canonical, digest, "promotion.authorization_lineage")?;
        let bytes = bounded_read(
            &canonical,
            MAX_MANIFEST_BYTES,
            "promotion.authorization_lineage",
        )?;
        let value = decode_strict_json(&bytes)?;
        let document: RawAuthorizationLineageManifest = serde_json::from_value(value.clone())
            .map_err(|_| {
                ServedModelError(
                    "promotion.authorization_lineage manifest typed schema is invalid".into(),
                )
            })?;
        match document {
            RawAuthorizationLineageManifest::V1(document) => {
                if document.disposition != "authorization_input_not_yet_runtime_bound"
                    || document.entries.len() != 6
                {
                    return Err(ServedModelError(
                        "promotion.authorization_lineage v1 manifest differs".into(),
                    ));
                }
                let (commit, tree, archive) = lineage_source(document.source, expected_commit)?;
                let entries_value = Value::Array(document.entries.clone());
                Ok(ValidatedLineageDocument {
                    schema_version: "ullm.sq8_authorization_lineage_input.v1",
                    source_commit: commit,
                    source_tree: tree,
                    source_archive: archive,
                    entries: document.entries,
                    entries_sha256: canonical_json_sha256(
                        &entries_value,
                        "promotion.authorization_lineage entries",
                    )?,
                    current_implementation_audit: None,
                    migrated_prefix: None,
                    v2_predecessor: None,
                })
            }
            RawAuthorizationLineageManifest::V2(document) => {
                if document.disposition != "authorization_input_not_yet_runtime_bound" {
                    return Err(ServedModelError(
                        "promotion.authorization_lineage v2 manifest differs".into(),
                    ));
                }
                let (commit, tree, archive) = lineage_source(document.source, expected_commit)?;
                let mut paths = HashSet::new();
                let mut digests = HashSet::new();
                let mut parsed = Vec::with_capacity(document.entries.len());
                let mut sources = Vec::with_capacity(document.entries.len());
                let mut current = Vec::new();
                let mut current_sources = HashSet::new();
                let mut capture_no_go = 0usize;
                let mut restore_no_go = 0usize;
                let mut actual_failure = 0usize;
                for (index, value) in document.entries.iter().enumerate() {
                    let (entry, source) = parse_v2_entry(value, index)?;
                    if !paths.insert(entry.path.clone()) || !digests.insert(entry.sha256.clone()) {
                        return Err(ServedModelError(
                            "promotion.authorization_lineage v2 entry is duplicated".into(),
                        ));
                    }
                    match entry.relation.as_str() {
                        "implementation_ready_current" => {
                            if !current_sources.insert(entry.source_commit.clone()) {
                                return Err(ServedModelError(
                                    "promotion.authorization_lineage current GO source is duplicated"
                                        .into(),
                                ));
                            }
                            current.push((
                                index,
                                entry.source_commit.clone(),
                                AuthorizationAuditIdentity {
                                    path: canonical_absolute_regular_file(
                                        entry.path.clone(),
                                        "promotion.authorization_lineage current GO path",
                                        true,
                                    )?,
                                    sha256: entry.sha256.clone(),
                                },
                            ));
                        }
                        "capture_implementation_no_go" => capture_no_go += 1,
                        "restore_implementation_no_go" => restore_no_go += 1,
                        "actual_failure" => actual_failure += 1,
                        _ => {}
                    }
                    parsed.push(entry);
                    sources.push(source);
                }
                if current.is_empty()
                    || capture_no_go < 2
                    || restore_no_go < 1
                    || actual_failure < 3
                {
                    return Err(ServedModelError(
                        "promotion.authorization_lineage v2 minimum history differs".into(),
                    ));
                }
                let (current_index, current_source, _) = current.last().ok_or_else(|| {
                    ServedModelError(
                        "promotion.authorization_lineage v2 minimum history differs".into(),
                    )
                })?;
                if *current_index + 1 != document.entries.len() {
                    return Err(ServedModelError(
                        "promotion.authorization_lineage latest current GO is not final".into(),
                    ));
                }
                if current_source != &commit {
                    return Err(ServedModelError(
                        "promotion.authorization_lineage current GO source differs".into(),
                    ));
                }
                let (migrated_prefix, v2_predecessor) = match document.predecessor {
                    RawAuthorizationLineagePredecessor::V1(predecessor) => {
                        let predecessor_digest = validate_sha256(
                            predecessor.sha256,
                            "promotion.authorization_lineage predecessor SHA-256",
                        )?;
                        let predecessor_path = canonical_absolute_regular_file(
                            predecessor.path,
                            "promotion.authorization_lineage predecessor path",
                            true,
                        )?;
                        let previous = validate_lineage_file(
                            &predecessor_path,
                            &predecessor_digest,
                            None,
                            seen,
                        )?;
                        if previous.schema_version != "ullm.sq8_authorization_lineage_input.v1" {
                            return Err(ServedModelError(
                                "promotion.authorization_lineage migration predecessor differs"
                                    .into(),
                            ));
                        }
                        let migrated = migrate_v1_entries(&previous.entries)?;
                        let migrated_value = Value::Array(migrated.clone());
                        let migrated_sha = canonical_json_sha256(
                            &migrated_value,
                            "promotion.authorization_lineage migrated prefix",
                        )?;
                        if predecessor.migrated_prefix_count != migrated.len()
                            || predecessor.migrated_prefix_sha256 != migrated_sha
                            || document.entries.len() != migrated.len() + 2
                            || document.entries[..migrated.len()] != migrated
                            || parsed.get(6).map(|entry| entry.relation.as_str())
                                != Some("actual_failure")
                            || parsed.get(6).map(|entry| entry.source_commit.as_str())
                                != Some(previous.source_commit.as_str())
                            || sources
                                .get(6)
                                .and_then(|source| source.get("source_provenance"))
                                != Some(&serde_json::json!({
                                    "tree_sha256": previous.source_tree,
                                    "archive_sha256": previous.source_archive,
                                }))
                            || parsed.get(7).map(|entry| entry.relation.as_str())
                                != Some("implementation_ready_current")
                        {
                            return Err(ServedModelError(
                                "promotion.authorization_lineage v1 migration differs".into(),
                            ));
                        }
                        (Some((migrated.len(), migrated_sha)), None)
                    }
                    RawAuthorizationLineagePredecessor::V2(predecessor) => {
                        let predecessor_digest = validate_sha256(
                            predecessor.sha256,
                            "promotion.authorization_lineage predecessor SHA-256",
                        )?;
                        let predecessor_entries_sha = validate_sha256(
                            predecessor.entries_sha256,
                            "promotion.authorization_lineage predecessor entries SHA-256",
                        )?;
                        let predecessor_path = canonical_absolute_regular_file(
                            predecessor.path,
                            "promotion.authorization_lineage predecessor path",
                            true,
                        )?;
                        let previous = validate_lineage_file(
                            &predecessor_path,
                            &predecessor_digest,
                            None,
                            seen,
                        )?;
                        if previous.schema_version != "ullm.sq8_authorization_lineage_input.v2"
                            || predecessor.entry_count != previous.entries.len()
                            || predecessor_entries_sha != previous.entries_sha256
                            || document.entries.len() != previous.entries.len() + 2
                            || document.entries[..previous.entries.len()] != previous.entries
                            || parsed
                                .get(previous.entries.len())
                                .map(|entry| entry.relation.as_str())
                                != Some("actual_failure")
                            || parsed
                                .get(previous.entries.len())
                                .map(|entry| entry.source_commit.as_str())
                                != Some(previous.source_commit.as_str())
                            || parsed
                                .get(previous.entries.len() + 1)
                                .map(|entry| entry.relation.as_str())
                                != Some("implementation_ready_current")
                        {
                            return Err(ServedModelError(
                                "promotion.authorization_lineage is not append-only".into(),
                            ));
                        }
                        (
                            previous.migrated_prefix,
                            Some((predecessor.entry_count, predecessor_entries_sha)),
                        )
                    }
                };
                let entries_value = Value::Array(document.entries.clone());
                Ok(ValidatedLineageDocument {
                    schema_version: "ullm.sq8_authorization_lineage_input.v2",
                    source_commit: commit,
                    source_tree: tree,
                    source_archive: archive,
                    entries: document.entries,
                    entries_sha256: canonical_json_sha256(
                        &entries_value,
                        "promotion.authorization_lineage entries",
                    )?,
                    current_implementation_audit: current.pop().map(|(_, _, identity)| identity),
                    migrated_prefix,
                    v2_predecessor,
                })
            }
        }
    })();
    seen.remove(&canonical);
    result
}

fn parse_authorization_lineage(
    raw: RawAuthorizationLineageIdentity,
    source_commit: &str,
) -> Result<ParsedAuthorizationLineage> {
    let (schema_version, input, runtime, sha256, entries_sha256, entry_count, current_raw) =
        match raw {
            RawAuthorizationLineageIdentity::V1(raw) => (
                "ullm.sq8_authorization_lineage_ref.v1",
                raw.input_path,
                raw.runtime_path,
                raw.sha256,
                raw.entries_sha256,
                None,
                None,
            ),
            RawAuthorizationLineageIdentity::V2(raw) => (
                "ullm.sq8_authorization_lineage_ref.v2",
                raw.input_path,
                raw.runtime_path,
                raw.sha256,
                raw.entries_sha256,
                Some(raw.entry_count),
                Some(raw.current_implementation_audit),
            ),
        };
    let input_path =
        canonical_absolute_regular_file(input, "promotion.authorization_lineage.input_path", true)?;
    let runtime_path = canonical_absolute_regular_file(
        runtime,
        "promotion.authorization_lineage.runtime_path",
        true,
    )?;
    let sha256 = validate_sha256(sha256, "promotion.authorization_lineage.sha256")?;
    let entries_sha256 = validate_sha256(
        entries_sha256,
        "promotion.authorization_lineage.entries_sha256",
    )?;
    let mut validated = Vec::new();
    for path in [&input_path, &runtime_path] {
        let mut seen = HashSet::new();
        validated.push(validate_lineage_file(
            path,
            &sha256,
            Some(source_commit),
            &mut seen,
        )?);
    }
    if validated[0].schema_version
        != if entry_count.is_some() {
            "ullm.sq8_authorization_lineage_input.v2"
        } else {
            "ullm.sq8_authorization_lineage_input.v1"
        }
        || validated[0].entries_sha256 != entries_sha256
        || validated[1].entries_sha256 != entries_sha256
        || validated[0].entries != validated[1].entries
    {
        return Err(ServedModelError(
            "promotion.authorization_lineage manifest differs".into(),
        ));
    }
    let current_implementation_audit = if let Some(current_raw) = current_raw {
        let current_path = canonical_absolute_regular_file(
            current_raw.path,
            "promotion.authorization_lineage.current_implementation_audit.path",
            true,
        )?;
        let current_sha = validate_sha256(
            current_raw.sha256,
            "promotion.authorization_lineage.current_implementation_audit.sha256",
        )?;
        verify_file_sha256(
            &current_path,
            &current_sha,
            "promotion.authorization_lineage.current_implementation_audit",
        )?;
        let current = AuthorizationAuditIdentity {
            path: current_path,
            sha256: current_sha,
        };
        if Some(&current) != validated[0].current_implementation_audit.as_ref()
            || Some(&current) != validated[1].current_implementation_audit.as_ref()
            || entry_count != Some(validated[0].entries.len())
        {
            return Err(ServedModelError(
                "promotion.authorization_lineage current GO differs".into(),
            ));
        }
        Some(current)
    } else {
        if validated[0].current_implementation_audit.is_some() {
            return Err(ServedModelError(
                "promotion.authorization_lineage v1 authorization differs".into(),
            ));
        }
        None
    };
    Ok(ParsedAuthorizationLineage {
        identity: AuthorizationLineageIdentity {
            input_path,
            runtime_path,
            sha256,
            entries_sha256,
            schema_version: schema_version.to_owned(),
            entry_count,
            current_implementation_audit,
        },
        migrated_prefix: validated[0].migrated_prefix.clone(),
        v2_predecessor: validated[0].v2_predecessor.clone(),
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
    let parsed_audit = raw
        .authorization_audit
        .map(|value| parse_authorization_audit(value, &source_commit))
        .transpose()?;
    let parsed_lineage = raw
        .authorization_lineage
        .map(|value| parse_authorization_lineage(value, &source_commit))
        .transpose()?;
    let readiness = raw.readiness.map(parse_readiness).transpose()?;
    if parsed_audit.is_some() && (parsed_lineage.is_none() || readiness.is_none()) {
        return Err(ServedModelError(
            "authorized promotion requires audit, lineage, and readiness".into(),
        ));
    }
    if let (Some(audit), Some(lineage)) = (&parsed_audit, &parsed_lineage) {
        match (&audit.lineage, lineage.identity.entry_count) {
            (None, None) => {}
            (
                Some(AuditLineageBinding::MigratedV2 {
                    manifest_sha256,
                    entries_sha256,
                    entry_count: audit_entry_count,
                    migrated_prefix_count: audit_prefix_count,
                    migrated_prefix_sha256: audit_prefix_sha256,
                }),
                Some(entry_count),
            ) => {
                let Some((migrated_prefix_count, migrated_prefix_sha256)) =
                    lineage.migrated_prefix.as_ref()
                else {
                    return Err(ServedModelError(
                        "promotion authorization lineage audit binding differs".into(),
                    ));
                };
                if lineage.v2_predecessor.is_some()
                    || manifest_sha256 != &lineage.identity.sha256
                    || entries_sha256 != &lineage.identity.entries_sha256
                    || *audit_entry_count != entry_count
                    || *audit_prefix_count != *migrated_prefix_count
                    || audit_prefix_sha256 != migrated_prefix_sha256
                {
                    return Err(ServedModelError(
                        "promotion authorization lineage audit binding differs".into(),
                    ));
                }
            }
            (
                Some(AuditLineageBinding::CurrentV2 {
                    manifest_sha256,
                    entries_sha256,
                    entry_count: audit_entry_count,
                    predecessor_entry_count: audit_predecessor_count,
                    predecessor_entries_sha256: audit_predecessor_sha256,
                }),
                Some(entry_count),
            ) => {
                let Some((predecessor_entry_count, predecessor_entries_sha256)) =
                    lineage.v2_predecessor.as_ref()
                else {
                    return Err(ServedModelError(
                        "promotion authorization lineage audit binding differs".into(),
                    ));
                };
                if manifest_sha256 != &lineage.identity.sha256
                    || entries_sha256 != &lineage.identity.entries_sha256
                    || *audit_entry_count != entry_count
                    || *audit_predecessor_count != *predecessor_entry_count
                    || audit_predecessor_sha256 != predecessor_entries_sha256
                {
                    return Err(ServedModelError(
                        "promotion authorization lineage audit binding differs".into(),
                    ));
                }
            }
            _ => {
                return Err(ServedModelError(
                    "promotion authorization lineage audit schema differs".into(),
                ));
            }
        }
    }
    let authorization_audit = parsed_audit.map(|value| value.identity);
    let authorization_lineage = parsed_lineage.map(|value| value.identity);
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

    fn write_immutable_json(root: &Path, name: &str, value: &Value) -> (PathBuf, String) {
        let path = root.join(name);
        let bytes = serde_json::to_vec(value).unwrap();
        let digest = sha256_bytes(&bytes);
        write_immutable(&path, &bytes);
        (path, digest)
    }

    fn first_v2_authorized_fixture() -> AuthorizationFixture {
        let mut fixture = authorized_fixture(
            "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "implementation_ready",
        );
        let current_commit = "a".repeat(40);
        let predecessor_commit = "b".repeat(40);
        let predecessor_tree = "c".repeat(40);
        let predecessor_archive = "d".repeat(64);
        let request = |byte: char| format!("sq8-promotion-{}", byte.to_string().repeat(64));
        let capture = |verdict: &str, eligible: bool, auditor: &str| {
            json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
                "auditor_task_id": auditor,
                "audited_source": {
                    "commit": predecessor_commit,
                    "tree_sha256": predecessor_tree,
                    "archive_sha256": predecessor_archive,
                },
                "verdict": verdict,
                "actual": "not_executed",
                "authorization": {
                    "eligible_for_fresh_authorization_builder": eligible,
                },
                "reason_codes": if verdict == "implementation_no_go" {
                    json!(["fixture_no_go"])
                } else {
                    json!([])
                },
            })
        };
        let mut receipts = Vec::new();
        receipts.push(write_immutable_json(
            &fixture.root,
            "entry-0.json",
            &capture("implementation_ready", true, "fixture-0"),
        ));
        receipts.push(write_immutable_json(
            &fixture.root,
            "entry-1.json",
            &capture("implementation_no_go", false, "fixture-1"),
        ));
        receipts.push(write_immutable_json(
            &fixture.root,
            "entry-2.json",
            &capture("implementation_no_go", false, "fixture-2"),
        ));
        for (index, request_byte) in [(3, '3'), (4, '4')] {
            let request_id = request(request_byte);
            receipts.push(write_immutable_json(
                &fixture.root,
                &format!("entry-{index}.json"),
                &json!({
                    "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                    "status": "actual_failed",
                    "request_id": request_id,
                    "source_commit": predecessor_commit,
                    "actual": {"status": "failed", "request_id": request_id},
                }),
            ));
        }
        receipts.push(write_immutable_json(
            &fixture.root,
            "entry-5.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
                "audited_source": {
                    "commit": predecessor_commit,
                    "tree_sha256": predecessor_tree,
                    "archive_sha256": predecessor_archive,
                },
                "fixed_request_id": request('5'),
                "verdict": "implementation_no_go",
                "actual": "not_executed",
                "reason_code": "restore_retry_terminal_identity_not_fail_closed",
            }),
        ));
        let v1_relations = [
            "implementation_go_eligible_for_fresh_runtime_audit",
            "superseded_capture_implementation_no_go",
            "superseded_capture_implementation_no_go",
            "consumed_actual_failure_latest",
            "consumed_actual_failure_predecessor",
            "superseded_restore_implementation_no_go",
        ];
        let schemas = [
            "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
        ];
        let v1_entries: Vec<Value> = (0..6)
            .map(|index| {
                let (path, digest) = &receipts[index];
                let mut entry = json!({
                    "relation": v1_relations[index],
                    "path": path,
                    "sha256": digest,
                    "schema_version": schemas[index],
                    "consumed": index != 0,
                    "reusable_as_runtime_authorization": false,
                });
                let object = entry.as_object_mut().unwrap();
                match index {
                    0 => {
                        object.insert("verdict".into(), json!("implementation_ready"));
                        object.insert("actual".into(), json!("not_executed"));
                    }
                    1 | 2 => {
                        object.insert("verdict".into(), json!("implementation_no_go"));
                        object.insert("actual".into(), json!("not_executed"));
                        object.insert("reason_codes".into(), json!(["fixture_no_go"]));
                    }
                    3 | 4 => {
                        object.insert("status".into(), json!("actual_failed"));
                        object.insert("actual_status".into(), json!("failed"));
                        object.insert(
                            "request_id".into(),
                            json!(request(char::from(b'0' + index as u8))),
                        );
                    }
                    _ => {
                        object.insert("verdict".into(), json!("implementation_no_go"));
                        object.insert("actual".into(), json!("not_executed"));
                        object.insert(
                            "reason_code".into(),
                            json!("restore_retry_terminal_identity_not_fail_closed"),
                        );
                    }
                }
                entry
            })
            .collect();
        let predecessor = json!({
            "schema_version": "ullm.sq8_authorization_lineage_input.v1",
            "disposition": "authorization_input_not_yet_runtime_bound",
            "source": {
                "commit": predecessor_commit,
                "tree_oid": predecessor_tree,
                "archive_sha256": predecessor_archive,
            },
            "entries": v1_entries,
        });
        let (predecessor_path, predecessor_sha) =
            write_immutable_json(&fixture.root, "lineage-predecessor.json", &predecessor);
        let migrated = migrate_v1_entries(predecessor["entries"].as_array().unwrap()).unwrap();
        let migrated_sha =
            canonical_json_sha256(&Value::Array(migrated.clone()), "fixture migrated entries")
                .unwrap();
        let latest_request = request('6');
        let (latest_path, latest_sha) = write_immutable_json(
            &fixture.root,
            "entry-6.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                "status": "actual_failed",
                "request_id": latest_request,
                "source_commit": predecessor_commit,
                "source_provenance": {
                    "tree_sha256": predecessor_tree,
                    "archive_sha256": predecessor_archive,
                },
                "actual": {"status": "failed", "request_id": latest_request},
            }),
        );
        let (current_path, current_sha) = write_immutable_json(
            &fixture.root,
            "entry-7.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
                "audited_source": {
                    "commit": current_commit,
                    "tree_sha256": "e".repeat(40),
                    "archive_sha256": "f".repeat(64),
                },
                "verdict": "implementation_ready",
                "actual": "not_executed",
                "authorization": {"eligible_for_fresh_authorization_builder": true},
            }),
        );
        let mut entries = migrated;
        entries.push(json!({
            "sequence": 6,
            "relation": "actual_failure",
            "path": latest_path,
            "sha256": latest_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed",
            "request_id": latest_request,
            "source_commit": predecessor_commit,
        }));
        entries.push(json!({
            "sequence": 7,
            "relation": "implementation_ready_current",
            "path": current_path,
            "sha256": current_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "status": "implementation_ready",
            "request_id": null,
            "source_commit": current_commit,
        }));
        let entries_sha =
            canonical_json_sha256(&Value::Array(entries.clone()), "fixture lineage entries")
                .unwrap();
        let lineage = json!({
            "schema_version": "ullm.sq8_authorization_lineage_input.v2",
            "disposition": "authorization_input_not_yet_runtime_bound",
            "source": {
                "commit": current_commit,
                "tree_oid": "e".repeat(40),
                "archive_sha256": "f".repeat(64),
            },
            "predecessor": {
                "schema_version": "ullm.sq8_authorization_lineage_input.v1",
                "path": predecessor_path,
                "sha256": predecessor_sha,
                "migrated_prefix_count": 6,
                "migrated_prefix_sha256": migrated_sha,
            },
            "entries": entries,
        });
        let lineage_bytes = serde_json::to_vec(&lineage).unwrap();
        let lineage_sha = sha256_bytes(&lineage_bytes);
        for name in ["lineage-input.json", "lineage-runtime.json"] {
            let path = fixture.root.join(name);
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            write_immutable(&path, &lineage_bytes);
        }
        fixture.value["promotion"]["authorization_lineage"] = json!({
            "schema_version": "ullm.sq8_authorization_lineage_ref.v2",
            "input_path": fixture.root.join("lineage-input.json"),
            "runtime_path": fixture.root.join("lineage-runtime.json"),
            "sha256": lineage_sha,
            "entries_sha256": entries_sha,
            "entry_count": 8,
            "current_implementation_audit": {
                "path": current_path,
                "sha256": current_sha,
            },
        });
        let audit_path = fixture.root.join("audit.json");
        let mut audit: Value = serde_json::from_slice(
            &bounded_read(&audit_path, MAX_MANIFEST_BYTES, "fixture audit").unwrap(),
        )
        .unwrap();
        audit["runtime"]["authorization_lineage_manifest"]["sha256"] = json!(lineage_sha);
        let topology = audit["topology"].as_object_mut().unwrap();
        topology.insert("artifact_payload_and_scale_bytes_hashed".into(), json!(1));
        topology.insert("package_regular_file_bytes".into(), json!(1));
        topology.insert(
            "authorization_lineage_entries_sha256".into(),
            json!(entries_sha),
        );
        topology.insert("authorization_lineage_entry_count".into(), json!(8));
        topology.insert(
            "authorization_lineage_migrated_prefix_count".into(),
            json!(6),
        );
        topology.insert(
            "authorization_lineage_migrated_prefix_sha256".into(),
            json!(migrated_sha),
        );
        topology.insert(
            "authorization_lineage_propagation_target_count".into(),
            json!(5),
        );
        topology.insert(
            "authorization_lineage_schema".into(),
            json!("ullm.sq8_authorization_lineage_input.v2"),
        );
        audit["tests"]["lineage_v1_authorization_rejection"] = json!("passed");
        audit["tests"]["lineage_v1_migration"] = json!("passed");
        let audit_bytes = serde_json::to_vec(&audit).unwrap();
        fs::set_permissions(&audit_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&audit_path, &audit_bytes);
        fixture.value["promotion"]["authorization_audit"]["sha256"] =
            json!(sha256_bytes(&audit_bytes));
        fixture
    }

    fn current_v2_authorized_fixture() -> AuthorizationFixture {
        let mut fixture = first_v2_authorized_fixture();
        let lineage_input = fixture.root.join("lineage-input.json");
        let lineage_runtime = fixture.root.join("lineage-runtime.json");
        let mut lineage: Value = serde_json::from_slice(
            &bounded_read(&lineage_input, MAX_MANIFEST_BYTES, "fixture lineage").unwrap(),
        )
        .unwrap();
        let predecessor_bytes = serde_json::to_vec(&lineage).unwrap();
        let predecessor_sha = sha256_bytes(&predecessor_bytes);
        let predecessor_entries_sha =
            canonical_json_sha256(&lineage["entries"], "fixture predecessor entries").unwrap();
        let predecessor_path = fixture.root.join("lineage-v2-predecessor.json");
        write_immutable(&predecessor_path, &predecessor_bytes);

        let previous_commit = "a".repeat(40);
        let current_commit = "b".repeat(40);
        let current_tree = "c".repeat(40);
        let current_archive = "d".repeat(64);
        let request = format!("sq8-promotion-{}", "8".repeat(64));
        let (failure_path, failure_sha) = write_immutable_json(
            &fixture.root,
            "entry-8.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                "status": "actual_failed",
                "request_id": request,
                "source_commit": previous_commit,
                "actual": {"status": "failed", "request_id": request},
            }),
        );
        let (current_path, current_sha) = write_immutable_json(
            &fixture.root,
            "entry-9.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
                "auditor_task_id": "fixture-9",
                "audited_source": {
                    "commit": current_commit,
                    "tree_sha256": current_tree,
                    "archive_sha256": current_archive,
                },
                "verdict": "implementation_ready",
                "actual": "not_executed",
                "authorization": {"eligible_for_fresh_authorization_builder": true},
            }),
        );
        lineage["source"] = json!({
            "commit": current_commit,
            "tree_oid": current_tree,
            "archive_sha256": current_archive,
        });
        lineage["predecessor"] = json!({
            "schema_version": "ullm.sq8_authorization_lineage_input.v2",
            "path": predecessor_path,
            "sha256": predecessor_sha,
            "entries_sha256": predecessor_entries_sha,
            "entry_count": 8,
        });
        lineage["entries"].as_array_mut().unwrap().push(json!({
            "sequence": 8,
            "relation": "actual_failure",
            "path": failure_path,
            "sha256": failure_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed",
            "request_id": request,
            "source_commit": previous_commit,
        }));
        lineage["entries"].as_array_mut().unwrap().push(json!({
            "sequence": 9,
            "relation": "implementation_ready_current",
            "path": current_path,
            "sha256": current_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "status": "implementation_ready",
            "request_id": null,
            "source_commit": current_commit,
        }));
        let lineage_bytes = serde_json::to_vec(&lineage).unwrap();
        let lineage_sha = sha256_bytes(&lineage_bytes);
        let entries_sha = canonical_json_sha256(&lineage["entries"], "fixture entries").unwrap();
        for path in [&lineage_input, &lineage_runtime] {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
            write_immutable(path, &lineage_bytes);
        }
        fixture.value["promotion"]["source_commit"] = json!(current_commit);
        fixture.value["promotion"]["authorization_lineage"] = json!({
            "schema_version": "ullm.sq8_authorization_lineage_ref.v2",
            "input_path": lineage_input,
            "runtime_path": lineage_runtime,
            "sha256": lineage_sha,
            "entries_sha256": entries_sha,
            "entry_count": 10,
            "current_implementation_audit": {
                "path": current_path,
                "sha256": current_sha,
            },
        });

        let audit_path = fixture.root.join("audit.json");
        let mut audit: Value = serde_json::from_slice(
            &bounded_read(&audit_path, MAX_MANIFEST_BYTES, "fixture audit").unwrap(),
        )
        .unwrap();
        audit["audited_source"] = json!({
            "commit": current_commit,
            "tree_sha256": current_tree,
            "archive_sha256": current_archive,
        });
        audit["runtime"]["authorization_lineage_manifest"]["sha256"] = json!(lineage_sha);
        let topology = audit["topology"].as_object_mut().unwrap();
        topology.remove("authorization_lineage_migrated_prefix_count");
        topology.remove("authorization_lineage_migrated_prefix_sha256");
        topology.insert(
            "historical_direct_authorization_reference_count".into(),
            json!(0),
        );
        topology.insert("authorization_lineage_entry_count".into(), json!(10));
        topology.insert(
            "authorization_lineage_entries_sha256".into(),
            json!(entries_sha),
        );
        topology.insert(
            "authorization_lineage_predecessor_entry_count".into(),
            json!(8),
        );
        topology.insert(
            "authorization_lineage_predecessor_entries_sha256".into(),
            json!(predecessor_entries_sha),
        );
        let tests = audit["tests"].as_object_mut().unwrap();
        tests.remove("lineage_v1_migration");
        tests.remove("worker_live_identity");
        tests.insert("lineage_v2_successor".into(), json!("passed"));
        tests.insert(
            "lineage_old_v2_authorization_rejection".into(),
            json!("passed"),
        );
        tests.insert("served_model_cpu_validation".into(), json!("passed"));
        let audit_bytes = serde_json::to_vec(&audit).unwrap();
        fs::set_permissions(&audit_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&audit_path, &audit_bytes);
        fixture.value["promotion"]["authorization_audit"]["sha256"] =
            json!(sha256_bytes(&audit_bytes));
        fixture
    }

    fn mutate_current_v2_audit(
        fixture: &mut AuthorizationFixture,
        mutate: impl FnOnce(&mut Value),
    ) {
        let audit_path = PathBuf::from(
            fixture.value["promotion"]["authorization_audit"]["path"]
                .as_str()
                .unwrap(),
        );
        let mut audit: Value = serde_json::from_slice(
            &bounded_read(&audit_path, MAX_MANIFEST_BYTES, "fixture audit").unwrap(),
        )
        .unwrap();
        mutate(&mut audit);
        let bytes = serde_json::to_vec(&audit).unwrap();
        fs::set_permissions(&audit_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&audit_path, &bytes);
        fixture.value["promotion"]["authorization_audit"]["sha256"] = json!(sha256_bytes(&bytes));
    }

    fn assert_current_v2_audit_rejected(mutate: impl FnOnce(&mut Value)) {
        let mut fixture = current_v2_authorized_fixture();
        mutate_current_v2_audit(&mut fixture, mutate);
        let raw = serde_json::to_vec(&fixture.value).unwrap();
        assert!(load_served_model_bytes(&fixture.manifest_path, &raw).is_err());
    }

    fn validate_mutated_first_v2(
        mutate: impl FnOnce(&mut Value),
    ) -> Result<ValidatedLineageDocument> {
        let fixture = first_v2_authorized_fixture();
        let path = fixture.root.join("lineage-input.json");
        let mut value: Value = serde_json::from_slice(
            &bounded_read(&path, MAX_MANIFEST_BYTES, "fixture lineage").unwrap(),
        )
        .unwrap();
        mutate(&mut value);
        let bytes = serde_json::to_vec(&value).unwrap();
        let digest = sha256_bytes(&bytes);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&path, &bytes);
        validate_lineage_file(&path, &digest, Some(&"a".repeat(40)), &mut HashSet::new())
    }

    fn validate_mutated_first_v2_predecessor(
        sequence: usize,
        mutate: impl FnOnce(&mut Value, &mut Value),
    ) -> Result<ValidatedLineageDocument> {
        let fixture = first_v2_authorized_fixture();
        let lineage_path = fixture.root.join("lineage-input.json");
        let mut lineage: Value = serde_json::from_slice(
            &bounded_read(&lineage_path, MAX_MANIFEST_BYTES, "fixture lineage").unwrap(),
        )
        .unwrap();
        let predecessor_path = PathBuf::from(lineage["predecessor"]["path"].as_str().unwrap());
        let mut predecessor: Value = serde_json::from_slice(
            &bounded_read(&predecessor_path, MAX_MANIFEST_BYTES, "fixture predecessor").unwrap(),
        )
        .unwrap();
        let receipt_path =
            PathBuf::from(predecessor["entries"][sequence]["path"].as_str().unwrap());
        let mut receipt: Value = serde_json::from_slice(
            &bounded_read(&receipt_path, MAX_MANIFEST_BYTES, "fixture receipt").unwrap(),
        )
        .unwrap();
        mutate(&mut predecessor["entries"][sequence], &mut receipt);

        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&receipt_path, &receipt_bytes);
        predecessor["entries"][sequence]["sha256"] = json!(sha256_bytes(&receipt_bytes));

        let predecessor_bytes = serde_json::to_vec(&predecessor).unwrap();
        fs::set_permissions(&predecessor_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&predecessor_path, &predecessor_bytes);
        lineage["predecessor"]["sha256"] = json!(sha256_bytes(&predecessor_bytes));

        let lineage_bytes = serde_json::to_vec(&lineage).unwrap();
        let lineage_digest = sha256_bytes(&lineage_bytes);
        fs::set_permissions(&lineage_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&lineage_path, &lineage_bytes);
        validate_lineage_file(
            &lineage_path,
            &lineage_digest,
            Some(&"a".repeat(40)),
            &mut HashSet::new(),
        )
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
    fn portable_first_v2_authorization_lineage_is_typed() {
        let fixture = first_v2_authorized_fixture();
        let raw = serde_json::to_vec(&fixture.value).unwrap();
        let model = load_served_model_bytes(&fixture.manifest_path, &raw).unwrap();
        let lineage = model.promotion.authorization_lineage.unwrap();
        assert_eq!(
            lineage.schema_version,
            "ullm.sq8_authorization_lineage_ref.v2"
        );
        assert_eq!(lineage.entry_count, Some(8));
        assert!(lineage.current_implementation_audit.is_some());
    }

    #[test]
    fn portable_current_v2_authorization_audit_is_typed_and_bound() {
        let fixture = current_v2_authorized_fixture();
        let raw = serde_json::to_vec(&fixture.value).unwrap();
        let model = load_served_model_bytes(&fixture.manifest_path, &raw).unwrap();
        assert!(model.promotion.authorization_audit.is_some());
        let lineage = model.promotion.authorization_lineage.unwrap();
        assert_eq!(lineage.entry_count, Some(10));
        assert_eq!(
            lineage
                .current_implementation_audit
                .expect("current implementation GO")
                .sha256,
            fixture.value["promotion"]["authorization_lineage"]
                ["current_implementation_audit"]["sha256"]
                .as_str()
                .unwrap()
        );
    }

    #[test]
    fn portable_current_v2_audit_variant_tamper_matrix_is_rejected() {
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]
                .as_object_mut()
                .unwrap()
                .insert("unknown".into(), json!(true));
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]
                .as_object_mut()
                .unwrap()
                .remove("historical_direct_authorization_reference_count");
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["historical_direct_authorization_reference_count"] = json!("0");
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["historical_direct_authorization_reference_count"] = json!(1);
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["authorization_lineage_entry_count"] = json!(9);
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["authorization_lineage_predecessor_entry_count"] = json!(7);
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["authorization_lineage_predecessor_entry_count"] = json!(usize::MAX);
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["authorization_lineage_entries_sha256"] = json!("0".repeat(64));
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["topology"]["authorization_lineage_predecessor_entries_sha256"] =
                json!("0".repeat(64));
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["tests"]["gpu_or_service_execution"] = json!(true);
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["tests"]
                .as_object_mut()
                .unwrap()
                .remove("served_model_cpu_validation");
        });
        assert_current_v2_audit_rejected(|audit| {
            audit["tests"]
                .as_object_mut()
                .unwrap()
                .insert("worker_live_identity".into(), json!("passed"));
        });
        assert_current_v2_audit_rejected(|audit| {
            let tests = audit["tests"].as_object_mut().unwrap();
            tests.remove("lineage_v2_successor");
            tests.remove("lineage_old_v2_authorization_rejection");
            tests.remove("served_model_cpu_validation");
            tests.insert("lineage_v1_migration".into(), json!("passed"));
            tests.insert("worker_live_identity".into(), json!("passed"));
        });
    }

    #[test]
    fn portable_v2_reference_tamper_matrix_is_rejected() {
        let fixture = first_v2_authorized_fixture();
        let mut cases = Vec::new();

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), Value::Bool(true));
        cases.push(value);

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]
            .as_object_mut()
            .unwrap()
            .remove("current_implementation_audit");
        cases.push(value);

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]["entry_count"] = json!("8");
        cases.push(value);

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]["entry_count"] = json!(9);
        cases.push(value);

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]["entries_sha256"] = json!("0".repeat(64));
        cases.push(value);

        let mut value = fixture.value.clone();
        value["promotion"]["authorization_lineage"]["current_implementation_audit"]["sha256"] =
            json!("0".repeat(64));
        cases.push(value);

        for value in cases {
            let raw = serde_json::to_vec(&value).unwrap();
            assert!(load_served_model_bytes(&fixture.manifest_path, &raw).is_err());
        }
    }

    #[test]
    fn portable_v2_manifest_tamper_matrix_is_rejected() {
        assert!(
            validate_mutated_first_v2(|value| {
                value
                    .as_object_mut()
                    .unwrap()
                    .insert("unknown".into(), Value::Bool(true));
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["entries"][7]["sequence"] = json!(8);
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["entries"][7]["path"] = value["entries"][0]["path"].clone();
                value["entries"][7]["sha256"] = value["entries"][0]["sha256"].clone();
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["entries"][7]["source_commit"] = json!("b".repeat(40));
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["predecessor"]["migrated_prefix_sha256"] = json!("0".repeat(64));
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["predecessor"]["sha256"] = json!("0".repeat(64));
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2(|value| {
                value["entries"].as_array_mut().unwrap().push(json!({}));
            })
            .is_err()
        );
    }

    #[test]
    fn portable_v1_migration_discarded_field_tamper_matrix_is_rejected() {
        assert!(
            validate_mutated_first_v2_predecessor(0, |entry, receipt| {
                entry["actual"] = json!("executed");
                receipt["actual"] = json!("executed");
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2_predecessor(1, |entry, receipt| {
                entry["reason_codes"] = json!([]);
                receipt["reason_codes"] = json!([]);
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2_predecessor(2, |entry, receipt| {
                entry["reason_codes"] = json!(["fixture_no_go", "fixture_no_go"]);
                receipt["reason_codes"] = json!(["fixture_no_go", "fixture_no_go"]);
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2_predecessor(3, |entry, receipt| {
                entry["actual_status"] = json!("succeeded");
                receipt["actual"]["status"] = json!("succeeded");
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2_predecessor(4, |entry, receipt| {
                entry["status"] = json!("pending");
                receipt["status"] = json!("pending");
            })
            .is_err()
        );
        assert!(
            validate_mutated_first_v2_predecessor(5, |entry, receipt| {
                entry["reason_code"] = json!("wrong_reason");
                receipt["reason_code"] = json!("wrong_reason");
            })
            .is_err()
        );
    }

    #[test]
    fn portable_subsequent_v2_is_append_only() {
        let fixture = first_v2_authorized_fixture();
        let lineage_input = fixture.root.join("lineage-input.json");
        let lineage_runtime = fixture.root.join("lineage-runtime.json");
        let mut previous: Value = serde_json::from_slice(
            &bounded_read(&lineage_input, MAX_MANIFEST_BYTES, "fixture lineage").unwrap(),
        )
        .unwrap();
        let previous_bytes = serde_json::to_vec(&previous).unwrap();
        let previous_sha = sha256_bytes(&previous_bytes);
        let previous_entries_sha =
            canonical_json_sha256(&previous["entries"], "fixture predecessor entries").unwrap();
        let predecessor_path = fixture.root.join("lineage-v2-predecessor.json");
        write_immutable(&predecessor_path, &previous_bytes);
        let request = format!("sq8-promotion-{}", "8".repeat(64));
        let (failure_path, failure_sha) = write_immutable_json(
            &fixture.root,
            "entry-8.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                "status": "actual_failed",
                "request_id": request,
                "source_commit": "a".repeat(40),
                "actual": {"status": "failed", "request_id": request},
            }),
        );
        let (current_path, current_sha) = write_immutable_json(
            &fixture.root,
            "entry-9.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
                "auditor_task_id": "fixture-9",
                "audited_source": {"commit": "b".repeat(40)},
                "verdict": "implementation_ready",
                "actual": "not_executed",
                "authorization": {"eligible_for_fresh_authorization_builder": true},
            }),
        );
        previous["source"] = json!({
            "commit": "b".repeat(40),
            "tree_oid": "c".repeat(40),
            "archive_sha256": "d".repeat(64),
        });
        previous["predecessor"] = json!({
            "schema_version": "ullm.sq8_authorization_lineage_input.v2",
            "path": predecessor_path,
            "sha256": previous_sha,
            "entries_sha256": previous_entries_sha,
            "entry_count": 8,
        });
        previous["entries"].as_array_mut().unwrap().push(json!({
            "sequence": 8,
            "relation": "actual_failure",
            "path": failure_path,
            "sha256": failure_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed",
            "request_id": request,
            "source_commit": "a".repeat(40),
        }));
        previous["entries"].as_array_mut().unwrap().push(json!({
            "sequence": 9,
            "relation": "implementation_ready_current",
            "path": current_path,
            "sha256": current_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "status": "implementation_ready",
            "request_id": null,
            "source_commit": "b".repeat(40),
        }));
        let lineage_bytes = serde_json::to_vec(&previous).unwrap();
        let lineage_sha = sha256_bytes(&lineage_bytes);
        let entries_sha = canonical_json_sha256(&previous["entries"], "fixture entries").unwrap();
        for path in [&lineage_input, &lineage_runtime] {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            write_immutable(&path, &lineage_bytes);
        }
        let raw_reference = |current: AuthorizationAuditIdentity| {
            RawAuthorizationLineageIdentity::V2(RawAuthorizationLineageReferenceV2 {
                input_path: lineage_input.to_string_lossy().into_owned(),
                runtime_path: lineage_runtime.to_string_lossy().into_owned(),
                sha256: lineage_sha.clone(),
                entries_sha256: entries_sha.clone(),
                entry_count: 10,
                current_implementation_audit: RawAuthorizationAuditIdentity {
                    path: current.path.to_string_lossy().into_owned(),
                    sha256: current.sha256,
                },
            })
        };
        let current = AuthorizationAuditIdentity {
            path: current_path.clone(),
            sha256: current_sha.clone(),
        };
        let parsed =
            parse_authorization_lineage(raw_reference(current.clone()), &"b".repeat(40)).unwrap();
        assert_eq!(parsed.identity.entry_count, Some(10));
        assert_eq!(parsed.identity.current_implementation_audit, Some(current));
        let old = AuthorizationAuditIdentity {
            path: PathBuf::from(previous["entries"][7]["path"].as_str().unwrap()),
            sha256: previous["entries"][7]["sha256"]
                .as_str()
                .unwrap()
                .to_owned(),
        };
        assert!(parse_authorization_lineage(raw_reference(old), &"b".repeat(40)).is_err());

        let validate = |value: &Value, expected_commit: &str| {
            let bytes = serde_json::to_vec(value).unwrap();
            let digest = sha256_bytes(&bytes);
            fs::set_permissions(&lineage_input, fs::Permissions::from_mode(0o600)).unwrap();
            write_immutable(&lineage_input, &bytes);
            validate_lineage_file(
                &lineage_input,
                &digest,
                Some(expected_commit),
                &mut HashSet::new(),
            )
        };

        let mut tampered = previous.clone();
        tampered["entries"].as_array_mut().unwrap().swap(8, 9);
        tampered["entries"][8]["sequence"] = json!(8);
        tampered["entries"][9]["sequence"] = json!(9);
        assert!(validate(&tampered, &"b".repeat(40)).is_err());

        let mut go_only = previous.clone();
        go_only["entries"].as_array_mut().unwrap().remove(8);
        go_only["entries"][8]["sequence"] = json!(8);
        assert!(validate(&go_only, &"b".repeat(40)).is_err());

        let request_after = format!("sq8-promotion-{}", "9".repeat(64));
        let (after_path, after_sha) = write_immutable_json(
            &fixture.root,
            "entry-10.json",
            &json!({
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                "status": "actual_failed",
                "request_id": request_after,
                "source_commit": "b".repeat(40),
                "actual": {"status": "failed", "request_id": request_after},
            }),
        );
        let mut failure_only = previous.clone();
        failure_only["entries"].as_array_mut().unwrap().push(json!({
            "sequence": 10,
            "relation": "actual_failure",
            "path": after_path,
            "sha256": after_sha,
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed",
            "request_id": request_after,
            "source_commit": "b".repeat(40),
        }));
        assert!(validate(&failure_only, &"b".repeat(40)).is_err());

        let mut fake_source = previous.clone();
        fake_source["entries"][9]["source_commit"] = json!("c".repeat(40));
        assert!(validate(&fake_source, &"b".repeat(40)).is_err());

        let mut rewritten = previous.clone();
        rewritten["entries"].as_array_mut().unwrap().swap(0, 1);
        rewritten["entries"][0]["sequence"] = json!(0);
        rewritten["entries"][1]["sequence"] = json!(1);
        assert!(validate(&rewritten, &"b".repeat(40)).is_err());

        let mut duplicate_source = previous.clone();
        let current_receipt = json!({
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "auditor_task_id": "fixture-9",
            "audited_source": {"commit": "a".repeat(40)},
            "verdict": "implementation_ready",
            "actual": "not_executed",
            "authorization": {"eligible_for_fresh_authorization_builder": true},
        });
        let current_bytes = serde_json::to_vec(&current_receipt).unwrap();
        fs::set_permissions(&current_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_immutable(&current_path, &current_bytes);
        duplicate_source["source"]["commit"] = json!("a".repeat(40));
        duplicate_source["entries"][9]["sha256"] = json!(sha256_bytes(&current_bytes));
        duplicate_source["entries"][9]["source_commit"] = json!("a".repeat(40));
        assert!(validate(&duplicate_source, &"a".repeat(40)).is_err());
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
    fn first_v2_authorized_sq8_manifest_is_cpu_loadable_when_available() {
        let path = PathBuf::from(
            "/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-de76c4c3ceb3c69b/served-model.json",
        );
        if !path.exists() {
            return;
        }
        assert_eq!(
            sha256_file(&path).unwrap(),
            "31ba7f6483a5baf7d84f8b45a5d86d02c2c22d72d229ca74cfe593192e98ccdd"
        );
        let worker = PathBuf::from(
            "/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-de76c4c3ceb3c69b/ullm-aq4-worker",
        );
        assert_eq!(
            sha256_file(&worker).unwrap(),
            "b4c3df3dd704b42ca6a6a2d353cf49fa95065fda737443f7da322bf7985e71ae"
        );
        let model = load_served_model(path).unwrap();
        let lineage = model.promotion.authorization_lineage.unwrap();
        assert_eq!(
            lineage.schema_version,
            "ullm.sq8_authorization_lineage_ref.v2"
        );
        assert_eq!(lineage.entry_count, Some(8));
        assert_eq!(
            lineage
                .current_implementation_audit
                .expect("v2 current implementation GO")
                .sha256,
            "058bc7f90c1c6cd93e2c5dae4a9a207749dc15bfc52e2d26203345fe3ebe01b4"
        );
    }

    #[test]
    fn current_v2_authorized_sq8_manifest_is_cpu_loadable_when_available() {
        let path = PathBuf::from(
            "/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-08044245855b9bc2/served-model.json",
        );
        if !path.exists() {
            return;
        }
        assert_eq!(
            sha256_file(&path).unwrap(),
            "b1f2a3a88ea24d65298129c065e77fede46711975ded40ea3a0a802634d6db43"
        );
        let model = load_served_model(path).unwrap();
        assert!(model.promotion.authorization_audit.is_some());
        let lineage = model.promotion.authorization_lineage.unwrap();
        assert_eq!(lineage.entry_count, Some(10));
        assert_eq!(
            lineage
                .current_implementation_audit
                .expect("current implementation GO")
                .sha256,
            "8b7a0ab9d3bd6ea672bed7b435a89176f178a30f9a02daf879e1ee42bb73465d"
        );
    }

    #[test]
    fn current_v2_unauthorized_sq8_manifest_is_cpu_loadable_when_available() {
        let path = PathBuf::from(
            "/tmp/ullm-sq8-overlay-gpu-promotion-6ad51ac5-5c7d71d2-unauthorized-v2/served-model.json",
        );
        if !path.exists() {
            return;
        }
        assert_eq!(
            sha256_file(&path).unwrap(),
            "484ac20f4a9828152c895cd6064371c1851b34dece64a996bf445c431a29d21e"
        );
        let model = load_served_model(path).unwrap();
        assert!(model.promotion.authorization_audit.is_none());
        assert_eq!(
            model
                .promotion
                .authorization_lineage
                .expect("current v2 lineage")
                .entry_count,
            Some(10)
        );
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
