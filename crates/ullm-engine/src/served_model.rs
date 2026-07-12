// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded, fail-closed loader for `ullm.served_model.v1`.

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

pub const SERVED_MODEL_SCHEMA_VERSION: &str = "ullm.served_model.v1";
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
pub struct PromotionContract {
    pub source_commit: String,
    pub receipt: PathBuf,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProfileSnapshot {
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
}

impl WorkerProfileSnapshot {
    pub fn into_worker_profile(self) -> crate::sq8_worker_protocol::Sq8WorkerProfile {
        crate::sq8_worker_protocol::Sq8WorkerProfile {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerBackendKind {
    Sq8,
    Aq4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerStartupConfig {
    pub artifact_dir: Option<PathBuf>,
    pub package_dir: PathBuf,
    pub profile: WorkerProfileSnapshot,
    pub required_environment: Vec<String>,
}

impl ServedModel {
    pub fn profile_snapshot(&self) -> WorkerProfileSnapshot {
        WorkerProfileSnapshot {
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
        if self.worker.protocol != "ullm.worker.v1" {
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
        })
    }
}

pub fn load_served_model(path: impl AsRef<Path>) -> Result<ServedModel> {
    let manifest_path = safe_regular_file(path.as_ref(), "served-model manifest")?;
    let raw = bounded_read(&manifest_path, MAX_MANIFEST_BYTES, "served-model manifest")?;
    let value = decode_strict_json(&raw)?;
    validate_exact_shape(&value)?;
    let raw_manifest: RawManifest = serde_json::from_value(value)
        .map_err(|_| ServedModelError("manifest typed schema is invalid".into()))?;
    if raw_manifest.schema_version != SERVED_MODEL_SCHEMA_VERSION {
        return Err(ServedModelError(
            "manifest schema_version is unsupported".into(),
        ));
    }
    let base = manifest_path
        .parent()
        .ok_or_else(|| ServedModelError("manifest path has no parent".into()))?;
    let public = parse_public(raw_manifest.public)?;
    let generation = parse_generation(raw_manifest.generation, &public)?;
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
    let product = parse_product(raw_manifest.product, base)?;
    let promotion = parse_promotion(raw_manifest.promotion, base)?;
    Ok(ServedModel {
        manifest_path,
        manifest_sha256: sha256_bytes(&raw),
        public,
        generation,
        format,
        tokenizer,
        worker,
        product,
        promotion,
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

fn parse_promotion(raw: RawPromotion, base: &Path) -> Result<PromotionContract> {
    let receipt = safe_regular_file(
        &resolve_root(base, &raw.receipt, "promotion.receipt")?,
        "promotion.receipt",
    )?;
    let digest = validate_sha256(raw.receipt_sha256, "promotion.receipt_sha256")?;
    verify_file_sha256(&receipt, &digest, "promotion.receipt")?;
    Ok(PromotionContract {
        source_commit: bounded_text(raw.source_commit, "promotion.source_commit", 256)?,
        receipt,
        receipt_sha256: digest,
    })
}

fn validate_exact_shape(value: &Value) -> Result<()> {
    exact_keys(
        value,
        &[
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "promotion",
        ],
        "manifest",
    )?;
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
    exact_keys(
        &value["promotion"],
        &["source_commit", "receipt", "receipt_sha256"],
        "promotion",
    )
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

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../services/openai-gateway/tests/fixtures/served-model")
            .join(name)
            .join("served-model.json")
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
