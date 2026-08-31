//! Local, admin-only model-library discovery for the bundled WebUI.
//!
//! The browser never reads the operator filesystem. Directory enumeration,
//! persisted selection, GGUF verification, and lifecycle registration all
//! remain inside the loopback sLLM server process.

use serde::{Deserialize, Serialize};
use sllm_core::{
    DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES, DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES,
    GEMMA4_12B_IT_FINGERPRINT, GEMMA4_MOE_MODEL_FINGERPRINT, GEMMA4_MTP_FINGERPRINT, GgufValue,
    MINIMAX_M3_CAPACITY_ADMISSION_BYTES, MINISTRAL3_GGUF_ARCHITECTURE, MINISTRAL3_MODEL_ALIAS,
    MINISTRAL3_MODEL_LOCK_FINGERPRINT, MINISTRAL3_WEIGHT_RESIDENT_BYTES,
    QWEN35_MOE_MODEL_FINGERPRINT, ReviewedModelLock, VerifiedGguf,
    build_verified_gemma4_mtp_weight_load_plan, builtin_reviewed_model_lock,
    gemma4_mtp_pair_semantic_id, parse_gemma4_mtp_model_lock, read_derived_gguf_lock,
    verify_derived_gguf, verify_gguf_gemma4_mtp,
};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::dynamic_model_plan_preflight_v1;

pub const MAX_MODEL_LIBRARY_FILES_V1: usize = 256;
pub const MAX_MODEL_LIBRARY_DIRECTORIES_V1: usize = 256;
const MAX_PERSISTED_STATE_BYTES: usize = 16 * 1024;
const MAX_DIRECTORY_ENTRIES_INSPECTED: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLibraryDeviceV1 {
    pub device_index: u32,
    pub target: String,
    pub name: String,
    pub total_memory_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLibraryRegistrationV1 {
    pub alias: String,
    pub gguf_path: PathBuf,
    /// A derived GGUF carries a sidecar lock; direct official artifacts do
    /// not.  Keeping this optional prevents an official file from being
    /// represented as an sLLM-derived conversion.
    pub derived_lock_path: Option<PathBuf>,
    pub architecture: String,
    pub model_identity: String,
    pub plan_identity: String,
    pub resident_bytes: u64,
    pub device_index: u32,
    pub target: String,
    pub mtp_assistant_gguf_path: Option<PathBuf>,
    pub mtp_assistant_derived_lock_path: Option<PathBuf>,
    pub mtp_assistant_identity: Option<String>,
    pub mtp_semantic_pair_identity: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLibraryModelV1 {
    pub alias: String,
    pub file_name: String,
    pub size_bytes: u64,
    pub architecture: String,
    pub supported_architecture: bool,
    pub compatible: bool,
    pub reason: Option<String>,
    pub mtp_companion_file_name: Option<String>,
    pub mtp_companion_for: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidMtpAssistantV1 {
    target_identity: String,
    assistant_identity: String,
    semantic_pair_identity: String,
    derived_lock_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MtpAssistantCandidateV1 {
    gguf_path: PathBuf,
    alias: String,
    file_name: String,
    size_bytes: u64,
    validation: Result<ValidMtpAssistantV1, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingModelV1 {
    sort_path: PathBuf,
    alias: String,
    file_name: String,
    size_bytes: u64,
    architecture: String,
    registration: ModelLibraryRegistrationV1,
    device_total_memory_bytes: u64,
    mtp_companion_file_name: Option<String>,
    mtp_pair_target_identity: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLibrarySnapshotV1 {
    pub schema_version: &'static str,
    pub selected_path: Option<String>,
    pub models: Vec<ModelLibraryModelV1>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLibraryDirectoryV1 {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLibraryBrowseV1 {
    pub schema_version: &'static str,
    pub current_path: String,
    pub parent_path: Option<String>,
    pub directories: Vec<ModelLibraryDirectoryV1>,
}

#[derive(Debug, Deserialize)]
struct PersistedStateV1 {
    schema_version: String,
    selected_path: String,
}

#[derive(Serialize)]
struct PersistedStateRefV1<'a> {
    schema_version: &'static str,
    selected_path: &'a str,
}

struct StateV1 {
    selected_path: Option<PathBuf>,
    models: Vec<ModelLibraryModelV1>,
    registered_aliases: BTreeSet<String>,
    error: Option<String>,
}

type RegisterFn = dyn Fn(ModelLibraryRegistrationV1) -> Result<(), String> + Send + Sync;
type UnregisterFn = dyn Fn(&str) -> Result<(), String> + Send + Sync;

#[derive(Clone)]
pub struct ModelLibraryV1 {
    state: Arc<Mutex<StateV1>>,
    refresh: Arc<Mutex<()>>,
    persistence_path: PathBuf,
    initial_path: PathBuf,
    device: Option<ModelLibraryDeviceV1>,
    register: Arc<RegisterFn>,
    unregister: Arc<UnregisterFn>,
}

impl ModelLibraryV1 {
    pub fn open<R, U>(
        persistence_path: PathBuf,
        initial_path: PathBuf,
        device: Option<ModelLibraryDeviceV1>,
        register: R,
        unregister: U,
    ) -> Result<Self, String>
    where
        R: Fn(ModelLibraryRegistrationV1) -> Result<(), String> + Send + Sync + 'static,
        U: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        let initial_path = canonical_directory(&initial_path)?;
        let selected_path = read_persisted_path(&persistence_path)?;
        let value = Self {
            state: Arc::new(Mutex::new(StateV1 {
                selected_path: None,
                models: Vec::new(),
                registered_aliases: BTreeSet::new(),
                error: None,
            })),
            refresh: Arc::new(Mutex::new(())),
            persistence_path,
            initial_path,
            device,
            register: Arc::new(register),
            unregister: Arc::new(unregister),
        };
        if let Some(path) = selected_path {
            if let Err(error) = value.refresh_path(path, false) {
                let mut state = value.state.lock().expect("model library mutex poisoned");
                state.error = Some(error);
            }
        }
        Ok(value)
    }

    pub fn snapshot(&self) -> ModelLibrarySnapshotV1 {
        let state = self.state.lock().expect("model library mutex poisoned");
        ModelLibrarySnapshotV1 {
            schema_version: "sllm-model-library-v1",
            selected_path: state
                .selected_path
                .as_ref()
                .map(|path| path.display().to_string()),
            models: state.models.clone(),
            error: state.error.clone(),
        }
    }

    pub(crate) fn selected_path(&self) -> Option<PathBuf> {
        let selected = self
            .state
            .lock()
            .expect("model library mutex poisoned")
            .selected_path
            .clone()?;
        canonical_directory(&selected)
            .ok()
            .filter(|canonical| canonical == &selected)
    }

    pub fn browse(&self, requested: Option<&Path>) -> Result<ModelLibraryBrowseV1, String> {
        let selected = self
            .state
            .lock()
            .expect("model library mutex poisoned")
            .selected_path
            .clone();
        let path = requested
            .map(Path::to_path_buf)
            .or(selected)
            .unwrap_or_else(|| self.initial_path.clone());
        let path = canonical_directory(&path)?;
        let mut directories = Vec::new();
        let entries = fs::read_dir(&path).map_err(|_| "directory could not be read".to_owned())?;
        for entry in entries.take(MAX_DIRECTORY_ENTRIES_INSPECTED) {
            if directories.len() == MAX_MODEL_LIBRARY_DIRECTORIES_V1 {
                break;
            }
            let entry = match entry {
                Ok(value) => value,
                Err(_) => continue,
            };
            let metadata = match fs::symlink_metadata(entry.path()) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            directories.push(ModelLibraryDirectoryV1 {
                name,
                path: entry.path().display().to_string(),
            });
        }
        directories.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(ModelLibraryBrowseV1 {
            schema_version: "sllm-model-library-browse-v1",
            current_path: path.display().to_string(),
            parent_path: path.parent().map(|parent| parent.display().to_string()),
            directories,
        })
    }

    pub fn select(&self, path: &Path) -> Result<ModelLibrarySnapshotV1, String> {
        let path = canonical_directory(path)?;
        self.refresh_path(path, true)?;
        Ok(self.snapshot())
    }

    pub fn rescan(&self) -> Result<ModelLibrarySnapshotV1, String> {
        let path = self
            .state
            .lock()
            .expect("model library mutex poisoned")
            .selected_path
            .clone()
            .ok_or_else(|| "no model folder is selected".to_owned())?;
        self.refresh_path(path, false)?;
        Ok(self.snapshot())
    }

    fn refresh_path(&self, path: PathBuf, persist_selected: bool) -> Result<(), String> {
        let _refresh = self
            .refresh
            .lock()
            .expect("model library refresh mutex poisoned");
        let path = canonical_directory(&path)?;
        let previous_aliases = self
            .state
            .lock()
            .expect("model library mutex poisoned")
            .registered_aliases
            .clone();
        for alias in &previous_aliases {
            if let Err(error) = (self.unregister)(alias) {
                let error = format!(
                    "unload model {alias} before changing or rescanning its folder: {error}"
                );
                self.state
                    .lock()
                    .expect("model library mutex poisoned")
                    .error = Some(error.clone());
                return Err(error);
            }
            let mut state = self.state.lock().expect("model library mutex poisoned");
            state.registered_aliases.remove(alias);
            if let Some(model) = state.models.iter_mut().find(|model| model.alias == *alias) {
                model.compatible = false;
                model.reason = Some(
                    "The model was unloaded while refreshing the selected folder; rescan to register it again."
                        .to_owned(),
                );
            }
        }

        let (models, registered_aliases) = self.scan(&path);
        let mut state = self.state.lock().expect("model library mutex poisoned");
        state.selected_path = Some(path.clone());
        state.models = models;
        state.registered_aliases = registered_aliases;
        state.error = None;
        drop(state);
        if persist_selected {
            if let Err(error) = persist_path(&self.persistence_path, &path) {
                self.state
                    .lock()
                    .expect("model library mutex poisoned")
                    .error = Some(error.clone());
                return Err(error);
            }
        }
        Ok(())
    }

    fn scan(&self, path: &Path) -> (Vec<ModelLibraryModelV1>, BTreeSet<String>) {
        let mut paths = Vec::new();
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.take(MAX_DIRECTORY_ENTRIES_INSPECTED).flatten() {
                if paths.len() == MAX_MODEL_LIBRARY_FILES_V1 {
                    break;
                }
                let entry_path = entry.path();
                let is_gguf = entry_path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("gguf"));
                let regular = fs::symlink_metadata(&entry_path)
                    .is_ok_and(|metadata| !metadata.file_type().is_symlink() && metadata.is_file());
                if is_gguf && regular {
                    paths.push(entry_path);
                }
            }
        }
        paths.sort();

        let mut aliases = BTreeSet::new();
        let mut registered = BTreeSet::new();
        let mut rows = Vec::with_capacity(paths.len());
        let mut pending = Vec::new();
        let mut assistants = Vec::new();
        for gguf_path in paths {
            let file_name = gguf_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("model.gguf")
                .to_owned();
            let path_alias = model_alias(&gguf_path);
            let size_bytes = fs::metadata(&gguf_path).map_or(0, |value| value.len());
            let verified = match VerifiedGguf::open(&gguf_path) {
                Ok(value) => value,
                Err(_) => {
                    rows.push((
                        gguf_path,
                        incompatible_model(
                            path_alias,
                            file_name,
                            size_bytes,
                            "unknown",
                            false,
                            "The file is not a valid GGUF artifact.",
                        ),
                    ));
                    continue;
                }
            };
            let architecture = verified.architecture().to_owned();
            // Ministral 3 is a direct official GGUF identity.  Its public
            // alias is fixed by the reviewed model lock rather than derived
            // from a user-controlled filename.
            let alias = if architecture == MINISTRAL3_GGUF_ARCHITECTURE {
                MINISTRAL3_MODEL_ALIAS.to_owned()
            } else {
                path_alias
            };
            if architecture == "gemma4mtp" {
                assistants.push(MtpAssistantCandidateV1 {
                    validation: inspect_mtp_assistant(&gguf_path, &verified),
                    gguf_path,
                    alias,
                    file_name,
                    size_bytes,
                });
                continue;
            }
            if architecture == "deepseek4" {
                let reason = format!(
                    "DeepSeek V4 is recognized, but production loading is unavailable. The reviewed official tensor payload requires at least {DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES} resident bytes before KV cache and workspace."
                );
                rows.push((
                    gguf_path,
                    incompatible_model(alias, file_name, size_bytes, &architecture, true, &reason),
                ));
                continue;
            }
            if architecture == "minimax-m3" {
                let reason = format!(
                    "MiniMax M3 is recognized under the MiniMax Community License, but production loading is unavailable. The official manifest is internally inconsistent, so fail-closed admission requires at least {MINIMAX_M3_CAPACITY_ADMISSION_BYTES} resident bytes before KV cache and workspace."
                );
                rows.push((
                    gguf_path,
                    incompatible_model(alias, file_name, size_bytes, &architecture, true, &reason),
                ));
                continue;
            }
            if architecture == "diffusion-gemma" {
                let reason = format!(
                    "DiffusionGemma 26B-A4B is recognized under Apache-2.0, but production loading is unavailable. The reviewed official BF16 shard files require at least {DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES} resident bytes before KV cache and workspace."
                );
                rows.push((
                    gguf_path,
                    incompatible_model(alias, file_name, size_bytes, &architecture, true, &reason),
                ));
                continue;
            }
            let supported_architecture = matches!(
                architecture.as_str(),
                "qwen35" | "qwen35moe" | "gemma4" | "gemma4moe" | MINISTRAL3_GGUF_ARCHITECTURE
            );
            if !supported_architecture {
                rows.push((
                    gguf_path,
                    incompatible_model(
                        alias,
                        file_name,
                        size_bytes,
                        &architecture,
                        false,
                        "This GGUF architecture is not implemented by sLLM.",
                    ),
                ));
                continue;
            }
            if !aliases.insert(alias.clone()) {
                rows.push((
                    gguf_path,
                    incompatible_model(
                        alias,
                        file_name,
                        size_bytes,
                        &architecture,
                        true,
                        "Another GGUF in this folder resolves to the same model alias.",
                    ),
                ));
                continue;
            }
            let Some(device) = self.device.as_ref() else {
                rows.push((
                    gguf_path,
                    incompatible_model(
                        alias,
                        file_name,
                        size_bytes,
                        &architecture,
                        true,
                        "No supported AMD GPU was detected on device 0.",
                    ),
                ));
                continue;
            };

            // The official Ministral 3 path intentionally branches before
            // sibling derived-lock enforcement.  The preflight opens the
            // retained official GGUF, authenticates its full-file identity,
            // verifies its canonical text-only catalog, and builds the exact
            // resident plan.  Metadata alone must never admit this source.
            if architecture == MINISTRAL3_GGUF_ARCHITECTURE {
                let registration = match ministral3_registration_with_preflight(
                    gguf_path.clone(),
                    device,
                    |path| {
                        crate::production::ministral3_model_plan_preflight_v1(path)
                            .map_err(|error| error.to_string())
                    },
                ) {
                    Ok(value) => value,
                    Err(_) => {
                        rows.push((
                            gguf_path,
                            incompatible_model(
                                alias,
                                file_name,
                                size_bytes,
                                &architecture,
                                true,
                                "The official Ministral 3 GGUF verifier or runtime weight-plan verification failed.",
                            ),
                        ));
                        continue;
                    }
                };
                if registration.resident_bytes > device.total_memory_bytes {
                    rows.push((
                        gguf_path,
                        incompatible_model(
                            alias,
                            file_name,
                            size_bytes,
                            &architecture,
                            true,
                            "The model's resident weights exceed device 0 memory.",
                        ),
                    ));
                    continue;
                }
                pending.push(PendingModelV1 {
                    sort_path: gguf_path,
                    alias: registration.alias.clone(),
                    file_name,
                    size_bytes,
                    architecture: registration.architecture.clone(),
                    registration,
                    device_total_memory_bytes: device.total_memory_bytes,
                    mtp_companion_file_name: None,
                    mtp_pair_target_identity: None,
                });
                continue;
            }

            let derived_lock_path = derived_lock_path(&gguf_path);
            let derived = match read_derived_gguf_lock(&derived_lock_path) {
                Ok(value) => value,
                Err(_) => {
                    rows.push((
                        gguf_path,
                        incompatible_model(
                            alias,
                            file_name,
                            size_bytes,
                            &architecture,
                            true,
                            "A matching canonical .derived-lock.json file is required.",
                        ),
                    ));
                    continue;
                }
            };
            let model_identity = if derived.semantic_model_id.starts_with("qwen35moe:") {
                QWEN35_MOE_MODEL_FINGERPRINT.to_owned()
            } else if derived.semantic_model_id.starts_with("gemma4moe:") {
                if derived.semantic_model_id != format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}")
                    || architecture != "gemma4moe"
                    || derived.source_lock_fingerprints.as_slice() != [GEMMA4_MOE_MODEL_FINGERPRINT]
                {
                    rows.push((
                        gguf_path,
                        incompatible_model(
                            alias,
                            file_name,
                            size_bytes,
                            &architecture,
                            true,
                            "The Gemma 4 MoE GGUF does not match the reviewed source identity.",
                        ),
                    ));
                    continue;
                }
                GEMMA4_MOE_MODEL_FINGERPRINT.to_owned()
            } else {
                match builtin_reviewed_model_lock(&derived.source_lock_fingerprints) {
                    Ok(ReviewedModelLock::Qwen35(lock)) => lock.fingerprint().to_owned(),
                    Ok(ReviewedModelLock::Gemma4(lock)) => lock.fingerprint().to_owned(),
                    Ok(ReviewedModelLock::Ministral3(_)) => {
                        rows.push((
                            gguf_path,
                            incompatible_model(
                                alias,
                                file_name,
                                size_bytes,
                                &architecture,
                                true,
                                "A direct official Ministral 3 model lock cannot be used as a derived GGUF source lock.",
                            ),
                        ));
                        continue;
                    }
                    Err(_) => {
                        rows.push((
                            gguf_path,
                            incompatible_model(
                                alias,
                                file_name,
                                size_bytes,
                                &architecture,
                                true,
                                "The GGUF does not match a reviewed sLLM model lock.",
                            ),
                        ));
                        continue;
                    }
                }
            };
            let mtp_pair_target_identity = mtp_target_identity(
                &architecture,
                &model_identity,
                &derived.semantic_model_id,
                &derived.source_lock_fingerprints,
            );
            let (plan_identity, resident_bytes) =
                match dynamic_model_plan_preflight_v1(&gguf_path, &derived) {
                    Ok(value) => value,
                    Err(_) => {
                        rows.push((
                            gguf_path,
                            incompatible_model(
                                alias,
                                file_name,
                                size_bytes,
                                &architecture,
                                true,
                                "GGUF identity or runtime weight-plan verification failed.",
                            ),
                        ));
                        continue;
                    }
                };
            if resident_bytes > device.total_memory_bytes {
                rows.push((
                    gguf_path,
                    incompatible_model(
                        alias,
                        file_name,
                        size_bytes,
                        &architecture,
                        true,
                        "The model's resident weights exceed device 0 memory.",
                    ),
                ));
                continue;
            }
            let registration = ModelLibraryRegistrationV1 {
                alias: alias.clone(),
                gguf_path: gguf_path.clone(),
                derived_lock_path: Some(derived_lock_path),
                architecture: architecture.clone(),
                model_identity,
                plan_identity,
                resident_bytes,
                device_index: device.device_index,
                target: device.target.clone(),
                mtp_assistant_gguf_path: None,
                mtp_assistant_derived_lock_path: None,
                mtp_assistant_identity: None,
                mtp_semantic_pair_identity: None,
            };
            pending.push(PendingModelV1 {
                sort_path: gguf_path,
                alias,
                file_name,
                size_bytes,
                architecture,
                registration,
                device_total_memory_bytes: device.total_memory_bytes,
                mtp_companion_file_name: None,
                mtp_pair_target_identity,
            });
        }

        rows.extend(resolve_mtp_companions(&mut pending, &assistants));
        for model in pending {
            match (self.register)(model.registration) {
                Ok(()) => {
                    registered.insert(model.alias.clone());
                    rows.push((
                        model.sort_path,
                        ModelLibraryModelV1 {
                            alias: model.alias,
                            file_name: model.file_name,
                            size_bytes: model.size_bytes,
                            architecture: model.architecture,
                            supported_architecture: true,
                            compatible: true,
                            reason: None,
                            mtp_companion_file_name: model.mtp_companion_file_name,
                            mtp_companion_for: None,
                        },
                    ));
                }
                Err(error) => rows.push((
                    model.sort_path,
                    ModelLibraryModelV1 {
                        alias: model.alias,
                        file_name: model.file_name,
                        size_bytes: model.size_bytes,
                        architecture: model.architecture,
                        supported_architecture: true,
                        compatible: false,
                        reason: Some(format!("The model could not be registered: {error}")),
                        mtp_companion_file_name: model.mtp_companion_file_name,
                        mtp_companion_for: None,
                    },
                )),
            }
        }
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        (
            rows.into_iter().map(|(_, model)| model).collect(),
            registered,
        )
    }
}

/// Resolve the immutable identity fields for a direct official Ministral 3
/// registration.  The preflight callback is kept injectable so unit tests can
/// exercise registration construction without requiring the 6.8 GiB fixture;
/// production always supplies the full official verifier/preflight.
fn ministral3_registration_fields_with_preflight(
    gguf_path: &Path,
    preflight: impl Fn(&Path) -> Result<(String, u64), String>,
) -> Result<(String, String, u64), String> {
    let (plan_identity, resident_bytes) = preflight(gguf_path)?;
    if resident_bytes != MINISTRAL3_WEIGHT_RESIDENT_BYTES {
        return Err(format!(
            "Ministral 3 resident-byte plan differs from the reviewed exact value: expected {}, got {resident_bytes}",
            MINISTRAL3_WEIGHT_RESIDENT_BYTES
        ));
    }
    let Some(plan_digest) = plan_identity.strip_prefix("sha256:") else {
        return Err("Ministral 3 resident plan identity is not a SHA-256 digest".to_owned());
    };
    if plan_digest.len() != 64 || !plan_digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("Ministral 3 resident plan identity is not a SHA-256 digest".to_owned());
    }
    Ok((
        MINISTRAL3_MODEL_LOCK_FINGERPRINT.to_owned(),
        plan_identity,
        resident_bytes,
    ))
}

fn ministral3_registration_with_preflight(
    gguf_path: PathBuf,
    device: &ModelLibraryDeviceV1,
    preflight: impl Fn(&Path) -> Result<(String, u64), String>,
) -> Result<ModelLibraryRegistrationV1, String> {
    let (model_identity, plan_identity, resident_bytes) =
        ministral3_registration_fields_with_preflight(&gguf_path, preflight)?;
    Ok(ModelLibraryRegistrationV1 {
        alias: MINISTRAL3_MODEL_ALIAS.to_owned(),
        gguf_path,
        derived_lock_path: None,
        architecture: MINISTRAL3_GGUF_ARCHITECTURE.to_owned(),
        model_identity,
        plan_identity,
        resident_bytes,
        device_index: device.device_index,
        target: device.target.clone(),
        mtp_assistant_gguf_path: None,
        mtp_assistant_derived_lock_path: None,
        mtp_assistant_identity: None,
        mtp_semantic_pair_identity: None,
    })
}

fn mtp_target_identity(
    architecture: &str,
    model_identity: &str,
    semantic_model_id: &str,
    source_lock_fingerprints: &[String],
) -> Option<String> {
    (architecture == "gemma4"
        && model_identity == GEMMA4_12B_IT_FINGERPRINT
        && semantic_model_id == format!("gemma4:{GEMMA4_12B_IT_FINGERPRINT}")
        && source_lock_fingerprints.first().map(String::as_str) == Some(GEMMA4_12B_IT_FINGERPRINT))
    .then(|| model_identity.to_owned())
}

fn inspect_mtp_assistant(
    gguf_path: &Path,
    verified: &VerifiedGguf,
) -> Result<ValidMtpAssistantV1, String> {
    if verified.architecture() != "gemma4mtp" || !verified.is_assistant_only() {
        return Err("The Gemma 4 MTP artifact is not a canonical assistant companion.".to_owned());
    }
    let target_identity = gguf_metadata_string(verified, "gemma4mtp.target_fingerprint")?;
    let assistant_identity = gguf_metadata_string(verified, "gemma4mtp.assistant_fingerprint")?;
    let semantic_pair_identity = gguf_metadata_string(verified, "gemma4mtp.semantic_pair_id")?;
    let expected_pair =
        gemma4_mtp_pair_semantic_id(GEMMA4_12B_IT_FINGERPRINT, GEMMA4_MTP_FINGERPRINT);
    if target_identity != GEMMA4_12B_IT_FINGERPRINT
        || assistant_identity != GEMMA4_MTP_FINGERPRINT
        || semantic_pair_identity != expected_pair
    {
        return Err(
            "The Gemma 4 MTP companion pair metadata does not match the reviewed target and assistant identities."
                .to_owned(),
        );
    }
    let derived_lock_path = derived_lock_path(gguf_path);
    let derived = read_derived_gguf_lock(&derived_lock_path).map_err(|_| {
        "A matching canonical .derived-lock.json file is required for this Gemma 4 MTP companion."
            .to_owned()
    })?;
    if derived.semantic_model_id != semantic_pair_identity
        || derived.source_lock_fingerprints.as_slice()
            != [
                GEMMA4_12B_IT_FINGERPRINT.to_owned(),
                GEMMA4_MTP_FINGERPRINT.to_owned(),
            ]
    {
        return Err(
            "The Gemma 4 MTP companion derived lock does not match its pair metadata.".to_owned(),
        );
    }
    verify_derived_gguf(derived, gguf_path).map_err(|_| {
        "The Gemma 4 MTP companion GGUF does not match its derived lock identity.".to_owned()
    })?;
    Ok(ValidMtpAssistantV1 {
        target_identity,
        assistant_identity,
        semantic_pair_identity,
        derived_lock_path,
    })
}

/// Returns resident bytes from the exact canonical assistant GGUF load plan.
/// This is deliberately called only after pair uniqueness has been established
/// so an unattached assistant is never treated as a standalone model.
fn verified_mtp_assistant_resident_bytes(gguf_path: &Path) -> Result<u64, String> {
    let derived_path = derived_lock_path(gguf_path);
    let derived = read_derived_gguf_lock(&derived_path)
        .map_err(|_| "A matching canonical .derived-lock.json file is required.".to_owned())?;
    let verified_derived = verify_derived_gguf(derived, gguf_path).map_err(|_| {
        "The Gemma 4 MTP companion GGUF does not match its derived lock identity.".to_owned()
    })?;
    let assistant_lock = parse_gemma4_mtp_model_lock(include_bytes!(
        "../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
    ))
    .map_err(|_| "The canonical Gemma 4 MTP assistant lock is invalid.".to_owned())?;
    let target_lock = match builtin_reviewed_model_lock(&[GEMMA4_12B_IT_FINGERPRINT.to_owned()])
        .map_err(|_| "The reviewed Gemma 4 target lock is unavailable.".to_owned())?
    {
        ReviewedModelLock::Gemma4(lock) => lock,
        ReviewedModelLock::Qwen35(_) | ReviewedModelLock::Ministral3(_) => {
            return Err("The reviewed Gemma 4 target lock has the wrong architecture.".to_owned());
        }
    };
    let source = verify_gguf_gemma4_mtp(verified_derived.gguf, &assistant_lock, &target_lock)
        .map_err(|_| {
            "The Gemma 4 MTP companion is not the canonical assistant artifact.".to_owned()
        })?;
    let plan = build_verified_gemma4_mtp_weight_load_plan(&assistant_lock, &source)
        .map_err(|_| "The Gemma 4 MTP companion weight plan is invalid.".to_owned())?;
    Ok(plan.total_destination_bytes)
}

fn gguf_metadata_string(verified: &VerifiedGguf, key: &str) -> Result<String, String> {
    match verified.metadata_value(key) {
        Some(GgufValue::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(format!(
            "The Gemma 4 MTP companion metadata {key} is missing or invalid."
        )),
    }
}

fn resolve_mtp_companions(
    targets: &mut [PendingModelV1],
    assistants: &[MtpAssistantCandidateV1],
) -> Vec<(PathBuf, ModelLibraryModelV1)> {
    resolve_mtp_companions_with_resident_resolver(
        targets,
        assistants,
        verified_mtp_assistant_resident_bytes,
    )
}

fn resolve_mtp_companions_with_resident_resolver(
    targets: &mut [PendingModelV1],
    assistants: &[MtpAssistantCandidateV1],
    resident_resolver: impl Fn(&Path) -> Result<u64, String>,
) -> Vec<(PathBuf, ModelLibraryModelV1)> {
    let mut rows = Vec::with_capacity(assistants.len());
    for assistant in assistants {
        let (reason, companion_for) = match &assistant.validation {
            Err(reason) => (reason.clone(), None),
            Ok(valid) => {
                let assistant_count = assistants
                    .iter()
                    .filter(|candidate| {
                        candidate.validation.as_ref().is_ok_and(|other| {
                            other.target_identity == valid.target_identity
                                && other.assistant_identity == valid.assistant_identity
                                && other.semantic_pair_identity == valid.semantic_pair_identity
                        })
                    })
                    .count();
                let matching_targets = targets
                    .iter()
                    .enumerate()
                    .filter(|(_, target)| {
                        target.mtp_pair_target_identity.as_deref()
                            == Some(valid.target_identity.as_str())
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                if assistant_count != 1 {
                    (
                        "Multiple Gemma 4 MTP assistant GGUFs match this target; no companion was attached."
                            .to_owned(),
                        None,
                    )
                } else if matching_targets.is_empty() {
                    (
                        "This Gemma 4 MTP assistant requires exactly one matching reviewed target GGUF in the selected folder."
                            .to_owned(),
                        None,
                    )
                } else if matching_targets.len() != 1 {
                    (
                        "Multiple target GGUFs match this Gemma 4 MTP assistant; no companion was attached."
                            .to_owned(),
                        None,
                    )
                } else {
                    let target = &mut targets[matching_targets[0]];
                    if target.registration.target != "gfx1201" {
                        (
                            format!(
                                "Gemma 4 MTP companion requires exact gfx1201 target; selected target is {}; no companion was attached.",
                                target.registration.target
                            ),
                            None,
                        )
                    } else {
                        match resident_resolver(&assistant.gguf_path) {
                            Err(error) => (
                                format!(
                                    "The Gemma 4 MTP companion failed canonical resident-plan verification: {error}; no companion was attached."
                                ),
                                None,
                            ),
                            Ok(assistant_resident_bytes) => match target
                                .registration
                                .resident_bytes
                                .checked_add(assistant_resident_bytes)
                            {
                                None => (
                                    "The target and Gemma 4 MTP assistant resident-byte total overflowed; no companion was attached."
                                        .to_owned(),
                                    None,
                                ),
                                Some(resident_bytes)
                                    if resident_bytes > target.device_total_memory_bytes => (
                                    format!(
                                        "Insufficient VRAM: target and Gemma 4 MTP assistant resident weights require {resident_bytes} bytes, exceeding selected device memory of {} bytes; no companion was attached.",
                                        target.device_total_memory_bytes
                                    ),
                                    None,
                                ),
                                Some(resident_bytes) => {
                                    target.registration.resident_bytes = resident_bytes;
                                    target.registration.mtp_assistant_gguf_path =
                                        Some(assistant.gguf_path.clone());
                                    target.registration.mtp_assistant_derived_lock_path =
                                        Some(valid.derived_lock_path.clone());
                                    target.registration.mtp_assistant_identity =
                                        Some(valid.assistant_identity.clone());
                                    target.registration.mtp_semantic_pair_identity =
                                        Some(valid.semantic_pair_identity.clone());
                                    target.mtp_companion_file_name =
                                        Some(assistant.file_name.clone());
                                    (
                                        format!(
                                            "Companion assistant for {}; it cannot be loaded as a standalone model.",
                                            target.alias
                                        ),
                                        Some(target.alias.clone()),
                                    )
                                }
                            },
                        }
                    }
                }
            }
        };
        rows.push((
            assistant.gguf_path.clone(),
            ModelLibraryModelV1 {
                alias: assistant.alias.clone(),
                file_name: assistant.file_name.clone(),
                size_bytes: assistant.size_bytes,
                architecture: "gemma4mtp".to_owned(),
                supported_architecture: true,
                compatible: false,
                reason: Some(reason),
                mtp_companion_file_name: None,
                mtp_companion_for: companion_for,
            },
        ));
    }
    rows
}

fn incompatible_model(
    alias: String,
    file_name: String,
    size_bytes: u64,
    architecture: &str,
    supported_architecture: bool,
    reason: &str,
) -> ModelLibraryModelV1 {
    ModelLibraryModelV1 {
        alias,
        file_name,
        size_bytes,
        architecture: architecture.to_owned(),
        supported_architecture,
        compatible: false,
        reason: Some(reason.to_owned()),
        mtp_companion_file_name: None,
        mtp_companion_for: None,
    }
}

fn model_alias(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let mut alias = String::with_capacity(stem.len().min(128));
    let mut separator = false;
    for byte in stem.bytes().take(128) {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_') {
            alias.push(byte as char);
            separator = false;
        } else if !separator && !alias.is_empty() {
            alias.push('-');
            separator = true;
        }
    }
    while alias.ends_with('-') {
        alias.pop();
    }
    if alias.is_empty() {
        "model".to_owned()
    } else {
        alias
    }
}

fn derived_lock_path(gguf_path: &Path) -> PathBuf {
    let stem = gguf_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    gguf_path.with_file_name(format!("{stem}.derived-lock.json"))
}

fn canonical_directory(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("model folder path must be absolute".to_owned());
    }
    let metadata =
        fs::symlink_metadata(path).map_err(|_| "model folder could not be inspected".to_owned())?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("model folder must be a non-symlink directory".to_owned());
    }
    fs::canonicalize(path).map_err(|_| "model folder could not be resolved".to_owned())
}

fn read_persisted_path(path: &Path) -> Result<Option<PathBuf>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("model-library state could not be inspected".to_owned()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("model-library state must be a regular non-symlink file".to_owned());
    }
    if metadata.len() > MAX_PERSISTED_STATE_BYTES as u64 {
        return Err("model-library state exceeds its size bound".to_owned());
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|file| {
            file.take((MAX_PERSISTED_STATE_BYTES + 1) as u64)
                .read_to_end(&mut bytes)
        })
        .map_err(|_| "model-library state could not be read".to_owned())?;
    let state: PersistedStateV1 =
        serde_json::from_slice(&bytes).map_err(|_| "model-library state is invalid".to_owned())?;
    if state.schema_version != "sllm-model-library-state-v1" {
        return Err("model-library state schema is unsupported".to_owned());
    }
    Ok(Some(PathBuf::from(state.selected_path)))
}

fn persist_path(state_path: &Path, selected_path: &Path) -> Result<(), String> {
    let parent = state_path
        .parent()
        .ok_or_else(|| "model-library state path has no parent".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|_| "model-library state directory could not be created".to_owned())?;
    let selected = selected_path.display().to_string();
    let bytes = serde_json::to_vec(&PersistedStateRefV1 {
        schema_version: "sllm-model-library-state-v1",
        selected_path: &selected,
    })
    .map_err(|_| "model-library state could not be serialized".to_owned())?;
    let temp = state_path.with_extension(format!("tmp-{}", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temp)
        .map_err(|_| "temporary model-library state could not be created".to_owned())?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| "model-library state could not be written".to_owned())?;
        fs::rename(&temp, state_path)
            .map_err(|_| "model-library state could not be published".to_owned())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{
        DerivedGgufConverter, DerivedGgufLock, GGUF_ALIGNMENT, GgufArray, GgufTensorRecipeV1,
        GgufTensorType, GgufValue, GgufWritePlan, GgufWriteReport, GgufWriteTensor,
        SLLM_EXTENSION_VERSION_KEY, SLLM_GGUF_EXTENSION_VERSION, SLLM_TENSOR_RECIPE_KEY,
        SLLM_TENSOR_RECIPE_SHA256_KEY, write_gguf,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("sllm-{label}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_device() -> ModelLibraryDeviceV1 {
        ModelLibraryDeviceV1 {
            device_index: 0,
            target: "gfx1201".to_owned(),
            name: "test GPU".to_owned(),
            total_memory_bytes: u64::MAX,
        }
    }

    #[test]
    fn direct_ministral3_registration_fields_bind_reviewed_identity_without_fixture() {
        let path = Path::new("/models/Ministral-3-3B-Instruct-2512-BF16.gguf");
        let expected_plan = format!("sha256:{}", "a".repeat(64));
        let device = test_device();
        let registration =
            ministral3_registration_with_preflight(path.to_owned(), &device, |candidate| {
                assert_eq!(candidate, path);
                Ok((expected_plan.clone(), MINISTRAL3_WEIGHT_RESIDENT_BYTES))
            })
            .expect("injected direct preflight");
        assert_eq!(registration.alias, MINISTRAL3_MODEL_ALIAS);
        assert_eq!(registration.architecture, MINISTRAL3_GGUF_ARCHITECTURE);
        assert_eq!(
            registration.model_identity,
            MINISTRAL3_MODEL_LOCK_FINGERPRINT
        );
        assert_eq!(registration.plan_identity, expected_plan);
        assert_eq!(
            registration.resident_bytes,
            MINISTRAL3_WEIGHT_RESIDENT_BYTES
        );
        assert!(registration.derived_lock_path.is_none());
        assert!(registration.mtp_assistant_gguf_path.is_none());
        assert!(registration.mtp_assistant_derived_lock_path.is_none());
        assert!(registration.mtp_assistant_identity.is_none());
        assert!(registration.mtp_semantic_pair_identity.is_none());
    }

    #[test]
    fn direct_ministral3_resident_byte_drift_fails_closed() {
        let error =
            ministral3_registration_fields_with_preflight(Path::new("/models/model.gguf"), |_| {
                Ok((
                    format!("sha256:{}", "a".repeat(64)),
                    MINISTRAL3_WEIGHT_RESIDENT_BYTES + 1,
                ))
            })
            .expect_err("resident-byte drift");
        assert!(error.contains("resident-byte plan differs"));
    }

    fn pending_target(alias: &str) -> PendingModelV1 {
        let gguf_path = PathBuf::from(format!("/models/{alias}.gguf"));
        PendingModelV1 {
            sort_path: gguf_path.clone(),
            alias: alias.to_owned(),
            file_name: format!("{alias}.gguf"),
            size_bytes: 100,
            architecture: "gemma4".to_owned(),
            registration: ModelLibraryRegistrationV1 {
                alias: alias.to_owned(),
                gguf_path,
                derived_lock_path: Some(PathBuf::from(format!(
                    "/models/{alias}.derived-lock.json"
                ))),
                architecture: "gemma4".to_owned(),
                model_identity: GEMMA4_12B_IT_FINGERPRINT.to_owned(),
                plan_identity: "plan".to_owned(),
                resident_bytes: 100,
                device_index: 0,
                target: "gfx1201".to_owned(),
                mtp_assistant_gguf_path: None,
                mtp_assistant_derived_lock_path: None,
                mtp_assistant_identity: None,
                mtp_semantic_pair_identity: None,
            },
            device_total_memory_bytes: u64::MAX,
            mtp_companion_file_name: None,
            mtp_pair_target_identity: Some(GEMMA4_12B_IT_FINGERPRINT.to_owned()),
        }
    }

    fn valid_assistant(file_name: &str) -> MtpAssistantCandidateV1 {
        let pair = gemma4_mtp_pair_semantic_id(GEMMA4_12B_IT_FINGERPRINT, GEMMA4_MTP_FINGERPRINT);
        let gguf_path = PathBuf::from(format!("/models/{file_name}"));
        MtpAssistantCandidateV1 {
            alias: model_alias(&gguf_path),
            file_name: file_name.to_owned(),
            size_bytes: 80,
            gguf_path,
            validation: Ok(ValidMtpAssistantV1 {
                target_identity: GEMMA4_12B_IT_FINGERPRINT.to_owned(),
                assistant_identity: GEMMA4_MTP_FINGERPRINT.to_owned(),
                semantic_pair_identity: pair,
                derived_lock_path: PathBuf::from(format!(
                    "/models/{}.derived-lock.json",
                    file_name.trim_end_matches(".gguf")
                )),
            }),
        }
    }

    fn write_structural_mtp_gguf(
        root: &Path,
        target_identity: &str,
        assistant_identity: &str,
    ) -> (PathBuf, GgufWriteReport, String) {
        let gguf_path = root.join("assistant.gguf");
        let pair = gemma4_mtp_pair_semantic_id(target_identity, assistant_identity);
        let recipe = GgufTensorRecipeV1 {
            schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
            semantic_model_id: pair.clone(),
            source_lock_fingerprints: vec![
                target_identity.to_owned(),
                assistant_identity.to_owned(),
            ],
            bindings: Vec::new(),
            logical_shapes: Vec::new(),
            static_fp8_kv: Vec::new(),
            known_unconsumed_tensors: vec!["model.norm.weight".to_owned()],
        };
        let source_ranges = concat!(
            "[{\"name\":\"model.norm.weight\",",
            "\"source_file\":\"model.safetensors\",\"dtype\":\"BF16\",",
            "\"shape\":[4],\"data_offsets\":[0,8],",
            "\"absolute_byte_range\":[5368,5376]}]"
        );
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".to_owned(),
                GgufValue::String("gemma4mtp".to_owned()),
            ),
            (
                "general.alignment".to_owned(),
                GgufValue::U32(GGUF_ALIGNMENT as u32),
            ),
            (
                "gemma4mtp.role".to_owned(),
                GgufValue::String("assistant".to_owned()),
            ),
            (
                "gemma4mtp.target_fingerprint".to_owned(),
                GgufValue::String(target_identity.to_owned()),
            ),
            (
                "gemma4mtp.assistant_fingerprint".to_owned(),
                GgufValue::String(assistant_identity.to_owned()),
            ),
            (
                "gemma4mtp.semantic_pair_id".to_owned(),
                GgufValue::String(pair.clone()),
            ),
            (
                "gemma4mtp.tensor_catalog_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "3".repeat(64))),
            ),
            (
                "gemma4mtp.source_model_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "4".repeat(64))),
            ),
            (
                "gemma4mtp.source_header_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "5".repeat(64))),
            ),
            (
                "gemma4mtp.layer_mapping".to_owned(),
                GgufValue::Array(GgufArray::U32(vec![0, 1, 2, 3])),
            ),
            (
                "gemma4mtp.kv_mapping".to_owned(),
                GgufValue::Array(GgufArray::U32(vec![46, 46, 46, 47])),
            ),
            (
                "gemma4mtp.layer_types".to_owned(),
                GgufValue::Array(GgufArray::String(vec![
                    "sliding_attention".to_owned(),
                    "sliding_attention".to_owned(),
                    "sliding_attention".to_owned(),
                    "full_attention".to_owned(),
                ])),
            ),
            (
                "gemma4mtp.tokenizer_identity".to_owned(),
                GgufValue::String("{}".to_owned()),
            ),
            (
                "gemma4mtp.source_ranges".to_owned(),
                GgufValue::String(source_ranges.to_owned()),
            ),
        ]);
        metadata.insert(
            SLLM_EXTENSION_VERSION_KEY.to_owned(),
            GgufValue::U32(SLLM_GGUF_EXTENSION_VERSION),
        );
        metadata.insert(
            SLLM_TENSOR_RECIPE_KEY.to_owned(),
            GgufValue::String(recipe.canonical_json().unwrap()),
        );
        metadata.insert(
            SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
            GgufValue::String(recipe.digest().unwrap()),
        );
        let report = write_gguf(
            &gguf_path,
            &GgufWritePlan {
                metadata,
                tensors: vec![GgufWriteTensor {
                    name: "model.norm.weight".to_owned(),
                    source_name: "model.norm.weight".to_owned(),
                    dimensions: vec![4],
                    tensor_type: GgufTensorType::Bf16,
                }],
            },
            |_, _, length| Ok(vec![0_u8; length]),
        )
        .unwrap();
        (gguf_path, report, pair)
    }

    fn write_architecture_only_gguf(root: &Path, architecture: &str) -> PathBuf {
        let gguf_path = root.join(format!("{architecture}.gguf"));
        write_gguf(
            &gguf_path,
            &GgufWritePlan {
                metadata: BTreeMap::from([
                    (
                        "general.architecture".to_owned(),
                        GgufValue::String(architecture.to_owned()),
                    ),
                    (
                        "general.alignment".to_owned(),
                        GgufValue::U32(GGUF_ALIGNMENT as u32),
                    ),
                ]),
                tensors: vec![GgufWriteTensor {
                    name: "model.norm.weight".to_owned(),
                    source_name: "model.norm.weight".to_owned(),
                    dimensions: vec![1],
                    tensor_type: GgufTensorType::F32,
                }],
            },
            |_, _, length| Ok(vec![0_u8; length]),
        )
        .unwrap();
        gguf_path
    }

    fn write_parser_only_architecture_gguf(
        root: &Path,
        file_name: &str,
        architecture: &str,
    ) -> PathBuf {
        fn push_string(output: &mut Vec<u8>, value: &str) {
            output.extend_from_slice(&(value.len() as u64).to_le_bytes());
            output.extend_from_slice(value.as_bytes());
        }

        let gguf_path = root.join(file_name);
        let mut output = Vec::new();
        output.extend_from_slice(b"GGUF");
        output.extend_from_slice(&3_u32.to_le_bytes());
        output.extend_from_slice(&1_u64.to_le_bytes());
        output.extend_from_slice(&2_u64.to_le_bytes());

        push_string(&mut output, "general.architecture");
        output.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut output, architecture);
        push_string(&mut output, "general.alignment");
        output.extend_from_slice(&4_u32.to_le_bytes());
        output.extend_from_slice(&(GGUF_ALIGNMENT as u32).to_le_bytes());

        push_string(&mut output, "model.norm.weight");
        output.extend_from_slice(&1_u32.to_le_bytes());
        output.extend_from_slice(&1_u64.to_le_bytes());
        output.extend_from_slice(&GgufTensorType::F32.raw().to_le_bytes());
        output.extend_from_slice(&0_u64.to_le_bytes());
        while output.len() % (GGUF_ALIGNMENT as usize) != 0 {
            output.push(0);
        }
        output.extend_from_slice(&0_f32.to_le_bytes());
        fs::write(&gguf_path, output).unwrap();
        gguf_path
    }

    fn write_parser_only_diffusion_gemma_gguf(root: &Path) -> PathBuf {
        write_parser_only_architecture_gguf(root, "diffusion-gemma.gguf", "diffusion-gemma")
    }

    #[test]
    fn browse_lists_only_real_child_directories_in_sorted_order() {
        let root = temp_dir("library-browse");
        fs::create_dir(root.join("zeta")).unwrap();
        fs::create_dir(root.join("alpha")).unwrap();
        fs::write(root.join("plain.txt"), b"x").unwrap();
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            None,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let listing = library.browse(Some(&root)).unwrap();
        assert_eq!(listing.directories.len(), 2);
        assert_eq!(listing.directories[0].name, "alpha");
        assert_eq!(listing.directories[1].name, "zeta");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selected_folder_is_persisted_and_restored_without_a_gpu() {
        let root = temp_dir("library-state");
        let models = root.join("models");
        fs::create_dir(&models).unwrap();
        let state_path = root.join("config/state.json");
        let library = ModelLibraryV1::open(
            state_path.clone(),
            root.clone(),
            None,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let selected = library.select(&models).unwrap();
        assert_eq!(
            selected.selected_path.as_deref(),
            Some(models.to_str().unwrap())
        );
        let restored =
            ModelLibraryV1::open(state_path, root.clone(), None, |_| Ok(()), |_| Ok(())).unwrap();
        assert_eq!(restored.snapshot().selected_path, selected.selected_path);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_unload_failure_keeps_lifecycle_state_retryable_and_does_not_persist_new_path() {
        let root = temp_dir("library-partial-unload");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir(&first).unwrap();
        fs::create_dir(&second).unwrap();
        let state_path = root.join("config/state.json");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured_calls = Arc::clone(&calls);
        let fail_b_once = Arc::new(AtomicBool::new(true));
        let captured_failure = Arc::clone(&fail_b_once);
        let library = ModelLibraryV1::open(
            state_path.clone(),
            root.clone(),
            None,
            |_| Ok(()),
            move |alias| {
                captured_calls.lock().unwrap().push(alias.to_owned());
                if alias == "b" && captured_failure.swap(false, Ordering::SeqCst) {
                    Err("injected unload failure".to_owned())
                } else {
                    Ok(())
                }
            },
        )
        .unwrap();
        library.select(&first).unwrap();
        {
            let mut state = library.state.lock().unwrap();
            state.registered_aliases = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
            state.models = ["a", "b"]
                .into_iter()
                .map(|alias| ModelLibraryModelV1 {
                    alias: alias.to_owned(),
                    file_name: format!("{alias}.gguf"),
                    size_bytes: 1,
                    architecture: "qwen35".to_owned(),
                    supported_architecture: true,
                    compatible: true,
                    reason: None,
                    mtp_companion_file_name: None,
                    mtp_companion_for: None,
                })
                .collect();
        }

        let error = library
            .select(&second)
            .expect_err("injected unload failure");
        assert!(error.contains("unload model b"));
        let snapshot = library.snapshot();
        assert_eq!(snapshot.selected_path.as_deref(), first.to_str());
        assert_eq!(
            read_persisted_path(&state_path).unwrap(),
            Some(first.clone())
        );
        assert_eq!(calls.lock().unwrap().as_slice(), ["a", "b"]);
        {
            let state = library.state.lock().unwrap();
            assert_eq!(state.registered_aliases, BTreeSet::from(["b".to_owned()]));
            assert!(!state.models[0].compatible);
            assert!(state.models[1].compatible);
        }

        let recovered = library.select(&second).unwrap();
        assert_eq!(recovered.selected_path.as_deref(), second.to_str());
        assert_eq!(read_persisted_path(&state_path).unwrap(), Some(second));
        assert_eq!(calls.lock().unwrap().as_slice(), ["a", "b", "b"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn aliases_are_local_names_and_never_paths() {
        assert_eq!(model_alias(Path::new("/models/My Model!.gguf")), "My-Model");
        assert_eq!(model_alias(Path::new("/models/---.gguf")), "model");
    }

    #[test]
    fn invalid_gguf_is_retained_as_a_disabled_catalog_row() {
        let root = temp_dir("library-invalid-gguf");
        fs::write(root.join("broken.gguf"), b"not a GGUF").unwrap();
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            None,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        assert_eq!(snapshot.models[0].file_name, "broken.gguf");
        assert!(!snapshot.models[0].compatible);
        assert_eq!(
            snapshot.models[0].reason.as_deref(),
            Some("The file is not a valid GGUF artifact.")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reviewed_deepseek4_is_visible_but_never_registered_as_production_ready() {
        let root = temp_dir("library-deepseek4-foundation");
        write_architecture_only_gguf(&root, "deepseek4");
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        let model = &snapshot.models[0];
        assert_eq!(model.architecture, "deepseek4");
        assert!(model.supported_architecture);
        assert!(!model.compatible);
        assert!(model.reason.as_deref().is_some_and(|reason| {
            reason.contains("production loading is unavailable")
                && reason.contains(&DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES.to_string())
        }));
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reviewed_minimax_m3_is_visible_but_never_registered_as_production_ready() {
        let root = temp_dir("library-minimax-m3-foundation");
        write_architecture_only_gguf(&root, "minimax-m3");
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        let model = &snapshot.models[0];
        assert_eq!(model.architecture, "minimax-m3");
        assert!(model.supported_architecture);
        assert!(!model.compatible);
        assert!(model.reason.as_deref().is_some_and(|reason| {
            reason.contains("production loading is unavailable")
                && reason.contains("MiniMax Community License")
                && reason.contains("internally inconsistent")
                && reason.contains(&MINIMAX_M3_CAPACITY_ADMISSION_BYTES.to_string())
        }));
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reviewed_diffusion_gemma_is_visible_but_never_registered_as_production_ready() {
        let root = temp_dir("library-diffusion-gemma-foundation");
        write_parser_only_diffusion_gemma_gguf(&root);
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        let model = &snapshot.models[0];
        assert_eq!(model.architecture, "diffusion-gemma");
        assert!(model.supported_architecture);
        assert!(!model.compatible);
        assert!(model.reason.as_deref().is_some_and(|reason| {
            reason.contains("production loading is unavailable")
                && reason.contains("Apache-2.0")
                && reason.contains(&DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES.to_string())
        }));
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ministral3_metadata_alone_never_registers_without_the_official_verifier() {
        let root = temp_dir("library-ministral3-metadata-only");
        write_parser_only_architecture_gguf(&root, "ministral3.gguf", MINISTRAL3_GGUF_ARCHITECTURE);
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        let model = &snapshot.models[0];
        assert_eq!(model.alias, MINISTRAL3_MODEL_ALIAS);
        assert_eq!(model.architecture, MINISTRAL3_GGUF_ARCHITECTURE);
        assert!(model.supported_architecture);
        assert!(!model.compatible);
        assert!(
            model
                .reason
                .as_deref()
                .is_some_and(|reason| { reason.contains("official Ministral 3 GGUF verifier") })
        );
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_companion_is_attached_only_to_one_reviewed_target_registration() {
        let mut targets = vec![pending_target("target")];
        let assistant = valid_assistant("assistant.gguf");
        let rows = resolve_mtp_companions_with_resident_resolver(
            &mut targets,
            std::slice::from_ref(&assistant),
            |_| Ok(80),
        );

        assert_eq!(rows.len(), 1);
        assert!(!rows[0].1.compatible);
        assert_eq!(rows[0].1.mtp_companion_for.as_deref(), Some("target"));
        assert_eq!(
            targets[0].registration.derived_lock_path.as_deref(),
            Some(Path::new("/models/target.derived-lock.json"))
        );
        assert_eq!(
            targets[0].registration.mtp_assistant_gguf_path.as_deref(),
            Some(assistant.gguf_path.as_path())
        );
        assert_eq!(
            targets[0]
                .registration
                .mtp_assistant_derived_lock_path
                .as_deref(),
            Some(Path::new("/models/assistant.derived-lock.json"))
        );
        assert_eq!(
            targets[0].registration.mtp_assistant_identity.as_deref(),
            Some(GEMMA4_MTP_FINGERPRINT)
        );
        assert_eq!(
            targets[0].mtp_companion_file_name.as_deref(),
            Some("assistant.gguf")
        );
        assert_eq!(targets[0].registration.resident_bytes, 180);
    }

    #[test]
    fn companion_resident_byte_overflow_fails_closed_and_keeps_target_only() {
        let mut targets = vec![pending_target("target")];
        targets[0].registration.resident_bytes = u64::MAX;
        let assistant = valid_assistant("assistant.gguf");
        let rows = resolve_mtp_companions_with_resident_resolver(
            &mut targets,
            std::slice::from_ref(&assistant),
            |_| Ok(80),
        );

        assert_eq!(rows.len(), 1);
        assert!(!rows[0].1.compatible);
        assert_eq!(rows[0].1.mtp_companion_for, None);
        assert!(
            rows[0]
                .1
                .reason
                .as_deref()
                .is_some_and(|reason| { reason.contains("resident-byte total overflowed") })
        );
        assert_eq!(targets[0].registration.resident_bytes, u64::MAX);
        assert!(targets[0].registration.mtp_assistant_gguf_path.is_none());
    }

    #[test]
    fn companion_requires_exact_gfx1201_target_and_keeps_other_targets_unattached() {
        for target_name in ["gfx1030", "gfx942"] {
            let mut targets = vec![pending_target("target")];
            targets[0].registration.target = target_name.to_owned();
            let assistant = valid_assistant("assistant.gguf");
            let rows = resolve_mtp_companions_with_resident_resolver(
                &mut targets,
                std::slice::from_ref(&assistant),
                |_| Ok(80),
            );

            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].1.mtp_companion_for, None);
            assert!(
                rows[0]
                    .1
                    .reason
                    .as_deref()
                    .is_some_and(|reason| { reason.contains("requires exact gfx1201 target") })
            );
            assert_eq!(targets[0].registration.resident_bytes, 100);
            assert!(targets[0].registration.mtp_assistant_gguf_path.is_none());
        }
    }

    #[test]
    fn companion_resident_bytes_must_fit_selected_device_including_exact_boundary() {
        let assistant = valid_assistant("assistant.gguf");

        let mut fits = vec![pending_target("fits")];
        fits[0].device_total_memory_bytes = 180;
        let rows = resolve_mtp_companions_with_resident_resolver(
            &mut fits,
            std::slice::from_ref(&assistant),
            |_| Ok(80),
        );
        assert_eq!(rows[0].1.mtp_companion_for.as_deref(), Some("fits"));
        assert_eq!(fits[0].registration.resident_bytes, 180);

        let mut over = vec![pending_target("over")];
        over[0].device_total_memory_bytes = 179;
        let rows = resolve_mtp_companions_with_resident_resolver(
            &mut over,
            std::slice::from_ref(&assistant),
            |_| Ok(80),
        );
        assert_eq!(rows[0].1.mtp_companion_for, None);
        assert!(
            rows[0]
                .1
                .reason
                .as_deref()
                .is_some_and(|reason| { reason.contains("Insufficient VRAM") })
        );
        assert_eq!(over[0].registration.resident_bytes, 100);
        assert!(over[0].registration.mtp_assistant_gguf_path.is_none());
    }

    #[test]
    fn reviewed_nvfp4_target_with_quant_artifact_source_is_pair_eligible() {
        let semantic = format!("gemma4:{GEMMA4_12B_IT_FINGERPRINT}");
        let quant_artifact = format!("sha256:{}", "7".repeat(64));
        let sources = vec![GEMMA4_12B_IT_FINGERPRINT.to_owned(), quant_artifact];
        assert_eq!(
            mtp_target_identity("gemma4", GEMMA4_12B_IT_FINGERPRINT, &semantic, &sources,)
                .as_deref(),
            Some(GEMMA4_12B_IT_FINGERPRINT)
        );
        let reversed = vec![sources[1].clone(), sources[0].clone()];
        assert!(
            mtp_target_identity("gemma4", GEMMA4_12B_IT_FINGERPRINT, &semantic, &reversed,)
                .is_none()
        );
        assert!(
            mtp_target_identity(
                "gemma4",
                GEMMA4_12B_IT_FINGERPRINT,
                "gemma4:wrong",
                &sources,
            )
            .is_none()
        );
    }

    #[test]
    fn target_only_and_ambiguous_companion_sets_remain_unattached() {
        let mut target_only = vec![pending_target("target")];
        assert!(resolve_mtp_companions(&mut target_only, &[]).is_empty());
        assert!(
            target_only[0]
                .registration
                .mtp_assistant_gguf_path
                .is_none()
        );

        let mut ambiguous_target = vec![pending_target("target")];
        let assistants = vec![
            valid_assistant("assistant-a.gguf"),
            valid_assistant("assistant-b.gguf"),
        ];
        let rows = resolve_mtp_companions(&mut ambiguous_target, &assistants);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|(_, row)| !row.compatible));
        assert!(rows.iter().all(|(_, row)| row.mtp_companion_for.is_none()));
        assert!(rows.iter().all(|(_, row)| {
            row.reason
                .as_deref()
                .is_some_and(|reason| reason.contains("Multiple"))
        }));
        assert!(
            ambiguous_target[0]
                .registration
                .mtp_assistant_gguf_path
                .is_none()
        );

        let mut duplicate_targets = vec![pending_target("target-a"), pending_target("target-b")];
        let rows =
            resolve_mtp_companions(&mut duplicate_targets, std::slice::from_ref(&assistants[0]));
        assert_eq!(rows[0].1.mtp_companion_for, None);
        assert!(
            duplicate_targets
                .iter()
                .all(|target| target.registration.mtp_assistant_gguf_path.is_none())
        );

        let mut noncanonical_target = vec![pending_target("target")];
        noncanonical_target[0].mtp_pair_target_identity = None;
        let rows = resolve_mtp_companions(
            &mut noncanonical_target,
            std::slice::from_ref(&assistants[0]),
        );
        assert_eq!(rows[0].1.mtp_companion_for, None);
        assert!(
            noncanonical_target[0]
                .registration
                .mtp_assistant_gguf_path
                .is_none()
        );
    }

    #[test]
    fn canonical_assistant_is_never_registered_standalone_and_requires_lock_then_target() {
        let root = temp_dir("library-mtp-assistant");
        let (_gguf_path, report, pair) =
            write_structural_mtp_gguf(&root, GEMMA4_12B_IT_FINGERPRINT, GEMMA4_MTP_FINGERPRINT);
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        let missing_lock = library.select(&root).unwrap();
        assert_eq!(missing_lock.models.len(), 1);
        assert_eq!(missing_lock.models[0].architecture, "gemma4mtp");
        assert!(!missing_lock.models[0].compatible);
        assert!(
            missing_lock.models[0]
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("derived-lock"))
        );
        assert!(registrations.lock().unwrap().is_empty());

        let derived = DerivedGgufLock::new(
            pair,
            vec![
                GEMMA4_12B_IT_FINGERPRINT.to_owned(),
                GEMMA4_MTP_FINGERPRINT.to_owned(),
            ],
            DerivedGgufConverter {
                repository: "sllm-test".to_owned(),
                commit: "0".repeat(40),
                arguments: vec!["--test".to_owned()],
                effective_config: BTreeMap::new(),
                environment: BTreeMap::new(),
            },
            &report,
        )
        .unwrap();
        fs::write(
            root.join("assistant.derived-lock.json"),
            derived.canonical_json().unwrap(),
        )
        .unwrap();
        let without_target = library.rescan().unwrap();
        assert_eq!(without_target.models.len(), 1);
        assert!(!without_target.models[0].compatible);
        assert_eq!(without_target.models[0].mtp_companion_for, None);
        assert!(
            without_target.models[0]
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("matching reviewed target"))
        );
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mismatched_pair_metadata_is_disabled_before_derived_lock_or_registration() {
        let root = temp_dir("library-mtp-mismatch");
        let wrong_target = format!("sha256:{}", "a".repeat(64));
        write_structural_mtp_gguf(&root, &wrong_target, GEMMA4_MTP_FINGERPRINT);
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&registrations);
        let library = ModelLibraryV1::open(
            root.join("state.json"),
            root.clone(),
            Some(test_device()),
            move |registration| {
                captured.lock().unwrap().push(registration);
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
        let snapshot = library.select(&root).unwrap();
        assert_eq!(snapshot.models.len(), 1);
        assert!(!snapshot.models[0].compatible);
        assert!(
            snapshot.models[0]
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("pair metadata"))
        );
        assert!(registrations.lock().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
