//! Bounded Hugging Face CLI integration for the loopback WebUI.
//!
//! This is deliberately not a general command runner. Search, inspection, and
//! downloads are translated into fixed `hf` subcommands from validated,
//! structured inputs. Downloads always target the currently selected model
//! library directory.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use crate::model_library::ModelLibraryV1;

const MAX_SEARCH_QUERY_BYTES: usize = 128;
const MAX_REPO_ID_BYTES: usize = 192;
const MAX_FILE_NAME_BYTES: usize = 255;
const MAX_SEARCH_RESULTS: usize = 20;
const MAX_DOWNLOAD_JOBS: usize = 32;
const MAX_ERROR_CHARS: usize = 600;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceStatusV1 {
    pub schema_version: &'static str,
    pub cli_available: bool,
    pub cli_version: Option<String>,
    pub auth_state: String,
    pub authenticated: bool,
    pub username: Option<String>,
    pub active_downloads: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceModelV1 {
    pub repo_id: String,
    pub revision: String,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
    pub private: bool,
    pub last_modified: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceSearchV1 {
    pub schema_version: &'static str,
    pub query: String,
    pub models: Vec<HuggingFaceModelV1>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceGgufFileV1 {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub derived_lock_path: Option<String>,
    pub download_command: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceFilesV1 {
    pub schema_version: &'static str,
    pub repo_id: String,
    pub revision: String,
    pub selected_path: String,
    pub files: Vec<HuggingFaceGgufFileV1>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HuggingFaceDownloadJobV1 {
    pub schema_version: &'static str,
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub file_path: String,
    pub destination: String,
    pub command: String,
    pub state: String,
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawModelV1 {
    id: String,
    sha: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    last_modified: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawRepoFileV1 {
    path: String,
    #[serde(default)]
    size: u64,
    lfs: Option<RawLfsV1>,
}

#[derive(Debug, Deserialize)]
struct RawLfsV1 {
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WhoAmIV1 {
    user: String,
}

#[derive(Debug)]
pub(crate) struct HuggingFaceErrorV1 {
    pub param: Option<&'static str>,
    pub message: String,
}

impl HuggingFaceErrorV1 {
    fn invalid(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param),
            message: message.into(),
        }
    }

    fn operation(message: impl Into<String>) -> Self {
        Self {
            param: None,
            message: message.into(),
        }
    }
}

struct DownloadJobsV1 {
    next_id: u64,
    jobs: VecDeque<HuggingFaceDownloadJobV1>,
}

#[derive(Clone)]
pub(crate) struct HuggingFaceHubV1 {
    model_library: ModelLibraryV1,
    jobs: Arc<Mutex<DownloadJobsV1>>,
}

impl HuggingFaceHubV1 {
    pub(crate) fn new(model_library: ModelLibraryV1) -> Self {
        Self {
            model_library,
            jobs: Arc::new(Mutex::new(DownloadJobsV1 {
                next_id: 1,
                jobs: VecDeque::new(),
            })),
        }
    }

    pub(crate) fn status(&self) -> HuggingFaceStatusV1 {
        let version = Command::new("hf").arg("--version").output().ok();
        let cli_available = version
            .as_ref()
            .is_some_and(|output| output.status.success());
        let cli_version = version.and_then(|output| {
            output.status.success().then(|| {
                String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .chars()
                    .take(64)
                    .collect::<String>()
            })
        });
        let whoami_output = cli_available
            .then(|| {
                Command::new("hf")
                    .args(["auth", "whoami", "--format", "json"])
                    .output()
                    .ok()
            })
            .flatten();
        let whoami = whoami_output
            .as_ref()
            .filter(|output| output.status.success())
            .and_then(|output| serde_json::from_slice::<WhoAmIV1>(&output.stdout).ok());
        let auth_state = if whoami.is_some() {
            "authenticated"
        } else if whoami_output.as_ref().is_some_and(|output| {
            String::from_utf8_lossy(&output.stderr)
                .to_ascii_lowercase()
                .contains("not logged in")
        }) {
            "unauthenticated"
        } else {
            "unknown"
        };
        let active_downloads = self
            .jobs
            .lock()
            .expect("Hugging Face job mutex poisoned")
            .jobs
            .iter()
            .filter(|job| matches!(job.state.as_str(), "queued" | "running"))
            .count();
        HuggingFaceStatusV1 {
            schema_version: "sllm-hugging-face-status-v1",
            cli_available,
            cli_version,
            auth_state: auth_state.to_owned(),
            authenticated: whoami.is_some(),
            username: whoami.map(|value| value.user),
            active_downloads,
        }
    }

    pub(crate) fn search(&self, query: &str) -> Result<HuggingFaceSearchV1, HuggingFaceErrorV1> {
        let query = validate_query(query)?;
        let output = Command::new("hf")
            .args([
                "models",
                "ls",
                "--search",
                query,
                "--filter",
                "gguf",
                "--sort",
                "downloads",
                "--limit",
                "20",
                "--expand",
                "downloads,likes,gated,private,sha,lastModified",
                "--format",
                "json",
            ])
            .output()
            .map_err(|_| HuggingFaceErrorV1::operation("Hugging Face CLI is not available"))?;
        if !output.status.success() {
            return Err(command_error(
                "Hugging Face model search failed",
                &output.stderr,
            ));
        }
        let raw: Vec<RawModelV1> = serde_json::from_slice(&output.stdout)
            .map_err(|_| HuggingFaceErrorV1::operation("Hugging Face search output was invalid"))?;
        let models = raw
            .into_iter()
            .take(MAX_SEARCH_RESULTS)
            .filter_map(|model| {
                let revision = model.sha?;
                validate_repo_id(&model.id).ok()?;
                validate_revision(&revision).ok()?;
                Some(HuggingFaceModelV1 {
                    repo_id: model.id,
                    revision,
                    downloads: model.downloads,
                    likes: model.likes,
                    gated: gated_value(&model.gated),
                    private: model.private,
                    last_modified: model.last_modified,
                })
            })
            .collect();
        Ok(HuggingFaceSearchV1 {
            schema_version: "sllm-hugging-face-search-v1",
            query: query.to_owned(),
            models,
        })
    }

    pub(crate) fn files(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> Result<HuggingFaceFilesV1, HuggingFaceErrorV1> {
        validate_repo_id(repo_id)?;
        validate_revision(revision)?;
        let selected = self.selected_path()?;
        let output = Command::new("hf")
            .args([
                "models",
                "ls",
                "--revision",
                revision,
                "--format",
                "json",
                "--",
                repo_id,
            ])
            .output()
            .map_err(|_| HuggingFaceErrorV1::operation("Hugging Face CLI is not available"))?;
        if !output.status.success() {
            return Err(command_error(
                "Hugging Face repository inspection failed",
                &output.stderr,
            ));
        }
        let raw: Vec<RawRepoFileV1> = serde_json::from_slice(&output.stdout).map_err(|_| {
            HuggingFaceErrorV1::operation("Hugging Face repository output was invalid")
        })?;
        let root_names = raw
            .iter()
            .filter(|file| validate_root_file_name(&file.path).is_ok())
            .map(|file| file.path.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut files = raw
            .into_iter()
            .filter(|file| is_root_gguf(&file.path))
            .filter_map(|file| {
                validate_root_file_name(&file.path).ok()?;
                let derived_lock_path =
                    derived_lock_name(&file.path).filter(|name| root_names.contains(name.as_str()));
                let command = download_command(
                    repo_id,
                    revision,
                    &file.path,
                    derived_lock_path.as_deref(),
                    &selected,
                );
                Some(HuggingFaceGgufFileV1 {
                    path: file.path,
                    size_bytes: file.size,
                    sha256: file.lfs.and_then(|lfs| lfs.sha256),
                    derived_lock_path,
                    download_command: command,
                })
            })
            .collect::<Vec<_>>();
        files.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(HuggingFaceFilesV1 {
            schema_version: "sllm-hugging-face-files-v1",
            repo_id: repo_id.to_owned(),
            revision: revision.to_owned(),
            selected_path: selected.display().to_string(),
            files,
        })
    }

    pub(crate) fn start_download(
        &self,
        repo_id: &str,
        revision: &str,
        file_path: &str,
        derived_lock_path: Option<&str>,
    ) -> Result<HuggingFaceDownloadJobV1, HuggingFaceErrorV1> {
        validate_repo_id(repo_id)?;
        validate_revision(revision)?;
        validate_gguf_file_name(file_path)?;
        if let Some(lock) = derived_lock_path {
            validate_derived_lock(file_path, lock)?;
        }
        let selected = self.selected_path()?;
        let command = download_command(repo_id, revision, file_path, derived_lock_path, &selected);
        let mut jobs = self.jobs.lock().expect("Hugging Face job mutex poisoned");
        if jobs
            .jobs
            .iter()
            .any(|job| matches!(job.state.as_str(), "queued" | "running"))
        {
            return Err(HuggingFaceErrorV1::operation(
                "another Hugging Face download is already running",
            ));
        }
        while jobs.jobs.len() >= MAX_DOWNLOAD_JOBS {
            jobs.jobs.pop_front();
        }
        let id = format!("hf-download-{}", jobs.next_id);
        jobs.next_id = jobs.next_id.saturating_add(1);
        let job = HuggingFaceDownloadJobV1 {
            schema_version: "sllm-hugging-face-download-v1",
            id: id.clone(),
            repo_id: repo_id.to_owned(),
            revision: revision.to_owned(),
            file_path: file_path.to_owned(),
            destination: selected.display().to_string(),
            command,
            state: "queued".to_owned(),
            message: None,
        };
        jobs.jobs.push_back(job.clone());
        drop(jobs);

        let jobs = Arc::clone(&self.jobs);
        let library = self.model_library.clone();
        let args = download_args(repo_id, revision, file_path, derived_lock_path, &selected);
        let worker = std::thread::Builder::new().name(id.clone()).spawn(move || {
            update_job(&jobs, &id, |job| job.state = "running".to_owned());
            let output = Command::new("hf").args(&args).output();
            match output {
                Ok(output) if output.status.success() => {
                    let rescan = library.rescan();
                    update_job(&jobs, &id, |job| {
                        job.state = "completed".to_owned();
                        job.message = Some(match rescan {
                            Ok(_) => {
                                "Download completed and the model folder was rescanned.".to_owned()
                            }
                            Err(error) => format!(
                                "Download completed, but the automatic rescan failed: {error}"
                            ),
                        });
                    });
                }
                Ok(output) => {
                    let error = command_error("Hugging Face download failed", &output.stderr);
                    update_job(&jobs, &id, |job| {
                        job.state = "failed".to_owned();
                        job.message = Some(error.message);
                    });
                }
                Err(_) => update_job(&jobs, &id, |job| {
                    job.state = "failed".to_owned();
                    job.message = Some("Hugging Face CLI is not available".to_owned());
                }),
            }
        });
        if worker.is_err() {
            update_job(&self.jobs, &job.id, |stored| {
                stored.state = "failed".to_owned();
                stored.message = Some("download worker could not be started".to_owned());
            });
            return Err(HuggingFaceErrorV1::operation(
                "download worker could not be started",
            ));
        }
        Ok(job)
    }

    pub(crate) fn download_job(
        &self,
        id: &str,
    ) -> Result<HuggingFaceDownloadJobV1, HuggingFaceErrorV1> {
        if id.len() > 64 || !id.starts_with("hf-download-") {
            return Err(HuggingFaceErrorV1::invalid(
                "id",
                "download job ID is invalid",
            ));
        }
        self.jobs
            .lock()
            .expect("Hugging Face job mutex poisoned")
            .jobs
            .iter()
            .find(|job| job.id == id)
            .cloned()
            .ok_or_else(|| HuggingFaceErrorV1::invalid("id", "download job was not found"))
    }

    fn selected_path(&self) -> Result<PathBuf, HuggingFaceErrorV1> {
        self.model_library.selected_path().ok_or_else(|| {
            HuggingFaceErrorV1::operation("select a model folder before using Hugging Face")
        })
    }
}

fn update_job(
    jobs: &Arc<Mutex<DownloadJobsV1>>,
    id: &str,
    update: impl FnOnce(&mut HuggingFaceDownloadJobV1),
) {
    if let Some(job) = jobs
        .lock()
        .expect("Hugging Face job mutex poisoned")
        .jobs
        .iter_mut()
        .find(|job| job.id == id)
    {
        update(job);
    }
}

fn validate_query(query: &str) -> Result<&str, HuggingFaceErrorV1> {
    let query = query.trim();
    if query.is_empty()
        || query.len() > MAX_SEARCH_QUERY_BYTES
        || query.chars().any(char::is_control)
    {
        return Err(HuggingFaceErrorV1::invalid(
            "query",
            "search query is invalid",
        ));
    }
    Ok(query)
}

fn validate_repo_id(repo_id: &str) -> Result<(), HuggingFaceErrorV1> {
    let parts = repo_id.split('/').collect::<Vec<_>>();
    let valid_parts = matches!(parts.len(), 1 | 2)
        && parts.iter().all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        });
    if repo_id.is_empty() || repo_id.len() > MAX_REPO_ID_BYTES || !valid_parts {
        return Err(HuggingFaceErrorV1::invalid(
            "repo_id",
            "Hugging Face repository ID is invalid",
        ));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), HuggingFaceErrorV1> {
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(HuggingFaceErrorV1::invalid(
            "revision",
            "revision must be a full commit SHA",
        ));
    }
    Ok(())
}

fn validate_root_file_name(file: &str) -> Result<(), HuggingFaceErrorV1> {
    if file.is_empty()
        || file.len() > MAX_FILE_NAME_BYTES
        || file.starts_with('-')
        || file == "."
        || file == ".."
        || file.contains(['/', '\\', '\0'])
        || file.chars().any(char::is_control)
    {
        return Err(HuggingFaceErrorV1::invalid(
            "file_path",
            "only a safe repository-root file can be downloaded",
        ));
    }
    Ok(())
}

fn validate_gguf_file_name(file: &str) -> Result<(), HuggingFaceErrorV1> {
    validate_root_file_name(file)?;
    if !file.to_ascii_lowercase().ends_with(".gguf") {
        return Err(HuggingFaceErrorV1::invalid(
            "file_path",
            "download file must be a GGUF artifact",
        ));
    }
    Ok(())
}

fn validate_derived_lock(gguf: &str, lock: &str) -> Result<(), HuggingFaceErrorV1> {
    validate_root_file_name(lock)?;
    if derived_lock_name(gguf).as_deref() != Some(lock) {
        return Err(HuggingFaceErrorV1::invalid(
            "derived_lock_path",
            "derived lock must match the selected GGUF file",
        ));
    }
    Ok(())
}

fn is_root_gguf(file: &str) -> bool {
    validate_root_file_name(file).is_ok() && file.to_ascii_lowercase().ends_with(".gguf")
}

fn derived_lock_name(gguf: &str) -> Option<String> {
    if !gguf.to_ascii_lowercase().ends_with(".gguf") {
        return None;
    }
    let stem = &gguf[..gguf.len() - ".gguf".len()];
    Some(format!("{stem}.derived-lock.json"))
}

fn gated_value(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Null => false,
        Value::String(value) => !value.eq_ignore_ascii_case("false"),
        _ => true,
    }
}

fn download_args(
    repo_id: &str,
    revision: &str,
    file_path: &str,
    derived_lock_path: Option<&str>,
    destination: &Path,
) -> Vec<std::ffi::OsString> {
    let mut args = vec![
        "download".into(),
        "--revision".into(),
        revision.into(),
        "--local-dir".into(),
        destination.as_os_str().to_owned(),
        "--quiet".into(),
        "--".into(),
        repo_id.into(),
        file_path.into(),
    ];
    if let Some(lock) = derived_lock_path {
        args.push(lock.into());
    }
    args
}

fn download_command(
    repo_id: &str,
    revision: &str,
    file_path: &str,
    derived_lock_path: Option<&str>,
    destination: &Path,
) -> String {
    let mut values = vec![
        "hf".to_owned(),
        "download".to_owned(),
        "--revision".to_owned(),
        shell_quote(revision),
        "--local-dir".to_owned(),
        shell_quote(&destination.display().to_string()),
        "--quiet".to_owned(),
        "--".to_owned(),
        shell_quote(repo_id),
        shell_quote(file_path),
    ];
    if let Some(lock) = derived_lock_path {
        values.push(shell_quote(lock));
    }
    values.join(" ")
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn command_error(prefix: &str, stderr: &[u8]) -> HuggingFaceErrorV1 {
    let detail = String::from_utf8_lossy(stderr)
        .chars()
        .filter(|value| !value.is_control() || matches!(value, '\n' | '\t'))
        .take(MAX_ERROR_CHARS)
        .collect::<String>()
        .trim()
        .to_owned();
    if detail.is_empty() {
        HuggingFaceErrorV1::operation(prefix)
    } else {
        HuggingFaceErrorV1::operation(format!("{prefix}: {detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        derived_lock_name, download_command, gated_value, shell_quote, validate_derived_lock,
        validate_repo_id, validate_revision, validate_root_file_name,
    };
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn command_is_revision_pinned_and_shell_quotes_destination() {
        let command = download_command(
            "owner/model",
            "0123456789abcdef0123456789abcdef01234567",
            "model.gguf",
            Some("model.derived-lock.json"),
            Path::new("/srv/Model Store/operator's"),
        );
        assert_eq!(
            command,
            "hf download --revision '0123456789abcdef0123456789abcdef01234567' --local-dir '/srv/Model Store/operator'\"'\"'s' --quiet -- 'owner/model' 'model.gguf' 'model.derived-lock.json'"
        );
    }

    #[test]
    fn structured_inputs_reject_option_and_path_injection() {
        assert!(validate_repo_id("owner/model").is_ok());
        assert!(validate_repo_id("owner/model/extra").is_err());
        assert!(validate_root_file_name("model.gguf").is_ok());
        assert!(validate_root_file_name("--token").is_err());
        assert!(validate_root_file_name("../model.gguf").is_err());
        assert!(validate_revision("main").is_err());
        assert!(validate_revision("0123456789abcdef0123456789abcdef01234567").is_ok());
    }

    #[test]
    fn companion_lock_must_match_the_selected_gguf() {
        assert_eq!(
            derived_lock_name("model.gguf").as_deref(),
            Some("model.derived-lock.json")
        );
        assert!(validate_derived_lock("model.gguf", "model.derived-lock.json").is_ok());
        assert!(validate_derived_lock("model.gguf", "other.derived-lock.json").is_err());
    }

    #[test]
    fn gated_status_accepts_boolean_and_manual_modes() {
        assert!(!gated_value(&json!(false)));
        assert!(!gated_value(&json!(null)));
        assert!(gated_value(&json!(true)));
        assert!(gated_value(&json!("manual")));
        assert_eq!(shell_quote("plain"), "'plain'");
    }
}
