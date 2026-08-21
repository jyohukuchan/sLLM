//! Credential parsing and authorization primitives for the HTTP service.
//!
//! This module deliberately keeps only a digest of each credential in memory.
//! It is independent of the HTTP router so that the same authorization rules
//! can be used by the public and administrative surfaces.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use sha2::{Digest, Sha256};

const DIGEST_BYTES: usize = 32;
pub const MAX_CREDENTIAL_KEYS: usize = 32;
pub const MAX_CREDENTIAL_TOKEN_BYTES: usize = 4096;
pub const MAX_CREDENTIAL_FILE_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CredentialRoleV1 {
    User,
    Admin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CredentialRoleMaskV1 {
    User,
    Admin,
}

impl CredentialRoleMaskV1 {
    const fn allows(self, requested: CredentialRoleV1) -> bool {
        match (self, requested) {
            (Self::User, CredentialRoleV1::User)
            | (Self::Admin, CredentialRoleV1::User)
            | (Self::Admin, CredentialRoleV1::Admin) => true,
            (Self::User, CredentialRoleV1::Admin) => false,
        }
    }
}

#[derive(Clone)]
struct CredentialEntryV1 {
    digest: [u8; DIGEST_BYTES],
    role: CredentialRoleMaskV1,
}

#[derive(Clone, Default)]
struct CredentialSnapshotV1 {
    entries: Vec<CredentialEntryV1>,
    open_mode: bool,
}

/// A credential store that can be shared by request handlers and reloaded by
/// an administrative control path.
///
/// The store never retains plaintext credentials.  In open mode user requests
/// are accepted without a header; administrative requests remain disabled
/// until an explicit admin credential is configured.
#[derive(Clone)]
pub struct CredentialStoreV1 {
    snapshot: Arc<RwLock<CredentialSnapshotV1>>,
    key_file: Option<PathBuf>,
}

impl fmt::Debug for CredentialStoreV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self
            .snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        formatter
            .debug_struct("CredentialStoreV1")
            .field("open_mode", &snapshot.open_mode)
            .field("credential_count", &snapshot.entries.len())
            .field("key_file_configured", &self.key_file.is_some())
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CredentialErrorV1 {
    InvalidToken,
    EmptyKeyFile,
    MalformedKeyFileEntry,
    TooManyKeys,
    KeyFileTooLarge,
    KeyFileNotRegular,
    KeyFilePermissions,
    KeyFileNotConfigured,
    KeyFileIo,
    DuplicateToken,
    StorePoisoned,
}

impl fmt::Display for CredentialErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidToken => "credential token is invalid",
            Self::EmptyKeyFile => "credential key file is empty",
            Self::MalformedKeyFileEntry => "credential key file entry is malformed",
            Self::TooManyKeys => "credential key file has too many keys",
            Self::KeyFileTooLarge => "credential key file is too large",
            Self::KeyFileNotRegular => "credential key file is not a regular file",
            Self::KeyFilePermissions => "credential key file permissions are too broad",
            Self::KeyFileNotConfigured => "credential key file reload is not configured",
            Self::KeyFileIo => "credential key file could not be read",
            Self::DuplicateToken => "credential key file contains a duplicate token",
            Self::StorePoisoned => "credential store is unavailable",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CredentialErrorV1 {}

impl CredentialStoreV1 {
    /// Construct an open store.  User authorization succeeds without a key;
    /// admin authorization remains denied.
    pub fn open() -> Self {
        Self::from_snapshot(CredentialSnapshotV1 {
            entries: Vec::new(),
            open_mode: true,
        })
    }

    /// Construct the legacy single-user-key configuration.  This preserves
    /// the pre-Phase39 `--api-key-env` behavior while using the new matcher.
    pub fn from_user_key(token: impl AsRef<str>) -> Result<Self, CredentialErrorV1> {
        Self::from_keys([(CredentialRoleV1::User, token.as_ref())])
    }

    /// Construct an in-memory store with one or more user/admin credentials.
    pub fn from_keys<I, S>(keys: I) -> Result<Self, CredentialErrorV1>
    where
        I: IntoIterator<Item = (CredentialRoleV1, S)>,
        S: AsRef<str>,
    {
        let snapshot = parse_entries(keys)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Construct a store backed by a key file and load its contents now.
    pub fn from_key_file(path: impl AsRef<Path>) -> Result<Self, CredentialErrorV1> {
        let path = path.as_ref().to_path_buf();
        let snapshot = read_key_file(&path)?;
        Ok(Self {
            snapshot: Arc::new(RwLock::new(snapshot)),
            key_file: Some(path),
        })
    }

    fn from_snapshot(snapshot: CredentialSnapshotV1) -> Self {
        Self {
            snapshot: Arc::new(RwLock::new(snapshot)),
            key_file: None,
        }
    }

    /// Reload the configured key file.  Parsing and validation complete before
    /// the lock is acquired for the swap, so a failed reload keeps old keys.
    pub fn reload(&self) -> Result<(), CredentialErrorV1> {
        let path = self
            .key_file
            .as_deref()
            .ok_or(CredentialErrorV1::KeyFileNotConfigured)?;
        let parsed = read_key_file(path)?;
        let mut guard = self
            .snapshot
            .write()
            .map_err(|_| CredentialErrorV1::StorePoisoned)?;
        *guard = parsed;
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .open_mode
    }

    pub fn credential_count(&self) -> usize {
        self.snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .len()
    }

    pub fn has_admin_credentials(&self) -> bool {
        let snapshot = self
            .snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot
            .entries
            .iter()
            .any(|entry| entry.role == CredentialRoleMaskV1::Admin)
    }

    /// Authorize a user endpoint from the raw value of the Authorization
    /// header.  Admin credentials are intentionally accepted for this role.
    pub fn authorize_user(&self, raw_authorization: Option<&str>) -> bool {
        self.authorize(CredentialRoleV1::User, raw_authorization)
    }

    /// Authorize an administrative endpoint from the raw value of the
    /// Authorization header.  User credentials never authorize this role.
    pub fn authorize_admin(&self, raw_authorization: Option<&str>) -> bool {
        self.authorize(CredentialRoleV1::Admin, raw_authorization)
    }

    fn authorize(&self, requested: CredentialRoleV1, raw_authorization: Option<&str>) -> bool {
        let snapshot = self
            .snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if snapshot.open_mode {
            return requested == CredentialRoleV1::User;
        }
        let Some(token) = bearer_token(raw_authorization) else {
            return false;
        };
        let digest = digest_token(token);
        let mut matched = 0_u8;
        for entry in &snapshot.entries {
            // Do not short-circuit either the fixed-size comparison or the
            // scan across all configured keys.
            let equal = constant_time_eq_32(&digest, &entry.digest) as u8;
            let role_allowed = entry.role.allows(requested) as u8;
            matched |= equal & role_allowed;
        }
        matched != 0
    }
}

fn parse_entries<I, S>(keys: I) -> Result<CredentialSnapshotV1, CredentialErrorV1>
where
    I: IntoIterator<Item = (CredentialRoleV1, S)>,
    S: AsRef<str>,
{
    let mut entries: Vec<CredentialEntryV1> = Vec::new();
    for (role, token) in keys {
        if entries.len() == MAX_CREDENTIAL_KEYS {
            return Err(CredentialErrorV1::TooManyKeys);
        }
        let token = token.as_ref();
        validate_token(token)?;
        let digest = digest_token(token);
        let mut duplicate = 0_u8;
        for entry in &entries {
            duplicate |= constant_time_eq_32(&entry.digest, &digest) as u8;
        }
        if duplicate != 0 {
            return Err(CredentialErrorV1::DuplicateToken);
        }
        entries.push(CredentialEntryV1 {
            digest,
            role: match role {
                CredentialRoleV1::User => CredentialRoleMaskV1::User,
                CredentialRoleV1::Admin => CredentialRoleMaskV1::Admin,
            },
        });
    }
    if entries.is_empty() {
        return Err(CredentialErrorV1::EmptyKeyFile);
    }
    Ok(CredentialSnapshotV1 {
        entries,
        open_mode: false,
    })
}

fn validate_token(token: &str) -> Result<(), CredentialErrorV1> {
    if token.is_empty()
        || token.len() > MAX_CREDENTIAL_TOKEN_BYTES
        || token
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
    {
        return Err(CredentialErrorV1::InvalidToken);
    }
    Ok(())
}

fn digest_token(token: &str) -> [u8; DIGEST_BYTES] {
    let digest = Sha256::digest(token.as_bytes());
    let mut result = [0_u8; DIGEST_BYTES];
    result.copy_from_slice(&digest);
    result
}

#[inline(never)]
fn constant_time_eq_32(left: &[u8; DIGEST_BYTES], right: &[u8; DIGEST_BYTES]) -> bool {
    let mut difference = 0_u8;
    for index in 0..DIGEST_BYTES {
        difference |= left[index] ^ right[index];
    }
    difference == 0
}

fn bearer_token(raw_authorization: Option<&str>) -> Option<&str> {
    let raw = raw_authorization?;
    let (scheme, token) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("Bearer") || token.is_empty() {
        return None;
    }
    validate_token(token).ok()?;
    Some(token)
}

fn read_key_file(path: &Path) -> Result<CredentialSnapshotV1, CredentialErrorV1> {
    let file = open_key_file(path)?;
    let metadata = file.metadata().map_err(|_| CredentialErrorV1::KeyFileIo)?;
    if !metadata.is_file() {
        return Err(CredentialErrorV1::KeyFileNotRegular);
    }
    if metadata.len() > MAX_CREDENTIAL_FILE_BYTES as u64 {
        return Err(CredentialErrorV1::KeyFileTooLarge);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(CredentialErrorV1::KeyFilePermissions);
        }
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((MAX_CREDENTIAL_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| CredentialErrorV1::KeyFileIo)?;
    if bytes.len() > MAX_CREDENTIAL_FILE_BYTES {
        return Err(CredentialErrorV1::KeyFileTooLarge);
    }
    let text = String::from_utf8(bytes).map_err(|_| CredentialErrorV1::MalformedKeyFileEntry)?;
    let mut lines = text.split('\n').collect::<Vec<_>>();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        return Err(CredentialErrorV1::EmptyKeyFile);
    }
    let entries = lines
        .into_iter()
        .map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let (role, token) = line
                .split_once(':')
                .ok_or(CredentialErrorV1::MalformedKeyFileEntry)?;
            let role = match role {
                "user" => CredentialRoleV1::User,
                "admin" => CredentialRoleV1::Admin,
                _ => return Err(CredentialErrorV1::MalformedKeyFileEntry),
            };
            if token.is_empty() {
                return Err(CredentialErrorV1::MalformedKeyFileEntry);
            }
            Ok((role, token))
        })
        .collect::<Result<Vec<_>, CredentialErrorV1>>()?;
    parse_entries(entries)
}

fn open_key_file(path: &Path) -> Result<File, CredentialErrorV1> {
    let metadata = fs::symlink_metadata(path).map_err(|_| CredentialErrorV1::KeyFileIo)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CredentialErrorV1::KeyFileNotRegular);
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    options.open(path).map_err(|_| CredentialErrorV1::KeyFileIo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

    fn temporary_key_file(contents: &str) -> PathBuf {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sllm-security-test-{id}"));
        fs::write(&path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        path
    }

    fn remove(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn open_and_legacy_user_key_roles_are_separated() {
        let open = CredentialStoreV1::open();
        assert!(open.authorize_user(None));
        assert!(!open.authorize_admin(None));

        let store = CredentialStoreV1::from_user_key("user-secret").unwrap();
        assert!(store.authorize_user(Some("Bearer user-secret")));
        assert!(!store.authorize_admin(Some("Bearer user-secret")));
        assert!(!store.authorize_user(Some("Bearer other")));
    }

    #[test]
    fn admin_keys_authorize_both_surfaces() {
        let store = CredentialStoreV1::from_keys([
            (CredentialRoleV1::User, "user-secret"),
            (CredentialRoleV1::Admin, "admin-secret"),
        ])
        .unwrap();
        assert!(store.authorize_user(Some("Bearer user-secret")));
        assert!(store.authorize_user(Some("Bearer admin-secret")));
        assert!(store.authorize_admin(Some("Bearer admin-secret")));
        assert!(!store.authorize_admin(Some("Bearer user-secret")));
        assert!(!store.authorize_user(Some("Basic admin-secret")));
    }

    #[test]
    fn malformed_and_duplicate_keys_are_rejected_without_secret_in_error() {
        for keys in [
            vec![(CredentialRoleV1::User, "")],
            vec![(CredentialRoleV1::User, "contains whitespace")],
            vec![
                (CredentialRoleV1::User, "same"),
                (CredentialRoleV1::Admin, "same"),
            ],
        ] {
            let error = CredentialStoreV1::from_keys(keys).unwrap_err();
            assert!(!error.to_string().contains("same"));
            assert!(!format!("{error:?}").contains("same"));
        }
    }

    #[test]
    fn key_file_parser_accepts_roles_and_rejects_malformed_lines() {
        let path = temporary_key_file("user:file-user\nadmin:file-admin\n");
        let store = CredentialStoreV1::from_key_file(&path).unwrap();
        assert!(store.authorize_user(Some("Bearer file-user")));
        assert!(store.authorize_admin(Some("Bearer file-admin")));
        remove(&path);

        for contents in ["", "user:\n", "owner:token\n", "user:has space\n"] {
            let path = temporary_key_file(contents);
            assert!(
                CredentialStoreV1::from_key_file(&path).is_err(),
                "{contents:?}"
            );
            remove(&path);
        }

        let path = temporary_key_file("user:token:with-colon\n");
        let store = CredentialStoreV1::from_key_file(&path).unwrap();
        assert!(store.authorize_user(Some("Bearer token:with-colon")));
        remove(&path);
    }

    #[test]
    fn failed_reload_retains_previous_snapshot_and_success_rotates() {
        let path = temporary_key_file("user:old-key\n");
        let store = CredentialStoreV1::from_key_file(&path).unwrap();
        fs::write(&path, "user:new-key\n").unwrap();
        assert!(store.reload().is_ok());
        assert!(!store.authorize_user(Some("Bearer old-key")));
        assert!(store.authorize_user(Some("Bearer new-key")));
        fs::write(&path, "user:invalid key\n").unwrap();
        assert!(store.reload().is_err());
        assert!(store.authorize_user(Some("Bearer new-key")));
        remove(&path);
    }

    #[cfg(unix)]
    #[test]
    fn key_file_rejects_group_or_other_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let path = temporary_key_file("user:file-key\n");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        assert!(matches!(
            CredentialStoreV1::from_key_file(&path),
            Err(CredentialErrorV1::KeyFilePermissions)
        ));
        remove(&path);
    }

    #[cfg(unix)]
    #[test]
    fn key_file_rejects_symlinks() {
        use std::os::unix::fs::symlink;
        let target = temporary_key_file("user:file-key\n");
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let link = std::env::temp_dir().join(format!("sllm-security-test-link-{id}"));
        symlink(&target, &link).unwrap();
        assert!(matches!(
            CredentialStoreV1::from_key_file(&link),
            Err(CredentialErrorV1::KeyFileNotRegular)
        ));
        remove(&link);
        remove(&target);
    }

    #[test]
    fn key_file_rejects_more_than_limits() {
        let too_many = (0..=MAX_CREDENTIAL_KEYS)
            .map(|index| format!("user:key-{index}\n"))
            .collect::<String>();
        let path = temporary_key_file(&too_many);
        assert!(matches!(
            CredentialStoreV1::from_key_file(&path),
            Err(CredentialErrorV1::TooManyKeys)
        ));
        remove(&path);

        let too_large = format!("user:{}\n", "a".repeat(MAX_CREDENTIAL_TOKEN_BYTES + 1));
        let path = temporary_key_file(&too_large);
        assert!(matches!(
            CredentialStoreV1::from_key_file(&path),
            Err(CredentialErrorV1::InvalidToken)
        ));
        remove(&path);
    }
}
