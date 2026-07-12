// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Streamed verified evidence for identity F32 adapter weights.
//!
//! The opaque receipt proves that one structurally admitted occurrence matched
//! a trusted manifest and a finite raw F32 little-endian regular file at
//! verification time. It is not a GPU upload, residency, allocator, or
//! same-bytes-at-upload capability. Transforms and quantized payload evidence
//! remain unsupported until later evidence stages.
//! Component checks and descriptor opens are not an atomic
//! `openat2`/`O_NOFOLLOW` provenance guarantee. Content substitution is instead
//! fail-closed by trusted digests, opened-file snapshots, and final path
//! identity checks; this receipt does not claim race-free origin provenance.

use std::{
    fmt,
    fs::{self, File, Metadata},
    io::Read,
    path::{Component, Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::{
    adapter_admission::{ResolvedAdmittedWeight, StructurallyAdmittedAdapter},
    model_graph::{NodeId, NumericalFormat, TensorLayout, WeightId},
    package::{VerifiedPassthroughDescriptor, parse_verified_passthrough_descriptors},
};

const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_DESCRIPTORS: usize = 4_096;
const MAX_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_ERROR_MESSAGE_BYTES: usize = 1_024;

/// Fixed SHA-256 digest parsed only from canonical lowercase hexadecimal text.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Parses exactly 64 lowercase hexadecimal characters.
    pub fn parse(value: &str) -> Result<Self, VerifiedEvidenceError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(error(
                VerifiedEvidenceStage::Digest,
                VerifiedEvidenceErrorClass::Invalid,
                "SHA-256 digest must be 64 lowercase hexadecimal characters",
            ));
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            bytes[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        }
        Ok(Self(bytes))
    }

    /// Returns the raw digest bytes.
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns lowercase hexadecimal text using a fallible bounded allocation.
    pub fn as_hex(&self) -> Result<String, VerifiedEvidenceError> {
        let mut output = String::new();
        output.try_reserve_exact(64).map_err(|_| {
            error(
                VerifiedEvidenceStage::Digest,
                VerifiedEvidenceErrorClass::Resource,
                "digest text allocation failed",
            )
        })?;
        use fmt::Write as _;
        for byte in self.0 {
            write!(&mut output, "{byte:02x}").map_err(|_| {
                error(
                    VerifiedEvidenceStage::Digest,
                    VerifiedEvidenceErrorClass::Resource,
                    "digest text formatting failed",
                )
            })?;
        }
        Ok(output)
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Stable verification stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedEvidenceStage {
    Digest,
    Manifest,
    Occurrence,
    Recipe,
    Payload,
    Binding,
}

/// Stable verification failure class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedEvidenceErrorClass {
    Invalid,
    Resource,
    Io,
    Integrity,
    Unsupported,
}

/// Bounded diagnostic without untrusted path or payload content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEvidenceError {
    stage: VerifiedEvidenceStage,
    class: VerifiedEvidenceErrorClass,
    message: &'static str,
}

impl VerifiedEvidenceError {
    pub const fn stage(&self) -> VerifiedEvidenceStage {
        self.stage
    }

    pub const fn class(&self) -> VerifiedEvidenceErrorClass {
        self.class
    }

    pub const fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for VerifiedEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?}/{:?}: {}",
            self.stage, self.class, self.message
        )
    }
}

impl std::error::Error for VerifiedEvidenceError {}

/// Trusted, bounded manifest evidence rooted in an externally supplied digest.
pub struct VerifiedPackageManifest {
    package_root: PathBuf,
    manifest_digest: Sha256Digest,
    descriptors: Vec<VerifiedPassthroughDescriptor>,
}

impl fmt::Debug for VerifiedPackageManifest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedPackageManifest")
            .field("manifest_digest", &self.manifest_digest)
            .field("descriptor_count", &self.descriptors.len())
            .finish_non_exhaustive()
    }
}

/// Opaque verified canonical-weight evidence bound to one structural token.
///
/// This receipt cannot authorize upload or prove that a later upload uses the
/// same bytes. A later resident-evidence stage must establish those facts.
pub struct VerifiedCanonicalWeightReceipt<'a> {
    admitted: &'a StructurallyAdmittedAdapter<'a>,
    node_id: &'a NodeId,
    weight_slot: usize,
    logical_id: &'a WeightId,
    manifest_digest: Sha256Digest,
    source_digest: Sha256Digest,
    recipe_digest: Sha256Digest,
    canonical_digest: Sha256Digest,
    binding_digest: Sha256Digest,
    shape: &'a [usize],
    bytes: u64,
    chunks: usize,
}

impl fmt::Debug for VerifiedCanonicalWeightReceipt<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedCanonicalWeightReceipt")
            .field("node_id", &self.node_id)
            .field("weight_slot", &self.weight_slot)
            .field("logical_id", &self.logical_id)
            .field("canonical_digest", &self.canonical_digest)
            .finish_non_exhaustive()
    }
}

impl<'a> VerifiedCanonicalWeightReceipt<'a> {
    pub fn admission_id(&self) -> &str {
        self.admitted.admission_id()
    }

    pub fn graph_id(&self) -> &str {
        self.admitted.graph_id()
    }

    pub const fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub const fn weight_slot(&self) -> usize {
        self.weight_slot
    }

    pub const fn logical_id(&self) -> &WeightId {
        self.logical_id
    }

    pub const fn manifest_digest(&self) -> &Sha256Digest {
        &self.manifest_digest
    }

    pub const fn source_digest(&self) -> &Sha256Digest {
        &self.source_digest
    }

    pub const fn recipe_digest(&self) -> &Sha256Digest {
        &self.recipe_digest
    }

    pub const fn canonical_digest(&self) -> &Sha256Digest {
        &self.canonical_digest
    }

    pub const fn binding_digest(&self) -> &Sha256Digest {
        &self.binding_digest
    }

    pub const fn shape(&self) -> &[usize] {
        self.shape
    }

    pub const fn format(&self) -> NumericalFormat {
        NumericalFormat::F32
    }

    pub const fn layout(&self) -> TensorLayout {
        TensorLayout::RowMajor
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn chunks(&self) -> usize {
        self.chunks
    }

    pub fn matches_admission(&self, admitted: &StructurallyAdmittedAdapter<'_>) -> bool {
        std::ptr::eq(self.admitted, admitted)
    }
}

/// Opens and verifies a bounded manifest against an external trust-root digest.
pub fn open_verified_package_manifest(
    package_root: impl AsRef<Path>,
    manifest_relative_path: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    manifest_limit: usize,
) -> Result<VerifiedPackageManifest, VerifiedEvidenceError> {
    if manifest_limit == 0 || manifest_limit > MAX_MANIFEST_BYTES {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Resource,
            "manifest byte limit must be in 1..=16MiB",
        ));
    }
    let expected = Sha256Digest::parse(expected_manifest_sha256)?;
    validate_root_argument(package_root.as_ref())?;
    let package_root = fs::canonicalize(package_root.as_ref()).map_err(|_| {
        error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Io,
            "package root canonicalization failed",
        )
    })?;
    if !package_root
        .metadata()
        .map(|value| value.is_dir())
        .unwrap_or(false)
    {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Invalid,
            "package root is not a directory",
        ));
    }
    let relative = validate_relative_path(
        manifest_relative_path.as_ref(),
        VerifiedEvidenceStage::Manifest,
    )?;
    let (mut file, path, before) =
        open_regular_beneath(&package_root, relative, VerifiedEvidenceStage::Manifest)?;
    let bytes = read_exact_bounded_manifest(&mut file, &before, manifest_limit)?;
    let actual = digest_bytes(&bytes);
    if actual != expected {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Integrity,
            "manifest digest does not match external trust root",
        ));
    }
    verify_stable_file(
        &file,
        &path,
        &before,
        &package_root,
        relative,
        VerifiedEvidenceStage::Manifest,
    )?;
    let descriptors =
        parse_verified_passthrough_descriptors(&bytes, MAX_DESCRIPTORS).map_err(|message| {
            let class = if message.starts_with("resource:") {
                VerifiedEvidenceErrorClass::Resource
            } else {
                VerifiedEvidenceErrorClass::Invalid
            };
            error(
                VerifiedEvidenceStage::Manifest,
                class,
                "verified manifest passthrough descriptors are invalid",
            )
        })?;
    for descriptor in &descriptors {
        validate_relative_path(
            Path::new(&descriptor.relative_path),
            VerifiedEvidenceStage::Manifest,
        )?;
    }
    Ok(VerifiedPackageManifest {
        package_root,
        manifest_digest: actual,
        descriptors,
    })
}

/// Verifies one Identity F32 RowMajor occurrence by streaming its regular file.
///
/// `chunk_bytes` must be 4-byte aligned and in `4..=1MiB`. The receipt proves
/// verification only; it does not retain or authorize later upload bytes.
pub fn verify_identity_f32_weight<'a>(
    admitted: &'a StructurallyAdmittedAdapter<'a>,
    package: &VerifiedPackageManifest,
    node_id: &NodeId,
    weight_slot: usize,
    chunk_bytes: usize,
) -> Result<VerifiedCanonicalWeightReceipt<'a>, VerifiedEvidenceError> {
    verify_identity_f32_weight_with_hook(
        admitted,
        package,
        node_id,
        weight_slot,
        chunk_bytes,
        |_| {},
    )
}

fn verify_identity_f32_weight_with_hook<'a, F>(
    admitted: &'a StructurallyAdmittedAdapter<'a>,
    package: &VerifiedPackageManifest,
    node_id: &NodeId,
    weight_slot: usize,
    chunk_bytes: usize,
    after_payload_read: F,
) -> Result<VerifiedCanonicalWeightReceipt<'a>, VerifiedEvidenceError>
where
    F: FnOnce(&Path),
{
    if chunk_bytes == 0 || chunk_bytes > MAX_CHUNK_BYTES || chunk_bytes % 4 != 0 {
        return Err(error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Resource,
            "payload chunk size must be 4-byte aligned and in 4..=1MiB",
        ));
    }
    let resolved = admitted
        .resolve_weight_occurrence(node_id, weight_slot)
        .ok_or_else(|| {
            error(
                VerifiedEvidenceStage::Occurrence,
                VerifiedEvidenceErrorClass::Invalid,
                "weight occurrence is not structurally admitted",
            )
        })?;
    validate_identity_contract(&resolved)?;
    let descriptor = package
        .descriptors
        .binary_search_by(|descriptor| {
            descriptor
                .name
                .as_str()
                .cmp(&resolved.binding.physical_tensor_name)
        })
        .ok()
        .and_then(|index| package.descriptors.get(index))
        .ok_or_else(|| {
            error(
                VerifiedEvidenceStage::Manifest,
                VerifiedEvidenceErrorClass::Unsupported,
                "no exact verified passthrough descriptor exists for binding",
            )
        })?;
    validate_descriptor(descriptor, &resolved)?;
    let expected_payload_digest = Sha256Digest::parse(&descriptor.payload_sha256)?;
    let binding_digest = resolved.binding.content_sha256.as_deref().ok_or_else(|| {
        error(
            VerifiedEvidenceStage::Binding,
            VerifiedEvidenceErrorClass::Unsupported,
            "logical binding has no trusted payload digest",
        )
    })?;
    let binding_payload_digest = Sha256Digest::parse(binding_digest)?;
    if binding_payload_digest != expected_payload_digest {
        return Err(error(
            VerifiedEvidenceStage::Binding,
            VerifiedEvidenceErrorClass::Integrity,
            "binding digest does not match verified manifest descriptor",
        ));
    }

    let relative = validate_relative_path(
        Path::new(&descriptor.relative_path),
        VerifiedEvidenceStage::Payload,
    )?;
    let (mut file, path, before) = open_regular_beneath(
        &package.package_root,
        relative,
        VerifiedEvidenceStage::Payload,
    )?;
    if before.len() != descriptor.payload_bytes {
        return Err(error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Integrity,
            "payload regular-file length does not match descriptor",
        ));
    }
    let (source_digest, chunks) =
        stream_finite_f32(&mut file, descriptor.payload_bytes, chunk_bytes)?;
    after_payload_read(&path);
    if source_digest != expected_payload_digest {
        return Err(error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Integrity,
            "streamed payload digest does not match descriptor",
        ));
    }
    verify_stable_file(
        &file,
        &path,
        &before,
        &package.package_root,
        relative,
        VerifiedEvidenceStage::Payload,
    )?;

    let recipe_digest = digest_identity_recipe(
        &resolved.use_record.recipe.source_shape,
        &resolved.weight.tensor.shape,
    );
    let canonical_digest = source_digest;
    let evidence_binding_digest = digest_binding_evidence(
        package.manifest_digest,
        &resolved,
        recipe_digest,
        source_digest,
        canonical_digest,
    );
    Ok(VerifiedCanonicalWeightReceipt {
        admitted,
        node_id: resolved.node_id,
        weight_slot,
        logical_id: &resolved.weight.id,
        manifest_digest: package.manifest_digest,
        source_digest,
        recipe_digest,
        canonical_digest,
        binding_digest: evidence_binding_digest,
        shape: &resolved.weight.tensor.shape,
        bytes: descriptor.payload_bytes,
        chunks,
    })
}

#[cfg(test)]
fn verify_identity_f32_weight_with_test_hook<'a, F>(
    admitted: &'a StructurallyAdmittedAdapter<'a>,
    package: &VerifiedPackageManifest,
    node_id: &NodeId,
    weight_slot: usize,
    chunk_bytes: usize,
    after_payload_read: F,
) -> Result<VerifiedCanonicalWeightReceipt<'a>, VerifiedEvidenceError>
where
    F: FnOnce(&Path),
{
    verify_identity_f32_weight_with_hook(
        admitted,
        package,
        node_id,
        weight_slot,
        chunk_bytes,
        after_payload_read,
    )
}

fn validate_identity_contract(
    resolved: &ResolvedAdmittedWeight<'_>,
) -> Result<(), VerifiedEvidenceError> {
    if resolved.weight.tensor.format != NumericalFormat::F32
        || resolved.weight.tensor.layout != TensorLayout::RowMajor
    {
        return Err(error(
            VerifiedEvidenceStage::Recipe,
            VerifiedEvidenceErrorClass::Unsupported,
            "initial verifier supports only F32 RowMajor logical weights",
        ));
    }
    if !resolved.use_record.recipe.steps.is_empty()
        || resolved.use_record.recipe.source_shape != resolved.weight.tensor.shape
    {
        return Err(error(
            VerifiedEvidenceStage::Recipe,
            VerifiedEvidenceErrorClass::Unsupported,
            "initial verifier supports only exact Identity recipes",
        ));
    }
    Ok(())
}

fn validate_descriptor(
    descriptor: &VerifiedPassthroughDescriptor,
    resolved: &ResolvedAdmittedWeight<'_>,
) -> Result<(), VerifiedEvidenceError> {
    if descriptor.dtype != "F32" || descriptor.encoding != "raw_safetensors_payload" {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Unsupported,
            "passthrough dtype or encoding is unsupported",
        ));
    }
    if descriptor.shape.len() != resolved.weight.tensor.shape.len()
        || descriptor
            .shape
            .iter()
            .zip(&resolved.weight.tensor.shape)
            .any(|(declared, expected)| u64::try_from(*expected).ok() != Some(*declared))
    {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Integrity,
            "passthrough shape does not match logical weight",
        ));
    }
    let elements = resolved
        .weight
        .tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| {
            let dimension = u64::try_from(*dimension).ok()?;
            product.checked_mul(dimension)
        })
        .ok_or_else(|| {
            error(
                VerifiedEvidenceStage::Manifest,
                VerifiedEvidenceErrorClass::Resource,
                "logical payload element count overflows",
            )
        })?;
    let expected_bytes = elements.checked_mul(4).ok_or_else(|| {
        error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Resource,
            "logical payload byte count overflows",
        )
    })?;
    if expected_bytes != descriptor.payload_bytes {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Integrity,
            "passthrough byte count does not match logical weight",
        ));
    }
    Ok(())
}

fn validate_root_argument(path: &Path) -> Result<(), VerifiedEvidenceError> {
    if path.as_os_str().is_empty() {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Invalid,
            "package root argument is empty",
        ));
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(error(
                    VerifiedEvidenceStage::Manifest,
                    VerifiedEvidenceErrorClass::Invalid,
                    "package root argument contains a parent component",
                ));
            }
            Component::Normal(part) => {
                current.push(part);
                let metadata = fs::symlink_metadata(&current).map_err(|_| {
                    error(
                        VerifiedEvidenceStage::Manifest,
                        VerifiedEvidenceErrorClass::Io,
                        "package root component metadata failed",
                    )
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(error(
                        VerifiedEvidenceStage::Manifest,
                        VerifiedEvidenceErrorClass::Invalid,
                        "package root symbolic links are not admitted",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_relative_path<'a>(
    path: &'a Path,
    stage: VerifiedEvidenceStage,
) -> Result<&'a Path, VerifiedEvidenceError> {
    let text = path.to_str().ok_or_else(|| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Invalid,
            "relative path is not valid UTF-8",
        )
    })?;
    if text.is_empty()
        || text.starts_with('/')
        || text.ends_with('/')
        || text.split('/').any(|part| part.is_empty())
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(error(
            stage,
            VerifiedEvidenceErrorClass::Invalid,
            "path must be normalized and relative without parent components",
        ));
    }
    Ok(path)
}

fn open_regular_beneath(
    root: &Path,
    relative: &Path,
    stage: VerifiedEvidenceStage,
) -> Result<(File, PathBuf, Metadata), VerifiedEvidenceError> {
    validate_no_symlink_components(root, relative, stage)?;
    let path = root.join(relative);
    let file = File::open(&path).map_err(|_| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Io,
            "regular file open failed",
        )
    })?;
    let handle_metadata = file.metadata().map_err(|_| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Io,
            "opened file metadata failed",
        )
    })?;
    if !handle_metadata.is_file() {
        return Err(error(
            stage,
            VerifiedEvidenceErrorClass::Invalid,
            "opened path is not a regular file",
        ));
    }
    let path_metadata = fs::metadata(&path).map_err(|_| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Io,
            "path metadata failed after open",
        )
    })?;
    if !same_file_identity(&handle_metadata, &path_metadata) {
        return Err(error(
            stage,
            VerifiedEvidenceErrorClass::Integrity,
            "opened file identity differs from path identity",
        ));
    }
    Ok((file, path, handle_metadata))
}

fn validate_no_symlink_components(
    root: &Path,
    relative: &Path,
    stage: VerifiedEvidenceStage,
) -> Result<(), VerifiedEvidenceError> {
    let mut current = root.to_path_buf();
    let component_count = relative.components().count();
    for (index, component) in relative.components().enumerate() {
        let Component::Normal(component) = component else {
            return Err(error(
                stage,
                VerifiedEvidenceErrorClass::Invalid,
                "path contains a non-normal component",
            ));
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current).map_err(|_| {
            error(
                stage,
                VerifiedEvidenceErrorClass::Io,
                "path component metadata failed",
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(error(
                stage,
                VerifiedEvidenceErrorClass::Invalid,
                "symbolic links are not admitted",
            ));
        }
        if index + 1 < component_count && !metadata.is_dir() {
            return Err(error(
                stage,
                VerifiedEvidenceErrorClass::Invalid,
                "intermediate path component is not a directory",
            ));
        }
    }
    Ok(())
}

fn read_exact_bounded_manifest(
    file: &mut File,
    before: &Metadata,
    limit: usize,
) -> Result<Vec<u8>, VerifiedEvidenceError> {
    let expected = usize::try_from(before.len()).map_err(|_| {
        error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Resource,
            "manifest file length exceeds usize",
        )
    })?;
    if expected == 0 || expected > limit {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Resource,
            "manifest file length exceeds configured limit",
        ));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(expected).map_err(|_| {
        error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Resource,
            "manifest byte allocation failed",
        )
    })?;
    let mut scratch = [0_u8; 8 * 1024];
    loop {
        let read = file.read(&mut scratch).map_err(|_| {
            error(
                VerifiedEvidenceStage::Manifest,
                VerifiedEvidenceErrorClass::Io,
                "manifest read failed",
            )
        })?;
        if read == 0 {
            break;
        }
        let end = bytes.len().checked_add(read).ok_or_else(|| {
            error(
                VerifiedEvidenceStage::Manifest,
                VerifiedEvidenceErrorClass::Resource,
                "manifest read length overflows",
            )
        })?;
        if end > expected {
            return Err(error(
                VerifiedEvidenceStage::Manifest,
                VerifiedEvidenceErrorClass::Integrity,
                "manifest grew while being read",
            ));
        }
        bytes.extend_from_slice(&scratch[..read]);
    }
    if bytes.len() != expected {
        return Err(error(
            VerifiedEvidenceStage::Manifest,
            VerifiedEvidenceErrorClass::Integrity,
            "manifest was truncated while being read",
        ));
    }
    Ok(bytes)
}

fn stream_finite_f32(
    file: &mut File,
    expected_bytes: u64,
    chunk_bytes: usize,
) -> Result<(Sha256Digest, usize), VerifiedEvidenceError> {
    if expected_bytes == 0 || expected_bytes % 4 != 0 {
        return Err(error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Invalid,
            "F32 payload byte length must be positive and 4-byte aligned",
        ));
    }
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(chunk_bytes).map_err(|_| {
        error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Resource,
            "payload scratch allocation failed",
        )
    })?;
    scratch.resize(chunk_bytes, 0);
    let mut remaining = expected_bytes;
    let mut chunks = 0_usize;
    let mut hasher = Sha256::new();
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(chunk_bytes as u64)).map_err(|_| {
            error(
                VerifiedEvidenceStage::Payload,
                VerifiedEvidenceErrorClass::Resource,
                "payload chunk length conversion failed",
            )
        })?;
        file.read_exact(&mut scratch[..wanted]).map_err(|_| {
            error(
                VerifiedEvidenceStage::Payload,
                VerifiedEvidenceErrorClass::Integrity,
                "payload ended before declared byte count",
            )
        })?;
        for bytes in scratch[..wanted].chunks_exact(4) {
            if !f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).is_finite() {
                return Err(error(
                    VerifiedEvidenceStage::Payload,
                    VerifiedEvidenceErrorClass::Integrity,
                    "payload contains non-finite F32 value",
                ));
            }
        }
        hasher.update(&scratch[..wanted]);
        remaining -= wanted as u64;
        chunks = chunks.checked_add(1).ok_or_else(|| {
            error(
                VerifiedEvidenceStage::Payload,
                VerifiedEvidenceErrorClass::Resource,
                "payload chunk count overflows",
            )
        })?;
    }
    let mut extra = [0_u8; 1];
    if file.read(&mut extra).map_err(|_| {
        error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Io,
            "payload EOF check failed",
        )
    })? != 0
    {
        return Err(error(
            VerifiedEvidenceStage::Payload,
            VerifiedEvidenceErrorClass::Integrity,
            "payload contains appended bytes",
        ));
    }
    Ok((digest_result(hasher), chunks))
}

fn verify_stable_file(
    file: &File,
    path: &Path,
    before: &Metadata,
    root: &Path,
    relative: &Path,
    stage: VerifiedEvidenceStage,
) -> Result<(), VerifiedEvidenceError> {
    let after = file.metadata().map_err(|_| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Io,
            "opened file metadata failed after read",
        )
    })?;
    validate_no_symlink_components(root, relative, stage)?;
    let path_after = fs::metadata(path).map_err(|_| {
        error(
            stage,
            VerifiedEvidenceErrorClass::Io,
            "path metadata failed after read",
        )
    })?;
    if !same_complete_snapshot(before, &after) || !same_file_identity(&after, &path_after) {
        return Err(error(
            stage,
            VerifiedEvidenceErrorClass::Integrity,
            "file identity or metadata changed during verification",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &Metadata, right: &Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.is_file() == right.is_file()
}

#[cfg(unix)]
fn same_complete_snapshot(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

#[cfg(not(unix))]
fn same_complete_snapshot(left: &Metadata, right: &Metadata) -> bool {
    same_file_identity(left, right)
}

fn digest_bytes(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    digest_result(hasher)
}

fn digest_result(hasher: Sha256) -> Sha256Digest {
    let result = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&result);
    Sha256Digest(bytes)
}

fn digest_identity_recipe(source_shape: &[usize], final_shape: &[usize]) -> Sha256Digest {
    let mut hasher = domain_hasher(b"ullm.adapter.recipe.identity.v1");
    hash_shape(&mut hasher, source_shape);
    hash_u64(&mut hasher, 0);
    hash_shape(&mut hasher, final_shape);
    digest_result(hasher)
}

fn digest_binding_evidence(
    manifest_digest: Sha256Digest,
    resolved: &ResolvedAdmittedWeight<'_>,
    recipe_digest: Sha256Digest,
    source_digest: Sha256Digest,
    canonical_digest: Sha256Digest,
) -> Sha256Digest {
    let mut hasher = domain_hasher(b"ullm.adapter.binding.evidence.v1");
    hash_bytes(&mut hasher, manifest_digest.bytes());
    hash_bytes(&mut hasher, resolved.spec.admission_id.as_bytes());
    hash_bytes(&mut hasher, resolved.graph.graph_id.as_bytes());
    hash_bytes(&mut hasher, resolved.node_id.as_str().as_bytes());
    hash_u64(&mut hasher, resolved.use_record.weight_slot as u64);
    hash_bytes(&mut hasher, resolved.weight.id.as_str().as_bytes());
    hash_bytes(
        &mut hasher,
        resolved.binding.physical_tensor_name.as_bytes(),
    );
    hash_bytes(&mut hasher, recipe_digest.bytes());
    hash_bytes(&mut hasher, source_digest.bytes());
    hash_bytes(&mut hasher, b"F32");
    hash_bytes(&mut hasher, b"RowMajor");
    hash_shape(&mut hasher, &resolved.weight.tensor.shape);
    hash_bytes(&mut hasher, canonical_digest.bytes());
    digest_result(hasher)
}

fn domain_hasher(domain: &[u8]) -> Sha256 {
    let mut hasher = Sha256::new();
    hash_bytes(&mut hasher, domain);
    hasher
}

fn hash_shape(hasher: &mut Sha256, shape: &[usize]) {
    hash_u64(hasher, shape.len() as u64);
    for dimension in shape {
        hash_u64(hasher, *dimension as u64);
    }
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

fn error(
    stage: VerifiedEvidenceStage,
    class: VerifiedEvidenceErrorClass,
    message: &'static str,
) -> VerifiedEvidenceError {
    let message = if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        message
    } else {
        "verified evidence diagnostic exceeded bounded limit"
    };
    VerifiedEvidenceError {
        stage,
        class,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapter_admission::{
            AdapterAdmissionSpec, CanonicalTransform, CanonicalWeightUse, CanonicalizationRecipe,
            validate_structural_adapter_admission,
        },
        model_graph::{
            GraphNode, GraphNodeKind, GraphValue, ModelGraph, TensorSpec, ValueId, WeightBinding,
            WeightBindings, WeightSpec,
        },
        state_schema::StateSchema,
    };
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(1);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let nonce = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ullm-verified-evidence-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(path.join("weights")).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Clone)]
    struct TinyFixture {
        graph: ModelGraph,
        bindings: WeightBindings,
        states: StateSchema,
        admission: AdapterAdmissionSpec,
    }

    fn id_value(value: &str) -> ValueId {
        ValueId::new(value).unwrap()
    }

    fn id_node(value: &str) -> NodeId {
        NodeId::new(value).unwrap()
    }

    fn id_weight(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
    }

    fn tensor(shape: &[usize], format: NumericalFormat, layout: TensorLayout) -> TensorSpec {
        TensorSpec::new(shape.to_vec(), format, layout).unwrap()
    }

    fn tiny_fixture(payload_digest: Option<String>, non_identity: bool) -> TinyFixture {
        let input = id_value("input");
        let output = id_value("output");
        let node = id_node("linear");
        let weight = id_weight("weight");
        let weight_tensor = tensor(&[2, 2], NumericalFormat::F32, TensorLayout::RowMajor);
        let graph = ModelGraph {
            graph_id: "verified-test-graph".into(),
            inputs: vec![input.clone()],
            outputs: vec![output.clone()],
            values: vec![
                GraphValue {
                    id: input.clone(),
                    tensor: tensor(&[1, 2], NumericalFormat::F32, TensorLayout::TokensHidden),
                },
                GraphValue {
                    id: output.clone(),
                    tensor: tensor(&[1, 2], NumericalFormat::F32, TensorLayout::TokensHidden),
                },
            ],
            weights: vec![WeightSpec {
                id: weight.clone(),
                tensor: weight_tensor.clone(),
            }],
            nodes: vec![GraphNode {
                id: node.clone(),
                inputs: vec![input],
                outputs: vec![output],
                weights: vec![weight.clone()],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            }],
        };
        let bindings = WeightBindings {
            bindings: vec![WeightBinding {
                logical_id: weight.clone(),
                physical_tensor_name: "fixture.weight".into(),
                tensor: weight_tensor,
                content_sha256: payload_digest,
            }],
        };
        let recipe = if non_identity {
            CanonicalizationRecipe {
                source_shape: vec![4],
                steps: vec![CanonicalTransform::Reshape { shape: vec![2, 2] }],
            }
        } else {
            CanonicalizationRecipe {
                source_shape: vec![2, 2],
                steps: vec![],
            }
        };
        TinyFixture {
            graph,
            bindings,
            states: StateSchema::new("verified-test-states", vec![]).unwrap(),
            admission: AdapterAdmissionSpec {
                admission_id: "verified-test-admission".into(),
                graph_id: "verified-test-graph".into(),
                node_layers: vec![],
                weight_uses: vec![CanonicalWeightUse {
                    node_id: node,
                    weight_slot: 0,
                    logical_id: weight,
                    recipe,
                }],
            },
        }
    }

    fn finite_payload() -> Vec<u8> {
        [1.0_f32, -2.5, 3.25, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect()
    }

    fn write_manifest(
        root: &Path,
        payload: &[u8],
        dtype: &str,
        shape: &str,
        payload_bytes: u64,
        encoding: &str,
        payload_digest: Option<&str>,
        payload_path: &str,
    ) -> (Sha256Digest, Sha256Digest) {
        let source_digest = digest_bytes(payload);
        fs::write(root.join("weights/weight.raw"), payload).unwrap();
        let digest_field = payload_digest
            .map(|value| format!(",\"payload_sha256\":\"{value}\""))
            .unwrap_or_default();
        let manifest = format!(
            "{{\"passthrough_tensors\":[{{\"name\":\"fixture.weight\",\"dtype\":\"{dtype}\",\"shape\":{shape},\"elements\":4,\"payload_bytes\":{payload_bytes},\"payload_encoding\":\"{encoding}\"{digest_field},\"payload_file\":\"{payload_path}\"}}]}}"
        );
        fs::write(root.join("manifest.json"), manifest.as_bytes()).unwrap();
        (digest_bytes(manifest.as_bytes()), source_digest)
    }

    fn open_manifest(root: &Path, digest: Sha256Digest) -> VerifiedPackageManifest {
        open_verified_package_manifest(root, "manifest.json", &digest.to_string(), 1024 * 1024)
            .unwrap()
    }

    fn independent_u64(hasher: &mut Sha256, value: u64) {
        hasher.update(value.to_le_bytes());
    }

    fn independent_bytes(hasher: &mut Sha256, value: &[u8]) {
        independent_u64(hasher, value.len() as u64);
        hasher.update(value);
    }

    fn independent_shape(hasher: &mut Sha256, shape: &[usize]) {
        independent_u64(hasher, shape.len() as u64);
        for dimension in shape {
            independent_u64(hasher, *dimension as u64);
        }
    }

    fn independent_finish(hasher: Sha256) -> Sha256Digest {
        let result = hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&result);
        Sha256Digest(bytes)
    }

    #[test]
    fn identity_f32_receipt_is_chunk_invariant_and_instance_bound() {
        let root = TestDir::new();
        let payload = finite_payload();
        let (_, source) = write_manifest(
            &root.0,
            &payload,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&digest_bytes(&payload).to_string()),
            "weights/weight.raw",
        );
        let manifest_bytes = fs::read(root.0.join("manifest.json")).unwrap();
        let manifest_digest = digest_bytes(&manifest_bytes);
        let package = open_manifest(&root.0, manifest_digest);
        let fixture = tiny_fixture(Some(source.to_string()), false);
        let admitted = validate_structural_adapter_admission(
            &fixture.graph,
            &fixture.bindings,
            &fixture.states,
            &fixture.admission,
        )
        .unwrap();
        let receipt4 =
            verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 4).unwrap();
        let receipt8 =
            verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8).unwrap();
        let receipt12 =
            verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 12).unwrap();
        let receipt_large =
            verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, MAX_CHUNK_BYTES)
                .unwrap();
        assert_eq!(receipt4.source_digest(), &source);
        assert_eq!(receipt4.source_digest(), receipt4.canonical_digest());
        assert_eq!(receipt4.binding_digest(), receipt8.binding_digest());
        assert_eq!(receipt8.binding_digest(), receipt12.binding_digest());
        assert_eq!(receipt8.binding_digest(), receipt_large.binding_digest());
        assert_eq!(
            (
                receipt4.chunks(),
                receipt8.chunks(),
                receipt12.chunks(),
                receipt_large.chunks()
            ),
            (4, 2, 2, 1)
        );
        assert_eq!(receipt4.shape(), &[2, 2]);
        assert_eq!(receipt4.bytes(), 16);
        assert_eq!(receipt4.format(), NumericalFormat::F32);
        assert_eq!(receipt4.layout(), TensorLayout::RowMajor);
        assert!(receipt4.matches_admission(&admitted));
        assert_eq!(receipt4.manifest_digest(), &manifest_digest);
        assert_eq!(receipt4.node_id(), &id_node("linear"));
        assert_eq!(receipt4.logical_id(), &id_weight("weight"));

        let mut expected_recipe = Sha256::new();
        independent_bytes(&mut expected_recipe, b"ullm.adapter.recipe.identity.v1");
        independent_shape(&mut expected_recipe, &[2, 2]);
        independent_u64(&mut expected_recipe, 0);
        independent_shape(&mut expected_recipe, &[2, 2]);
        let expected_recipe = independent_finish(expected_recipe);
        assert_eq!(receipt4.recipe_digest(), &expected_recipe);

        let mut expected_binding = Sha256::new();
        independent_bytes(&mut expected_binding, b"ullm.adapter.binding.evidence.v1");
        independent_bytes(&mut expected_binding, manifest_digest.bytes());
        independent_bytes(&mut expected_binding, b"verified-test-admission");
        independent_bytes(&mut expected_binding, b"verified-test-graph");
        independent_bytes(&mut expected_binding, b"linear");
        independent_u64(&mut expected_binding, 0);
        independent_bytes(&mut expected_binding, b"weight");
        independent_bytes(&mut expected_binding, b"fixture.weight");
        independent_bytes(&mut expected_binding, expected_recipe.bytes());
        independent_bytes(&mut expected_binding, source.bytes());
        independent_bytes(&mut expected_binding, b"F32");
        independent_bytes(&mut expected_binding, b"RowMajor");
        independent_shape(&mut expected_binding, &[2, 2]);
        independent_bytes(&mut expected_binding, source.bytes());
        assert_eq!(
            receipt4.binding_digest(),
            &independent_finish(expected_binding)
        );

        let cloned = fixture.clone();
        let other = validate_structural_adapter_admission(
            &cloned.graph,
            &cloned.bindings,
            &cloned.states,
            &cloned.admission,
        )
        .unwrap();
        assert!(!receipt4.matches_admission(&other));

        let mut independent = Sha256::new();
        independent.update(&payload);
        assert_eq!(source.bytes().as_slice(), independent.finalize().as_slice());
        assert_eq!(source.as_hex().unwrap(), source.to_string());
    }

    #[test]
    fn manifest_trust_root_digest_path_and_limits_fail_closed() {
        let root = TestDir::new();
        let payload = finite_payload();
        let (manifest, source) = write_manifest(
            &root.0,
            &payload,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&digest_bytes(&payload).to_string()),
            "weights/weight.raw",
        );
        assert!(
            open_verified_package_manifest(&root.0, "manifest.json", &source.to_string(), 1024)
                .is_err()
        );
        assert!(
            open_verified_package_manifest(
                &root.0,
                "manifest.json",
                &manifest.to_string().to_uppercase(),
                1024
            )
            .is_err()
        );
        assert!(
            open_verified_package_manifest(
                root.0.join("weights/.."),
                "manifest.json",
                &manifest.to_string(),
                1024
            )
            .is_err()
        );
        open_verified_package_manifest(
            root.0.join("."),
            "manifest.json",
            &manifest.to_string(),
            1024,
        )
        .unwrap();
        assert!(
            open_verified_package_manifest(&root.0, "manifest.json", &manifest.to_string(), 0)
                .is_err()
        );
        assert!(
            open_verified_package_manifest(&root.0, "manifest.json", &manifest.to_string(), 8)
                .is_err()
        );
        assert!(
            open_verified_package_manifest(
                &root.0,
                "manifest.json",
                &manifest.to_string(),
                MAX_MANIFEST_BYTES + 1
            )
            .is_err()
        );
        assert!(
            open_verified_package_manifest(
                &root.0,
                "../manifest.json",
                &manifest.to_string(),
                1024
            )
            .is_err()
        );
        assert!(
            open_verified_package_manifest(
                &root.0,
                root.0.join("manifest.json"),
                &manifest.to_string(),
                1024
            )
            .is_err()
        );
        assert!(open_verified_package_manifest(&root.0, "", &manifest.to_string(), 1024).is_err());
    }

    #[test]
    fn manifest_lexical_amplification_is_publicly_classified_as_resource() {
        let root = TestDir::new();
        let mut deep = "[".repeat(65);
        deep.push('0');
        deep.push_str(&"]".repeat(65));
        let huge_string = format!("{{\"x\":\"{}\"}}", "a".repeat(4_097));
        let mut token_heavy = String::from("[");
        for index in 0..131_072 {
            if index != 0 {
                token_heavy.push(',');
            }
            token_heavy.push('0');
        }
        token_heavy.push(']');
        for manifest in [deep, huge_string, token_heavy] {
            fs::write(root.0.join("manifest.json"), manifest.as_bytes()).unwrap();
            let error = open_verified_package_manifest(
                &root.0,
                "manifest.json",
                &digest_bytes(manifest.as_bytes()).to_string(),
                1024 * 1024,
            )
            .unwrap_err();
            assert_eq!(error.class(), VerifiedEvidenceErrorClass::Resource);
            assert_eq!(error.stage(), VerifiedEvidenceStage::Manifest);
        }
    }

    #[test]
    fn descriptor_digest_dtype_shape_bytes_and_encoding_fail_closed() {
        let cases = [
            ("F32", "[2,2]", 16, "raw_safetensors_payload", None),
            (
                "F32",
                "[2,2]",
                16,
                "raw_safetensors_payload",
                Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            ),
            (
                "BF16",
                "[2,2]",
                16,
                "raw_safetensors_payload",
                Some("valid"),
            ),
            ("F32", "[1,4]", 16, "raw_safetensors_payload", Some("valid")),
            ("F32", "[2,2]", 12, "raw_safetensors_payload", Some("valid")),
            ("F32", "[2,2]", 16, "raw", Some("valid")),
        ];
        for (dtype, shape, bytes, encoding, digest_kind) in cases {
            let root = TestDir::new();
            let payload = finite_payload();
            let valid = digest_bytes(&payload).to_string();
            let digest = match digest_kind {
                None => None,
                Some("valid") => Some(valid.as_str()),
                Some(value) => Some(value),
            };
            let (manifest, source) = write_manifest(
                &root.0,
                &payload,
                dtype,
                shape,
                bytes,
                encoding,
                digest,
                "weights/weight.raw",
            );
            let opened = open_verified_package_manifest(
                &root.0,
                "manifest.json",
                &manifest.to_string(),
                4096,
            );
            let failed = match opened {
                Err(_) => true,
                Ok(package) => {
                    let fixture = tiny_fixture(Some(source.to_string()), false);
                    let admitted = validate_structural_adapter_admission(
                        &fixture.graph,
                        &fixture.bindings,
                        &fixture.states,
                        &fixture.admission,
                    )
                    .unwrap();
                    verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8)
                        .is_err()
                }
            };
            assert!(failed);
        }
        for payload_path in ["../weight.raw", "/tmp/weight.raw", "weights//weight.raw"] {
            let root = TestDir::new();
            let payload = finite_payload();
            let valid = digest_bytes(&payload).to_string();
            let (manifest, _) = write_manifest(
                &root.0,
                &payload,
                "F32",
                "[2,2]",
                16,
                "raw_safetensors_payload",
                Some(&valid),
                payload_path,
            );
            assert!(
                open_verified_package_manifest(
                    &root.0,
                    "manifest.json",
                    &manifest.to_string(),
                    4096
                )
                .is_err()
            );
        }
    }

    #[test]
    fn binding_occurrence_recipe_chunk_and_payload_integrity_fail_closed() {
        let root = TestDir::new();
        let payload = finite_payload();
        let source = digest_bytes(&payload);
        let (manifest, _) = write_manifest(
            &root.0,
            &payload,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&source.to_string()),
            "weights/weight.raw",
        );
        let package = open_manifest(&root.0, manifest);

        for binding in [None, Some("0".repeat(64))] {
            let fixture = tiny_fixture(binding, false);
            let admitted = validate_structural_adapter_admission(
                &fixture.graph,
                &fixture.bindings,
                &fixture.states,
                &fixture.admission,
            )
            .unwrap();
            assert!(
                verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8).is_err()
            );
        }
        let wrong_declared = "0".repeat(64);
        let (wrong_manifest, _) = write_manifest(
            &root.0,
            &payload,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&wrong_declared),
            "weights/weight.raw",
        );
        let wrong_package = open_manifest(&root.0, wrong_manifest);
        let wrong_fixture = tiny_fixture(Some(wrong_declared), false);
        let wrong_admitted = validate_structural_adapter_admission(
            &wrong_fixture.graph,
            &wrong_fixture.bindings,
            &wrong_fixture.states,
            &wrong_fixture.admission,
        )
        .unwrap();
        assert!(
            verify_identity_f32_weight(&wrong_admitted, &wrong_package, &id_node("linear"), 0, 8)
                .is_err()
        );
        let fixture = tiny_fixture(Some(source.to_string()), false);
        let admitted = validate_structural_adapter_admission(
            &fixture.graph,
            &fixture.bindings,
            &fixture.states,
            &fixture.admission,
        )
        .unwrap();
        assert!(
            verify_identity_f32_weight(&admitted, &package, &id_node("missing"), 0, 8).is_err()
        );
        assert!(verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 1, 8).is_err());
        for chunk in [0, 2, MAX_CHUNK_BYTES + 4] {
            assert!(
                verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, chunk)
                    .is_err()
            );
        }

        let transformed = tiny_fixture(Some(source.to_string()), true);
        let transformed_admission = validate_structural_adapter_admission(
            &transformed.graph,
            &transformed.bindings,
            &transformed.states,
            &transformed.admission,
        )
        .unwrap();
        let error =
            verify_identity_f32_weight(&transformed_admission, &package, &id_node("linear"), 0, 8)
                .unwrap_err();
        assert_eq!(error.class(), VerifiedEvidenceErrorClass::Unsupported);

        let mut bf16 = tiny_fixture(Some(source.to_string()), false);
        for value in &mut bf16.graph.values {
            value.tensor.format = NumericalFormat::Bf16;
        }
        bf16.graph.weights[0].tensor.format = NumericalFormat::Bf16;
        bf16.bindings.bindings[0].tensor.format = NumericalFormat::Bf16;
        let bf16_admission = validate_structural_adapter_admission(
            &bf16.graph,
            &bf16.bindings,
            &bf16.states,
            &bf16.admission,
        )
        .unwrap();
        let error = verify_identity_f32_weight(&bf16_admission, &package, &id_node("linear"), 0, 8)
            .unwrap_err();
        assert_eq!(error.class(), VerifiedEvidenceErrorClass::Unsupported);

        let nonfinite = [1.0_f32, f32::NAN, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        fs::write(root.0.join("weights/weight.raw"), &nonfinite).unwrap();
        let nonfinite_digest = digest_bytes(&nonfinite);
        let fixture = tiny_fixture(Some(nonfinite_digest.to_string()), false);
        let (manifest, _) = write_manifest(
            &root.0,
            &nonfinite,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&nonfinite_digest.to_string()),
            "weights/weight.raw",
        );
        let package = open_manifest(&root.0, manifest);
        let admitted = validate_structural_adapter_admission(
            &fixture.graph,
            &fixture.bindings,
            &fixture.states,
            &fixture.admission,
        )
        .unwrap();
        assert!(verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8).is_err());
    }

    #[test]
    fn quantized_only_manifest_is_unsupported_without_component_digests() {
        let root = TestDir::new();
        let manifest =
            br#"{"tensors":[{"name":"aq.weight","dtype":"AQ4_0","shape":[2,2],"elements":4}]}"#;
        fs::write(root.0.join("manifest.json"), manifest).unwrap();
        let package = open_manifest(&root.0, digest_bytes(manifest));
        let payload = finite_payload();
        let source = digest_bytes(&payload);
        let fixture = tiny_fixture(Some(source.to_string()), false);
        let admitted = validate_structural_adapter_admission(
            &fixture.graph,
            &fixture.bindings,
            &fixture.states,
            &fixture.admission,
        )
        .unwrap();
        let error =
            verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8).unwrap_err();
        assert_eq!(error.class(), VerifiedEvidenceErrorClass::Unsupported);
    }

    #[cfg(unix)]
    #[test]
    fn final_and_intermediate_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let payload = finite_payload();
        let source = digest_bytes(&payload);
        let root = TestDir::new();
        let (manifest, _) = write_manifest(
            &root.0,
            &payload,
            "F32",
            "[2,2]",
            16,
            "raw_safetensors_payload",
            Some(&source.to_string()),
            "weights/weight.raw",
        );
        symlink(".", root.0.join("linked-root")).unwrap();
        assert!(
            open_verified_package_manifest(
                root.0.join("linked-root"),
                "manifest.json",
                &manifest.to_string(),
                4096
            )
            .is_err()
        );

        for intermediate in [false, true] {
            let root = TestDir::new();
            let (manifest, _) = write_manifest(
                &root.0,
                &payload,
                "F32",
                "[2,2]",
                16,
                "raw_safetensors_payload",
                Some(&source.to_string()),
                "weights/weight.raw",
            );
            if intermediate {
                fs::rename(root.0.join("weights"), root.0.join("real-weights")).unwrap();
                symlink("real-weights", root.0.join("weights")).unwrap();
            } else {
                fs::rename(
                    root.0.join("weights/weight.raw"),
                    root.0.join("real-weight.raw"),
                )
                .unwrap();
                symlink("../real-weight.raw", root.0.join("weights/weight.raw")).unwrap();
            }
            let package = open_manifest(&root.0, manifest);
            let fixture = tiny_fixture(Some(source.to_string()), false);
            let admitted = validate_structural_adapter_admission(
                &fixture.graph,
                &fixture.bindings,
                &fixture.states,
                &fixture.admission,
            )
            .unwrap();
            assert!(
                verify_identity_f32_weight(&admitted, &package, &id_node("linear"), 0, 8).is_err()
            );
        }

        let root = TestDir::new();
        fs::write(root.0.join("real-manifest.json"), b"{}").unwrap();
        symlink("real-manifest.json", root.0.join("manifest.json")).unwrap();
        assert!(
            open_verified_package_manifest(
                &root.0,
                "manifest.json",
                &digest_bytes(b"{}").to_string(),
                1024
            )
            .is_err()
        );
    }

    #[test]
    fn metadata_snapshot_detects_append_truncate_rewrite_and_path_replace() {
        enum Mutation {
            Append,
            Truncate,
            Rewrite,
            Replace,
        }
        for mutation in [
            Mutation::Append,
            Mutation::Truncate,
            Mutation::Rewrite,
            Mutation::Replace,
        ] {
            let root = TestDir::new();
            let relative = Path::new("weights/weight.raw");
            fs::write(root.0.join(relative), finite_payload()).unwrap();
            let canonical = fs::canonicalize(&root.0).unwrap();
            let (file, path, before) =
                open_regular_beneath(&canonical, relative, VerifiedEvidenceStage::Payload).unwrap();
            match mutation {
                Mutation::Append => {
                    use std::io::Write;
                    let mut writer = fs::OpenOptions::new().append(true).open(&path).unwrap();
                    writer.write_all(&[0, 0, 0, 0]).unwrap();
                    writer.sync_all().unwrap();
                }
                Mutation::Truncate => {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(&path)
                        .unwrap()
                        .set_len(4)
                        .unwrap();
                }
                Mutation::Rewrite => {
                    fs::write(&path, vec![7_u8; before.len() as usize]).unwrap();
                }
                Mutation::Replace => {
                    let replacement = root.0.join("replacement.raw");
                    fs::write(&replacement, finite_payload()).unwrap();
                    fs::rename(replacement, &path).unwrap();
                }
            }
            assert!(
                verify_stable_file(
                    &file,
                    &path,
                    &before,
                    &canonical,
                    relative,
                    VerifiedEvidenceStage::Payload
                )
                .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn public_verifier_wiring_rejects_after_read_append_rewrite_and_replace() {
        use std::io::Write;

        enum Mutation {
            Append,
            Rewrite,
            Replace,
        }
        for mutation in [Mutation::Append, Mutation::Rewrite, Mutation::Replace] {
            let root = TestDir::new();
            let payload = finite_payload();
            let source = digest_bytes(&payload);
            let (manifest, _) = write_manifest(
                &root.0,
                &payload,
                "F32",
                "[2,2]",
                16,
                "raw_safetensors_payload",
                Some(&source.to_string()),
                "weights/weight.raw",
            );
            let package = open_manifest(&root.0, manifest);
            let fixture = tiny_fixture(Some(source.to_string()), false);
            let admitted = validate_structural_adapter_admission(
                &fixture.graph,
                &fixture.bindings,
                &fixture.states,
                &fixture.admission,
            )
            .unwrap();
            let result = verify_identity_f32_weight_with_test_hook(
                &admitted,
                &package,
                &id_node("linear"),
                0,
                8,
                |path| match mutation {
                    Mutation::Append => {
                        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
                        file.write_all(&[0, 0, 0, 0]).unwrap();
                        file.sync_all().unwrap();
                    }
                    Mutation::Rewrite => {
                        fs::write(path, vec![9_u8; payload.len()]).unwrap();
                    }
                    Mutation::Replace => {
                        let replacement = root.0.join("replacement-after-read.raw");
                        fs::write(&replacement, &payload).unwrap();
                        fs::rename(replacement, path).unwrap();
                    }
                },
            );
            let error = result.unwrap_err();
            assert_eq!(error.class(), VerifiedEvidenceErrorClass::Integrity);
        }
    }
}
