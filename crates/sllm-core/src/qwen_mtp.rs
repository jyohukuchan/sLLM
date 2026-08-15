//! Exact Qwen3.5-4B MTP component manifest.

use crate::{
    ModelLock, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID, QWEN35_4B_REVISION, TensorDType,
    TensorDescriptor, VerifiedCache,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const QWEN35_MTP_TENSOR_COUNT: usize = 15;
pub const QWEN35_MTP_HIDDEN_SIZE: u64 = 2_560;
pub const QWEN35_MTP_INTERMEDIATE_SIZE: u64 = 9_216;
pub const QWEN35_MTP_DRAFT_WIDTH: usize = 2;

const MANIFEST_DOMAIN: &[u8] = b"sLLM-qwen35-mtp-manifest-v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenMtpTensor {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub source_file: String,
    pub source_range: [u64; 2],
    pub byte_size: u64,
    pub source_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenMtpManifest {
    pub repo_id: String,
    pub resolved_revision: String,
    pub model_fingerprint: String,
    pub shared_embedding: String,
    pub shared_output: String,
    pub resident_bytes: u64,
    pub tensors: Vec<QwenMtpTensor>,
    digest: [u8; 32],
}

impl QwenMtpManifest {
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn digest_hex(&self) -> String {
        format!("sha256:{}", hex(&self.digest))
    }
}

pub fn build_verified_qwen35_mtp_manifest(
    lock: &ModelLock,
    cache: &VerifiedCache,
) -> Result<QwenMtpManifest, QwenMtpError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(QwenMtpError::Invalid(
            "verified cache fingerprint differs from model lock".to_owned(),
        ));
    }
    build_qwen35_mtp_manifest(lock, cache.tensors())
}

pub fn build_qwen35_mtp_manifest<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> Result<QwenMtpManifest, QwenMtpError> {
    validate_lock(lock)?;
    let locked_files = lock
        .model
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.sha256.as_str()))
        .collect::<BTreeMap<_, _>>();
    if locked_files.len() != lock.model.files.len() {
        return Err(QwenMtpError::Invalid(
            "model lock contains duplicate files".to_owned(),
        ));
    }

    let expected = expected_shapes();
    let mut observed = BTreeSet::new();
    let mut tensors = Vec::with_capacity(QWEN35_MTP_TENSOR_COUNT);
    let mut resident_bytes = 0_u64;
    let mut has_shared_embedding = false;
    for descriptor in descriptors {
        if descriptor.tensor_name == "model.language_model.embed_tokens.weight" {
            has_shared_embedding = true;
        }
        if !descriptor.tensor_name.starts_with("mtp.") {
            continue;
        }
        let shape = expected
            .get(descriptor.tensor_name.as_str())
            .ok_or_else(|| {
                QwenMtpError::Invalid(format!("unknown MTP tensor: {}", descriptor.tensor_name))
            })?;
        if !observed.insert(descriptor.tensor_name.as_str()) {
            return Err(QwenMtpError::Invalid(format!(
                "duplicate MTP tensor: {}",
                descriptor.tensor_name
            )));
        }
        if descriptor.dtype != TensorDType::Bf16 || descriptor.shape.as_slice() != shape.as_slice()
        {
            return Err(QwenMtpError::Invalid(format!(
                "MTP tensor shape or dtype differs: {}",
                descriptor.tensor_name
            )));
        }
        let source_sha256 = locked_files
            .get(descriptor.source_file.as_str())
            .copied()
            .ok_or_else(|| {
                QwenMtpError::Invalid(format!(
                    "MTP tensor source is not locked: {}",
                    descriptor.tensor_name
                ))
            })?;
        if descriptor.absolute_byte_range[0] >= descriptor.absolute_byte_range[1]
            || descriptor.absolute_byte_range[1] - descriptor.absolute_byte_range[0]
                != descriptor.byte_size
        {
            return Err(QwenMtpError::Invalid(format!(
                "MTP tensor range differs: {}",
                descriptor.tensor_name
            )));
        }
        resident_bytes = resident_bytes
            .checked_add(descriptor.byte_size)
            .ok_or_else(|| QwenMtpError::Invalid("MTP resident bytes overflowed".to_owned()))?;
        tensors.push(QwenMtpTensor {
            name: descriptor.tensor_name.clone(),
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            source_range: descriptor.absolute_byte_range,
            byte_size: descriptor.byte_size,
            source_sha256: source_sha256.to_owned(),
        });
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    if !has_shared_embedding {
        return Err(QwenMtpError::Invalid(
            "shared embedding/output tensor is missing".to_owned(),
        ));
    }
    if tensors.len() != QWEN35_MTP_TENSOR_COUNT || observed.len() != expected.len() {
        return Err(QwenMtpError::Invalid(format!(
            "MTP tensor set differs: observed={}, expected={QWEN35_MTP_TENSOR_COUNT}",
            tensors.len()
        )));
    }
    let digest = digest_manifest(lock, resident_bytes, &tensors);
    Ok(QwenMtpManifest {
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        model_fingerprint: lock.fingerprint().to_owned(),
        shared_embedding: "model.language_model.embed_tokens.weight".to_owned(),
        shared_output: "model.language_model.embed_tokens.weight".to_owned(),
        resident_bytes,
        tensors,
        digest,
    })
}

fn validate_lock(lock: &ModelLock) -> Result<(), QwenMtpError> {
    let text = &lock.model.architecture.text_config;
    if lock.schema_version != "model-lock-v1"
        || lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
        || lock.model.architecture.mtp.tensor_prefix != "mtp."
        || lock.model.architecture.mtp.tensor_count != QWEN35_MTP_TENSOR_COUNT as u64
        || text.mtp_num_hidden_layers != 1
        || !text.tie_word_embeddings
    {
        return Err(QwenMtpError::Invalid(
            "model is not the fixed shared-embedding Qwen3.5-4B MTP contract".to_owned(),
        ));
    }
    Ok(())
}

fn expected_shapes() -> BTreeMap<&'static str, Vec<u64>> {
    BTreeMap::from([
        ("mtp.fc.weight", vec![2_560, 5_120]),
        ("mtp.layers.0.input_layernorm.weight", vec![2_560]),
        ("mtp.layers.0.mlp.down_proj.weight", vec![2_560, 9_216]),
        ("mtp.layers.0.mlp.gate_proj.weight", vec![9_216, 2_560]),
        ("mtp.layers.0.mlp.up_proj.weight", vec![9_216, 2_560]),
        ("mtp.layers.0.post_attention_layernorm.weight", vec![2_560]),
        ("mtp.layers.0.self_attn.k_norm.weight", vec![256]),
        ("mtp.layers.0.self_attn.k_proj.weight", vec![1_024, 2_560]),
        ("mtp.layers.0.self_attn.o_proj.weight", vec![2_560, 4_096]),
        ("mtp.layers.0.self_attn.q_norm.weight", vec![256]),
        ("mtp.layers.0.self_attn.q_proj.weight", vec![8_192, 2_560]),
        ("mtp.layers.0.self_attn.v_proj.weight", vec![1_024, 2_560]),
        ("mtp.norm.weight", vec![2_560]),
        ("mtp.pre_fc_norm_embedding.weight", vec![2_560]),
        ("mtp.pre_fc_norm_hidden.weight", vec![2_560]),
    ])
}

fn digest_manifest(lock: &ModelLock, resident_bytes: u64, tensors: &[QwenMtpTensor]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(MANIFEST_DOMAIN);
    hash.update(lock.model.repo_id.as_bytes());
    hash.update([0]);
    hash.update(lock.model.resolved_revision.as_bytes());
    hash.update([0]);
    hash.update(lock.fingerprint().as_bytes());
    hash.update(resident_bytes.to_le_bytes());
    for tensor in tensors {
        hash.update(tensor.name.as_bytes());
        hash.update([0]);
        hash.update([match tensor.dtype {
            TensorDType::Bf16 => 1,
            TensorDType::F32 => 2,
            TensorDType::F16 => 3,
            TensorDType::I32 => 4,
            TensorDType::I64 => 5,
            TensorDType::U8 => 6,
        }]);
        hash.update((tensor.shape.len() as u64).to_le_bytes());
        for dimension in &tensor.shape {
            hash.update(dimension.to_le_bytes());
        }
        hash.update(tensor.source_file.as_bytes());
        hash.update([0]);
        hash.update(tensor.source_sha256.as_bytes());
        hash.update(tensor.source_range[0].to_le_bytes());
        hash.update(tensor.source_range[1].to_le_bytes());
    }
    hash.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QwenMtpError {
    Invalid(String),
}

impl fmt::Display for QwenMtpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid Qwen3.5 MTP manifest: {message}"),
        }
    }
}

impl std::error::Error for QwenMtpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_manifest_has_all_fifteen_unique_shapes() {
        let shapes = expected_shapes();
        assert_eq!(shapes.len(), QWEN35_MTP_TENSOR_COUNT);
        assert_eq!(shapes["mtp.fc.weight"], [2_560, 5_120]);
        assert_eq!(
            shapes["mtp.layers.0.self_attn.q_proj.weight"],
            [8_192, 2_560]
        );
        assert!(!shapes.keys().any(|name| name.contains("embed_tokens")));
    }
}
