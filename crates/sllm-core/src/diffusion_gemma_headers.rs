//! Header-only safetensors catalog for the reviewed DiffusionGemma artifact.
//!
//! A caller must bounded-read exactly the eight-byte safetensors length field
//! plus the declared JSON header. Full shard bytes and tensor payloads are
//! deliberately rejected by the parser and are never materialized here.

use crate::diffusion_gemma::{
    DIFFUSION_GEMMA_CATALOG_SHA256, DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
    DIFFUSION_GEMMA_SHARD_COUNT, DIFFUSION_GEMMA_SHARD_FILE_BYTES, DIFFUSION_GEMMA_SHARDS,
    DIFFUSION_GEMMA_TENSOR_COUNT, DIFFUSION_GEMMA_TOTAL_PARAMETERS, DiffusionGemmaIndex,
};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const DIFFUSION_GEMMA_TENSOR_PAYLOAD_BYTES: u64 = DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES;
pub const DIFFUSION_GEMMA_HEADER_PREFIX_BYTES: u64 = 138_568;
pub const DIFFUSION_GEMMA_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES: u64 = 8;

/// SHA-256 over canonical header identities and every tensor's source shard,
/// dtype, shape, relative range, and absolute range. It covers no payload.
pub const DIFFUSION_GEMMA_HEADER_CATALOG_SHA256: &str =
    "fd2cdedb367cd6c9aa52af6463e73baff3df52477b9cc3d61b9c6c4213cdc86f";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaHeaderIdentity {
    pub file_name: &'static str,
    pub file_size: u64,
    /// Git LFS SHA-256 OID from the fixed-revision Hub metadata.
    pub lfs_sha256: &'static str,
    /// JSON header bytes, excluding the eight-byte length field.
    pub header_length: u64,
    /// SHA-256 of the eight-byte length field followed by the JSON header.
    pub header_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const DIFFUSION_GEMMA_HEADER_IDENTITIES: [DiffusionGemmaHeaderIdentity;
    DIFFUSION_GEMMA_SHARD_COUNT] = [
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00001-of-00011.safetensors",
        file_size: 4_732_780_476,
        lfs_sha256: "3efe137998af7d2bde4e3ab04ab3524823699a4ac3130adace5003ef40cceeb6",
        header_length: 5_552,
        header_sha256: "c7b4772d3cd3d9120a5e003f6c53e6b489a5a7930a262238b292cb4082f895be",
        indexed_tensor_count: 45,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00002-of-00011.safetensors",
        file_size: 4_884_578_006,
        lfs_sha256: "4a39d68c756fb26bbd2a54f2b8d550047ea98f3152f87ec75db825c8e17934a7",
        header_length: 8_136,
        header_sha256: "520e7b6f0eb2300ceee5df0bf501a862f25f1d4557b5449dedaaf61f2bb78b83",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00003-of-00011.safetensors",
        file_size: 4_913_414_742,
        lfs_sha256: "ac6083e3489215ca032501714b78832e5cc4c945a8dbbb905b5292a4e95bc75e",
        header_length: 8_008,
        header_sha256: "77ee0f0390ff0eb0eef5f4ae710ae8a2cb832c1f2793b46979270d9e2fd3eacf",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00004-of-00011.safetensors",
        file_size: 4_884_578_030,
        lfs_sha256: "865b66393de5a9c752e67beae1bd2c860c12786b72ad86ca9345c00b7b586e60",
        header_length: 8_160,
        header_sha256: "f309460fb6cfc514e6aa212caf55326b4ad62fcc759beb867ab10edb8f1edb63",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00005-of-00011.safetensors",
        file_size: 4_913_414_814,
        lfs_sha256: "a87e01bed77ad9d2234851267d99af583ff79c7a86d7494fe08ae0f9de9cd318",
        header_length: 8_080,
        header_sha256: "811297b84587f781aba57f487ac3b998dfe915b29aa63c3a2492dc03a4fea410",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00006-of-00011.safetensors",
        file_size: 4_884_578_070,
        lfs_sha256: "077e841b3b138fbc38df2c36665abdaafa96b8c96f968e2263a7452afaa912ab",
        header_length: 8_200,
        header_sha256: "b1bde87bed1f47234a87a6a6302c4e6d55b083d2e3cabddba2647328c76cbb55",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00007-of-00011.safetensors",
        file_size: 4_913_414_814,
        lfs_sha256: "13a18b3c04a7f19a16385dd8a0d7fd0b9cc89ef131e4294a063c31c7d90c58ef",
        header_length: 8_080,
        header_sha256: "a3397347e20e09153a78c9bcd32aa058cc65c4efc027aa2d6ca0bf5c37f6feb1",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00008-of-00011.safetensors",
        file_size: 4_884_578_070,
        lfs_sha256: "aca5d3bdfc84700bd55b475781cb46e5608104c4832d2abe111a119ddcb23ff7",
        header_length: 8_200,
        header_sha256: "5455c3185a7419ece95ccad00c4351e42775cb66dd71632b6327a1ed084f9fbb",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00009-of-00011.safetensors",
        file_size: 4_913_414_814,
        lfs_sha256: "8e2418867354e5cb356c0af8ccfdcc60bc363500674d6cf395991e7d6219eb29",
        header_length: 8_080,
        header_sha256: "2a0a44c2755e089466755c8c02d123546fdc50b2c044c49a0c19dd46fec7b597",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00010-of-00011.safetensors",
        file_size: 4_884_578_070,
        lfs_sha256: "93d564b7dd686464a5c068ff9665cd5d3bca399c2ce320aecd41bd011e3787d5",
        header_length: 8_200,
        header_sha256: "bd02dfa8059eb38c5b71ec8be36444170fb20dcd9f49b71a9ab7cbee5d57a62a",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaHeaderIdentity {
        file_name: "model-00011-of-00011.safetensors",
        file_size: 2_838_371_118,
        lfs_sha256: "afec047176bb2a05f078566576aec6bdb71ad4d041275d0d0c89473fda6d6d87",
        header_length: 59_784,
        header_sha256: "40daac48522b6a890b3d9d03710fdef0d3c8fc9cec313fad7fbe5c2b79f9566e",
        indexed_tensor_count: 412,
    },
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaSafetensorsDType {
    Bf16,
}

impl DiffusionGemmaSafetensorsDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
        }
    }

    fn parse(value: &str) -> Result<Self, DiffusionGemmaHeaderError> {
        match value {
            "BF16" => Ok(Self::Bf16),
            other => Err(invalid(format!(
                "unsupported safetensors dtype for the fixed BF16 artifact: {other}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaHeaderTensor {
    pub name: String,
    /// One-based official shard number.
    pub shard_index: u32,
    pub dtype: DiffusionGemmaSafetensorsDType,
    pub shape: Vec<u64>,
    /// Start-inclusive, end-exclusive offsets relative to the tensor payload.
    pub data_offsets: [u64; 2],
    /// Start-inclusive, end-exclusive offsets relative to the complete shard.
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaShardHeader {
    pub file_name: String,
    pub shard_index: u32,
    pub file_size: u64,
    pub header_length: u64,
    pub header_sha256: String,
    pub data_start: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<DiffusionGemmaHeaderTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaHeaderCatalog {
    shards: Vec<DiffusionGemmaShardHeader>,
    tensor_count: usize,
    shard_file_bytes: u64,
    payload_bytes: u64,
    catalog_sha256: String,
}

impl DiffusionGemmaHeaderCatalog {
    pub fn shards(&self) -> &[DiffusionGemmaShardHeader] {
        &self.shards
    }

    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    pub const fn shard_file_bytes(&self) -> u64 {
        self.shard_file_bytes
    }

    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn tensors(&self) -> impl Iterator<Item = &DiffusionGemmaHeaderTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn tensor(&self, name: &str) -> Option<&DiffusionGemmaHeaderTensor> {
        self.tensors().find(|tensor| tensor.name == name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaHeaderError {
    Invalid(String),
}

impl fmt::Display for DiffusionGemmaHeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(
                formatter,
                "invalid DiffusionGemma safetensors header catalog: {message}"
            ),
        }
    }
}

impl std::error::Error for DiffusionGemmaHeaderError {}

fn invalid(message: impl Into<String>) -> DiffusionGemmaHeaderError {
    DiffusionGemmaHeaderError::Invalid(message.into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct UniqueRawHeaderDocument {
    tensors: BTreeMap<String, RawHeaderTensor>,
}

impl<'de> Deserialize<'de> for UniqueRawHeaderDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = UniqueRawHeaderDocument;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header without duplicate keys")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut metadata = None;
                let mut tensors = BTreeMap::new();
                while let Some(name) = entries.next_key::<String>()? {
                    if name == "__metadata__" {
                        if metadata.is_some() {
                            return Err(serde::de::Error::custom("duplicate __metadata__"));
                        }
                        metadata = Some(entries.next_value::<BTreeMap<String, String>>()?);
                    } else {
                        let tensor = entries.next_value::<RawHeaderTensor>()?;
                        if tensors.insert(name.clone(), tensor).is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "duplicate tensor name: {name}"
                            )));
                        }
                    }
                }
                let metadata =
                    metadata.ok_or_else(|| serde::de::Error::custom("missing __metadata__"))?;
                if metadata.len() != 1 || metadata.get("format").map(String::as_str) != Some("pt") {
                    return Err(serde::de::Error::custom(
                        "safetensors metadata must be exactly format=pt",
                    ));
                }
                Ok(UniqueRawHeaderDocument { tensors })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn safe_tensor_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !name.contains("..")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.')
}

fn locked_header_identity(
    file_name: &str,
) -> Result<(usize, DiffusionGemmaHeaderIdentity), DiffusionGemmaHeaderError> {
    if file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
    {
        return Err(invalid("unsafe shard path"));
    }
    let found = DIFFUSION_GEMMA_HEADER_IDENTITIES
        .iter()
        .copied()
        .enumerate()
        .find(|(_, identity)| identity.file_name == file_name)
        .ok_or_else(|| invalid(format!("unknown shard: {file_name}")))?;
    let source_identity = DIFFUSION_GEMMA_SHARDS
        .get(found.0)
        .ok_or_else(|| invalid("header/source shard tables differ in length"))?;
    if source_identity.file_name != found.1.file_name
        || source_identity.size != found.1.file_size
        || source_identity.lfs_sha256 != found.1.lfs_sha256
        || source_identity.indexed_tensor_count != found.1.indexed_tensor_count
    {
        return Err(invalid("header/source shard identity tables differ"));
    }
    Ok(found)
}

pub fn diffusion_gemma_locked_header(file_name: &str) -> Option<DiffusionGemmaHeaderIdentity> {
    DIFFUSION_GEMMA_HEADER_IDENTITIES
        .iter()
        .copied()
        .find(|identity| identity.file_name == file_name)
}

fn checked_shape_bytes(
    shape: &[u64],
    dtype: DiffusionGemmaSafetensorsDType,
) -> Result<u64, DiffusionGemmaHeaderError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid("empty or zero-sized tensor shape is not reviewed"));
    }
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid("tensor shape byte count overflowed"))
    })?;
    elements
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| invalid("tensor shape byte count overflowed"))
}

fn validate_ranges(
    tensors: &[DiffusionGemmaHeaderTensor],
    payload_bytes: u64,
) -> Result<(), DiffusionGemmaHeaderError> {
    let mut order = Vec::new();
    order
        .try_reserve_exact(tensors.len())
        .map_err(|_| invalid("tensor range ordering allocation failed"))?;
    order.extend(0..tensors.len());
    order.sort_by_key(|index| {
        (
            tensors[*index].data_offsets[0],
            tensors[*index].data_offsets[1],
            tensors[*index].name.as_str(),
        )
    });
    let mut previous_end = 0_u64;
    for (ordinal, index) in order.into_iter().enumerate() {
        let tensor = &tensors[index];
        let [start, end] = tensor.data_offsets;
        if start >= end || end > payload_bytes {
            return Err(invalid(format!(
                "tensor range is outside payload: {}",
                tensor.name
            )));
        }
        if ordinal == 0 && start != 0 {
            return Err(invalid("tensor payload has a leading gap"));
        }
        if start < previous_end {
            return Err(invalid(format!("tensor ranges overlap: {}", tensor.name)));
        }
        if start != previous_end {
            return Err(invalid(format!(
                "tensor payload has a gap before {}",
                tensor.name
            )));
        }
        previous_end = end;
    }
    if previous_end != payload_bytes {
        return Err(invalid("tensor payload has a trailing gap"));
    }
    Ok(())
}

/// Parse and verify one official shard's bounded safetensors header prefix.
///
/// `bytes` must be exactly `8 + header_length` bytes. A full shard or a prefix
/// containing any payload byte is rejected.
pub fn parse_diffusion_gemma_safetensors_header(
    file_name: &str,
    bytes: &[u8],
) -> Result<DiffusionGemmaShardHeader, DiffusionGemmaHeaderError> {
    let (position, identity) = locked_header_identity(file_name)?;
    let length_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("header length field is truncated"))?
        .try_into()
        .expect("slice length checked");
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length != identity.header_length {
        return Err(invalid(format!(
            "header length differs for {file_name}: expected {}, got {header_length}",
            identity.header_length
        )));
    }
    let prefix_length = DIFFUSION_GEMMA_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
        .checked_add(header_length)
        .ok_or_else(|| invalid("header prefix length overflowed"))?;
    let prefix_length_usize = usize::try_from(prefix_length)
        .map_err(|_| invalid("header prefix does not fit host size"))?;
    if bytes.len() != prefix_length_usize {
        return Err(invalid(format!(
            "header prefix length differs: expected {prefix_length}, got {}",
            bytes.len()
        )));
    }
    let header_sha256 = format!("{:x}", Sha256::digest(bytes));
    if header_sha256 != identity.header_sha256 {
        return Err(invalid(format!("header SHA-256 differs for {file_name}")));
    }
    let document: UniqueRawHeaderDocument = serde_json::from_slice(&bytes[8..])
        .map_err(|error| invalid(format!("header JSON: {error}")))?;
    if document.tensors.len() != identity.indexed_tensor_count {
        return Err(invalid(format!(
            "header tensor count differs for {file_name}"
        )));
    }
    let data_start = prefix_length;
    if data_start > identity.file_size {
        return Err(invalid("header extends beyond shard file"));
    }
    let payload_bytes = identity.file_size - data_start;
    let shard_index = u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(document.tensors.len())
        .map_err(|_| invalid("header tensor allocation failed"))?;
    for (name, raw) in document.tensors {
        if !safe_tensor_name(&name) {
            return Err(invalid(format!("unsafe tensor name: {name}")));
        }
        let dtype = DiffusionGemmaSafetensorsDType::parse(&raw.dtype)?;
        let byte_size = checked_shape_bytes(&raw.shape, dtype)?;
        let [start, end] = raw.data_offsets;
        let range_size = end
            .checked_sub(start)
            .ok_or_else(|| invalid(format!("tensor range underflowed: {name}")))?;
        if range_size != byte_size {
            return Err(invalid(format!("tensor byte size differs: {name}")));
        }
        let absolute_start = data_start
            .checked_add(start)
            .ok_or_else(|| invalid(format!("absolute tensor range overflowed: {name}")))?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| invalid(format!("absolute tensor range overflowed: {name}")))?;
        tensors.push(DiffusionGemmaHeaderTensor {
            name,
            shard_index,
            dtype,
            shape: raw.shape,
            data_offsets: [start, end],
            absolute_byte_range: [absolute_start, absolute_end],
            byte_size,
        });
    }
    validate_ranges(&tensors, payload_bytes)?;
    Ok(DiffusionGemmaShardHeader {
        file_name: file_name.to_owned(),
        shard_index,
        file_size: identity.file_size,
        header_length,
        header_sha256,
        data_start,
        payload_bytes,
        tensors,
    })
}

fn canonical_catalog_sha256(shards: &[DiffusionGemmaShardHeader]) -> String {
    let mut header_rows: Vec<&DiffusionGemmaShardHeader> = shards.iter().collect();
    header_rows.sort_by_key(|shard| shard.file_name.as_str());
    let mut tensor_rows: Vec<&DiffusionGemmaHeaderTensor> = shards
        .iter()
        .flat_map(|shard| shard.tensors.iter())
        .collect();
    tensor_rows.sort_by_key(|tensor| tensor.name.as_str());
    let file_names_by_shard = shards
        .iter()
        .map(|shard| (shard.shard_index, shard.file_name.as_str()))
        .collect::<BTreeMap<_, _>>();

    let mut canonical = String::new();
    for shard in header_rows {
        use std::fmt::Write;
        let _ = writeln!(
            canonical,
            "header\t{}\t{}\t{}\t{}\t{}",
            shard.file_name,
            shard.file_size,
            shard.header_length,
            shard.header_sha256,
            shard.payload_bytes
        );
    }
    for tensor in tensor_rows {
        use std::fmt::Write;
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let [start, end] = tensor.data_offsets;
        let [absolute_start, absolute_end] = tensor.absolute_byte_range;
        let file_name = file_names_by_shard
            .get(&tensor.shard_index)
            .copied()
            .unwrap_or("<invalid-shard-index>");
        let _ = writeln!(
            canonical,
            "tensor\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tensor.name,
            file_name,
            tensor.dtype.as_str(),
            shape,
            start,
            end,
            absolute_start,
            absolute_end
        );
    }
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

/// Validate all 11 unique headers against the exact fixed-revision index.
pub fn validate_diffusion_gemma_header_catalog(
    headers: &[DiffusionGemmaShardHeader],
    index: &DiffusionGemmaIndex,
) -> Result<DiffusionGemmaHeaderCatalog, DiffusionGemmaHeaderError> {
    if headers.len() != DIFFUSION_GEMMA_SHARD_COUNT {
        return Err(invalid(format!(
            "header shard count differs: expected {}, got {}",
            DIFFUSION_GEMMA_SHARD_COUNT,
            headers.len()
        )));
    }
    if index.catalog_sha256() != DIFFUSION_GEMMA_CATALOG_SHA256
        || index.index_advertised_bytes() != DIFFUSION_GEMMA_TENSOR_PAYLOAD_BYTES
        || index.shard_file_bytes() != DIFFUSION_GEMMA_SHARD_FILE_BYTES
        || index.total_parameters() != DIFFUSION_GEMMA_TOTAL_PARAMETERS
        || index.tensor_count() != DIFFUSION_GEMMA_TENSOR_COUNT
    {
        return Err(invalid("typed safetensors index identity differs"));
    }

    let mut by_file = BTreeMap::new();
    for header in headers {
        let (position, identity) = locked_header_identity(&header.file_name)?;
        let expected_shard_index =
            u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
        let expected_data_start = DIFFUSION_GEMMA_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
            .checked_add(identity.header_length)
            .ok_or_else(|| invalid("header prefix length overflowed"))?;
        if header.shard_index != expected_shard_index
            || header.file_size != identity.file_size
            || header.header_length != identity.header_length
            || header.header_sha256 != identity.header_sha256
            || header.data_start != expected_data_start
            || header.data_start > header.file_size
            || header.payload_bytes != header.file_size - header.data_start
            || header.tensors.len() != identity.indexed_tensor_count
        {
            return Err(invalid(format!(
                "header identity differs: {}",
                header.file_name
            )));
        }
        if by_file.insert(header.file_name.as_str(), header).is_some() {
            return Err(invalid(format!(
                "duplicate shard header: {}",
                header.file_name
            )));
        }
    }
    if DIFFUSION_GEMMA_HEADER_IDENTITIES
        .iter()
        .any(|identity| !by_file.contains_key(identity.file_name))
    {
        return Err(invalid("missing shard header"));
    }

    let mut tensor_sources = BTreeMap::<&str, &DiffusionGemmaHeaderTensor>::new();
    let mut shard_file_bytes = 0_u64;
    let mut payload_bytes = 0_u64;
    let mut header_prefix_bytes = 0_u64;
    let mut tensor_count = 0_usize;
    for header in headers {
        validate_ranges(&header.tensors, header.payload_bytes)?;
        shard_file_bytes = shard_file_bytes
            .checked_add(header.file_size)
            .ok_or_else(|| invalid("shard file byte total overflowed"))?;
        payload_bytes = payload_bytes
            .checked_add(header.payload_bytes)
            .ok_or_else(|| invalid("payload byte total overflowed"))?;
        header_prefix_bytes = header_prefix_bytes
            .checked_add(header.data_start)
            .ok_or_else(|| invalid("header prefix byte total overflowed"))?;
        tensor_count = tensor_count
            .checked_add(header.tensors.len())
            .ok_or_else(|| invalid("header tensor count overflowed"))?;
        for tensor in &header.tensors {
            let expected_byte_size = checked_shape_bytes(&tensor.shape, tensor.dtype)?;
            let [start, end] = tensor.data_offsets;
            if tensor.byte_size != expected_byte_size
                || end.checked_sub(start) != Some(tensor.byte_size)
                || tensor.absolute_byte_range
                    != [
                        header.data_start.checked_add(start).ok_or_else(|| {
                            invalid(format!("absolute range overflowed: {}", tensor.name))
                        })?,
                        header.data_start.checked_add(end).ok_or_else(|| {
                            invalid(format!("absolute range overflowed: {}", tensor.name))
                        })?,
                    ]
                || tensor.absolute_byte_range[1] > header.file_size
                || tensor.shard_index != header.shard_index
            {
                return Err(invalid(format!(
                    "tensor range geometry differs: {}",
                    tensor.name
                )));
            }
            if tensor_sources
                .insert(tensor.name.as_str(), tensor)
                .is_some()
            {
                return Err(invalid(format!("duplicate tensor name: {}", tensor.name)));
            }
        }
    }
    if shard_file_bytes != DIFFUSION_GEMMA_SHARD_FILE_BYTES
        || payload_bytes != DIFFUSION_GEMMA_TENSOR_PAYLOAD_BYTES
        || header_prefix_bytes != DIFFUSION_GEMMA_HEADER_PREFIX_BYTES
        || tensor_count != DIFFUSION_GEMMA_TENSOR_COUNT
        || tensor_sources.len() != index.tensor_count()
    {
        return Err(invalid("catalog byte or tensor accounting differs"));
    }
    for (name, tensor) in &tensor_sources {
        let source_file = index
            .source_file(name)
            .ok_or_else(|| invalid(format!("header tensor is absent from index: {name}")))?;
        let header = headers
            .get(
                usize::try_from(tensor.shard_index)
                    .map_err(|_| invalid("tensor shard index does not fit host size"))?
                    .checked_sub(1)
                    .ok_or_else(|| invalid("tensor shard index is zero"))?,
            )
            .filter(|header| header.shard_index == tensor.shard_index)
            .or_else(|| {
                headers
                    .iter()
                    .find(|header| header.shard_index == tensor.shard_index)
            })
            .ok_or_else(|| invalid(format!("unknown tensor shard index: {name}")))?;
        if source_file != header.file_name {
            return Err(invalid(format!("header/index shard differs: {name}")));
        }
    }

    let catalog_sha256 = canonical_catalog_sha256(headers);
    if catalog_sha256 != DIFFUSION_GEMMA_HEADER_CATALOG_SHA256 {
        return Err(invalid(format!(
            "header catalog SHA-256 differs: got {catalog_sha256}"
        )));
    }
    let mut ordered = headers.to_vec();
    ordered.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    Ok(DiffusionGemmaHeaderCatalog {
        shards: ordered,
        tensor_count,
        shard_file_bytes,
        payload_bytes,
        catalog_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tensor(
        name: &str,
        dtype: DiffusionGemmaSafetensorsDType,
        shape: Vec<u64>,
        offsets: [u64; 2],
    ) -> DiffusionGemmaHeaderTensor {
        DiffusionGemmaHeaderTensor {
            name: name.to_owned(),
            shard_index: 1,
            dtype,
            shape,
            data_offsets: offsets,
            absolute_byte_range: [96 + offsets[0], 96 + offsets[1]],
            byte_size: offsets[1] - offsets[0],
        }
    }

    #[test]
    fn identity_table_fixes_header_file_payload_and_index_counts() {
        assert_eq!(DIFFUSION_GEMMA_HEADER_IDENTITIES.len(), 11);
        assert_eq!(
            DIFFUSION_GEMMA_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.indexed_tensor_count)
                .sum::<usize>(),
            DIFFUSION_GEMMA_TENSOR_COUNT
        );
        assert_eq!(
            DIFFUSION_GEMMA_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.file_size)
                .sum::<u64>(),
            DIFFUSION_GEMMA_SHARD_FILE_BYTES
        );
        assert_eq!(
            DIFFUSION_GEMMA_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.header_length + 8)
                .sum::<u64>(),
            DIFFUSION_GEMMA_HEADER_PREFIX_BYTES
        );
        assert_eq!(
            DIFFUSION_GEMMA_SHARD_FILE_BYTES - DIFFUSION_GEMMA_HEADER_PREFIX_BYTES,
            DIFFUSION_GEMMA_TENSOR_PAYLOAD_BYTES
        );
        assert!(
            DIFFUSION_GEMMA_HEADER_IDENTITIES
                .windows(2)
                .all(|pair| pair[0].file_name < pair[1].file_name)
        );
        assert_eq!(
            diffusion_gemma_locked_header("model-00011-of-00011.safetensors"),
            Some(DIFFUSION_GEMMA_HEADER_IDENTITIES[10])
        );
    }

    #[test]
    fn dtype_and_shape_boundaries_are_fail_closed() {
        assert_eq!(
            DiffusionGemmaSafetensorsDType::parse("BF16").unwrap(),
            DiffusionGemmaSafetensorsDType::Bf16
        );
        assert_eq!(DiffusionGemmaSafetensorsDType::Bf16.byte_width(), 2);
        assert!(DiffusionGemmaSafetensorsDType::parse("F32").is_err());
        assert!(DiffusionGemmaSafetensorsDType::parse("U8").is_err());
        assert!(checked_shape_bytes(&[], DiffusionGemmaSafetensorsDType::Bf16).is_err());
        assert!(checked_shape_bytes(&[0], DiffusionGemmaSafetensorsDType::Bf16).is_err());
        assert!(checked_shape_bytes(&[u64::MAX, 2], DiffusionGemmaSafetensorsDType::Bf16).is_err());
        assert_eq!(
            checked_shape_bytes(&[3, 5], DiffusionGemmaSafetensorsDType::Bf16).unwrap(),
            30
        );
    }

    #[test]
    fn duplicate_header_tensor_and_metadata_keys_are_rejected() {
        let duplicate_header = br#"{
            "__metadata__":{"format":"pt"},
            "a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},
            "a":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}
        }"#;
        assert!(serde_json::from_slice::<UniqueRawHeaderDocument>(duplicate_header).is_err());

        let duplicate_metadata = br#"{
            "__metadata__":{"format":"pt"},
            "__metadata__":{"format":"pt"},
            "a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}
        }"#;
        assert!(serde_json::from_slice::<UniqueRawHeaderDocument>(duplicate_metadata).is_err());
    }

    #[test]
    fn range_gaps_overlap_outside_and_overflow_are_rejected() {
        let contiguous = vec![
            synthetic_tensor("a", DiffusionGemmaSafetensorsDType::Bf16, vec![1], [0, 2]),
            synthetic_tensor("b", DiffusionGemmaSafetensorsDType::Bf16, vec![1], [2, 4]),
        ];
        assert!(validate_ranges(&contiguous, 4).is_ok());
        let mut leading_gap = contiguous.clone();
        leading_gap[0].data_offsets = [1, 3];
        assert!(validate_ranges(&leading_gap, 5).is_err());
        let mut middle_gap = contiguous.clone();
        middle_gap[1].data_offsets = [3, 5];
        assert!(validate_ranges(&middle_gap, 5).is_err());
        let mut overlap = contiguous.clone();
        overlap[1].data_offsets = [1, 3];
        assert!(validate_ranges(&overlap, 4).is_err());
        let mut outside = contiguous;
        outside[1].data_offsets = [2, 5];
        assert!(validate_ranges(&outside, 4).is_err());
        assert!(checked_shape_bytes(&[u64::MAX], DiffusionGemmaSafetensorsDType::Bf16).is_err());
    }

    #[test]
    fn unsafe_and_unknown_paths_are_rejected_before_bytes() {
        for path in [
            "../model-00001-of-00011.safetensors",
            "/model-00001-of-00011.safetensors",
            "model-00001-of-00011.safetensors/extra",
            "model-00001-of-00012.safetensors",
            "model-00000-of-00011.safetensors",
        ] {
            assert!(parse_diffusion_gemma_safetensors_header(path, &[]).is_err());
        }
        let wrong_length = [0_u8; 8];
        assert!(
            parse_diffusion_gemma_safetensors_header(
                "model-00001-of-00011.safetensors",
                &wrong_length
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_catalog_digest_is_input_order_independent() {
        let identity = DIFFUSION_GEMMA_HEADER_IDENTITIES[0];
        let first = DiffusionGemmaShardHeader {
            file_name: identity.file_name.to_owned(),
            shard_index: 1,
            file_size: identity.file_size,
            header_length: identity.header_length,
            header_sha256: identity.header_sha256.to_owned(),
            data_start: 96,
            payload_bytes: 4,
            tensors: vec![synthetic_tensor(
                "b",
                DiffusionGemmaSafetensorsDType::Bf16,
                vec![1],
                [2, 4],
            )],
        };
        let mut second = first.clone();
        second.tensors = vec![synthetic_tensor(
            "a",
            DiffusionGemmaSafetensorsDType::Bf16,
            vec![1],
            [0, 2],
        )];
        assert_eq!(
            canonical_catalog_sha256(&[first.clone(), second.clone()]),
            canonical_catalog_sha256(&[second, first])
        );
    }

    #[test]
    #[ignore = "requires fixed-revision 11 bounded header prefixes and official index"]
    fn exact_official_headers_match_index_geometry_and_catalog_digest() {
        use std::path::PathBuf;

        let header_root = std::env::var_os("SLLM_DIFFUSION_GEMMA_HEADER_DIR")
            .map(PathBuf::from)
            .expect("set SLLM_DIFFUSION_GEMMA_HEADER_DIR to the bounded header-prefix directory");
        let index_path = std::env::var_os("SLLM_DIFFUSION_GEMMA_INDEX_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/tmp/sllm-phase59.oVDb4B/model.safetensors.index.json")
            });
        let mut headers = Vec::new();
        headers
            .try_reserve_exact(DIFFUSION_GEMMA_SHARD_COUNT)
            .expect("header vector allocation");
        for identity in DIFFUSION_GEMMA_HEADER_IDENTITIES {
            let path = header_root.join(format!("{}.prefix", identity.file_name));
            let bytes = std::fs::read(path).expect("read bounded header prefix");
            assert_eq!(
                u64::try_from(bytes.len()).unwrap(),
                identity.header_length + DIFFUSION_GEMMA_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
            );
            let mut payload_probe = bytes.clone();
            payload_probe.push(0);
            assert!(
                parse_diffusion_gemma_safetensors_header(identity.file_name, &payload_probe)
                    .is_err()
            );
            headers.push(
                parse_diffusion_gemma_safetensors_header(identity.file_name, &bytes)
                    .expect("parse fixed header prefix"),
            );
        }
        let index_bytes = std::fs::read(index_path).expect("read fixed safetensors index");
        let index = crate::diffusion_gemma::validate_diffusion_gemma_index(&index_bytes)
            .expect("validate index");
        let catalog = validate_diffusion_gemma_header_catalog(&headers, &index)
            .expect("validate header catalog");
        assert_eq!(catalog.shards().len(), DIFFUSION_GEMMA_SHARD_COUNT);
        assert_eq!(catalog.tensor_count(), DIFFUSION_GEMMA_TENSOR_COUNT);
        assert_eq!(catalog.shard_file_bytes(), DIFFUSION_GEMMA_SHARD_FILE_BYTES);
        assert_eq!(
            catalog.payload_bytes(),
            DIFFUSION_GEMMA_TENSOR_PAYLOAD_BYTES
        );
        assert_eq!(
            catalog.catalog_sha256(),
            DIFFUSION_GEMMA_HEADER_CATALOG_SHA256
        );
        assert!(
            catalog
                .tensors()
                .all(|tensor| tensor.dtype == DiffusionGemmaSafetensorsDType::Bf16)
        );
        assert!(
            catalog
                .tensor("model.decoder.embed_tokens.weight")
                .is_some()
        );
    }
}
