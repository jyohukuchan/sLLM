//! Header-only safetensors catalog for the reviewed Ministral 3 3B artifact.
//!
//! Callers must provide exactly the eight-byte length field and the declared
//! JSON header.  No API in this module reads or hashes tensor payload bytes.
//! Full-file LFS identities are retained as remote evidence and are never
//! confused with the bounded header identities.

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const MINISTRAL3_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES: u64 = 8;
pub const MINISTRAL3_HEADER_PREFIX_BYTES: u64 = MINISTRAL3_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES;
pub const MINISTRAL3_HEADER_CATALOG_SHA256: &str =
    "e29562934027f6e6290080f6465349822afb3ce7f188aae97e27d947272cd31f";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ministral3HeaderIdentity {
    pub file_name: &'static str,
    pub file_size: u64,
    pub header_length: u64,
    pub header_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const MINISTRAL3_HEADER_IDENTITIES: [Ministral3HeaderIdentity; 2] = [
    Ministral3HeaderIdentity {
        file_name: "model-00001-of-00002.safetensors",
        file_size: 4_967_581_832,
        header_length: 47_232,
        header_sha256: "0a9c9a62103a14b6d5a9f04958e1df0137f9b296cd66eaaf79959b1d549839c9",
        indexed_tensor_count: 353,
    },
    Ministral3HeaderIdentity {
        file_name: "model-00002-of-00002.safetensors",
        file_size: 2_730_659_224,
        header_length: 13_712,
        header_sha256: "a1e93e8c71240094ee375c1641fd7dedb8ba250dd88ff74c8373dcfe0571756f",
        indexed_tensor_count: 105,
    },
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Ministral3SafetensorsDType {
    Bf16,
}

impl Ministral3SafetensorsDType {
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

    fn parse(value: &str) -> Result<Self, Ministral3HeaderError> {
        match value {
            "BF16" => Ok(Self::Bf16),
            other => Err(invalid(format!("unsupported safetensors dtype: {other}"))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3HeaderTensor {
    pub name: String,
    /// One-based shard number.
    pub shard_index: u32,
    pub dtype: Ministral3SafetensorsDType,
    pub shape: Vec<u64>,
    /// Start-inclusive, end-exclusive payload-relative offsets.
    pub data_offsets: [u64; 2],
    /// Start-inclusive, end-exclusive complete-file offsets.
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3ShardHeader {
    pub file_name: String,
    pub shard_index: u32,
    pub file_size: u64,
    pub header_length: u64,
    /// SHA-256 over the eight-byte length field and JSON header only.
    pub header_sha256: String,
    pub data_start: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<Ministral3HeaderTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3Index {
    total_parameters: u64,
    total_size: u64,
    weight_map: BTreeMap<String, String>,
}

impl Ministral3Index {
    pub const fn total_parameters(&self) -> u64 {
        self.total_parameters
    }

    pub const fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &str)> {
        self.weight_map
            .iter()
            .map(|(name, shard)| (name.as_str(), shard.as_str()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3HeaderCatalog {
    shards: Vec<Ministral3ShardHeader>,
    tensor_count: usize,
    physical_parameters: u64,
    payload_bytes: u64,
    file_bytes: u64,
    catalog_sha256: String,
}

impl Ministral3HeaderCatalog {
    pub fn shards(&self) -> &[Ministral3ShardHeader] {
        &self.shards
    }

    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Physical BF16 elements represented by header tensor rows.  This is
    /// lower than the index parameter count because tied embeddings are
    /// counted twice by the official index metadata.
    pub const fn physical_parameters(&self) -> u64 {
        self.physical_parameters
    }

    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub const fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn tensors(&self) -> impl Iterator<Item = &Ministral3HeaderTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn tensor(&self, name: &str) -> Option<&Ministral3HeaderTensor> {
        self.tensors().find(|tensor| tensor.name == name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3HeaderError {
    Invalid(String),
}

impl fmt::Display for Ministral3HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(
                    formatter,
                    "invalid Ministral 3 safetensors header: {message}"
                )
            }
        }
    }
}

impl std::error::Error for Ministral3HeaderError {}

fn invalid(message: impl Into<String>) -> Ministral3HeaderError {
    Ministral3HeaderError::Invalid(message.into())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMetadata {
    format: String,
}

struct RawHeaderDocument {
    metadata: RawMetadata,
    tensors: BTreeMap<String, RawTensor>,
}

impl<'de> Deserialize<'de> for RawHeaderDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = RawHeaderDocument;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a strict safetensors header with unique tensor names")
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
                            return Err(serde::de::Error::custom(
                                "duplicate safetensors metadata key",
                            ));
                        }
                        metadata = Some(entries.next_value::<RawMetadata>()?);
                    } else {
                        let tensor = entries.next_value::<RawTensor>()?;
                        if tensors.insert(name.clone(), tensor).is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "duplicate tensor name: {name}"
                            )));
                        }
                    }
                }
                let metadata = metadata
                    .ok_or_else(|| serde::de::Error::custom("safetensors metadata is missing"))?;
                Ok(RawHeaderDocument { metadata, tensors })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

#[derive(Debug)]
struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WeightMapVisitor;

        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a unique tensor-to-shard map")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((name, shard)) = entries.next_entry::<String, String>()? {
                    if values.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate index tensor name: {name}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(values))
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndexMetadata {
    total_parameters: u64,
    total_size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndex {
    metadata: RawIndexMetadata,
    weight_map: UniqueWeightMap,
}

fn validate_tensor_name(name: &str) -> Result<(), Ministral3HeaderError> {
    if name.is_empty()
        || name.starts_with('.')
        || name.ends_with('.')
        || name.contains("..")
        || name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.'))
    {
        return Err(invalid(format!("unsafe tensor name: {name}")));
    }
    Ok(())
}

fn locked_header_identity(
    file_name: &str,
) -> Result<(usize, Ministral3HeaderIdentity), Ministral3HeaderError> {
    if file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
    {
        return Err(invalid("unsafe shard path"));
    }
    MINISTRAL3_HEADER_IDENTITIES
        .iter()
        .copied()
        .enumerate()
        .find(|(_, identity)| identity.file_name == file_name)
        .ok_or_else(|| invalid(format!("unknown shard: {file_name}")))
}

fn checked_shape_bytes(
    shape: &[u64],
    dtype: Ministral3SafetensorsDType,
) -> Result<u64, Ministral3HeaderError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid("empty or zero tensor shape is not reviewed"));
    }
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid("tensor shape element count overflowed"))
    })?;
    elements
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| invalid("tensor shape byte count overflowed"))
}

fn validate_ranges(
    tensors: &mut [Ministral3HeaderTensor],
    payload_bytes: u64,
) -> Result<(), Ministral3HeaderError> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
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
        if start >= end || end > payload_bytes || end - start != tensor.byte_size {
            return Err(invalid(format!(
                "tensor range outside payload: {}",
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

fn parse_header_prefix(
    file_name: &str,
    bytes: &[u8],
) -> Result<Ministral3ShardHeader, Ministral3HeaderError> {
    let (position, identity) = locked_header_identity(file_name)?;
    let length_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("header length field is truncated"))?
        .try_into()
        .expect("slice length checked");
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length != identity.header_length {
        return Err(invalid(format!("header length differs for {file_name}")));
    }
    let prefix_length = MINISTRAL3_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
        .checked_add(header_length)
        .ok_or_else(|| invalid("header prefix length overflowed"))?;
    let prefix_length_usize = usize::try_from(prefix_length)
        .map_err(|_| invalid("header prefix does not fit host size"))?;
    if bytes.len() != prefix_length_usize {
        return Err(invalid(format!(
            "header prefix length differs for {file_name}"
        )));
    }
    let header_sha256 = format!("{:x}", Sha256::digest(bytes));
    if header_sha256 != identity.header_sha256 {
        return Err(invalid(format!("header SHA-256 differs for {file_name}")));
    }
    let document: RawHeaderDocument = serde_json::from_slice(&bytes[8..])
        .map_err(|error| invalid(format!("header JSON: {error}")))?;
    if document.metadata.format != "pt" {
        return Err(invalid("safetensors metadata format differs"));
    }
    if document.tensors.is_empty() || document.tensors.len() != identity.indexed_tensor_count {
        return Err(invalid(format!("tensor count differs for {file_name}")));
    }
    let data_start = prefix_length;
    if data_start > identity.file_size {
        return Err(invalid("header extends beyond shard file"));
    }
    let payload_bytes = identity.file_size - data_start;
    let shard_index = u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
    let mut tensors = Vec::with_capacity(document.tensors.len());
    for (name, raw) in document.tensors {
        validate_tensor_name(&name)?;
        let dtype = Ministral3SafetensorsDType::parse(&raw.dtype)?;
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
            .ok_or_else(|| invalid(format!("absolute range overflowed: {name}")))?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| invalid(format!("absolute range overflowed: {name}")))?;
        tensors.push(Ministral3HeaderTensor {
            name,
            shard_index,
            dtype,
            shape: raw.shape,
            data_offsets: [start, end],
            absolute_byte_range: [absolute_start, absolute_end],
            byte_size,
        });
    }
    validate_ranges(&mut tensors, payload_bytes)?;
    Ok(Ministral3ShardHeader {
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

/// Parse exactly one official shard header prefix.  Passing a full shard or a
/// truncated prefix is rejected before any catalog is constructed.
pub fn parse_ministral3_safetensors_header(
    file_name: &str,
    bytes: &[u8],
) -> Result<Ministral3ShardHeader, Ministral3HeaderError> {
    parse_header_prefix(file_name, bytes)
}

pub fn ministral3_locked_header(file_name: &str) -> Option<Ministral3HeaderIdentity> {
    locked_header_identity(file_name)
        .ok()
        .map(|(_, identity)| identity)
}

/// Validate the exact official safetensors index.  Its parameter total is
/// intentionally retained separately from physical header-derived numel.
pub fn validate_ministral3_index(bytes: &[u8]) -> Result<Ministral3Index, Ministral3HeaderError> {
    if bytes.len() != crate::ministral3::MINISTRAL3_INDEX_BYTES
        || format!("{:x}", Sha256::digest(bytes)) != crate::ministral3::MINISTRAL3_INDEX_SHA256
    {
        return Err(invalid("safetensors index identity differs"));
    }
    let raw: RawIndex = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("safetensors index JSON: {error}")))?;
    if raw.metadata.total_parameters != crate::ministral3::MINISTRAL3_INDEX_TOTAL_PARAMETERS
        || raw.metadata.total_size != crate::ministral3::MINISTRAL3_INDEX_TOTAL_SIZE
        || raw.weight_map.0.len() != crate::ministral3::MINISTRAL3_TENSOR_COUNT
    {
        return Err(invalid("index metadata or tensor count differs"));
    }
    let mut counts = [0_usize; 2];
    for (name, shard) in &raw.weight_map.0 {
        validate_tensor_name(name)?;
        let index = locked_header_identity(shard)?.0;
        counts[index] = counts[index]
            .checked_add(1)
            .ok_or_else(|| invalid("index shard tensor count overflowed"))?;
    }
    if counts != MINISTRAL3_HEADER_IDENTITIES.map(|identity| identity.indexed_tensor_count) {
        return Err(invalid("index shard coverage differs"));
    }
    Ok(Ministral3Index {
        total_parameters: raw.metadata.total_parameters,
        total_size: raw.metadata.total_size,
        weight_map: raw.weight_map.0,
    })
}

fn canonical_catalog_sha256(shards: &[Ministral3ShardHeader]) -> String {
    let mut header_rows: Vec<&Ministral3ShardHeader> = shards.iter().collect();
    header_rows.sort_by_key(|shard| shard.file_name.as_str());
    let mut tensor_rows: Vec<&Ministral3HeaderTensor> = shards
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
            "header\t{}\t{}\t{}",
            shard.file_name, shard.header_length, shard.header_sha256
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

/// Build the exact two-shard metadata-only catalog.
pub fn build_ministral3_header_catalog(
    headers: &[Ministral3ShardHeader],
) -> Result<Ministral3HeaderCatalog, Ministral3HeaderError> {
    if headers.len() != crate::ministral3::MINISTRAL3_SHARD_COUNT {
        return Err(invalid("header shard count differs"));
    }
    let mut by_file = BTreeMap::new();
    let mut by_index = BTreeMap::new();
    for header in headers {
        let (position, identity) = locked_header_identity(&header.file_name)?;
        let expected_index =
            u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
        if header.shard_index != expected_index
            || header.file_size != identity.file_size
            || header.header_length != identity.header_length
            || header.header_sha256 != identity.header_sha256
            || header.data_start != 8 + identity.header_length
            || header.payload_bytes != identity.file_size - header.data_start
            || header.tensors.len() != identity.indexed_tensor_count
        {
            return Err(invalid(format!(
                "header identity differs: {}",
                header.file_name
            )));
        }
        if by_file.insert(header.file_name.as_str(), header).is_some()
            || by_index.insert(header.shard_index, header).is_some()
        {
            return Err(invalid("duplicate shard header"));
        }
        let mut ranges = header.tensors.clone();
        validate_ranges(&mut ranges, header.payload_bytes)?;
        for tensor in &header.tensors {
            if tensor.shard_index != header.shard_index {
                return Err(invalid(format!("tensor shard mismatch: {}", tensor.name)));
            }
            let [start, end] = tensor.data_offsets;
            let absolute = [
                header
                    .data_start
                    .checked_add(start)
                    .ok_or_else(|| invalid("absolute range overflowed"))?,
                header
                    .data_start
                    .checked_add(end)
                    .ok_or_else(|| invalid("absolute range overflowed"))?,
            ];
            if tensor.dtype != Ministral3SafetensorsDType::Bf16
                || tensor.byte_size != checked_shape_bytes(&tensor.shape, tensor.dtype)?
                || end.checked_sub(start) != Some(tensor.byte_size)
                || tensor.absolute_byte_range != absolute
            {
                return Err(invalid(format!("tensor geometry differs: {}", tensor.name)));
            }
        }
    }
    if by_file.len() != crate::ministral3::MINISTRAL3_SHARD_COUNT
        || MINISTRAL3_HEADER_IDENTITIES
            .iter()
            .any(|identity| !by_file.contains_key(identity.file_name))
    {
        return Err(invalid("missing or extra shard header"));
    }
    let file_bytes = headers.iter().try_fold(0_u64, |sum, header| {
        sum.checked_add(header.file_size)
            .ok_or_else(|| invalid("shard file total overflowed"))
    })?;
    let header_bytes = headers.iter().try_fold(0_u64, |sum, header| {
        sum.checked_add(header.data_start)
            .ok_or_else(|| invalid("header total overflowed"))
    })?;
    let payload_bytes = headers.iter().try_fold(0_u64, |sum, header| {
        sum.checked_add(header.payload_bytes)
            .ok_or_else(|| invalid("header payload total overflowed"))
    })?;
    let tensor_count = headers.iter().try_fold(0_usize, |sum, header| {
        sum.checked_add(header.tensors.len())
            .ok_or_else(|| invalid("header tensor count overflowed"))
    })?;
    let physical_parameters = headers.iter().try_fold(0_u64, |sum, header| {
        let bytes = header.tensors.iter().try_fold(0_u64, |inner, tensor| {
            inner
                .checked_add(tensor.byte_size)
                .ok_or_else(|| invalid("physical parameter byte total overflowed"))
        })?;
        sum.checked_add(bytes / 2)
            .ok_or_else(|| invalid("physical parameter total overflowed"))
    })?;
    if file_bytes != crate::ministral3::MINISTRAL3_SHARD_FILE_BYTES
        || header_bytes != crate::ministral3::MINISTRAL3_HEADER_BYTES
        || payload_bytes != crate::ministral3::MINISTRAL3_INDEX_TOTAL_SIZE
        || tensor_count != crate::ministral3::MINISTRAL3_TENSOR_COUNT
        || physical_parameters != crate::ministral3::MINISTRAL3_PHYSICAL_PARAMETERS
    {
        return Err(invalid("header aggregate geometry differs"));
    }
    let mut names = BTreeMap::new();
    for header in headers {
        for tensor in &header.tensors {
            if names.insert(tensor.name.as_str(), tensor).is_some() {
                return Err(invalid(format!("duplicate tensor name: {}", tensor.name)));
            }
        }
    }
    let catalog_sha256 = canonical_catalog_sha256(headers);
    if catalog_sha256 != MINISTRAL3_HEADER_CATALOG_SHA256 {
        return Err(invalid("header catalog SHA-256 differs"));
    }
    let mut ordered = headers.to_vec();
    ordered.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    Ok(Ministral3HeaderCatalog {
        shards: ordered,
        tensor_count,
        physical_parameters,
        payload_bytes,
        file_bytes,
        catalog_sha256,
    })
}

/// Validate the header/index tensor-name and shard assignment contract.
pub fn validate_ministral3_header_catalog(
    headers: &[Ministral3ShardHeader],
    index: &Ministral3Index,
) -> Result<Ministral3HeaderCatalog, Ministral3HeaderError> {
    let catalog = build_ministral3_header_catalog(headers)?;
    if index.tensor_count() != catalog.tensor_count
        || index.total_parameters() != crate::ministral3::MINISTRAL3_INDEX_TOTAL_PARAMETERS
        || index.total_size() != catalog.payload_bytes
    {
        return Err(invalid("header/index aggregate metadata differs"));
    }
    let mut header_sources = BTreeMap::new();
    for header in headers {
        for tensor in &header.tensors {
            header_sources.insert(tensor.name.as_str(), header.file_name.as_str());
        }
    }
    for (name, shard) in index.tensors() {
        let source = header_sources
            .get(name)
            .ok_or_else(|| invalid(format!("header tensor missing from index: {name}")))?;
        if *source != shard {
            return Err(invalid(format!("header/index shard differs: {name}")));
        }
    }
    if header_sources.len() != index.tensor_count() {
        return Err(invalid("header has an extra tensor absent from index"));
    }
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_identity_geometry_and_parameter_distinction_are_fixed() {
        assert_eq!(MINISTRAL3_HEADER_IDENTITIES.len(), 2);
        assert_eq!(
            MINISTRAL3_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.file_size)
                .sum::<u64>(),
            crate::ministral3::MINISTRAL3_SHARD_FILE_BYTES
        );
        assert_eq!(
            MINISTRAL3_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.header_length + 8)
                .sum::<u64>(),
            crate::ministral3::MINISTRAL3_HEADER_BYTES
        );
        assert_eq!(
            MINISTRAL3_HEADER_IDENTITIES
                .iter()
                .map(|identity| identity.indexed_tensor_count)
                .sum::<usize>(),
            crate::ministral3::MINISTRAL3_TENSOR_COUNT
        );
        assert_eq!(
            crate::ministral3::MINISTRAL3_INDEX_TOTAL_PARAMETERS
                - crate::ministral3::MINISTRAL3_PHYSICAL_PARAMETERS,
            402_653_184
        );
    }

    #[test]
    fn duplicate_unknown_path_and_overflow_inputs_fail_closed() {
        let duplicate = br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]},"__metadata__":{"format":"pt"}}"#;
        assert!(serde_json::from_slice::<RawHeaderDocument>(duplicate).is_err());
        let duplicate_metadata = br#"{"__metadata__":{"format":"pt"},"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        assert!(serde_json::from_slice::<RawHeaderDocument>(duplicate_metadata).is_err());
        let unknown = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"unknown":{"x":1}}"#;
        assert!(serde_json::from_slice::<RawHeaderDocument>(unknown).is_err());
        assert!(locked_header_identity("../model-00001-of-00002.safetensors").is_err());
        assert!(locked_header_identity("model-00003-of-00002.safetensors").is_err());
        assert!(checked_shape_bytes(&[u64::MAX, 2], Ministral3SafetensorsDType::Bf16).is_err());
        assert!(checked_shape_bytes(&[0], Ministral3SafetensorsDType::Bf16).is_err());
    }

    #[test]
    fn malformed_prefixes_and_range_boundaries_are_rejected() {
        let file_name = MINISTRAL3_HEADER_IDENTITIES[0].file_name;
        assert!(parse_ministral3_safetensors_header(file_name, &[]).is_err());
        assert!(parse_ministral3_safetensors_header(file_name, &[0; 8]).is_err());
        let mut full = vec![0_u8; 8];
        full.extend_from_slice(&[0; 32]);
        assert!(parse_ministral3_safetensors_header(file_name, &full).is_err());
        assert!(
            parse_ministral3_safetensors_header("model-00001-of-00002.safetensors/../x", &[])
                .is_err()
        );
    }

    #[test]
    fn index_duplicate_map_is_rejected_before_identity_gate() {
        let duplicate = br#"{"metadata":{"total_parameters":1,"total_size":1},"weight_map":{"a":"model-00001-of-00002.safetensors","a":"model-00002-of-00002.safetensors"}}"#;
        assert!(serde_json::from_slice::<RawIndex>(duplicate).is_err());
        let unknown = br#"{"metadata":{"total_parameters":1,"total_size":1},"weight_map":{"a":"model-00003-of-00002.safetensors"}}"#;
        let parsed = serde_json::from_slice::<RawIndex>(unknown).unwrap();
        assert_eq!(parsed.weight_map.0.len(), 1);
        assert!(validate_ministral3_index(duplicate).is_err());
    }

    #[test]
    #[ignore = "requires SLLM_MINISTRAL3_HEADER_DIR containing exact official prefixes and index"]
    fn exact_official_header_fixture_passes() {
        use std::fs::{self, File};
        use std::io::{Read, Seek, SeekFrom};
        use std::path::{Path, PathBuf};

        fn read_prefix(path: &Path) -> Vec<u8> {
            let mut file = File::open(path).expect("open exact shard fixture");
            let mut length = [0_u8; 8];
            file.read_exact(&mut length).expect("read header length");
            let header_length = u64::from_le_bytes(length);
            let total = usize::try_from(header_length + 8).expect("bounded header");
            file.seek(SeekFrom::Start(0)).expect("rewind fixture");
            let mut bytes = vec![0_u8; total];
            file.read_exact(&mut bytes).expect("read bounded header");
            bytes
        }

        let dir = PathBuf::from(std::env::var("SLLM_MINISTRAL3_HEADER_DIR").expect("fixture env"));
        let index = fs::read(dir.join("model.safetensors.index.json")).expect("read exact index");
        let index = validate_ministral3_index(&index).expect("official index");
        let mut headers = Vec::new();
        for identity in MINISTRAL3_HEADER_IDENTITIES {
            let exact = dir.join(identity.file_name);
            let bounded = dir.join(format!("{}.header.bin", identity.file_name));
            let prefix = dir.join(format!("{}.prefix", identity.file_name));
            let path = if exact.exists() {
                exact
            } else if bounded.exists() {
                bounded
            } else {
                prefix
            };
            headers.push(
                parse_ministral3_safetensors_header(identity.file_name, &read_prefix(&path))
                    .expect("official shard header"),
            );
        }
        let catalog = validate_ministral3_header_catalog(&headers, &index).expect("catalog");
        assert_eq!(
            catalog.tensor_count(),
            crate::ministral3::MINISTRAL3_TENSOR_COUNT
        );
        assert_eq!(catalog.catalog_sha256(), MINISTRAL3_HEADER_CATALOG_SHA256);
        assert_eq!(
            catalog.physical_parameters(),
            crate::ministral3::MINISTRAL3_PHYSICAL_PARAMETERS
        );
    }
}
