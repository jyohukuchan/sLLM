//! Header-only safetensors contract for the reviewed MiniMax M3 artifact.
//!
//! This module reads only the eight-byte safetensors length field and the
//! bounded JSON header.  It never reads tensor payload bytes.  The index's
//! advertised total is intentionally kept separate from the payload bytes
//! derived from the 59 shard geometries: the official manifest is inconsistent
//! and admission must fail closed instead of silently normalizing it.

use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const MINIMAX_M3_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES: u64 = 8;
pub const MINIMAX_M3_SHARD_COUNT: usize = 59;
pub const MINIMAX_M3_TENSOR_COUNT: usize = 23_416;
pub const MINIMAX_M3_INDEX_BYTES: usize = 2_706_437;
pub const MINIMAX_M3_INDEX_SHA256: &str =
    "54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e";
pub const MINIMAX_M3_INDEX_TOTAL_SIZE: u64 = 869_157_697_024;
pub const MINIMAX_M3_SHARD_FILE_BYTES: u64 = 854_176_398_808;
pub const MINIMAX_M3_SHARD_PAYLOAD_BYTES: u64 = 854_172_958_720;
pub const MINIMAX_M3_HEADER_BYTES: u64 = 3_440_088;
/// Canonical SHA-256 of header identities and all 23,416 header tensor rows.
/// It covers metadata only and is not a payload hash.
pub const MINIMAX_M3_HEADER_CATALOG_SHA256: &str =
    "341285506267abca7bf50507d4bd39adf3eb430d1454d3f4dbfe74eb84b35982";

/// `(file size, JSON header length, SHA-256 of length field plus JSON header)`.
pub const MINIMAX_M3_SHARD_IDENTITIES: [(u64, u64, &str); MINIMAX_M3_SHARD_COUNT] = [
    (
        5583706344,
        1760,
        "d423c213f105c4e4f409f28b96756c583e730d62a9dd8bdecf5d540bf5446dc2",
    ),
    (
        10922058704,
        40392,
        "c99b935cb5ac369a26b6c170b3f75d6d8dd8dfeb323c96c32531273f62cb788e",
    ),
    (
        16079479008,
        63704,
        "5bf489b9efb44a93730b667af45935904cb9729965225601a09038fbf63da1ee",
    ),
    (
        16079479000,
        63696,
        "a35ae899433fdb0efe7f5df4d0dc7f6a8481c0adfe1821956603a38e0be338e3",
    ),
    (
        16079479000,
        63696,
        "7c7acd90bd04f24065be82874b4ef3eb53cd3efef8866c010a406e5107e253e9",
    ),
    (
        16079479000,
        63696,
        "d666aa0f672a1d054bed1aff99ad0f29a3d1b4757b86e4d8bbcc47f913c59b08",
    ),
    (
        16079479000,
        63696,
        "0c7b513bd7e8f3e24ebda310b87e71a3e8749d9b2c4dce22ae4ae72ae10eb11f",
    ),
    (
        16079479000,
        63696,
        "8966923160f4164efabb6eb95b4b0ab8f1f9e642e79f0e3962f032dac0cda21e",
    ),
    (
        16079479000,
        63696,
        "8905d081726ef39966bacc73e9aec6c1d7355c46a342f7249c110df2163bab57",
    ),
    (
        16079479408,
        64104,
        "d9784f6a5d291cc412f7f153ca70f6f44f11ffe30f5576c0a97382f0b21278de",
    ),
    (
        16079479408,
        64104,
        "1bba18e58af77a0faa0226d3a36f7f33c86c291fdc4d5b8826f6b0ba488be28d",
    ),
    (
        16079479408,
        64104,
        "f0c74098c5c934d965ab40be03578f768163026e9a3f49195968a61c027dee71",
    ),
    (
        16079479408,
        64104,
        "c46341b5a3d78a3322edbaff167abbb9209c120e403311aa29e3816c86e03b45",
    ),
    (
        16079479408,
        64104,
        "2377a1ad3070ca43e8a1b96e03cbc82dc1f570e2ae5ef9eb8e2c2493f262e448",
    ),
    (
        16079479408,
        64104,
        "736f94d6b0faa065aa0dce27c4b96ce4176f778e6b4d1b8027f08db1efc59a92",
    ),
    (
        16079479408,
        64104,
        "c56c6925f0300ef89afdff5a69f7082e430fc73c1fc68d806c1a64494bc6cbd0",
    ),
    (
        16079479408,
        64104,
        "91cdb1e0248c50f9a2dc8d268f32a8bb695dd78fcd1039ed9e901301aec1131f",
    ),
    (
        16079479408,
        64104,
        "96fa7b2de2a591558f6109ff76c2d4325adc02b15ee7ae78744fc49f9c2d8267",
    ),
    (
        16079479408,
        64104,
        "6f3dbd5ac324323f7ec762433b751f14f513a79535e15096e5a7e3331f904781",
    ),
    (
        16079479408,
        64104,
        "d65216215810a724edde2608f25126078802e7c3598691a791d05fe4265b966a",
    ),
    (
        16079479408,
        64104,
        "8400313e5c17004b9dc49b90032920c70ff175b50f28147b523ef0d7051705f8",
    ),
    (
        16079479408,
        64104,
        "d233765ebbab1205abadeed70dc8f98b4d1e36e0ec2846bc22c39905931ee2bf",
    ),
    (
        16079479408,
        64104,
        "b6786c6c72b6452bc3c513b7615a1b8eb7b26a18d2d58a9d1204faabc4c9626f",
    ),
    (
        16079479408,
        64104,
        "11f0ffee56604d0d647c250091c8563e0f474a781c61bbf13c0710b98f51c3fe",
    ),
    (
        5245548280,
        21232,
        "17e9b2c0c666ac40456c818f179dd49db9655b496b4589cbfb439b314e6e639b",
    ),
    (
        15302516656,
        59816,
        "41f6a363adb62dce9f30d4144e4b7c883eb34f71d8e2240d73a2a9d8d55c6cd9",
    ),
    (
        14833765536,
        59032,
        "7a88c73b54a4cf81b7d2a915d9b2f7dae22cc0a75775a474932fa9c113f919f1",
    ),
    (
        14833765536,
        59032,
        "33f23ca3fada1fce55197cec0cde6811af89aa332eefe6f8ba2c7091a930c663",
    ),
    (
        14833765536,
        59032,
        "6aab8f39481e5e44e0ed5a266c5591cd535e44f974d95e19d77c4da4ff7ead00",
    ),
    (
        14833765536,
        59032,
        "8403c50e3910669d9c86f5475da69276ebb39813840aa21cdf4d66e09011e905",
    ),
    (
        14833765536,
        59032,
        "54447bd6145b61562509696f7d1859ec4284cb63347279c3d2bfb9adf0cbf6b8",
    ),
    (
        14833765536,
        59032,
        "407b5e85a89f33df56322b52994298db9d3820276c2746cc9ba927c021f77429",
    ),
    (
        14833765536,
        59032,
        "3fc4823044601c436f8fd9fdb8e1f923c3f2cbedb21d9569ffc46cff1af4c1a1",
    ),
    (
        14833765536,
        59032,
        "7e0a611524f5779fa2a2f0c1660789c32af18c00181293cb4674f7634be7b74e",
    ),
    (
        14833765536,
        59032,
        "fa65f5e4f4e8003be2697d0e911201e1eedc4cd3d93c966fe32c3b8761554646",
    ),
    (
        14833765536,
        59032,
        "ef726cffd210a9a1e976f5669f507a5750e43b6d6d591c41bdcbec04c511591d",
    ),
    (
        13588051672,
        53968,
        "5a408c94010fb4b44a03e4eea7722aef4c578830ead7cb465bf4c5e542dd583a",
    ),
    (
        13588051672,
        53968,
        "e8bff437fa7f012a28ccbb3c0b3632b50c2004943449ce01cc9dc15d1671c28f",
    ),
    (
        13588051672,
        53968,
        "678a6b28e70d1eb2045033a807419095ce5445307ae803f285c6e17c96c0293e",
    ),
    (
        13588051672,
        53968,
        "0df66d88da5ce0d7f0ed7b05b4007dbc9244baf2d44df0b92ce73f1433d0c570",
    ),
    (
        13588051672,
        53968,
        "3651fd17bc4983b72962d6ae111401e835f5316b0cf6ad35d7102d6ce0fe8aee",
    ),
    (
        13588051672,
        53968,
        "504cb2c0dc20ec70bbd277f6616c02d56fb2b6816c069ae6bf87b6131c438bbd",
    ),
    (
        13588051672,
        53968,
        "a2c49d2f741ebab5eeae9ec9c3a6ec6bbb764c106733cc7decb0834d4420b2d4",
    ),
    (
        13588051672,
        53968,
        "1950b75f6414794ae3b482bc413a93cbad50445000ed4c389b4b8777c65888f3",
    ),
    (
        13588051672,
        53968,
        "323f62213157732357a8fd35b14d9fe3d827eb6fb92bb1e5f970e38fa9af01b4",
    ),
    (
        13588051672,
        53968,
        "d170daefc383dcfa9244a07fd6ccd5226f8ca5ce67a1a9048c661d80d994a14d",
    ),
    (
        13588051672,
        53968,
        "06e78945f2c8f291f770fcfbba2eec3add4f43a64b9b3a7565d61b706fe3c656",
    ),
    (
        13588051672,
        53968,
        "408512bdf7ed21a0851ab6f0d93adbf3d80d416dda32fa255dbad2831dabf96a",
    ),
    (
        13588051672,
        53968,
        "4802a27c719d58ab770a38d115e39c8230d474d9d4740ee1edcf7d60f8572670",
    ),
    (
        13588051672,
        53968,
        "277f135f0b3d9a74f107d056954c9efc7cad3fc8fa7ab0fc54b47ac2110c2d3a",
    ),
    (
        13588051672,
        53968,
        "57dc74c75f711d9d4da848ebbe73bfecdcd746ee8bd5d788cb23b979d9d5b452",
    ),
    (
        13588051672,
        53968,
        "84c085f52a68d10ccc72fceb5f5da905b498db4161f96f8e67a8cf1d7ba273c9",
    ),
    (
        13588051672,
        53968,
        "452faeaf9945e93e0bc200705beed2263e4a8eba5e49e4dc1056e32a01fd9cb3",
    ),
    (
        13588051672,
        53968,
        "a19d5004d541b0dc2d1082dede52ae0d02ee378378065a115deac2500cc0f392",
    ),
    (
        13588051672,
        53968,
        "0d03a6e32465061a8f11992a04a8e09457be7f8e81fc028796c6c4cef12a099d",
    ),
    (
        13588051672,
        53968,
        "6c84f05d7d22c1250355f8df47ca28d99d627e49024c7a685161a577c75caf56",
    ),
    (
        13588051672,
        53968,
        "051e47e22b4ff3d585531747d1d3039bba6c92d41d764f4bc29f26b04041d14a",
    ),
    (
        13588051672,
        53968,
        "319a02f14adfddeee9768debad2c5d49501be60e4667254b0ff34b0ac06a7214",
    ),
    (
        16099232552,
        131360,
        "2c17b5a347b603e04d92b533d8749fe3973cba24081963a903d7ba2e2d6c1ada",
    ),
];

/// Tensor count per shard in the reviewed official index.
pub const MINIMAX_M3_INDEXED_TENSOR_COUNTS: [usize; MINIMAX_M3_SHARD_COUNT] = [
    14, 277, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435,
    435, 435, 435, 435, 435, 146, 408, 401, 401, 401, 401, 401, 401, 401, 401, 401, 401, 367, 367,
    367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367,
    367, 917,
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3SafetensorsDType {
    Bf16,
    F32,
}

impl MiniMaxM3SafetensorsDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    fn parse(value: &str) -> Result<Self, MiniMaxM3HeaderError> {
        match value {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            other => Err(invalid(format!("unsupported safetensors dtype: {other}"))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3HeaderTensor {
    pub name: String,
    /// One-based shard number.
    pub shard_index: u32,
    pub dtype: MiniMaxM3SafetensorsDType,
    pub shape: Vec<u64>,
    /// Start-inclusive, end-exclusive payload-relative offsets.
    pub data_offsets: [u64; 2],
    /// Start-inclusive, end-exclusive complete-file offsets.
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3ShardHeader {
    pub file_name: String,
    pub shard_index: u32,
    pub file_size: u64,
    pub header_length: u64,
    pub header_sha256: String,
    pub data_start: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<MiniMaxM3HeaderTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3Index {
    total_size: u64,
    weight_map: BTreeMap<String, String>,
}

impl MiniMaxM3Index {
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
pub struct MiniMaxM3HeaderCatalog {
    shards: Vec<MiniMaxM3ShardHeader>,
    tensor_count: usize,
    payload_bytes: u64,
    catalog_sha256: String,
}

impl MiniMaxM3HeaderCatalog {
    pub fn shards(&self) -> &[MiniMaxM3ShardHeader] {
        &self.shards
    }

    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn tensors(&self) -> impl Iterator<Item = &MiniMaxM3HeaderTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn tensor(&self, name: &str) -> Option<&MiniMaxM3HeaderTensor> {
        self.tensors().find(|tensor| tensor.name == name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MiniMaxM3HeaderError {
    Invalid(String),
    ManifestTotalSizeMismatch {
        index_total_size: u64,
        shard_payload_bytes: u64,
    },
}

impl fmt::Display for MiniMaxM3HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(
                formatter,
                "invalid MiniMax M3 safetensors header: {message}"
            ),
            Self::ManifestTotalSizeMismatch {
                index_total_size,
                shard_payload_bytes,
            } => write!(
                formatter,
                "MiniMax M3 index total_size {index_total_size} differs from shard payload {shard_payload_bytes}"
            ),
        }
    }
}

impl std::error::Error for MiniMaxM3HeaderError {}

fn invalid(message: impl Into<String>) -> MiniMaxM3HeaderError {
    MiniMaxM3HeaderError::Invalid(message.into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct UniqueTensors(BTreeMap<String, RawTensor>);

impl<'de> Deserialize<'de> for UniqueTensors {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = UniqueTensors;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor map without duplicate keys")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tensors = BTreeMap::new();
                while let Some((name, tensor)) = entries.next_entry::<String, RawTensor>()? {
                    if tensors.insert(name.clone(), tensor).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name: {name}"
                        )));
                    }
                }
                Ok(UniqueTensors(tensors))
            }
        }

        deserializer.deserialize_map(TensorVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndexMetadata {
    total_size: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndex {
    metadata: RawIndexMetadata,
    weight_map: UniqueIndex,
}

struct UniqueIndex(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct IndexVisitor;
        impl<'de> Visitor<'de> for IndexVisitor {
            type Value = UniqueIndex;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor index without duplicate keys")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some((name, shard)) = entries.next_entry::<String, String>()? {
                    if map.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate index tensor name: {name}"
                        )));
                    }
                }
                Ok(UniqueIndex(map))
            }
        }
        deserializer.deserialize_map(IndexVisitor)
    }
}

fn expected_file_name(index: usize) -> String {
    format!("model-{:05}-of-00059.safetensors", index + 1)
}

fn shard_number(file_name: &str) -> Result<usize, MiniMaxM3HeaderError> {
    if file_name.contains('/') || file_name.contains('\\') || file_name.contains("..") {
        return Err(invalid("unsafe shard path"));
    }
    let prefix = "model-";
    let suffix = "-of-00059.safetensors";
    let middle = file_name
        .strip_prefix(prefix)
        .and_then(|name| name.strip_suffix(suffix))
        .ok_or_else(|| invalid(format!("invalid shard filename: {file_name}")))?;
    if middle.len() != 5 || !middle.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(format!("invalid shard number: {file_name}")));
    }
    let number = middle
        .parse::<usize>()
        .map_err(|_| invalid(format!("invalid shard number: {file_name}")))?;
    if number == 0 || number > MINIMAX_M3_SHARD_COUNT {
        return Err(invalid(format!("shard number out of range: {file_name}")));
    }
    Ok(number - 1)
}

fn checked_shape_bytes(
    shape: &[u64],
    dtype: MiniMaxM3SafetensorsDType,
) -> Result<u64, MiniMaxM3HeaderError> {
    if shape.is_empty() {
        return Err(invalid("scalar tensor shape is not reviewed"));
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
    tensors: &mut [MiniMaxM3HeaderTensor],
    payload_bytes: u64,
) -> Result<(), MiniMaxM3HeaderError> {
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
        let [start, end] = tensors[index].data_offsets;
        if start >= end
            || end > payload_bytes
            || end.checked_sub(start) != Some(tensors[index].byte_size)
        {
            return Err(invalid(format!(
                "tensor range outside payload: {}",
                tensors[index].name
            )));
        }
        if ordinal == 0 && start != 0 {
            return Err(invalid("tensor payload has a leading gap"));
        }
        if start < previous_end {
            return Err(invalid(format!(
                "tensor ranges overlap: {}",
                tensors[index].name
            )));
        }
        if start != previous_end {
            return Err(invalid(format!(
                "tensor payload has a gap before {}",
                tensors[index].name
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
) -> Result<MiniMaxM3ShardHeader, MiniMaxM3HeaderError> {
    let index = shard_number(file_name)?;
    let (file_size, expected_header_length, expected_sha) = MINIMAX_M3_SHARD_IDENTITIES[index];
    let length_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("header length field is truncated"))?
        .try_into()
        .expect("slice length checked");
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length != expected_header_length {
        return Err(invalid(format!("header length differs for {file_name}")));
    }
    let prefix_length = MINIMAX_M3_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
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
    if header_sha256 != expected_sha {
        return Err(invalid(format!("header SHA-256 differs for {file_name}")));
    }
    let document: UniqueTensors = serde_json::from_slice(&bytes[8..])
        .map_err(|error| invalid(format!("header JSON: {error}")))?;
    if document.0.is_empty() {
        return Err(invalid("header tensor map is empty"));
    }
    if prefix_length > file_size {
        return Err(invalid("header extends beyond shard file"));
    }
    let payload_bytes = file_size - prefix_length;
    let shard_index = u32::try_from(index + 1).map_err(|_| invalid("shard index overflowed"))?;
    let mut tensors = Vec::with_capacity(document.0.len());
    for (name, raw) in document.0 {
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
        let dtype = MiniMaxM3SafetensorsDType::parse(&raw.dtype)?;
        let byte_size = checked_shape_bytes(&raw.shape, dtype)?;
        let [start, end] = raw.data_offsets;
        if end.checked_sub(start) != Some(byte_size) {
            return Err(invalid(format!("tensor byte size differs: {name}")));
        }
        let absolute_start = prefix_length
            .checked_add(start)
            .ok_or_else(|| invalid(format!("absolute range overflowed: {name}")))?;
        let absolute_end = prefix_length
            .checked_add(end)
            .ok_or_else(|| invalid(format!("absolute range overflowed: {name}")))?;
        tensors.push(MiniMaxM3HeaderTensor {
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
    Ok(MiniMaxM3ShardHeader {
        file_name: file_name.to_owned(),
        shard_index,
        file_size,
        header_length,
        header_sha256,
        data_start: prefix_length,
        payload_bytes,
        tensors,
    })
}

/// Parse and verify one official shard's header prefix. Full shard bytes are
/// intentionally rejected; callers should bounded-read the prefix first.
pub fn parse_minimax_m3_safetensors_header(
    file_name: &str,
    bytes: &[u8],
) -> Result<MiniMaxM3ShardHeader, MiniMaxM3HeaderError> {
    parse_header_prefix(file_name, bytes)
}

pub fn minimax_m3_locked_shard(
    file_name: &str,
) -> Result<(u64, u64, &'static str), MiniMaxM3HeaderError> {
    let index = shard_number(file_name)?;
    Ok(MINIMAX_M3_SHARD_IDENTITIES[index])
}

/// Parse the exact official safetensors index. Its total_size is retained even
/// though it is known to disagree with the sum of shard payload geometries.
pub fn validate_minimax_m3_index(bytes: &[u8]) -> Result<MiniMaxM3Index, MiniMaxM3HeaderError> {
    if bytes.len() != MINIMAX_M3_INDEX_BYTES
        || format!("{:x}", Sha256::digest(bytes)) != MINIMAX_M3_INDEX_SHA256
    {
        return Err(invalid("safetensors index identity differs"));
    }
    let raw: RawIndex = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("safetensors index JSON: {error}")))?;
    if raw.metadata.total_size != MINIMAX_M3_INDEX_TOTAL_SIZE
        || raw.weight_map.0.len() != MINIMAX_M3_TENSOR_COUNT
    {
        return Err(invalid("safetensors index count or total differs"));
    }
    let mut counts = [0_usize; MINIMAX_M3_SHARD_COUNT];
    for (name, shard) in &raw.weight_map.0 {
        if name.is_empty()
            || name.starts_with('.')
            || name.ends_with('.')
            || name.contains("..")
            || name
                .bytes()
                .any(|byte| !(byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.'))
        {
            return Err(invalid(format!("unsafe index tensor name: {name}")));
        }
        let shard_index = shard_number(shard)?;
        counts[shard_index] = counts[shard_index]
            .checked_add(1)
            .ok_or_else(|| invalid("index shard tensor count overflowed"))?;
    }
    if counts != MINIMAX_M3_INDEXED_TENSOR_COUNTS {
        return Err(invalid("index shard coverage differs"));
    }
    Ok(MiniMaxM3Index {
        total_size: raw.metadata.total_size,
        weight_map: raw.weight_map.0,
    })
}

fn canonical_catalog_sha256(shards: &[MiniMaxM3ShardHeader]) -> String {
    let mut headers = shards.iter().collect::<Vec<_>>();
    headers.sort_by_key(|shard| shard.file_name.as_str());
    let mut tensors = shards
        .iter()
        .flat_map(|shard| shard.tensors.iter())
        .collect::<Vec<_>>();
    tensors.sort_by_key(|tensor| tensor.name.as_str());
    let mut canonical = String::new();
    for shard in headers {
        use std::fmt::Write;
        let _ = writeln!(
            canonical,
            "header\t{}\t{}\t{}\t{}",
            shard.file_name, shard.file_size, shard.header_length, shard.header_sha256
        );
    }
    for tensor in tensors {
        use std::fmt::Write;
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let [start, end] = tensor.data_offsets;
        let [absolute_start, absolute_end] = tensor.absolute_byte_range;
        let file_name = expected_file_name((tensor.shard_index - 1) as usize);
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

/// Build the exact header catalog without consulting the index. This is useful
/// for reporting the official manifest mismatch without treating it as a
/// successful loadable model.
pub fn build_minimax_m3_header_catalog(
    headers: &[MiniMaxM3ShardHeader],
) -> Result<MiniMaxM3HeaderCatalog, MiniMaxM3HeaderError> {
    if headers.len() != MINIMAX_M3_SHARD_COUNT {
        return Err(invalid("header shard count differs"));
    }
    let mut by_file = BTreeMap::new();
    let mut by_index = BTreeMap::new();
    for header in headers {
        let index = shard_number(&header.file_name)?;
        let (file_size, header_length, header_sha) = MINIMAX_M3_SHARD_IDENTITIES[index];
        let expected_index =
            u32::try_from(index + 1).map_err(|_| invalid("shard index overflowed"))?;
        if header.shard_index != expected_index
            || header.file_size != file_size
            || header.header_length != header_length
            || header.header_sha256 != header_sha
            || header.data_start != 8 + header_length
            || header.data_start > header.file_size
            || header.payload_bytes != header.file_size - header.data_start
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
            if tensor.byte_size != checked_shape_bytes(&tensor.shape, tensor.dtype)?
                || end.checked_sub(start) != Some(tensor.byte_size)
                || tensor.absolute_byte_range != absolute
            {
                return Err(invalid(format!("tensor geometry differs: {}", tensor.name)));
            }
        }
    }
    if by_file.len() != MINIMAX_M3_SHARD_COUNT
        || (0..MINIMAX_M3_SHARD_COUNT)
            .any(|index| !by_file.contains_key(expected_file_name(index).as_str()))
    {
        return Err(invalid("missing or extra shard header"));
    }
    let shard_file_bytes = headers.iter().try_fold(0_u64, |sum, header| {
        sum.checked_add(header.file_size)
            .ok_or_else(|| invalid("shard file total overflowed"))
    })?;
    if shard_file_bytes != MINIMAX_M3_SHARD_FILE_BYTES {
        return Err(invalid("shard file byte total differs"));
    }
    let payload_bytes = headers.iter().try_fold(0_u64, |sum, header| {
        sum.checked_add(header.payload_bytes)
            .ok_or_else(|| invalid("header payload total overflowed"))
    })?;
    let tensor_count = headers.iter().try_fold(0_usize, |sum, header| {
        sum.checked_add(header.tensors.len())
            .ok_or_else(|| invalid("header tensor count overflowed"))
    })?;
    if payload_bytes != MINIMAX_M3_SHARD_PAYLOAD_BYTES || tensor_count != MINIMAX_M3_TENSOR_COUNT {
        return Err(invalid("header payload or tensor count differs"));
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
    if catalog_sha256 != MINIMAX_M3_HEADER_CATALOG_SHA256 {
        return Err(invalid("header catalog SHA-256 differs"));
    }
    let mut ordered = headers.to_vec();
    ordered.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    Ok(MiniMaxM3HeaderCatalog {
        shards: ordered,
        tensor_count,
        payload_bytes,
        catalog_sha256,
    })
}

/// Validate header/index name and shard coverage, then fail closed on the
/// official index total-size inconsistency. Callers that need the metadata-only
/// catalog for diagnostics can use `build_minimax_m3_header_catalog` first.
pub fn validate_minimax_m3_header_catalog(
    headers: &[MiniMaxM3ShardHeader],
    index: &MiniMaxM3Index,
) -> Result<MiniMaxM3HeaderCatalog, MiniMaxM3HeaderError> {
    let catalog = build_minimax_m3_header_catalog(headers)?;
    if index.tensor_count() != catalog.tensor_count {
        return Err(invalid("header/index tensor count differs"));
    }
    let mut header_sources = BTreeMap::new();
    for header in headers {
        for tensor in &header.tensors {
            header_sources.insert(
                tensor.name.as_str(),
                expected_file_name((tensor.shard_index - 1) as usize),
            );
        }
    }
    for (name, shard) in index.tensors() {
        let source = header_sources
            .get(name)
            .ok_or_else(|| invalid(format!("header tensor missing from index: {name}")))?;
        if source != shard {
            return Err(invalid(format!("header/index shard differs: {name}")));
        }
    }
    if header_sources.len() != index.tensor_count() {
        return Err(invalid("header has an extra tensor absent from index"));
    }
    if index.total_size() != catalog.payload_bytes {
        return Err(MiniMaxM3HeaderError::ManifestTotalSizeMismatch {
            index_total_size: index.total_size(),
            shard_payload_bytes: catalog.payload_bytes,
        });
    }
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tensor(
        name: &str,
        shard_index: u32,
        dtype: MiniMaxM3SafetensorsDType,
        shape: Vec<u64>,
        offsets: [u64; 2],
    ) -> MiniMaxM3HeaderTensor {
        MiniMaxM3HeaderTensor {
            name: name.to_owned(),
            shard_index,
            dtype,
            shape,
            data_offsets: offsets,
            absolute_byte_range: [96 + offsets[0], 96 + offsets[1]],
            byte_size: offsets[1] - offsets[0],
        }
    }

    #[test]
    fn official_identity_and_count_tables_are_fixed() {
        assert_eq!(MINIMAX_M3_SHARD_IDENTITIES.len(), MINIMAX_M3_SHARD_COUNT);
        assert_eq!(
            MINIMAX_M3_INDEXED_TENSOR_COUNTS.iter().sum::<usize>(),
            MINIMAX_M3_TENSOR_COUNT
        );
        assert_eq!(MINIMAX_M3_SHARD_IDENTITIES[0].1, 1760);
        assert_eq!(MINIMAX_M3_SHARD_IDENTITIES[58].1, 131360);
        assert_eq!(
            MINIMAX_M3_SHARD_IDENTITIES
                .iter()
                .map(|row| row.0)
                .sum::<u64>(),
            MINIMAX_M3_SHARD_FILE_BYTES
        );
        assert_eq!(
            MINIMAX_M3_SHARD_IDENTITIES
                .iter()
                .map(|row| row.1 + 8)
                .sum::<u64>(),
            MINIMAX_M3_HEADER_BYTES
        );
    }

    #[test]
    fn duplicate_unknown_path_and_shape_overflow_are_rejected() {
        let duplicate = br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#;
        assert!(serde_json::from_slice::<UniqueTensors>(duplicate).is_err());
        let duplicate_index = br#"{"metadata":{"total_size":1},"weight_map":{"a":"model-00001-of-00059.safetensors","a":"model-00002-of-00059.safetensors"}}"#;
        assert!(serde_json::from_slice::<RawIndex>(duplicate_index).is_err());
        assert!(shard_number("../model-00001-of-00059.safetensors").is_err());
        assert!(shard_number("model-00060-of-00059.safetensors").is_err());
        assert!(checked_shape_bytes(&[u64::MAX, 2], MiniMaxM3SafetensorsDType::F32).is_err());
    }

    #[test]
    fn ranges_reject_gap_overlap_and_boundaries() {
        let mut rows = vec![
            synthetic_tensor("a", 1, MiniMaxM3SafetensorsDType::F32, vec![1], [0, 4]),
            synthetic_tensor("b", 1, MiniMaxM3SafetensorsDType::F32, vec![1], [4, 8]),
        ];
        assert!(validate_ranges(&mut rows, 8).is_ok());
        rows[1].data_offsets = [5, 9];
        assert!(validate_ranges(&mut rows, 9).is_err());
        rows[1].data_offsets = [2, 6];
        assert!(validate_ranges(&mut rows, 8).is_err());
        rows[1].data_offsets = [4, u64::MAX];
        assert!(validate_ranges(&mut rows, u64::MAX).is_err());
    }

    #[test]
    fn manifest_inconsistency_is_explicit_and_fail_closed() {
        let error = MiniMaxM3HeaderError::ManifestTotalSizeMismatch {
            index_total_size: MINIMAX_M3_INDEX_TOTAL_SIZE,
            shard_payload_bytes: MINIMAX_M3_SHARD_PAYLOAD_BYTES,
        };
        assert!(error.to_string().contains("differs"));
    }

    #[test]
    #[ignore = "requires the reviewed official MiniMax M3 index and 59 header prefixes"]
    fn official_header_prefixes_match_index_and_catalog() {
        use std::fs::File;
        use std::io::Read;
        use std::path::{Path, PathBuf};

        fn read_prefix(path: &Path) -> Vec<u8> {
            let mut file = File::open(path).expect("open shard/header prefix");
            let mut length = [0_u8; 8];
            file.read_exact(&mut length).expect("read header length");
            let header_length = u64::from_le_bytes(length);
            let total = usize::try_from(header_length + 8).expect("prefix fits");
            let mut bytes = vec![0_u8; total];
            bytes[..8].copy_from_slice(&length);
            file.read_exact(&mut bytes[8..]).expect("read header JSON");
            bytes
        }

        let root = PathBuf::from(
            std::env::var_os("SLLM_MINIMAX_M3_METADATA_DIR")
                .expect("set SLLM_MINIMAX_M3_METADATA_DIR to index and 59 header prefixes"),
        );
        let mut headers = Vec::with_capacity(MINIMAX_M3_SHARD_COUNT);
        for index in 0..MINIMAX_M3_SHARD_COUNT {
            let file_name = expected_file_name(index);
            let short = format!("model-{index:05}.header", index = index + 1);
            let candidates = [
                root.join(&file_name),
                root.join(format!("{file_name}.header")),
                root.join(format!("{file_name}.prefix")),
                root.join(&short),
                root.join(short.replace(".header", ".prefix")),
            ];
            let path = candidates
                .iter()
                .find(|candidate| candidate.is_file())
                .unwrap_or_else(|| panic!("missing header prefix for {file_name}"));
            headers.push(
                parse_minimax_m3_safetensors_header(&file_name, &read_prefix(path))
                    .expect("validate header"),
            );
        }
        let index = validate_minimax_m3_index(
            &std::fs::read(root.join("model.safetensors.index.json")).expect("read index"),
        )
        .expect("validate index");
        let catalog = build_minimax_m3_header_catalog(&headers).expect("build header catalog");
        assert_eq!(catalog.tensor_count(), MINIMAX_M3_TENSOR_COUNT);
        assert_eq!(catalog.payload_bytes(), MINIMAX_M3_SHARD_PAYLOAD_BYTES);
        assert_eq!(catalog.catalog_sha256(), MINIMAX_M3_HEADER_CATALOG_SHA256);
        assert!(matches!(
            validate_minimax_m3_header_catalog(&headers, &index),
            Err(MiniMaxM3HeaderError::ManifestTotalSizeMismatch { .. })
        ));
    }
}
