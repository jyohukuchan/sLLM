//! Header-only safetensors contract for the reviewed DeepSeek V4 artifact.
//!
//! A safetensors file starts with an eight-byte little-endian header length and
//! a JSON header.  This module intentionally reads only that prefix: no API in
//! this module materializes tensor payload bytes.  The reviewed header
//! identities are separate from the Hub LFS payload identities in
//! `deepseek_v4.rs`.

use crate::{
    DEEPSEEK_V4_SHARD_COUNT, DEEPSEEK_V4_SHARDS, DEEPSEEK_V4_TENSOR_COUNT,
    DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES, DeepSeekV4Index, classify_deepseek_v4_tensor,
};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const DEEPSEEK_V4_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES: u64 = 8;

/// SHA-256 covers the eight-byte length field and the JSON header prefix.
/// It does not cover the tensor payload.
pub const DEEPSEEK_V4_HEADER_CATALOG_SHA256: &str =
    "6d90aa665f26217f4488809b1fdf87a1459702aa4ec46c8b02b44ce66bd4afcc";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4HeaderIdentity {
    pub file_name: &'static str,
    pub header_length: u64,
    pub header_sha256: &'static str,
}

/// Reviewed SHA-256 and geometry for all 48 official shard headers.
pub const DEEPSEEK_V4_HEADER_IDENTITIES: [DeepSeekV4HeaderIdentity; DEEPSEEK_V4_SHARD_COUNT] = [
    DeepSeekV4HeaderIdentity {
        file_name: "model-00001-of-00048.safetensors",
        header_length: 88,
        header_sha256: "ec94f070d8173f2b3eaae1638b811b8cfa1082a59de9d377081d3008c4cc6247",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00002-of-00048.safetensors",
        header_length: 172232,
        header_sha256: "c0460c63141a938c627da59cc153dc79e0840439f7312e421c5026f8a9bca629",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00003-of-00048.safetensors",
        header_length: 172232,
        header_sha256: "d6b3a0ebca6aa59b76662c1dcc7bbc58fc8d4cdf2ff3fcfeabeb3eb61b1a9c5b",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00004-of-00048.safetensors",
        header_length: 173624,
        header_sha256: "a8b814ff108217c6ef71250b2ac9c6e184a5a02abaa10ffcc918a9317bac029a",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00005-of-00048.safetensors",
        header_length: 172656,
        header_sha256: "5ba5fd61cedb5e934a35d4b214d43316b94101ac08e068501179497af4234a7c",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00006-of-00048.safetensors",
        header_length: 173544,
        header_sha256: "6a227e0a48fb4a9bdc8ad8dd1842d587f949902972bad48ca8ff59f4a6584cc3",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00007-of-00048.safetensors",
        header_length: 172656,
        header_sha256: "3766bdc1fe3a069f6a7d87ea76e99a2daebb73b616f2cf8d3be8d6cd7a0a61d5",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00008-of-00048.safetensors",
        header_length: 173544,
        header_sha256: "fa087491626731378708d743dcf31dd043c1ba124ac42e04f5d5b6637280c9e7",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00009-of-00048.safetensors",
        header_length: 172656,
        header_sha256: "a0d1f5894bcdd9539839ea2bed44941a3062be6a7fec118dcd13fe91fb1c625c",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00010-of-00048.safetensors",
        header_length: 173544,
        header_sha256: "952616c610e5b6f1e88721389f7bc6fd012d88d14e7d1939f9b83a8afd553e2f",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00011-of-00048.safetensors",
        header_length: 172656,
        header_sha256: "f748fcb4e2d050c65fd87d15ad81122252fdb263aec93312645a668a974d21a5",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00012-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "26823212fdefc2d9cde829de94c5069cfaaabee27d65c8b1e810e3f3628153c0",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00013-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "17cf3e150276bb1af5983af6d1a3cec23d3bcb6d4d79138c95499fd7ad8a4d58",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00014-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "24194492d25306914b45c370fec78253273fb4d77c6cdff862f47a448b60d3f9",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00015-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "c5698c00efbc6c0b025dd5c4d3f262c3726f3182ae636511c116c71f71a6a1cb",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00016-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "bfa91569acb70c14d8db4c11f501f012c6b4e41ae286b4aa149f02503ae30fd4",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00017-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "05dc7169809d4a402fb05ddea1fef1ec99659a9e7fad20fb380223ba220e9ac0",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00018-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "f68b45f74c5e10a3be1c1a8d83ea874202d136c53a872e884058aa4687f788ae",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00019-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "95055e75864bf5286945a82894ad2640812a61d7583d672f682d1600b723a864",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00020-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "72b5179877a51baa5aca4f1db6f8948bc6f5b90b05c7114dd6ff82918bb9413e",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00021-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "755eb6c67ef5fa846af05c6a1aa03e1ba6b8cb98fda4e9f246cac1c7c2d719c6",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00022-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "f3fa7eeb2c0cc758f8a21933ddb03332acf387888c5cc704ac73679486ba39c8",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00023-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "f7c04ec1b635428abb166d4d316c94f9048cd665f25e0ea5671c0a83f7fb01c9",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00024-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "502c67fa0497c5dda867a6ace028d79af986a3212d111e2115dbcdea7f8aacd1",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00025-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "7d6fdf6cd35b7be3fff2887483e5833f78bd308c7fa23106ed86092564a44e87",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00026-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "575b3670a555390acc3cc96feccfb63f279422f54593949b2ca36adc6c272f4f",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00027-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "885b1bced21faa48df9fa82000681224e8a1d02874e576d788f78a576fe3e2d4",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00028-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "34c541ad9720071cadbc14b128c633bf094d6a6616d7885855810d516f4c9240",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00029-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "428c6a2601872d5461da38dd69b0e5cd2240af6bd826ed1eeace59037e8c4533",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00030-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "f7b9ae55931d4d21434164646676f8b8c71053a94f04aca7bd2718ec68eebad5",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00031-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "c9d429249b12313ae4e6ca8c54013afd801cc8950d602d32d275d0c5272c71bc",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00032-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "eb00bf03e1e05ddcf32dc655cf3e3f2ae8808684d8062c0e098cadf95e7f2601",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00033-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "0c6f363d5e14afae9eafbffe5d288b51a21f8041a7d17b07d4a93e69ffe032f8",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00034-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "24c9ae647d6a946cc3213becfcbfbdbbb6cb271c5034f42853fe52b47bd12fb4",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00035-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "774cd0b4dcc8820be658a70e4c9d39563f98ae43029a07b905108df84073cb89",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00036-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "5a1a058b40405c443fbcb4a3cffab4f817297f3b1bcc4529b8f01ebcc5e2cbf9",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00037-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "ab45b66cd8da367dc779e85e9320e2e4eb40a139cfc3cf97d910aeab0c56fd57",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00038-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "4e63803bfa0cf285702615fc4ccf6f0b434e6d14bfd416b02887397ebced6cbe",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00039-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "102c53b2de458bb744feb81ba04cde8b712ddea80f10dd6967506ec9b2ce8cbb",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00040-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "b0d457ee97e5cb77a908b9bb12f1da576b1c9207ca04c443363ae179df5ea90c",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00041-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "668a3efc56bcf1ee261bf151a608eaf7dd637fac4828431e3558dbcace97f12a",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00042-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "241b4fab5163b3886ff7517ef03686be804160cd444e1e0085a5a4321a4edfe2",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00043-of-00048.safetensors",
        header_length: 174224,
        header_sha256: "cb19acae159b658e26c3d7bdd166ab861a9cb55bea4320f5c3cff678ee04088b",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00044-of-00048.safetensors",
        header_length: 175120,
        header_sha256: "e5fb424355c49d76e16d0b18f911793d6dd611babf62821d65a31d2b6340e6c1",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00045-of-00048.safetensors",
        header_length: 392,
        header_sha256: "ed86fda4ac2c09c307a6578b21cbd610a24da5e6015f75309186c0e9941447cb",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00046-of-00048.safetensors",
        header_length: 167728,
        header_sha256: "52e716d914861449c3771082d615faa6103f0655afb8d297b54c1677c1704925",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00047-of-00048.safetensors",
        header_length: 167416,
        header_sha256: "fcbee7df8b778f5eb845d95277840b49c6430acf32e55e9b10a40a029308faa8",
    },
    DeepSeekV4HeaderIdentity {
        file_name: "model-00048-of-00048.safetensors",
        header_length: 168920,
        header_sha256: "a59014b7960c7a6bab53f5a72a3629b14af5e967bbd2bdb8e7e792ff72080ecc",
    },
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4SafetensorsDType {
    Bf16,
    F32,
    F8E4M3,
    F8E8M0,
    I8,
    I64,
}

impl DeepSeekV4SafetensorsDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::I8 => "I8",
            Self::I64 => "I64",
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::F8E4M3 | Self::F8E8M0 | Self::I8 => 1,
            Self::I64 => 8,
        }
    }

    fn parse(value: &str) -> Result<Self, DeepSeekV4HeaderError> {
        match value {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            "F8_E4M3" => Ok(Self::F8E4M3),
            "F8_E8M0" => Ok(Self::F8E8M0),
            "I8" => Ok(Self::I8),
            "I64" => Ok(Self::I64),
            other => Err(invalid(format!("unsupported safetensors dtype: {other}"))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4HeaderTensor {
    pub name: String,
    /// One-based shard number, matching the official file name.
    pub shard_index: u32,
    pub dtype: DeepSeekV4SafetensorsDType,
    pub shape: Vec<u64>,
    /// Start-inclusive, end-exclusive offsets relative to the tensor payload.
    pub data_offsets: [u64; 2],
    /// Start-inclusive, end-exclusive offsets relative to the complete shard.
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ShardHeader {
    pub file_name: String,
    /// One-based shard number.
    pub shard_index: u32,
    pub file_size: u64,
    pub header_length: u64,
    pub header_sha256: String,
    pub data_start: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<DeepSeekV4HeaderTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4HeaderCatalog {
    shards: Vec<DeepSeekV4ShardHeader>,
    tensor_count: usize,
    payload_bytes: u64,
    catalog_sha256: String,
}

impl DeepSeekV4HeaderCatalog {
    pub fn shards(&self) -> &[DeepSeekV4ShardHeader] {
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

    pub fn tensors(&self) -> impl Iterator<Item = &DeepSeekV4HeaderTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn tensor(&self, name: &str) -> Option<&DeepSeekV4HeaderTensor> {
        self.tensors().find(|tensor| tensor.name == name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4HeaderError {
    Invalid(String),
}

impl fmt::Display for DeepSeekV4HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(
                formatter,
                "invalid DeepSeek V4 safetensors header: {message}"
            ),
        }
    }
}

impl std::error::Error for DeepSeekV4HeaderError {}

fn invalid(message: impl Into<String>) -> DeepSeekV4HeaderError {
    DeepSeekV4HeaderError::Invalid(message.into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct UniqueRawHeaderTensors(BTreeMap<String, RawHeaderTensor>);

impl<'de> Deserialize<'de> for UniqueRawHeaderTensors {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TensorMapVisitor;

        impl<'de> Visitor<'de> for TensorMapVisitor {
            type Value = UniqueRawHeaderTensors;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor-name to descriptor object without duplicate keys")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tensors = BTreeMap::new();
                while let Some((name, tensor)) = entries.next_entry::<String, RawHeaderTensor>()? {
                    if tensors.insert(name.clone(), tensor).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name: {name}"
                        )));
                    }
                }
                Ok(UniqueRawHeaderTensors(tensors))
            }
        }

        deserializer.deserialize_map(TensorMapVisitor)
    }
}

fn locked_header_identity(
    file_name: &str,
) -> Result<(usize, DeepSeekV4HeaderIdentity, u64), DeepSeekV4HeaderError> {
    if file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
    {
        return Err(invalid("unsafe shard path"));
    }
    let (position, identity) = DEEPSEEK_V4_HEADER_IDENTITIES
        .iter()
        .copied()
        .enumerate()
        .find(|(_, identity)| identity.file_name == file_name)
        .ok_or_else(|| invalid(format!("unknown shard: {file_name}")))?;
    let file_size = DEEPSEEK_V4_SHARDS
        .get(position)
        .ok_or_else(|| invalid("header/shard identity table is inconsistent"))?
        .size;
    Ok((position, identity, file_size))
}

fn checked_shape_bytes(
    shape: &[u64],
    dtype: DeepSeekV4SafetensorsDType,
) -> Result<u64, DeepSeekV4HeaderError> {
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
    tensors: &mut [DeepSeekV4HeaderTensor],
    payload_bytes: u64,
) -> Result<(), DeepSeekV4HeaderError> {
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

fn parse_header_prefix(
    file_name: &str,
    bytes: &[u8],
) -> Result<(DeepSeekV4ShardHeader, DeepSeekV4HeaderIdentity), DeepSeekV4HeaderError> {
    let (position, identity, file_size) = locked_header_identity(file_name)?;
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
    let prefix_length = DEEPSEEK_V4_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES
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
    let document: UniqueRawHeaderTensors = serde_json::from_slice(&bytes[8..])
        .map_err(|error| invalid(format!("header JSON: {error}")))?;
    if document.0.is_empty() {
        return Err(invalid("header tensor map is empty"));
    }
    let data_start = prefix_length;
    if data_start > file_size {
        return Err(invalid("header extends beyond shard file"));
    }
    let payload_bytes = file_size - data_start;
    let shard_index = u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
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
        let dtype = DeepSeekV4SafetensorsDType::parse(&raw.dtype)?;
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
        tensors.push(DeepSeekV4HeaderTensor {
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
    Ok((
        DeepSeekV4ShardHeader {
            file_name: file_name.to_owned(),
            shard_index,
            file_size,
            header_length,
            header_sha256,
            data_start,
            payload_bytes,
            tensors,
        },
        identity,
    ))
}

/// Parse and verify one official shard's safetensors header prefix.
///
/// `bytes` must contain exactly the eight-byte length field plus the declared
/// JSON header. Passing a full shard is intentionally rejected by this API;
/// callers can bounded-read that prefix before calling it.
pub fn parse_deepseek_v4_safetensors_header(
    file_name: &str,
    bytes: &[u8],
) -> Result<DeepSeekV4ShardHeader, DeepSeekV4HeaderError> {
    parse_header_prefix(file_name, bytes).map(|(header, _)| header)
}

pub fn deepseek_v4_locked_header(file_name: &str) -> Option<DeepSeekV4HeaderIdentity> {
    DEEPSEEK_V4_HEADER_IDENTITIES
        .iter()
        .copied()
        .find(|identity| identity.file_name == file_name)
}

fn canonical_catalog_sha256(shards: &[DeepSeekV4ShardHeader]) -> String {
    let mut header_rows: Vec<&DeepSeekV4ShardHeader> = shards.iter().collect();
    header_rows.sort_by_key(|shard| shard.file_name.as_str());
    let mut tensor_rows: Vec<&DeepSeekV4HeaderTensor> = shards
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

/// Validate all shard headers against the reviewed index and return the
/// header-only catalog. Input order is irrelevant, but all 48 unique shards
/// are required. Tensor names, source shard, dtype, shape, and both relative
/// and absolute ranges are checked before the catalog digest is accepted.
pub fn validate_deepseek_v4_header_catalog(
    headers: &[DeepSeekV4ShardHeader],
    index: &DeepSeekV4Index,
) -> Result<DeepSeekV4HeaderCatalog, DeepSeekV4HeaderError> {
    if headers.len() != DEEPSEEK_V4_SHARD_COUNT {
        return Err(invalid(format!(
            "header shard count differs: expected {}, got {}",
            DEEPSEEK_V4_SHARD_COUNT,
            headers.len()
        )));
    }
    let mut by_file = BTreeMap::new();
    for header in headers {
        let (position, identity, file_size) = locked_header_identity(&header.file_name)?;
        let expected_shard_index =
            u32::try_from(position + 1).map_err(|_| invalid("shard index overflowed"))?;
        if header.shard_index != expected_shard_index
            || header.file_size != file_size
            || header.header_length != identity.header_length
            || header.header_sha256 != identity.header_sha256
            || header.data_start
                != DEEPSEEK_V4_SAFETENSORS_HEADER_LENGTH_FIELD_BYTES + identity.header_length
            || header.data_start > header.file_size
            || header.payload_bytes != header.file_size - header.data_start
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
    if by_file.len() != DEEPSEEK_V4_SHARD_COUNT
        || DEEPSEEK_V4_HEADER_IDENTITIES
            .iter()
            .any(|identity| !by_file.contains_key(identity.file_name))
    {
        return Err(invalid("missing or extra shard header"));
    }

    let mut tensor_sources = BTreeMap::<&str, &DeepSeekV4HeaderTensor>::new();
    let mut payload_bytes = 0_u64;
    let mut tensor_count = 0_usize;
    for header in headers {
        let mut range_check = header.tensors.clone();
        validate_ranges(&mut range_check, header.payload_bytes)?;
        payload_bytes = payload_bytes
            .checked_add(header.payload_bytes)
            .ok_or_else(|| invalid("header payload byte total overflowed"))?;
        tensor_count = tensor_count
            .checked_add(header.tensors.len())
            .ok_or_else(|| invalid("header tensor count overflowed"))?;
        for tensor in &header.tensors {
            classify_deepseek_v4_tensor(&tensor.name)
                .map_err(|error| invalid(format!("unknown tensor family: {error}")))?;
            let expected_byte_size = checked_shape_bytes(&tensor.shape, tensor.dtype)?;
            let [start, end] = tensor.data_offsets;
            if tensor.byte_size != expected_byte_size
                || end.checked_sub(start) != Some(tensor.byte_size)
                || tensor.absolute_byte_range
                    != [
                        header.data_start.checked_add(start).ok_or_else(|| {
                            invalid(format!("absolute tensor range overflowed: {}", tensor.name))
                        })?,
                        header.data_start.checked_add(end).ok_or_else(|| {
                            invalid(format!("absolute tensor range overflowed: {}", tensor.name))
                        })?,
                    ]
            {
                return Err(invalid(format!(
                    "tensor range geometry differs: {}",
                    tensor.name
                )));
            }
            if tensor.shard_index != header.shard_index {
                return Err(invalid(format!("tensor shard mismatch: {}", tensor.name)));
            }
            if tensor_sources
                .insert(tensor.name.as_str(), tensor)
                .is_some()
            {
                return Err(invalid(format!("duplicate tensor name: {}", tensor.name)));
            }
        }
    }
    if tensor_count != DEEPSEEK_V4_TENSOR_COUNT || index.tensor_count() != DEEPSEEK_V4_TENSOR_COUNT
    {
        return Err(invalid("header/index tensor count differs"));
    }
    if payload_bytes != DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES
        || index.total_size() != DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES
    {
        return Err(invalid("header/index payload byte total differs"));
    }
    if tensor_sources.len() != index.tensor_count() {
        return Err(invalid("header tensor name coverage differs"));
    }
    for (name, source_file) in index.tensors() {
        let tensor = tensor_sources
            .get(name)
            .ok_or_else(|| invalid(format!("header tensor is missing from index: {name}")))?;
        if source_file != tensor_file_name(headers, tensor)? {
            return Err(invalid(format!("header/index shard differs: {name}")));
        }
    }
    if tensor_sources
        .keys()
        .any(|name| index.source_file(name).is_none())
    {
        return Err(invalid("header has an extra tensor absent from index"));
    }
    let catalog_sha256 = canonical_catalog_sha256(headers);
    if catalog_sha256 != DEEPSEEK_V4_HEADER_CATALOG_SHA256 {
        return Err(invalid("header catalog SHA-256 differs"));
    }
    let mut ordered = headers.to_vec();
    ordered.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    Ok(DeepSeekV4HeaderCatalog {
        shards: ordered,
        tensor_count,
        payload_bytes,
        catalog_sha256,
    })
}

fn tensor_file_name<'a>(
    headers: &'a [DeepSeekV4ShardHeader],
    tensor: &DeepSeekV4HeaderTensor,
) -> Result<&'a str, DeepSeekV4HeaderError> {
    headers
        .iter()
        .find(|header| header.shard_index == tensor.shard_index)
        .map(|header| header.file_name.as_str())
        .ok_or_else(|| invalid(format!("tensor shard index is unknown: {}", tensor.name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tensor(
        name: &str,
        shard_index: u32,
        dtype: DeepSeekV4SafetensorsDType,
        shape: Vec<u64>,
        offsets: [u64; 2],
    ) -> DeepSeekV4HeaderTensor {
        let byte_size = offsets[1] - offsets[0];
        DeepSeekV4HeaderTensor {
            name: name.to_owned(),
            shard_index,
            dtype,
            shape,
            data_offsets: offsets,
            absolute_byte_range: [96 + offsets[0], 96 + offsets[1]],
            byte_size,
        }
    }

    #[test]
    fn official_header_identity_table_is_complete_and_bounded() {
        assert_eq!(DEEPSEEK_V4_HEADER_IDENTITIES.len(), 48);
        assert_eq!(DEEPSEEK_V4_HEADER_IDENTITIES[0].header_length, 88);
        assert_eq!(DEEPSEEK_V4_HEADER_IDENTITIES[44].header_length, 392);
        assert!(
            DEEPSEEK_V4_HEADER_IDENTITIES
                .iter()
                .all(|identity| identity.header_sha256.len() == 64)
        );
        assert!(
            DEEPSEEK_V4_HEADER_IDENTITIES
                .windows(2)
                .all(|pair| pair[0].file_name < pair[1].file_name)
        );
    }

    #[test]
    fn dtype_widths_and_shape_overflow_are_fail_closed() {
        assert_eq!(DeepSeekV4SafetensorsDType::Bf16.byte_width(), 2);
        assert_eq!(DeepSeekV4SafetensorsDType::F32.byte_width(), 4);
        assert_eq!(DeepSeekV4SafetensorsDType::F8E8M0.byte_width(), 1);
        assert!(checked_shape_bytes(&[u64::MAX, 2], DeepSeekV4SafetensorsDType::I8).is_err());
        assert!(checked_shape_bytes(&[], DeepSeekV4SafetensorsDType::I8).is_err());
    }

    #[test]
    fn duplicate_unknown_and_unsafe_header_entries_are_rejected() {
        let duplicate = br#"{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"I8","shape":[1],"data_offsets":[1,2]}}"#;
        assert!(serde_json::from_slice::<UniqueRawHeaderTensors>(duplicate).is_err());
        let unknown = br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let parsed: UniqueRawHeaderTensors = serde_json::from_slice(unknown).unwrap();
        assert!(DeepSeekV4SafetensorsDType::parse(&parsed.0["a"].dtype).is_err());
        let path = parse_deepseek_v4_safetensors_header("../model-00001-of-00048.safetensors", &[]);
        assert!(path.is_err());
        let mismatch =
            parse_deepseek_v4_safetensors_header("model-99999-of-00048.safetensors", &[]);
        assert!(mismatch.is_err());
    }

    #[test]
    fn range_boundaries_reject_gaps_overlap_and_overflow() {
        let mut contiguous = vec![
            synthetic_tensor("a", 1, DeepSeekV4SafetensorsDType::I8, vec![2], [0, 2]),
            synthetic_tensor("b", 1, DeepSeekV4SafetensorsDType::I8, vec![2], [2, 4]),
        ];
        assert!(validate_ranges(&mut contiguous, 4).is_ok());
        contiguous[1].data_offsets = [3, 5];
        assert!(validate_ranges(&mut contiguous, 5).is_err());
        contiguous[1].data_offsets = [1, 3];
        assert!(validate_ranges(&mut contiguous, 4).is_err());
        contiguous[1].data_offsets = [u64::MAX, u64::MAX];
        assert!(validate_ranges(&mut contiguous, u64::MAX).is_err());
    }

    #[test]
    fn canonical_digest_is_order_independent() {
        let first = DeepSeekV4ShardHeader {
            file_name: DEEPSEEK_V4_HEADER_IDENTITIES[0].file_name.to_owned(),
            shard_index: 1,
            file_size: DEEPSEEK_V4_SHARDS[0].size,
            header_length: 88,
            header_sha256: DEEPSEEK_V4_HEADER_IDENTITIES[0].header_sha256.to_owned(),
            data_start: 96,
            payload_bytes: 4,
            tensors: vec![synthetic_tensor(
                "b",
                1,
                DeepSeekV4SafetensorsDType::I8,
                vec![2],
                [2, 4],
            )],
        };
        let second = DeepSeekV4ShardHeader {
            file_name: DEEPSEEK_V4_HEADER_IDENTITIES[1].file_name.to_owned(),
            shard_index: 2,
            file_size: DEEPSEEK_V4_SHARDS[1].size,
            header_length: DEEPSEEK_V4_HEADER_IDENTITIES[1].header_length,
            header_sha256: DEEPSEEK_V4_HEADER_IDENTITIES[1].header_sha256.to_owned(),
            data_start: DEEPSEEK_V4_HEADER_IDENTITIES[1].header_length + 8,
            payload_bytes: 0,
            tensors: vec![],
        };
        let a = canonical_catalog_sha256(&[first.clone(), second.clone()]);
        let b = canonical_catalog_sha256(&[second, first]);
        assert_eq!(a, b);
    }

    #[test]
    #[ignore = "requires the reviewed official DeepSeek V4 index and 48 header prefixes"]
    fn official_header_prefixes_match_index_and_catalog() {
        use std::fs::File;
        use std::io::Read;
        use std::path::{Path, PathBuf};

        fn read_prefix(path: &Path) -> Vec<u8> {
            let mut file = File::open(path).expect("open shard/header prefix");
            let mut length = [0_u8; 8];
            file.read_exact(&mut length).expect("read header length");
            let header_length = u64::from_le_bytes(length);
            let total = usize::try_from(header_length + 8).expect("header prefix fits");
            let mut bytes = vec![0_u8; total];
            bytes[..8].copy_from_slice(&length);
            file.read_exact(&mut bytes[8..]).expect("read header JSON");
            bytes
        }

        let root = PathBuf::from(std::env::var_os("SLLM_DEEPSEEK_V4_METADATA_DIR").expect(
            "set SLLM_DEEPSEEK_V4_METADATA_DIR to headers and model.safetensors.index.json",
        ));
        let mut headers = Vec::with_capacity(DEEPSEEK_V4_SHARD_COUNT);
        for identity in DEEPSEEK_V4_HEADER_IDENTITIES {
            let short = format!(
                "model-{:05}.header",
                identity.file_name[6..11].parse::<u32>().unwrap()
            );
            let candidates = [
                root.join(identity.file_name),
                root.join(format!("{}.header", identity.file_name)),
                root.join(format!("{}.prefix", identity.file_name)),
                root.join(short.clone()),
                root.join(short.replace(".header", ".prefix")),
            ];
            let path = candidates
                .iter()
                .find(|candidate| candidate.is_file())
                .unwrap_or_else(|| panic!("missing header prefix for {}", identity.file_name));
            headers.push(
                parse_deepseek_v4_safetensors_header(identity.file_name, &read_prefix(path))
                    .expect("validate official header"),
            );
        }
        let index_bytes = std::fs::read(root.join("model.safetensors.index.json"))
            .expect("read official safetensors index");
        let index =
            crate::validate_deepseek_v4_index(&index_bytes).expect("validate official index");
        let catalog = validate_deepseek_v4_header_catalog(&headers, &index)
            .expect("validate official header catalog");
        assert_eq!(catalog.tensor_count(), DEEPSEEK_V4_TENSOR_COUNT);
        assert_eq!(catalog.payload_bytes(), DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES);
        assert_eq!(catalog.catalog_sha256(), DEEPSEEK_V4_HEADER_CATALOG_SHA256);
        assert_eq!(catalog.shards().len(), DEEPSEEK_V4_SHARD_COUNT);
    }
}
