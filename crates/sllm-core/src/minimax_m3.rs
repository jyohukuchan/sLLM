//! Reviewed source identity and host-only foundation contract for MiniMax M3.
//!
//! The official index metadata and the sum of the 59 Hub shard file sizes do
//! not agree. This module preserves that mismatch and always uses the larger
//! value for admission. Hub LFS OIDs below are remote identities; they are not
//! evidence that the full 854 GB shard set was downloaded and hashed locally.

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MINIMAX_M3_REPOSITORY: &str = "MiniMaxAI/MiniMax-M3";
pub const MINIMAX_M3_REVISION: &str = "f0e1c1e04d40177e4673a22097036854f536e9c0";
pub const MINIMAX_M3_LICENSE: &str = "MiniMax Community License";
pub const MINIMAX_M3_CONFIG_SHA256: &str =
    "c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b";
pub const MINIMAX_M3_INDEX_SHA256: &str =
    "54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e";
pub const MINIMAX_M3_CATALOG_SHA256: &str =
    "c19d8d997ac75026125a2766e27642e0d9f4d9dfea3e0ac7f046eea63cb1ee7b";
pub const MINIMAX_M3_CONFIG_BYTES: usize = 5_254;
pub const MINIMAX_M3_INDEX_BYTES: usize = 2_706_437;
pub const MINIMAX_M3_SHARD_COUNT: usize = 59;
pub const MINIMAX_M3_TENSOR_COUNT: usize = 23_416;
pub const MINIMAX_M3_SHARD_FILE_BYTES: u64 = 854_176_398_808;
/// Tensor payload bytes derived from the exact 59 safetensors headers.
pub const MINIMAX_M3_TENSOR_PAYLOAD_BYTES: u64 = 854_172_958_720;
/// The inconsistent total advertised by the fixed index metadata.
pub const MINIMAX_M3_INDEX_ADVERTISED_BYTES: u64 = 869_157_697_024;
pub const MINIMAX_M3_CAPACITY_ADMISSION_BYTES: u64 =
    if MINIMAX_M3_SHARD_FILE_BYTES > MINIMAX_M3_INDEX_ADVERTISED_BYTES {
        MINIMAX_M3_SHARD_FILE_BYTES
    } else {
        MINIMAX_M3_INDEX_ADVERTISED_BYTES
    };
pub const MINIMAX_M3_INDEX_METADATA_BYTES: u64 = MINIMAX_M3_INDEX_ADVERTISED_BYTES;
pub const MINIMAX_M3_MANIFEST_DELTA_BYTES: u64 = 14_981_298_216;
pub const MINIMAX_M3_ADMISSION_BASE_BYTES: u64 = MINIMAX_M3_CAPACITY_ADMISSION_BYTES;
pub const MINIMAX_M3_TEXT_LAYER_COUNT: u32 = 60;
pub const MINIMAX_M3_DENSE_LAYER_COUNT: u32 = 3;
pub const MINIMAX_M3_MOE_LAYER_COUNT: u32 = 57;
pub const MINIMAX_M3_MTP_MODULE_COUNT: u32 = 7;

const TEXT_ROOT_TENSOR_COUNT: usize = 3;
const DENSE_TEXT_TENSOR_COUNT: usize = 33;
const MOE_TEXT_TENSOR_COUNT: usize = 22_857;
const VISION_TENSOR_COUNT: usize = 515;
const MULTIMODAL_PROJECTOR_TENSOR_COUNT: usize = 4;
const PATCH_MERGE_PROJECTOR_TENSOR_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3ShardIdentity {
    pub file_name: &'static str,
    pub size: u64,
    /// Git LFS SHA-256 OID reported by the fixed-revision Hub API.
    pub lfs_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const MINIMAX_M3_SHARDS: [MiniMaxM3ShardIdentity; MINIMAX_M3_SHARD_COUNT] = [
    MiniMaxM3ShardIdentity {
        file_name: "model-00001-of-00059.safetensors",
        size: 5_583_706_344,
        lfs_sha256: "dd43571ea444cb0c966f2b2523d437252f81145f204062aea46521e202c32c9f",
        indexed_tensor_count: 14,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00002-of-00059.safetensors",
        size: 10_922_058_704,
        lfs_sha256: "f00337b2ad18c2d7ea3ede488ac40578b6faa4118bb8cfd5767ace9cbdadc26c",
        indexed_tensor_count: 277,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00003-of-00059.safetensors",
        size: 16_079_479_008,
        lfs_sha256: "db80e5edf8169e3c5cc098e032d556381c1a84b1738f84ac376ed7b39cafffb5",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00004-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "97990b9a6477674a32acbbcb8b9ae5ead5eff93e882e31c8da36c95b3e1c62b8",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00005-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "109c0245ed614bf8502edc1e83ef9575ec4aa210cb99fc34bf2db03245f3c305",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00006-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "d787b2366131f8fb9175aec5bb986d0b3f3fa3a6433b73124b089fe78a765bbb",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00007-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "c6ec7f70426ab2ee7decd771c1db66f231a57110fd382dc07c8a00d2b28b79a5",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00008-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "6463a6c079529088be571dff7d49714e5ba53e8be91d0365c85861f8e24813c5",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00009-of-00059.safetensors",
        size: 16_079_479_000,
        lfs_sha256: "b40f0eb7c619e133498a5e7c08cb5dc19e7d7ba410bbec95604af2d7a60560ea",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00010-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "dda2800d133243292ddc513c6840965719bf9be56b8e0688f1e0976f9a1f8718",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00011-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "2b9034b9465cc74d7597f772bebb2ab2a11927bccbafe3c945d83fe416b18b31",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00012-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "428bd21d923d485be8f3a9c07c05f11d4cbb17663b116501f7ce70920928e0ae",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00013-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "63e7bccc9bfde81758dbbc8f3c1d489265e5e7b8090cbd3c680f716169238d02",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00014-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "2bb43609c1f96762ff37e1ea96e5de13042e7ca449f456b6f8e8177adf3c7a6e",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00015-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "e8a996b3b76b09d956f38c37ec7a0849068ac4cf42da2b449055d6ab452d422a",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00016-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "468d354425f54ad3f8f4f434d611fa27c1164e5362610acf2e58a491c41157f5",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00017-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "6698280d2df1837e76674b1b93825b397390c80ff4790f5a3cc12fb555e4b86e",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00018-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "52982932f6fb23d4805317bf9785582b9e6db13d034669f45d8b186df7110b6e",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00019-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "999ed59667d5546aa47ed447c63d3a7523e1c011c0406c3db27248fb88789aa8",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00020-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "465b4c6a0b2aa62ee77364d885d25fc05348704fc721baec8a68be99ce96b2a9",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00021-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "10f62f292fe4914b80ba3b1304c88152d25dbb811a89ead2609ce7f831e80fc4",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00022-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "7a2a1e556b9dfe3915ceb28a0a6394372957c63063f692c740349a74b6aa177a",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00023-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "f8294c392c4daff5deca1fd3c0c22de13bbc79078a18986e8865a49527dbe275",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00024-of-00059.safetensors",
        size: 16_079_479_408,
        lfs_sha256: "5c54a7513a720e7d1a9709be756d5f78fa97d14b0e7ce9877494cebad681852f",
        indexed_tensor_count: 435,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00025-of-00059.safetensors",
        size: 5_245_548_280,
        lfs_sha256: "06562ebf1ab9a4ebf1c048dec2de812ca4bec77f1250b868dc90c72d96fe1137",
        indexed_tensor_count: 146,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00026-of-00059.safetensors",
        size: 15_302_516_656,
        lfs_sha256: "92d4b0199b5bfc57337d15ab057d9056bad11d37788274a1727a69d929d7caac",
        indexed_tensor_count: 408,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00027-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "3d707b6c6f9240204119bc781fd2d4f0640866c02de75a5b9d504aad73003c18",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00028-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "713c7a51b4dfaef7f36016e5f33fd0af6aa2842a0c424786e9163b2ceeeecb63",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00029-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "1c912118399e767c7eaf5d7096d78a91610df7a67ee1d11de141777d2a5c3797",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00030-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "20553bba3fd11ae151b2160c1b33f6dc5a7da0a6e9c6bd408fb300fcde4727eb",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00031-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "03d00987b21643b8a5821641dd829473433c9b0b8f98b48732ca25fcf27c5e48",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00032-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "a6bc4b5a14c34d6ddd6b784f2dd5c358d142bb7b50087329ab07b124461899b5",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00033-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "316fe078370010b9d95ff1d9bc7c21c6198610da8ed5d2d84d0415fc24bee642",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00034-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "c7d5bd0cb3e429d2e11ccf040d8c82790ca6c8937b25c0c882f91a56a3ad9edf",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00035-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "34e3b0020de4f56cdc68d1906ede8ee7948d6089eea4dc32aa753d7d56a49197",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00036-of-00059.safetensors",
        size: 14_833_765_536,
        lfs_sha256: "e2de321c02ddb663df351366d872e49fdc64a98cd18d5b58c30ab49a12f70b06",
        indexed_tensor_count: 401,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00037-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "42f199fd8def96ccdb7f171d46b88111ccdf65ac9118d681e37f5d85db3e02a4",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00038-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "8f647d846544f81871e5784777d59bfe7a1c2122539ff3c7dd86d45fba0fa005",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00039-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "decc8cd7c6f6ee1e7bb0cbc50bead1941090586382e024cf5e0ad193f60e4315",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00040-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "a109ca52fadfc1bd1fd3edc47e00145165c06cdbfc20c56e0e3ff0d9dae65873",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00041-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "f6964e4e2e412e66f6ab05889b242b7b042ebb648a569d695e2743de93c78ac2",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00042-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "911aa6743f831e9da2c7184a66bbc52d3989594fdb7a6b798ee94dc70e274ce7",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00043-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "bba6ca788435039fb3294090ae22369604c553fef57e40ae9ef9ed7f6f4e5850",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00044-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "ad708e127bb80ec97c25b9b01b58ddd490ebd6b3543e049a961a66cc58ed2e78",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00045-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "52436bdfb36c9404d337a824ca22688bdfdfbccfd5c32fc1ccfb3eb3473f14a9",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00046-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "c2773bd5375b2167a20e67b8e1daeee41f7249db06a96661194308d834b07a18",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00047-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "5c934370fa7c89da714c54c7a14d6db3562a9683ff48177ed18c95cdc94fe3c4",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00048-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "7b249f72d8097d2fd22f953da724045e4c8b84ec6455795ccae00474d15da3de",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00049-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "4b365073baee65d2c871437dab1fe76bee18444dbace55268d1999901eae2cd6",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00050-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "fc921404f960056d6ce64cc10dfc5509e881c6ea37be5b7c36980a0b71d37882",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00051-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "4902133d12d3eab6abfb872acb71ea19818387407f38468ea78b20616920894f",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00052-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "4cc81c60ee68e7a6a64e650f26918504382de2664bb7506152e86be2c1b4a159",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00053-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "427a70077722b14e0139b802dda584ec37abe219797347edbf79953f40a78f5e",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00054-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "cd78d9721e4e25f6f1bd9e55e575cba05006584569f50cb55de84393980b4867",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00055-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "889899c0b05447f84f4685267ce28b4eb7bc6bca8c51d9763eaa36214237851a",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00056-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "0254d37fb8adf83c1978319f1b567674f009780a77cf6ad68eb6c73e650cf084",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00057-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "ed20e5acdb3df314d240e4cc1af286a9e0a4831d7ee59f1eb324dfcf6fdcecf8",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00058-of-00059.safetensors",
        size: 13_588_051_672,
        lfs_sha256: "84c7dc7e7c8cbe32a1bb879983740ed4583d89ebb52585d090fe0e2908a07ec9",
        indexed_tensor_count: 367,
    },
    MiniMaxM3ShardIdentity {
        file_name: "model-00059-of-00059.safetensors",
        size: 16_099_232_552,
        lfs_sha256: "719c66930446908e42fd4f34535c375fe677f450276421ab506a97a219b10013",
        indexed_tensor_count: 917,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxM3ManifestState {
    Consistent {
        bytes: u64,
    },
    IndexMetadataExceedsShardFiles {
        index_metadata_bytes: u64,
        shard_file_bytes: u64,
        delta_bytes: u64,
    },
    ShardFilesExceedIndexMetadata {
        index_metadata_bytes: u64,
        shard_file_bytes: u64,
        delta_bytes: u64,
    },
}

impl MiniMaxM3ManifestState {
    pub const fn admission_base_bytes(self) -> u64 {
        match self {
            Self::Consistent { bytes } => bytes,
            Self::IndexMetadataExceedsShardFiles {
                index_metadata_bytes,
                shard_file_bytes,
                delta_bytes: _,
            }
            | Self::ShardFilesExceedIndexMetadata {
                index_metadata_bytes,
                shard_file_bytes,
                delta_bytes: _,
            } => {
                if index_metadata_bytes > shard_file_bytes {
                    index_metadata_bytes
                } else {
                    shard_file_bytes
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3ArtifactEvidence {
    pub hub_lfs_identity_fixed: bool,
    pub local_full_payload_sha256_verified: bool,
}

pub const MINIMAX_M3_ARTIFACT_EVIDENCE: MiniMaxM3ArtifactEvidence = MiniMaxM3ArtifactEvidence {
    hub_lfs_identity_fixed: true,
    local_full_payload_sha256_verified: false,
};

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3SparseAttentionConfig {
    pub index_dimension: u32,
    pub index_heads: u32,
    pub top_k_blocks: u32,
    pub block_size: u32,
    pub init_blocks: u32,
    pub local_blocks: u32,
    pub score_type: &'static str,
    pub enabled_layers: Vec<bool>,
    pub index_value_disabled_layers: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3TextConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_epsilon: f64,
    pub rotary_dimension: u32,
    pub rope_theta: u32,
    pub dense_intermediate_size: u32,
    pub shared_intermediate_size: u32,
    pub expert_count: u32,
    pub selected_expert_count: u32,
    pub shared_expert_count: u32,
    pub moe_layers: Vec<bool>,
    pub mtp_module_count: u32,
    pub nextn_predict_layers: u32,
    pub swiglu_alpha: f64,
    pub swiglu_limit: f64,
    pub routed_scaling_factor: f64,
    pub sparse_attention: MiniMaxM3SparseAttentionConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3VisionConfig {
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub layer_count: u32,
    pub intermediate_size: u32,
    pub patch_size: u32,
    pub image_size: u32,
    pub projection_dimension: u32,
    pub rope_mode: &'static str,
    pub rope_theta: f64,
    pub max_frames: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3MultimodalConfig {
    pub image_grid_count: u32,
    pub image_sequence_length: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub spatial_merge_size: u32,
    pub temporal_patch_size: u32,
    pub projector_hidden_size: u32,
    pub production_execution_enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3Config {
    pub text: MiniMaxM3TextConfig,
    pub vision: MiniMaxM3VisionConfig,
    pub multimodal: MiniMaxM3MultimodalConfig,
    /// The config advertises seven modules while the reviewed index contains
    /// no distinct MTP tensor family. This remains an identity/graph contract.
    pub indexed_mtp_tensor_count: usize,
    pub mtp_production_execution_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MiniMaxM3TensorSummary {
    pub text_root: usize,
    pub dense_text_layers: usize,
    pub moe_text_layers: usize,
    pub vision: usize,
    pub multimodal_projector: usize,
    pub patch_merge_projector: usize,
    pub mtp: usize,
}

impl MiniMaxM3TensorSummary {
    pub fn checked_total(self) -> Result<usize, MiniMaxM3ModelError> {
        self.text_root
            .checked_add(self.dense_text_layers)
            .and_then(|value| value.checked_add(self.moe_text_layers))
            .and_then(|value| value.checked_add(self.vision))
            .and_then(|value| value.checked_add(self.multimodal_projector))
            .and_then(|value| value.checked_add(self.patch_merge_projector))
            .and_then(|value| value.checked_add(self.mtp))
            .ok_or_else(|| invalid("tensor classification count overflowed"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxM3TensorClass {
    TextRoot,
    DenseTextLayer,
    MoeTextLayer,
    Vision,
    MultimodalProjector,
    PatchMergeProjector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3Index {
    index_metadata_bytes: u64,
    shard_file_bytes: u64,
    manifest_state: MiniMaxM3ManifestState,
    catalog_sha256: String,
    summary: MiniMaxM3TensorSummary,
    weight_map: BTreeMap<String, String>,
}

impl MiniMaxM3Index {
    pub const fn index_metadata_bytes(&self) -> u64 {
        self.index_metadata_bytes
    }

    pub const fn shard_file_bytes(&self) -> u64 {
        self.shard_file_bytes
    }

    pub const fn manifest_state(&self) -> MiniMaxM3ManifestState {
        self.manifest_state
    }

    pub const fn summary(&self) -> MiniMaxM3TensorSummary {
        self.summary
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn source_file(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &str)> {
        self.weight_map
            .iter()
            .map(|(name, shard)| (name.as_str(), shard.as_str()))
    }

    pub fn checked_admission_bytes(
        &self,
        resident_copy_count: u64,
        additional_bytes: u64,
    ) -> Result<u64, MiniMaxM3ModelError> {
        if resident_copy_count == 0 {
            return Err(invalid("resident copy count must be nonzero"));
        }
        self.index_metadata_bytes
            .max(self.shard_file_bytes)
            .checked_mul(resident_copy_count)
            .and_then(|value| value.checked_add(additional_bytes))
            .ok_or_else(|| invalid("admission byte count overflowed"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3CapacityDecision {
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub fits: bool,
    pub shortfall_bytes: u64,
    pub manifest_state: MiniMaxM3ManifestState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MiniMaxM3ModelError {
    Invalid(String),
}

impl fmt::Display for MiniMaxM3ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid MiniMax M3 artifact: {message}"),
        }
    }
}

impl std::error::Error for MiniMaxM3ModelError {}

fn invalid(message: impl Into<String>) -> MiniMaxM3ModelError {
    MiniMaxM3ModelError::Invalid(message.into())
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), MiniMaxM3ModelError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_locked_document(
    bytes: &[u8],
    expected_len: usize,
    expected_sha256: &str,
    label: &str,
) -> Result<(), MiniMaxM3ModelError> {
    ensure(
        bytes.len() == expected_len,
        format!(
            "{label} byte length {} does not match reviewed {expected_len}",
            bytes.len()
        ),
    )?;
    let actual = sha256_hex(bytes);
    ensure(
        actual == expected_sha256,
        format!("{label} SHA-256 {actual} does not match reviewed {expected_sha256}"),
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAutoMap {
    #[serde(rename = "AutoConfig")]
    auto_config: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawImageCompressionConfig {
    image_token_compression_method: String,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSparseAttentionConfig {
    use_sparse_attention: bool,
    sparse_index_dim: u32,
    sparse_num_index_heads: u32,
    sparse_topk_blocks: u32,
    sparse_block_size: u32,
    sparse_disable_index_value: Vec<u8>,
    sparse_score_type: String,
    sparse_init_block: u32,
    sparse_local_block: u32,
    sparse_attention_freq: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextConfig {
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f64,
    use_gemma_norm: bool,
    attention_output_gate: bool,
    rope_theta: u32,
    rotary_dim: u32,
    partial_rotary_factor: f64,
    hidden_act: String,
    use_qk_norm: bool,
    tie_word_embeddings: bool,
    dense_intermediate_size: u32,
    shared_intermediate_size: u32,
    num_local_experts: u32,
    num_experts_per_tok: u32,
    n_shared_experts: u32,
    scoring_func: String,
    use_routing_bias: bool,
    moe_layer_freq: Vec<u8>,
    qk_norm_type: String,
    num_mtp_modules: u32,
    num_nextn_predict_layers: u32,
    swiglu_alpha: f64,
    swiglu_limit: f64,
    routed_scaling_factor: f64,
    sparse_attention_config: RawSparseAttentionConfig,
    architectures: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVisionConfig {
    hidden_size: u32,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    intermediate_size: u32,
    patch_size: u32,
    image_size: u32,
    projection_dim: u32,
    position_embedding_type: String,
    rope_mode: String,
    rope_theta: f64,
    attention_dropout: f64,
    hidden_act: String,
    initializer_factor: f64,
    initializer_range: f64,
    layer_norm_eps: f64,
    model_type: String,
    num_channels: u32,
    vocab_size: u32,
    img_token_compression_config: RawImageCompressionConfig,
    vision_segment_max_frames: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    architectures: Vec<String>,
    auto_map: RawAutoMap,
    model_type: String,
    text_config: RawTextConfig,
    vision_config: RawVisionConfig,
    img_token_compression_config: RawImageCompressionConfig,
    image_grid_pinpoints: String,
    image_seq_length: u32,
    image_token_index: u32,
    video_token_index: u32,
    multimodal_projector_bias: bool,
    num_reward_heads: u32,
    process_image_mode: String,
    projector_hidden_act: String,
    vision_feature_layer: i32,
    vision_feature_select_strategy: String,
    torch_dtype: String,
    transformers_version: String,
    projector_hidden_size: u32,
}

const REVIEWED_IMAGE_GRID_PINPOINTS: &str = "[(336, 336), (336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016)]";

fn is_reviewed_schedule(values: &[u8]) -> bool {
    values.len() == MINIMAX_M3_TEXT_LAYER_COUNT as usize
        && values[..MINIMAX_M3_DENSE_LAYER_COUNT as usize]
            .iter()
            .all(|&value| value == 0)
        && values[MINIMAX_M3_DENSE_LAYER_COUNT as usize..]
            .iter()
            .all(|&value| value == 1)
}

fn same_f64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
}

fn validate_config_document(raw: RawConfig) -> Result<MiniMaxM3Config, MiniMaxM3ModelError> {
    ensure(
        raw.architectures == ["MiniMaxM3SparseForConditionalGeneration"],
        "root architecture changed",
    )?;
    ensure(
        raw.auto_map.auto_config == "configuration_minimax_m3_vl.MiniMaxM3VLConfig",
        "AutoConfig mapping changed",
    )?;
    ensure(raw.model_type == "minimax_m3_vl", "root model type changed")?;

    let text = raw.text_config;
    ensure(
        text.architectures == ["MiniMaxM3SparseForCausalLM"],
        "text architecture changed",
    )?;
    ensure(text.hidden_size == 6_144, "text hidden size changed")?;
    ensure(
        text.intermediate_size == 3_072,
        "text intermediate size changed",
    )?;
    ensure(
        text.num_hidden_layers == MINIMAX_M3_TEXT_LAYER_COUNT,
        "text layer count changed",
    )?;
    ensure(
        text.num_attention_heads == 64,
        "attention head count changed",
    )?;
    ensure(text.num_key_value_heads == 4, "KV head count changed")?;
    ensure(text.head_dim == 128, "attention head dimension changed")?;
    ensure(text.vocab_size == 200_064, "text vocabulary size changed")?;
    ensure(
        text.max_position_embeddings == 1_048_576,
        "text context length changed",
    )?;
    ensure(
        same_f64(text.rms_norm_eps, 1e-6),
        "RMS norm epsilon changed",
    )?;
    ensure(
        text.use_gemma_norm,
        "Gemma normalization must remain enabled",
    )?;
    ensure(
        !text.attention_output_gate,
        "attention output gate must remain disabled",
    )?;
    ensure(text.rope_theta == 5_000_000, "RoPE theta changed")?;
    ensure(text.rotary_dim == 64, "rotary dimension changed")?;
    ensure(
        same_f64(text.partial_rotary_factor, 0.5),
        "partial rotary factor changed",
    )?;
    ensure(text.hidden_act == "swigluoai", "text activation changed")?;
    ensure(text.use_qk_norm, "QK normalization must remain enabled")?;
    ensure(
        !text.tie_word_embeddings,
        "word embeddings must remain untied",
    )?;
    ensure(
        text.dense_intermediate_size == 12_288,
        "dense intermediate size changed",
    )?;
    ensure(
        text.shared_intermediate_size == 3_072,
        "shared expert intermediate size changed",
    )?;
    ensure(text.num_local_experts == 128, "expert count changed")?;
    ensure(
        text.num_experts_per_tok == 4,
        "selected expert count changed",
    )?;
    ensure(text.n_shared_experts == 1, "shared expert count changed")?;
    ensure(
        text.scoring_func == "sigmoid",
        "MoE scoring function changed",
    )?;
    ensure(
        text.use_routing_bias,
        "MoE routing bias must remain enabled",
    )?;
    ensure(
        is_reviewed_schedule(&text.moe_layer_freq),
        "MoE schedule must be dense layers 0..2 then MoE layers 3..59",
    )?;
    ensure(text.qk_norm_type == "per_head", "QK norm type changed")?;
    ensure(
        text.num_mtp_modules == MINIMAX_M3_MTP_MODULE_COUNT,
        "MTP module count changed",
    )?;
    ensure(
        text.num_nextn_predict_layers == 1,
        "next-token prediction layer count changed",
    )?;
    ensure(same_f64(text.swiglu_alpha, 1.702), "SwiGLU alpha changed")?;
    ensure(same_f64(text.swiglu_limit, 7.0), "SwiGLU limit changed")?;
    ensure(
        same_f64(text.routed_scaling_factor, 2.0),
        "routed expert scaling factor changed",
    )?;

    let sparse = text.sparse_attention_config;
    ensure(sparse.use_sparse_attention, "MSA must remain enabled")?;
    ensure(
        sparse.sparse_index_dim == 128,
        "MSA index dimension changed",
    )?;
    ensure(
        sparse.sparse_num_index_heads == 4,
        "MSA index head count changed",
    )?;
    ensure(sparse.sparse_topk_blocks == 16, "MSA top-k changed")?;
    ensure(sparse.sparse_block_size == 128, "MSA block size changed")?;
    ensure(
        is_reviewed_schedule(&sparse.sparse_disable_index_value),
        "MSA index-value schedule changed",
    )?;
    ensure(sparse.sparse_score_type == "max", "MSA score type changed")?;
    ensure(
        sparse.sparse_init_block == 0,
        "MSA initial block count changed",
    )?;
    ensure(
        sparse.sparse_local_block == 1,
        "MSA local block count changed",
    )?;
    ensure(
        is_reviewed_schedule(&sparse.sparse_attention_freq),
        "MSA layer schedule changed",
    )?;
    ensure(
        sparse.sparse_attention_freq == text.moe_layer_freq,
        "MSA and MoE layer schedules diverged",
    )?;
    ensure(
        sparse.sparse_disable_index_value == sparse.sparse_attention_freq,
        "MSA index-value and attention schedules diverged",
    )?;

    ensure(
        text.num_attention_heads % text.num_key_value_heads == 0,
        "attention heads are not divisible by KV heads",
    )?;
    let derived_rotary = (text.head_dim as f64 * text.partial_rotary_factor) as u32;
    ensure(
        derived_rotary == text.rotary_dim,
        "rotary dimension is inconsistent with head dimension",
    )?;
    ensure(
        text.num_experts_per_tok <= text.num_local_experts,
        "selected expert count exceeds expert count",
    )?;
    ensure(
        sparse.sparse_num_index_heads == text.num_key_value_heads,
        "MSA index heads do not match KV heads",
    )?;
    ensure(
        sparse.sparse_index_dim == text.head_dim,
        "MSA index dimension does not match attention head dimension",
    )?;

    let vision = raw.vision_config;
    ensure(vision.hidden_size == 1_280, "vision hidden size changed")?;
    ensure(
        vision.num_attention_heads == 16,
        "vision head count changed",
    )?;
    ensure(vision.num_hidden_layers == 32, "vision layer count changed")?;
    ensure(
        vision.intermediate_size == 5_120,
        "vision intermediate size changed",
    )?;
    ensure(vision.patch_size == 14, "vision patch size changed")?;
    ensure(vision.image_size == 2_016, "vision image size changed")?;
    ensure(
        vision.projection_dim == 6_144,
        "vision projection size changed",
    )?;
    ensure(
        vision.position_embedding_type == "rope",
        "vision position embedding type changed",
    )?;
    ensure(vision.rope_mode == "3d", "vision RoPE mode changed")?;
    ensure(
        same_f64(vision.rope_theta, 10_000.0),
        "vision RoPE theta changed",
    )?;
    ensure(
        same_f64(vision.attention_dropout, 0.0),
        "vision attention dropout changed",
    )?;
    ensure(vision.hidden_act == "gelu", "vision activation changed")?;
    ensure(
        same_f64(vision.initializer_factor, 1.0),
        "vision initializer factor changed",
    )?;
    ensure(
        same_f64(vision.initializer_range, 0.02),
        "vision initializer range changed",
    )?;
    ensure(
        same_f64(vision.layer_norm_eps, 1e-5),
        "vision layer norm epsilon changed",
    )?;
    ensure(
        vision.model_type == "clip_vision_model",
        "vision model type changed",
    )?;
    ensure(vision.num_channels == 3, "vision channel count changed")?;
    ensure(
        vision.vocab_size == 32_000,
        "vision vocabulary size changed",
    )?;
    ensure(
        vision.vision_segment_max_frames == 4,
        "vision maximum frame count changed",
    )?;
    ensure(
        vision.image_size % vision.patch_size == 0,
        "vision image size is not patch aligned",
    )?;

    let compression = &raw.img_token_compression_config;
    ensure(
        compression == &vision.img_token_compression_config,
        "root and vision image compression configs diverged",
    )?;
    ensure(
        compression.image_token_compression_method == "patch_merge",
        "image compression method changed",
    )?;
    ensure(
        compression.spatial_merge_size == 2,
        "spatial merge size changed",
    )?;
    ensure(
        compression.temporal_patch_size == 2,
        "temporal patch size changed",
    )?;
    ensure(
        raw.image_grid_pinpoints == REVIEWED_IMAGE_GRID_PINPOINTS,
        "image grid pinpoints changed",
    )?;
    ensure(raw.image_seq_length == 576, "image sequence length changed")?;
    ensure(raw.image_token_index == 200_025, "image token ID changed")?;
    ensure(raw.video_token_index == 200_026, "video token ID changed")?;
    ensure(
        raw.image_token_index < text.vocab_size && raw.video_token_index < text.vocab_size,
        "multimodal special token is outside the text vocabulary",
    )?;
    ensure(
        raw.multimodal_projector_bias,
        "multimodal projector bias must remain enabled",
    )?;
    ensure(raw.num_reward_heads == 0, "reward head count changed")?;
    ensure(
        raw.process_image_mode == "dynamic_res",
        "image process mode changed",
    )?;
    ensure(
        raw.projector_hidden_act == "gelu",
        "projector activation changed",
    )?;
    ensure(
        raw.vision_feature_layer == -1,
        "vision feature layer changed",
    )?;
    ensure(
        raw.vision_feature_select_strategy == "full",
        "vision feature selection strategy changed",
    )?;
    ensure(
        raw.torch_dtype == "bfloat16",
        "declared torch dtype changed",
    )?;
    ensure(
        raw.transformers_version == "4.52.4",
        "declared Transformers version changed",
    )?;
    ensure(
        raw.projector_hidden_size == 6_144,
        "projector hidden size changed",
    )?;
    ensure(
        raw.projector_hidden_size == text.hidden_size && vision.projection_dim == text.hidden_size,
        "vision/text projector dimensions diverged",
    )?;

    let moe_layers = text
        .moe_layer_freq
        .iter()
        .map(|&value| value == 1)
        .collect();
    let enabled_layers = sparse
        .sparse_attention_freq
        .iter()
        .map(|&value| value == 1)
        .collect();
    let index_value_disabled_layers = sparse
        .sparse_disable_index_value
        .iter()
        .map(|&value| value == 1)
        .collect();

    Ok(MiniMaxM3Config {
        text: MiniMaxM3TextConfig {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            layer_count: text.num_hidden_layers,
            attention_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            rms_norm_epsilon: text.rms_norm_eps,
            rotary_dimension: text.rotary_dim,
            rope_theta: text.rope_theta,
            dense_intermediate_size: text.dense_intermediate_size,
            shared_intermediate_size: text.shared_intermediate_size,
            expert_count: text.num_local_experts,
            selected_expert_count: text.num_experts_per_tok,
            shared_expert_count: text.n_shared_experts,
            moe_layers,
            mtp_module_count: text.num_mtp_modules,
            nextn_predict_layers: text.num_nextn_predict_layers,
            swiglu_alpha: text.swiglu_alpha,
            swiglu_limit: text.swiglu_limit,
            routed_scaling_factor: text.routed_scaling_factor,
            sparse_attention: MiniMaxM3SparseAttentionConfig {
                index_dimension: sparse.sparse_index_dim,
                index_heads: sparse.sparse_num_index_heads,
                top_k_blocks: sparse.sparse_topk_blocks,
                block_size: sparse.sparse_block_size,
                init_blocks: sparse.sparse_init_block,
                local_blocks: sparse.sparse_local_block,
                score_type: "max",
                enabled_layers,
                index_value_disabled_layers,
            },
        },
        vision: MiniMaxM3VisionConfig {
            hidden_size: vision.hidden_size,
            attention_heads: vision.num_attention_heads,
            layer_count: vision.num_hidden_layers,
            intermediate_size: vision.intermediate_size,
            patch_size: vision.patch_size,
            image_size: vision.image_size,
            projection_dimension: vision.projection_dim,
            rope_mode: "3d",
            rope_theta: vision.rope_theta,
            max_frames: vision.vision_segment_max_frames,
        },
        multimodal: MiniMaxM3MultimodalConfig {
            image_grid_count: 36,
            image_sequence_length: raw.image_seq_length,
            image_token_id: raw.image_token_index,
            video_token_id: raw.video_token_index,
            spatial_merge_size: compression.spatial_merge_size,
            temporal_patch_size: compression.temporal_patch_size,
            projector_hidden_size: raw.projector_hidden_size,
            production_execution_enabled: false,
        },
        indexed_mtp_tensor_count: 0,
        mtp_production_execution_enabled: false,
    })
}

/// Validates the exact reviewed config bytes and returns the typed foundation contract.
pub fn validate_minimax_m3_config(bytes: &[u8]) -> Result<MiniMaxM3Config, MiniMaxM3ModelError> {
    validate_locked_document(
        bytes,
        MINIMAX_M3_CONFIG_BYTES,
        MINIMAX_M3_CONFIG_SHA256,
        "config",
    )?;
    let raw: RawConfig = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("config JSON is not strict-valid: {error}")))?;
    validate_config_document(raw)
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
                formatter.write_str("a tensor-to-shard object with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((name, shard)) = map.next_entry::<String, String>()? {
                    if values.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name {name}"
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
    total_size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndex {
    metadata: RawIndexMetadata,
    weight_map: UniqueWeightMap,
}

fn parse_canonical_index(value: &str, limit: u32, label: &str) -> Result<u32, MiniMaxM3ModelError> {
    ensure(!value.is_empty(), format!("missing {label}"))?;
    ensure(
        value == "0" || !value.starts_with('0'),
        format!("{label} is not canonical decimal: {value}"),
    )?;
    ensure(
        value.bytes().all(|byte| byte.is_ascii_digit()),
        format!("{label} is not decimal: {value}"),
    )?;
    let parsed = value
        .parse::<u32>()
        .map_err(|_| invalid(format!("{label} overflowed: {value}")))?;
    ensure(
        parsed < limit,
        format!("{label} {parsed} is out of range 0..{limit}"),
    )?;
    Ok(parsed)
}

fn validate_tensor_name_characters(name: &str) -> Result<(), MiniMaxM3ModelError> {
    ensure(!name.is_empty(), "tensor name is empty")?;
    ensure(
        !name.starts_with('.') && !name.ends_with('.') && !name.contains(".."),
        format!("tensor name has an empty path component: {name}"),
    )?;
    ensure(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.'),
        format!("tensor name contains a forbidden character: {name}"),
    )
}

const TEXT_COMMON_SUFFIXES: [&str; 8] = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];
const DENSE_SUFFIXES: [&str; 3] = [
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
];
const MSA_SUFFIXES: [&str; 4] = [
    "self_attn.index_k_norm.weight",
    "self_attn.index_k_proj.weight",
    "self_attn.index_q_norm.weight",
    "self_attn.index_q_proj.weight",
];
const MOE_DIRECT_SUFFIXES: [&str; 5] = [
    "block_sparse_moe.e_score_correction_bias",
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
];
const VISION_LAYER_SUFFIXES: [&str; 16] = [
    "layer_norm1.bias",
    "layer_norm1.weight",
    "layer_norm2.bias",
    "layer_norm2.weight",
    "mlp.fc1.bias",
    "mlp.fc1.weight",
    "mlp.fc2.bias",
    "mlp.fc2.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
];

fn classify_text_layer_suffix(
    layer: u32,
    suffix: &str,
) -> Result<MiniMaxM3TensorClass, MiniMaxM3ModelError> {
    let class = if layer < MINIMAX_M3_DENSE_LAYER_COUNT {
        MiniMaxM3TensorClass::DenseTextLayer
    } else {
        MiniMaxM3TensorClass::MoeTextLayer
    };
    if TEXT_COMMON_SUFFIXES.contains(&suffix) {
        return Ok(class);
    }
    if layer < MINIMAX_M3_DENSE_LAYER_COUNT {
        ensure(
            DENSE_SUFFIXES.contains(&suffix),
            format!("unknown dense layer tensor suffix: {suffix}"),
        )?;
        return Ok(class);
    }
    if MSA_SUFFIXES.contains(&suffix) || MOE_DIRECT_SUFFIXES.contains(&suffix) {
        return Ok(class);
    }
    let routed = suffix
        .strip_prefix("block_sparse_moe.experts.")
        .ok_or_else(|| invalid(format!("unknown MoE layer tensor suffix: {suffix}")))?;
    let (expert, projection) = routed
        .split_once('.')
        .ok_or_else(|| invalid(format!("malformed routed expert tensor suffix: {suffix}")))?;
    parse_canonical_index(expert, 128, "expert index")?;
    ensure(
        matches!(projection, "w1.weight" | "w2.weight" | "w3.weight"),
        format!("unknown routed expert projection: {projection}"),
    )?;
    Ok(class)
}

/// Parses a reviewed MiniMax M3 tensor name with exact grammar and index ranges.
pub fn classify_minimax_m3_tensor(name: &str) -> Result<MiniMaxM3TensorClass, MiniMaxM3ModelError> {
    validate_tensor_name_characters(name)?;
    if matches!(
        name,
        "language_model.lm_head.weight"
            | "language_model.model.embed_tokens.weight"
            | "language_model.model.norm.weight"
    ) {
        return Ok(MiniMaxM3TensorClass::TextRoot);
    }
    if let Some(rest) = name.strip_prefix("language_model.model.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed text layer tensor: {name}")))?;
        let layer = parse_canonical_index(layer, MINIMAX_M3_TEXT_LAYER_COUNT, "text layer")?;
        return classify_text_layer_suffix(layer, suffix);
    }
    if matches!(
        name,
        "vision_tower.vision_model.embeddings.patch_embedding.weight"
            | "vision_tower.vision_model.pre_layrnorm.bias"
            | "vision_tower.vision_model.pre_layrnorm.weight"
    ) {
        return Ok(MiniMaxM3TensorClass::Vision);
    }
    if let Some(rest) = name.strip_prefix("vision_tower.vision_model.encoder.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed vision layer tensor: {name}")))?;
        parse_canonical_index(layer, 32, "vision layer")?;
        ensure(
            VISION_LAYER_SUFFIXES.contains(&suffix),
            format!("unknown vision layer tensor suffix: {suffix}"),
        )?;
        return Ok(MiniMaxM3TensorClass::Vision);
    }
    if matches!(
        name,
        "multi_modal_projector.linear_1.bias"
            | "multi_modal_projector.linear_1.weight"
            | "multi_modal_projector.linear_2.bias"
            | "multi_modal_projector.linear_2.weight"
    ) {
        return Ok(MiniMaxM3TensorClass::MultimodalProjector);
    }
    if matches!(
        name,
        "patch_merge_mlp.linear_1.bias"
            | "patch_merge_mlp.linear_1.weight"
            | "patch_merge_mlp.linear_2.bias"
            | "patch_merge_mlp.linear_2.weight"
    ) {
        return Ok(MiniMaxM3TensorClass::PatchMergeProjector);
    }
    Err(invalid(format!("unknown tensor family: {name}")))
}

/// Returns the fixed-revision Hub identity for an exact canonical shard name.
pub fn minimax_m3_locked_shard(file_name: &str) -> Option<&'static MiniMaxM3ShardIdentity> {
    MINIMAX_M3_SHARDS
        .iter()
        .find(|shard| shard.file_name == file_name)
}

/// Validates a shard name, file size, and Hub-reported LFS SHA-256 identity.
pub fn validate_minimax_m3_shard_lfs_identity(
    file_name: &str,
    size: u64,
    lfs_sha256: &str,
) -> Result<&'static MiniMaxM3ShardIdentity, MiniMaxM3ModelError> {
    ensure(
        !file_name.contains('/') && !file_name.contains('\\') && !file_name.contains(".."),
        format!("shard path is not a canonical base name: {file_name}"),
    )?;
    let shard = minimax_m3_locked_shard(file_name)
        .ok_or_else(|| invalid(format!("unknown shard file: {file_name}")))?;
    ensure(
        shard.size == size,
        format!(
            "shard {file_name} size {size} does not match reviewed {}",
            shard.size
        ),
    )?;
    ensure(
        lfs_sha256.len() == 64
            && lfs_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            && lfs_sha256.bytes().all(|byte| !byte.is_ascii_uppercase()),
        format!("shard {file_name} LFS OID is not lowercase SHA-256"),
    )?;
    ensure(
        shard.lfs_sha256 == lfs_sha256,
        format!("shard {file_name} LFS OID does not match reviewed identity"),
    )?;
    Ok(shard)
}

pub fn checked_minimax_m3_shard_file_bytes() -> Result<u64, MiniMaxM3ModelError> {
    MINIMAX_M3_SHARDS.iter().try_fold(0_u64, |total, shard| {
        total
            .checked_add(shard.size)
            .ok_or_else(|| invalid("shard file byte total overflowed"))
    })
}

pub fn minimax_m3_manifest_state(
    index_metadata_bytes: u64,
    shard_file_bytes: u64,
) -> MiniMaxM3ManifestState {
    match index_metadata_bytes.cmp(&shard_file_bytes) {
        std::cmp::Ordering::Equal => MiniMaxM3ManifestState::Consistent {
            bytes: index_metadata_bytes,
        },
        std::cmp::Ordering::Greater => MiniMaxM3ManifestState::IndexMetadataExceedsShardFiles {
            index_metadata_bytes,
            shard_file_bytes,
            delta_bytes: index_metadata_bytes - shard_file_bytes,
        },
        std::cmp::Ordering::Less => MiniMaxM3ManifestState::ShardFilesExceedIndexMetadata {
            index_metadata_bytes,
            shard_file_bytes,
            delta_bytes: shard_file_bytes - index_metadata_bytes,
        },
    }
}

fn increment_summary(
    summary: &mut MiniMaxM3TensorSummary,
    class: MiniMaxM3TensorClass,
) -> Result<(), MiniMaxM3ModelError> {
    let value = match class {
        MiniMaxM3TensorClass::TextRoot => &mut summary.text_root,
        MiniMaxM3TensorClass::DenseTextLayer => &mut summary.dense_text_layers,
        MiniMaxM3TensorClass::MoeTextLayer => &mut summary.moe_text_layers,
        MiniMaxM3TensorClass::Vision => &mut summary.vision,
        MiniMaxM3TensorClass::MultimodalProjector => &mut summary.multimodal_projector,
        MiniMaxM3TensorClass::PatchMergeProjector => &mut summary.patch_merge_projector,
    };
    *value = value
        .checked_add(1)
        .ok_or_else(|| invalid("tensor classification count overflowed"))?;
    Ok(())
}

fn validate_summary(summary: MiniMaxM3TensorSummary) -> Result<(), MiniMaxM3ModelError> {
    ensure(
        summary.text_root == TEXT_ROOT_TENSOR_COUNT,
        "text root coverage changed",
    )?;
    ensure(
        summary.dense_text_layers == DENSE_TEXT_TENSOR_COUNT,
        "dense text tensor coverage changed",
    )?;
    ensure(
        summary.moe_text_layers == MOE_TEXT_TENSOR_COUNT,
        "MoE text tensor coverage changed",
    )?;
    ensure(
        summary.vision == VISION_TENSOR_COUNT,
        "vision tensor coverage changed",
    )?;
    ensure(
        summary.multimodal_projector == MULTIMODAL_PROJECTOR_TENSOR_COUNT,
        "multimodal projector coverage changed",
    )?;
    ensure(
        summary.patch_merge_projector == PATCH_MERGE_PROJECTOR_TENSOR_COUNT,
        "patch-merge projector coverage changed",
    )?;
    ensure(
        summary.mtp == 0,
        "reviewed index unexpectedly contains MTP tensors",
    )?;
    ensure(
        summary.checked_total()? == MINIMAX_M3_TENSOR_COUNT,
        "classified tensor total changed",
    )
}

fn extract_layer(name: &str, prefix: &str, limit: u32) -> Option<u32> {
    let rest = name.strip_prefix(prefix)?;
    let (index, _) = rest.split_once('.')?;
    parse_canonical_index(index, limit, "coverage layer").ok()
}

fn catalog_sha256(weight_map: &BTreeMap<String, String>) -> Result<String, MiniMaxM3ModelError> {
    let mut hasher = Sha256::new();
    for (name, shard) in weight_map {
        let row = serde_json::to_vec(&(name, shard))
            .map_err(|error| invalid(format!("catalog row serialization failed: {error}")))?;
        hasher.update(row);
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_index_document(raw: RawIndex) -> Result<MiniMaxM3Index, MiniMaxM3ModelError> {
    ensure(
        raw.metadata.total_size == MINIMAX_M3_INDEX_ADVERTISED_BYTES,
        format!(
            "index advertised bytes {} do not match reviewed {}",
            raw.metadata.total_size, MINIMAX_M3_INDEX_ADVERTISED_BYTES
        ),
    )?;
    ensure(
        raw.weight_map.0.len() == MINIMAX_M3_TENSOR_COUNT,
        format!(
            "index tensor count {} does not match reviewed {}",
            raw.weight_map.0.len(),
            MINIMAX_M3_TENSOR_COUNT
        ),
    )?;

    let mut summary = MiniMaxM3TensorSummary::default();
    let mut shard_counts = BTreeMap::<&str, usize>::new();
    let mut text_layers = BTreeSet::new();
    let mut vision_layers = BTreeSet::new();
    for (name, shard_name) in &raw.weight_map.0 {
        let class = classify_minimax_m3_tensor(name)?;
        increment_summary(&mut summary, class)?;
        if let Some(layer) = extract_layer(
            name,
            "language_model.model.layers.",
            MINIMAX_M3_TEXT_LAYER_COUNT,
        ) {
            text_layers.insert(layer);
        }
        if let Some(layer) = extract_layer(name, "vision_tower.vision_model.encoder.layers.", 32) {
            vision_layers.insert(layer);
        }
        ensure(
            !shard_name.contains('/') && !shard_name.contains('\\') && !shard_name.contains(".."),
            format!("index shard path is not a canonical base name: {shard_name}"),
        )?;
        let shard = minimax_m3_locked_shard(shard_name)
            .ok_or_else(|| invalid(format!("index references unknown shard: {shard_name}")))?;
        let count = shard_counts.entry(shard.file_name).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("per-shard tensor count overflowed"))?;
    }
    validate_summary(summary)?;
    ensure(
        text_layers.len() == MINIMAX_M3_TEXT_LAYER_COUNT as usize,
        "text layer coverage is incomplete",
    )?;
    ensure(
        vision_layers.len() == 32,
        "vision layer coverage is incomplete",
    )?;
    ensure(
        shard_counts.len() == MINIMAX_M3_SHARD_COUNT,
        "index shard coverage is incomplete",
    )?;
    for shard in &MINIMAX_M3_SHARDS {
        let actual = shard_counts.get(shard.file_name).copied().unwrap_or(0);
        ensure(
            actual == shard.indexed_tensor_count,
            format!(
                "shard {} indexes {actual} tensors, reviewed {}",
                shard.file_name, shard.indexed_tensor_count
            ),
        )?;
    }

    let shard_file_bytes = checked_minimax_m3_shard_file_bytes()?;
    ensure(
        shard_file_bytes == MINIMAX_M3_SHARD_FILE_BYTES,
        "locked shard byte total changed",
    )?;
    let manifest_state = minimax_m3_manifest_state(raw.metadata.total_size, shard_file_bytes);
    ensure(
        manifest_state
            == MiniMaxM3ManifestState::IndexMetadataExceedsShardFiles {
                index_metadata_bytes: MINIMAX_M3_INDEX_ADVERTISED_BYTES,
                shard_file_bytes: MINIMAX_M3_SHARD_FILE_BYTES,
                delta_bytes: MINIMAX_M3_MANIFEST_DELTA_BYTES,
            },
        "index/shard manifest mismatch changed",
    )?;
    let digest = catalog_sha256(&raw.weight_map.0)?;
    ensure(
        digest == MINIMAX_M3_CATALOG_SHA256,
        format!("catalog SHA-256 {digest} does not match reviewed {MINIMAX_M3_CATALOG_SHA256}"),
    )?;

    Ok(MiniMaxM3Index {
        index_metadata_bytes: raw.metadata.total_size,
        shard_file_bytes,
        manifest_state,
        catalog_sha256: digest,
        summary,
        weight_map: raw.weight_map.0,
    })
}

/// Validates the exact reviewed index bytes, its grammar, and complete manifest coverage.
pub fn validate_minimax_m3_index(bytes: &[u8]) -> Result<MiniMaxM3Index, MiniMaxM3ModelError> {
    validate_locked_document(
        bytes,
        MINIMAX_M3_INDEX_BYTES,
        MINIMAX_M3_INDEX_SHA256,
        "index",
    )?;
    let raw: RawIndex = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("index JSON is not strict-valid: {error}")))?;
    validate_index_document(raw)
}

/// Makes a fail-closed capacity decision using the larger manifest byte count.
pub fn minimax_m3_capacity_decision(
    index: &MiniMaxM3Index,
    available_bytes: u64,
    resident_copy_count: u64,
    additional_bytes: u64,
) -> Result<MiniMaxM3CapacityDecision, MiniMaxM3ModelError> {
    let required_bytes = index.checked_admission_bytes(resident_copy_count, additional_bytes)?;
    Ok(MiniMaxM3CapacityDecision {
        required_bytes,
        available_bytes,
        fits: available_bytes >= required_bytes,
        shortfall_bytes: required_bytes.saturating_sub(available_bytes),
        manifest_state: index.manifest_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn foundation_index() -> MiniMaxM3Index {
        let manifest_state = minimax_m3_manifest_state(
            MINIMAX_M3_INDEX_ADVERTISED_BYTES,
            MINIMAX_M3_SHARD_FILE_BYTES,
        );
        MiniMaxM3Index {
            index_metadata_bytes: MINIMAX_M3_INDEX_ADVERTISED_BYTES,
            shard_file_bytes: MINIMAX_M3_SHARD_FILE_BYTES,
            manifest_state,
            catalog_sha256: MINIMAX_M3_CATALOG_SHA256.to_owned(),
            summary: MiniMaxM3TensorSummary {
                text_root: TEXT_ROOT_TENSOR_COUNT,
                dense_text_layers: DENSE_TEXT_TENSOR_COUNT,
                moe_text_layers: MOE_TEXT_TENSOR_COUNT,
                vision: VISION_TENSOR_COUNT,
                multimodal_projector: MULTIMODAL_PROJECTOR_TENSOR_COUNT,
                patch_merge_projector: PATCH_MERGE_PROJECTOR_TENSOR_COUNT,
                mtp: 0,
            },
            weight_map: BTreeMap::new(),
        }
    }

    #[test]
    fn config_and_index_serde_are_strict() {
        let duplicate = br#"{
            "architectures": [],
            "architectures": []
        }"#;
        let error = serde_json::from_slice::<RawConfig>(duplicate)
            .expect_err("duplicate config key must fail")
            .to_string();
        assert!(error.contains("duplicate field `architectures`"), "{error}");

        let error = serde_json::from_slice::<RawConfig>(br#"{"unexpected": 1}"#)
            .expect_err("unknown config key must fail")
            .to_string();
        assert!(error.contains("unknown field `unexpected`"), "{error}");

        let error = serde_json::from_slice::<RawConfig>(br#"{}"#)
            .expect_err("missing config key must fail")
            .to_string();
        assert!(error.contains("missing field"), "{error}");

        let duplicate_weight = br#"{
            "metadata": {"total_size": 1},
            "weight_map": {"tensor.weight": "a", "tensor.weight": "b"}
        }"#;
        let error = serde_json::from_slice::<RawIndex>(duplicate_weight)
            .expect_err("duplicate tensor key must fail")
            .to_string();
        assert!(
            error.contains("duplicate tensor name tensor.weight"),
            "{error}"
        );

        let overflow = br#"{
            "metadata": {"total_size": 18446744073709551616},
            "weight_map": {}
        }"#;
        assert!(serde_json::from_slice::<RawIndex>(overflow).is_err());
    }

    #[test]
    fn reviewed_schedule_covers_dense_and_sparse_boundaries() {
        let mut schedule = vec![1_u8; MINIMAX_M3_TEXT_LAYER_COUNT as usize];
        schedule[..MINIMAX_M3_DENSE_LAYER_COUNT as usize].fill(0);
        assert!(is_reviewed_schedule(&schedule));

        for length in [59, 61] {
            let mut changed = schedule.clone();
            changed.resize(length, 1);
            assert!(!is_reviewed_schedule(&changed));
        }
        for (index, value) in [(2, 1), (3, 0), (59, 2)] {
            let mut changed = schedule.clone();
            changed[index] = value;
            assert!(!is_reviewed_schedule(&changed));
        }
    }

    #[test]
    fn exact_tensor_grammar_checks_architecture_boundaries() {
        let accepted = [
            (
                "language_model.lm_head.weight",
                MiniMaxM3TensorClass::TextRoot,
            ),
            (
                "language_model.model.layers.2.mlp.up_proj.weight",
                MiniMaxM3TensorClass::DenseTextLayer,
            ),
            (
                "language_model.model.layers.3.self_attn.index_q_proj.weight",
                MiniMaxM3TensorClass::MoeTextLayer,
            ),
            (
                "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight",
                MiniMaxM3TensorClass::MoeTextLayer,
            ),
            (
                "language_model.model.layers.59.block_sparse_moe.experts.127.w3.weight",
                MiniMaxM3TensorClass::MoeTextLayer,
            ),
            (
                "vision_tower.vision_model.encoder.layers.31.self_attn.out_proj.bias",
                MiniMaxM3TensorClass::Vision,
            ),
            (
                "multi_modal_projector.linear_2.weight",
                MiniMaxM3TensorClass::MultimodalProjector,
            ),
            (
                "patch_merge_mlp.linear_1.bias",
                MiniMaxM3TensorClass::PatchMergeProjector,
            ),
        ];
        for (name, expected) in accepted {
            assert_eq!(classify_minimax_m3_tensor(name), Ok(expected), "{name}");
        }

        for name in [
            "language_model.model.layers.02.mlp.up_proj.weight",
            "language_model.model.layers.3.mlp.up_proj.weight",
            "language_model.model.layers.2.self_attn.index_q_proj.weight",
            "language_model.model.layers.60.input_layernorm.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.128.w1.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.01.w1.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.0.w4.weight",
            "language_model.model.layers.3.mtp.0.weight",
            "vision_tower.vision_model.encoder.layers.32.layer_norm1.weight",
            "../model.weight",
            "model//weight",
        ] {
            assert!(classify_minimax_m3_tensor(name).is_err(), "accepted {name}");
        }
    }

    #[test]
    fn locked_shards_reject_paths_and_identity_changes() {
        assert_eq!(MINIMAX_M3_SHARDS.len(), MINIMAX_M3_SHARD_COUNT);
        assert_eq!(
            checked_minimax_m3_shard_file_bytes().expect("sum locked shard bytes"),
            MINIMAX_M3_SHARD_FILE_BYTES
        );
        assert_eq!(
            MINIMAX_M3_SHARDS
                .iter()
                .map(|shard| shard.indexed_tensor_count)
                .sum::<usize>(),
            MINIMAX_M3_TENSOR_COUNT
        );
        let first = MINIMAX_M3_SHARDS[0];
        assert_eq!(
            validate_minimax_m3_shard_lfs_identity(first.file_name, first.size, first.lfs_sha256),
            Ok(&first)
        );
        assert!(
            validate_minimax_m3_shard_lfs_identity(
                "../model-00001-of-00059.safetensors",
                first.size,
                first.lfs_sha256
            )
            .is_err()
        );
        assert!(
            validate_minimax_m3_shard_lfs_identity(
                first.file_name,
                first.size + 1,
                first.lfs_sha256
            )
            .is_err()
        );
        assert!(
            validate_minimax_m3_shard_lfs_identity(first.file_name, first.size, &"0".repeat(64))
                .is_err()
        );
    }

    #[test]
    fn manifest_mismatch_is_typed_and_admission_uses_the_larger_side() {
        let official = minimax_m3_manifest_state(
            MINIMAX_M3_INDEX_ADVERTISED_BYTES,
            MINIMAX_M3_SHARD_FILE_BYTES,
        );
        assert_eq!(
            official,
            MiniMaxM3ManifestState::IndexMetadataExceedsShardFiles {
                index_metadata_bytes: MINIMAX_M3_INDEX_ADVERTISED_BYTES,
                shard_file_bytes: MINIMAX_M3_SHARD_FILE_BYTES,
                delta_bytes: MINIMAX_M3_MANIFEST_DELTA_BYTES,
            }
        );
        assert_eq!(
            official.admission_base_bytes(),
            MINIMAX_M3_CAPACITY_ADMISSION_BYTES
        );

        let reverse = minimax_m3_manifest_state(7, 11);
        assert_eq!(
            reverse,
            MiniMaxM3ManifestState::ShardFilesExceedIndexMetadata {
                index_metadata_bytes: 7,
                shard_file_bytes: 11,
                delta_bytes: 4,
            }
        );
        assert_eq!(reverse.admission_base_bytes(), 11);
        assert_eq!(minimax_m3_manifest_state(13, 13).admission_base_bytes(), 13);
    }

    #[test]
    fn capacity_decision_checks_boundaries_and_overflow() {
        let index = foundation_index();
        let required = MINIMAX_M3_CAPACITY_ADMISSION_BYTES + 17;
        let below = minimax_m3_capacity_decision(&index, required - 1, 1, 17)
            .expect("finite capacity decision");
        assert!(!below.fits);
        assert_eq!(below.shortfall_bytes, 1);
        let exact = minimax_m3_capacity_decision(&index, required, 1, 17)
            .expect("boundary capacity decision");
        assert!(exact.fits);
        assert_eq!(exact.shortfall_bytes, 0);
        assert_eq!(exact.required_bytes, required);
        assert!(minimax_m3_capacity_decision(&index, u64::MAX, 0, 0).is_err());
        assert!(minimax_m3_capacity_decision(&index, u64::MAX, u64::MAX, 0).is_err());
        assert!(minimax_m3_capacity_decision(&index, u64::MAX, 1, u64::MAX).is_err());
    }

    #[test]
    fn tensor_summary_coverage_is_fail_closed() {
        let valid = foundation_index().summary;
        validate_summary(valid).expect("reviewed family coverage");
        let mut missing_vision = valid;
        missing_vision.vision -= 1;
        missing_vision.mtp += 1;
        assert!(validate_summary(missing_vision).is_err());
        assert!(
            MiniMaxM3TensorSummary {
                text_root: usize::MAX,
                dense_text_layers: 1,
                ..Default::default()
            }
            .checked_total()
            .is_err()
        );
    }

    fn metadata_file(directory: &Path, candidates: &[&str]) -> PathBuf {
        candidates
            .iter()
            .map(|candidate| directory.join(candidate))
            .find(|path| path.is_file())
            .unwrap_or_else(|| {
                panic!(
                    "none of the metadata candidates exist under {}: {candidates:?}",
                    directory.display()
                )
            })
    }

    #[test]
    #[ignore = "requires fixed-revision MiniMax M3 metadata via SLLM_MINIMAX_M3_METADATA_DIR"]
    fn official_metadata_matches_locked_foundation() {
        let directory = PathBuf::from(
            std::env::var("SLLM_MINIMAX_M3_METADATA_DIR")
                .expect("set SLLM_MINIMAX_M3_METADATA_DIR"),
        );
        let config_path = metadata_file(&directory, &["config.json"]);
        let index_path = metadata_file(&directory, &["model.safetensors.index.json", "index.json"]);
        let config = validate_minimax_m3_config(
            &fs::read(&config_path).expect("read fixed-revision config"),
        )
        .expect("validate fixed-revision config");
        let index =
            validate_minimax_m3_index(&fs::read(&index_path).expect("read fixed-revision index"))
                .expect("validate fixed-revision index");

        assert_eq!(config.text.layer_count, 60);
        assert_eq!(
            config
                .text
                .moe_layers
                .iter()
                .filter(|&&value| value)
                .count(),
            57
        );
        assert_eq!(config.text.mtp_module_count, 7);
        assert_eq!(config.indexed_mtp_tensor_count, 0);
        assert!(!config.mtp_production_execution_enabled);
        assert!(!config.multimodal.production_execution_enabled);
        assert_eq!(index.tensor_count(), MINIMAX_M3_TENSOR_COUNT);
        assert_eq!(index.catalog_sha256(), MINIMAX_M3_CATALOG_SHA256);
        assert_eq!(index.summary().mtp, 0);
        assert_eq!(
            index.manifest_state().admission_base_bytes(),
            MINIMAX_M3_CAPACITY_ADMISSION_BYTES
        );
    }
}
