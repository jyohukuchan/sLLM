//! Reviewed source identity and host-side container contract for DeepSeek V4 Flash.
//!
//! This module does not make the checkpoint production-loadable. It freezes the
//! official configuration, Hugging Face LFS shard identities, and safetensors
//! index catalog so later conversion and execution work can fail closed before
//! downloading or allocating the 166.9 GB tensor payload.

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DEEPSEEK_V4_REPOSITORY: &str = "deepseek-ai/DeepSeek-V4-Flash-0731";
pub const DEEPSEEK_V4_REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
pub const DEEPSEEK_V4_LICENSE: &str = "MIT";
pub const DEEPSEEK_V4_CONFIG_SHA256: &str =
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023";
pub const DEEPSEEK_V4_INDEX_SHA256: &str =
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b";
pub const DEEPSEEK_V4_CATALOG_SHA256: &str =
    "189e5e292be34b1378ac96dafd0e6255d582aaae28a1558a21f0d629938d5c8c";
pub const DEEPSEEK_V4_CONFIG_BYTES: usize = 1_888;
pub const DEEPSEEK_V4_INDEX_BYTES: usize = 5_602_871;
pub const DEEPSEEK_V4_TENSOR_COUNT: usize = 72_317;
pub const DEEPSEEK_V4_SHARD_COUNT: usize = 48;
pub const DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES: u64 = 166_878_536_440;
pub const DEEPSEEK_V4_SHARD_FILE_BYTES: u64 = 166_886_535_336;
pub const DEEPSEEK_V4_MAIN_LAYER_COUNT: u32 = 43;
pub const DEEPSEEK_V4_HASH_ROUTED_LAYER_COUNT: u32 = 3;
pub const DEEPSEEK_V4_CONFIG_NEXTN_PREDICT_LAYER_COUNT: u32 = 1;
pub const DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT: u32 = 3;

const ROOT_TENSOR_COUNT: usize = 6;
const HASH_LAYER_TENSOR_COUNT: usize = 4_706;
const NEXT_TOKEN_LAYER_TENSOR_COUNT: usize = 58_179;
const DSPARK_TARGET_LAYER_TENSOR_COUNT: usize = 4_721;
const DSPARK_STAGE_TENSOR_COUNT: usize = 4_705;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ShardIdentity {
    pub file_name: &'static str,
    /// Complete Git LFS object size, including the safetensors header.
    pub size: u64,
    /// Git LFS SHA-256 OID. This is not evidence that local payload bytes were read.
    pub lfs_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const DEEPSEEK_V4_SHARDS: [DeepSeekV4ShardIdentity; DEEPSEEK_V4_SHARD_COUNT] = [
    DeepSeekV4ShardIdentity {
        file_name: "model-00001-of-00048.safetensors",
        size: 1_059_061_856,
        lfs_sha256: "f3668ba4cccf1ca6a7eb84e888fb92c1cdc7204d472ba9db771e6fd3abf6b874",
        indexed_tensor_count: 1,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00002-of-00048.safetensors",
        size: 3_566_321_192,
        lfs_sha256: "77b26c939a0e25b3113c8d6bb04e1901a748bd4a7d2589e3bfdaabdf1e9bba14",
        indexed_tensor_count: 1_565,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00003-of-00048.safetensors",
        size: 3_566_321_192,
        lfs_sha256: "412abf4c906faadc221ef0cb50f90fe20bde8454a08ad4dc2364b6b79e7fda5c",
        indexed_tensor_count: 1_565,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00004-of-00048.safetensors",
        size: 3_596_229_272,
        lfs_sha256: "9610f56bc587fb0ff9a8b68a60299482ee8c433fe5b5587e4257aca98add4a2e",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00005-of-00048.safetensors",
        size: 3_568_768_976,
        lfs_sha256: "f87a5ac7b8becc31f9c3169afd3a6f33fb82b4af9e21022e3755a10bc28f0180",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00006-of-00048.safetensors",
        size: 3_590_024_776,
        lfs_sha256: "4a4f3764e3fc772b9fba67f0a44ef68e18f178b6f00faa80b75db549e51894cd",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00007-of-00048.safetensors",
        size: 3_568_768_976,
        lfs_sha256: "df81bb80e27a689e01fa579eebd6499f86e0b6105f7fea18961aa5eebbbee9bc",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00008-of-00048.safetensors",
        size: 3_590_024_776,
        lfs_sha256: "224968d2b27f8669365ec08657a768dfec40da0585f85f302a31495931f6a526",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00009-of-00048.safetensors",
        size: 3_568_768_976,
        lfs_sha256: "04d69ef1071fff8721c62968c200a5583122b59b015e9ef9b2978bfed271b2b7",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00010-of-00048.safetensors",
        size: 3_590_024_776,
        lfs_sha256: "627145f4ebeb1cc3f5bdd03416b8cb7370b3c96974853cbeb8e5516ad5713e49",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00011-of-00048.safetensors",
        size: 3_568_768_976,
        lfs_sha256: "e4b8e601dcbebe902e0102e7b098b670a121cb8b9564dd719fc41d782c8416e0",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00012-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "64ed4e5f6126ba029c462c9d5fca0fc907c5f855b4ba01194d79560f6db16e42",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00013-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "8dfe199d07c07ddd141c2c0136a2237f1161250a1a03ebe8deaabac93440da1d",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00014-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "45db2f540f825f92453c50335e49aede58cca56bc578d1787c12a0fbca6593e5",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00015-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "5810381a0f05b7381c002d299ed6ac19e42eba8070dd17e2703546944d84f292",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00016-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "e0530b7024771b0ce2df9b40bcc2232578f3300178487ec216863b0b2835617b",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00017-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "ed11130247118b185ade893c0109bad896dd394cb1e066ce4fce044176261d94",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00018-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "e393fea96da2a3414ef089354fc32e1c8891954de40958c84d2c2ecf80365b25",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00019-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "a74ca4d3e8e82ce20c458bb8b1900110b753793ad4c58d08f38995f719c616f7",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00020-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "9f556769926e60309e8defe45ab59fc8b26ae460d30c190cd746a3d78c11e2c2",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00021-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "1671cce7f90d781f796b5ca6bf32dd1aeb740abcb2735e41ffd28f62485ce005",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00022-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "decd67a4bd97a75fa36861d2ad3067afeefa6a04a20da997fe6c19f171e70132",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00023-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "c61a3e179cdbee19bfb8cbc4e111928ce2f1e1f0f4729d7c0cd5634354a4689d",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00024-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "fc27aeb4233534f6f7781dcfe57127a3908ae10fc025c5d86dc0682057f8b2fe",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00025-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "a66b6b8d5821b68f5b511e4f91e12025cd07d0fa6d0b71e722d825a2d6d878ca",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00026-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "657b89314fbaf6eee4acce24b3baf7e5fd2c5986a96ad85b08d90539cde869fe",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00027-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "fb01f21a0da0446b0bdf25a127ab19a6b06006acd8735f06a9ebfe34423fd7f5",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00028-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "b2fd5cbbb639f16e673bc484e5cca16b52a58bf2ab4bd62592e0c5408712ad7c",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00029-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "9ec2fdf900275daeac0980490c5c731cc7868b151ce1de5698f48418de4fa5f0",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00030-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "9ed3c317bf967d32133ad3a068ee4c56aae9784bc8b7da694482437f37dc1782",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00031-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "d5078c3fca3e6370043606ead7856e0b8fe67a9aab52c415769f29934c4d7f5d",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00032-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "163653848f002718d3deaa6ce48885483fc1f2e12e50e44a47477c73ccd91393",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00033-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "f2cffd43f2a5f491f4691f8694e6bc08239158e143ae7063dc04f0eb0259214a",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00034-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "0f94945121474cfdb6a9ab175914d3811ffbf08e6cc54082e14e473c755d18d8",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00035-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "9cb6a316989f7c7385e3ec2bd42ffe766ee126c70ef3466742849982ee1b0f0f",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00036-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "7e6761421fe944c2143eb897b983085891b421c148c6c17fb5cd8eaa9bdaa497",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00037-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "a59d662f1143596d56c452a1230b717ce43edf678207398c573a2503b0f72c91",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00038-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "137fa617a74ba8e73fd76bb1010c7a85d791aaa150006cb66faa04a83e9e730f",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00039-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "a29af1aa519d7ce726235ea2c2b38146d756290cda7a82d90c4d4438155b53e4",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00040-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "8bc93d8a7d1987dc86b14e22b1d8f42ec31da92c56edd9f312daf43f33a6a206",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00041-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "fd312e7fdd6cb5796df356a7f0314f124851dc149991e9fb02c5bed45cc4ba05",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00042-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "4d19bf368083c9a183cb0849f316ec17b62f859ca824c0586e779657efb6e6a6",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00043-of-00048.safetensors",
        size: 3_568_770_544,
        lfs_sha256: "b7103842ceb70848f9804f55c193d6a57f43174a587cba42b61b5c1bc4e1303d",
        indexed_tensor_count: 1_569,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00044-of-00048.safetensors",
        size: 3_590_026_352,
        lfs_sha256: "422d3889fa20c238b7f97464c14df0bcf3328f189c294f41a3a334421dc560c7",
        indexed_tensor_count: 1_576,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00045-of-00048.safetensors",
        size: 1_059_332_516,
        lfs_sha256: "a5be6aed7b84fc87ec42b5d24ba0b0d67f253a3906fcd99c13f4f7be5958fc00",
        indexed_tensor_count: 5,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00046-of-00048.safetensors",
        size: 3_610_455_184,
        lfs_sha256: "5db924ca907e0d93acd975bd5079c3662717f9ac709f23d079bd8f816d29d9dd",
        indexed_tensor_count: 1_568,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00047-of-00048.safetensors",
        size: 3_560_111_960,
        lfs_sha256: "62816173f9f6e136b20b48e3b6f16613ac9ea02b5603f636928b253244a548bd",
        indexed_tensor_count: 1_565,
    },
    DeepSeekV4ShardIdentity {
        file_name: "model-00048-of-00048.safetensors",
        size: 3_692_775_244,
        lfs_sha256: "cc43742bd24ae6bcdea343a91442f6f66aed2cfebcc6b235470204851ce2f8a9",
        indexed_tensor_count: 1_572,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4Compression {
    Uncompressed,
    Csa4To1,
    Hca128To1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4Quantization {
    pub activation_dynamic: bool,
    pub value_format: &'static str,
    pub scale_format: &'static str,
    pub block_shape: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4YarnRope {
    pub factor: u32,
    pub original_max_position_embeddings: u32,
    pub beta_fast: u32,
    pub beta_slow: u32,
    pub theta: u32,
    pub compressed_theta: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4Config {
    pub hidden_size: u32,
    pub layer_count: u32,
    pub hash_layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub index_top_k: u32,
    pub expert_count: u32,
    pub selected_expert_count: u32,
    pub shared_expert_count: u32,
    pub expert_intermediate_size: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub sliding_window: u32,
    pub hc_multiplier: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub compression: Vec<DeepSeekV4Compression>,
    pub dspark_block_size: u32,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: [u32; 3],
    pub dspark_markov_rank: u32,
    pub next_token_prediction_layers: u32,
    pub rope: DeepSeekV4YarnRope,
    pub quantization: DeepSeekV4Quantization,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4TensorClass {
    NextTokenRoot,
    HashRoutedLayer,
    NextTokenLayer,
    DsparkTargetLayer,
    DsparkStage,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeepSeekV4TensorSummary {
    pub next_token_root: usize,
    pub hash_routed_layers: usize,
    pub next_token_layers: usize,
    pub dspark_target_layers: usize,
    pub dspark_stages: usize,
}

/// Layer identity keeps two similarly named upstream concepts distinct.
///
/// `num_nextn_predict_layers == 1` is the root config field, while the exact
/// checkpoint index contains three `mtp.*` DSpark stages. The reviewed artifact
/// requires this 1-versus-3 pairing; callers must not infer one from the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LayerSummary {
    pub main_layers: u32,
    pub hash_routed_main_layers: u32,
    pub config_nextn_predict_layers: u32,
    pub dspark_checkpoint_stages: u32,
    pub dspark_target_layer_ids: [u32; 3],
}

pub const fn deepseek_v4_layer_summary() -> DeepSeekV4LayerSummary {
    DeepSeekV4LayerSummary {
        main_layers: DEEPSEEK_V4_MAIN_LAYER_COUNT,
        hash_routed_main_layers: DEEPSEEK_V4_HASH_ROUTED_LAYER_COUNT,
        config_nextn_predict_layers: DEEPSEEK_V4_CONFIG_NEXTN_PREDICT_LAYER_COUNT,
        dspark_checkpoint_stages: DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT,
        dspark_target_layer_ids: [40, 41, 42],
    }
}

impl DeepSeekV4TensorSummary {
    pub fn checked_total(self) -> Result<usize, DeepSeekV4ModelError> {
        self.next_token_root
            .checked_add(self.hash_routed_layers)
            .and_then(|value| value.checked_add(self.next_token_layers))
            .and_then(|value| value.checked_add(self.dspark_target_layers))
            .and_then(|value| value.checked_add(self.dspark_stages))
            .ok_or_else(|| invalid("tensor classification count overflowed"))
    }

    fn checked_increment(
        &mut self,
        class: DeepSeekV4TensorClass,
    ) -> Result<(), DeepSeekV4ModelError> {
        let count = match class {
            DeepSeekV4TensorClass::NextTokenRoot => &mut self.next_token_root,
            DeepSeekV4TensorClass::HashRoutedLayer => &mut self.hash_routed_layers,
            DeepSeekV4TensorClass::NextTokenLayer => &mut self.next_token_layers,
            DeepSeekV4TensorClass::DsparkTargetLayer => &mut self.dspark_target_layers,
            DeepSeekV4TensorClass::DsparkStage => &mut self.dspark_stages,
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("tensor classification count overflowed"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4Index {
    total_size: u64,
    catalog_sha256: String,
    summary: DeepSeekV4TensorSummary,
    weight_map: BTreeMap<String, String>,
}

impl DeepSeekV4Index {
    pub const fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub const fn summary(&self) -> DeepSeekV4TensorSummary {
        self.summary
    }

    pub fn source_file(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    /// Iterate the exact reviewed source tensor catalog in canonical name
    /// order. Callers may derive bounded conversion metadata from the catalog,
    /// but this does not prove any shard payload byte was read locally.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &str)> {
        self.weight_map
            .iter()
            .map(|(name, shard)| (name.as_str(), shard.as_str()))
    }

    /// Checked lower-bound accounting for copies of the source tensor payload.
    /// Conversion padding, runtime workspaces, KV, and shard headers are excluded.
    pub fn checked_resident_bytes(
        &self,
        resident_copy_count: u64,
        additional_bytes: u64,
    ) -> Result<u64, DeepSeekV4ModelError> {
        if resident_copy_count == 0 {
            return Err(invalid("resident copy count must be nonzero"));
        }
        self.total_size
            .checked_mul(resident_copy_count)
            .and_then(|value| value.checked_add(additional_bytes))
            .ok_or_else(|| invalid("source payload resident bytes overflowed"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4ModelError {
    Invalid(String),
}

impl fmt::Display for DeepSeekV4ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid DeepSeek V4 artifact: {message}"),
        }
    }
}

impl std::error::Error for DeepSeekV4ModelError {}

fn invalid(message: impl Into<String>) -> DeepSeekV4ModelError {
    DeepSeekV4ModelError::Invalid(message.into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeepSeekV4Quantization {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    scale_fmt: String,
    weight_block_size: [u32; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeepSeekV4Rope {
    beta_fast: u32,
    beta_slow: u32,
    factor: u32,
    original_max_position_embeddings: u32,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeepSeekV4Config {
    architectures: Vec<String>,
    attention_bias: bool,
    attention_dropout: f64,
    bos_token_id: u32,
    eos_token_id: u32,
    expert_dtype: String,
    hc_eps: f64,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,
    head_dim: u32,
    hidden_act: String,
    hidden_size: u32,
    index_head_dim: u32,
    index_n_heads: u32,
    index_topk: u32,
    initializer_range: f64,
    max_position_embeddings: u32,
    model_type: String,
    moe_intermediate_size: u32,
    n_routed_experts: u32,
    n_shared_experts: u32,
    norm_topk_prob: bool,
    num_attention_heads: u32,
    num_experts_per_tok: u32,
    num_hash_layers: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_nextn_predict_layers: u32,
    o_groups: u32,
    o_lora_rank: u32,
    q_lora_rank: u32,
    qk_rope_head_dim: u32,
    quantization_config: RawDeepSeekV4Quantization,
    rms_norm_eps: f64,
    rope_scaling: RawDeepSeekV4Rope,
    rope_theta: u32,
    routed_scaling_factor: f64,
    scoring_func: String,
    sliding_window: u32,
    swiglu_limit: f64,
    tie_word_embeddings: bool,
    topk_method: String,
    torch_dtype: String,
    transformers_version: String,
    use_cache: bool,
    vocab_size: u32,
    compress_rope_theta: u32,
    compress_ratios: Vec<u32>,
    dspark_block_size: u32,
    dspark_noise_token_id: u32,
    dspark_target_layer_ids: [u32; 3],
    dspark_markov_rank: u32,
}

const REVIEWED_COMPRESSION_RATIOS: [u32; 46] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
];

fn same_f64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
}

fn checked_product(values: &[u64], label: &str) -> Result<u64, DeepSeekV4ModelError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{label} overflowed")))
    })
}

fn validate_config_document(
    raw: RawDeepSeekV4Config,
) -> Result<DeepSeekV4Config, DeepSeekV4ModelError> {
    if raw.architectures.as_slice() != ["DeepseekV4ForCausalLM"]
        || raw.model_type != "deepseek_v4"
        || raw.expert_dtype != "fp4"
        || raw.hidden_act != "silu"
        || raw.scoring_func != "sqrtsoftplus"
        || raw.topk_method != "noaux_tc"
        || raw.torch_dtype != "bfloat16"
        || raw.transformers_version != "4.57.1"
    {
        return Err(invalid("architecture or format identity differs"));
    }
    if raw.attention_bias
        || !same_f64(raw.attention_dropout, 0.0)
        || raw.bos_token_id != 0
        || raw.eos_token_id != 1
        || !raw.norm_topk_prob
        || raw.tie_word_embeddings
        || !raw.use_cache
    {
        return Err(invalid("token or boolean semantic field differs"));
    }
    if !same_f64(raw.hc_eps, 1.0e-6)
        || !same_f64(raw.initializer_range, 0.02)
        || !same_f64(raw.rms_norm_eps, 1.0e-6)
        || !same_f64(raw.routed_scaling_factor, 1.5)
        || !same_f64(raw.swiglu_limit, 10.0)
    {
        return Err(invalid("floating-point semantic field differs"));
    }
    for (actual, expected, label) in [
        (raw.hidden_size, 4_096, "hidden_size"),
        (raw.num_hidden_layers, 43, "num_hidden_layers"),
        (raw.num_hash_layers, 3, "num_hash_layers"),
        (raw.num_attention_heads, 64, "num_attention_heads"),
        (raw.num_key_value_heads, 1, "num_key_value_heads"),
        (raw.head_dim, 512, "head_dim"),
        (raw.qk_rope_head_dim, 64, "qk_rope_head_dim"),
        (raw.index_n_heads, 64, "index_n_heads"),
        (raw.index_head_dim, 128, "index_head_dim"),
        (raw.index_topk, 512, "index_topk"),
        (raw.n_routed_experts, 256, "n_routed_experts"),
        (raw.num_experts_per_tok, 6, "num_experts_per_tok"),
        (raw.n_shared_experts, 1, "n_shared_experts"),
        (raw.moe_intermediate_size, 2_048, "moe_intermediate_size"),
        (
            raw.max_position_embeddings,
            1_048_576,
            "max_position_embeddings",
        ),
        (raw.vocab_size, 129_280, "vocab_size"),
        (raw.sliding_window, 128, "sliding_window"),
        (raw.hc_mult, 4, "hc_mult"),
        (raw.hc_sinkhorn_iters, 20, "hc_sinkhorn_iters"),
        (raw.q_lora_rank, 1_024, "q_lora_rank"),
        (raw.o_lora_rank, 1_024, "o_lora_rank"),
        (raw.o_groups, 8, "o_groups"),
        (raw.num_nextn_predict_layers, 1, "num_nextn_predict_layers"),
        (raw.dspark_block_size, 5, "dspark_block_size"),
        (raw.dspark_noise_token_id, 128_799, "dspark_noise_token_id"),
        (raw.dspark_markov_rank, 256, "dspark_markov_rank"),
        (raw.rope_theta, 10_000, "rope_theta"),
        (raw.compress_rope_theta, 160_000, "compress_rope_theta"),
    ] {
        if actual != expected {
            return Err(invalid(format!("config field differs: {label}")));
        }
    }
    if raw.dspark_target_layer_ids != [40, 41, 42]
        || raw.compress_ratios.as_slice() != REVIEWED_COMPRESSION_RATIOS
        || raw.rope_scaling.kind != "yarn"
        || raw.rope_scaling.beta_fast != 32
        || raw.rope_scaling.beta_slow != 1
        || raw.rope_scaling.factor != 16
        || raw.rope_scaling.original_max_position_embeddings != 65_536
    {
        return Err(invalid("compression, DSpark, or YaRN schedule differs"));
    }
    let quant = &raw.quantization_config;
    if quant.activation_scheme != "dynamic"
        || quant.fmt != "e4m3"
        || quant.quant_method != "fp8"
        || quant.scale_fmt != "ue8m0"
        || quant.weight_block_size != [128, 128]
    {
        return Err(invalid("FP4/FP8 artifact recipe differs"));
    }

    // Hash routing applies to the first three main layers; it does not add
    // layers. The extra schedule rows correspond to the three checkpoint
    // DSpark stages, despite the separate root `num_nextn_predict_layers = 1`.
    let schedule_length = raw
        .num_hidden_layers
        .checked_add(DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT)
        .ok_or_else(|| invalid("compression schedule length overflowed"))?;
    if usize::try_from(schedule_length).ok() != Some(raw.compress_ratios.len())
        || raw.num_hash_layers > raw.num_hidden_layers
        || raw.num_hash_layers != DEEPSEEK_V4_HASH_ROUTED_LAYER_COUNT
        || raw.num_nextn_predict_layers != DEEPSEEK_V4_CONFIG_NEXTN_PREDICT_LAYER_COUNT
        || raw.dspark_target_layer_ids.len() != DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT as usize
        || raw.dspark_noise_token_id >= raw.vocab_size
        || raw.num_experts_per_tok > raw.n_routed_experts
        || raw.index_topk < raw.sliding_window
    {
        return Err(invalid("config dimensions are mutually inconsistent"));
    }
    if checked_product(
        &[u64::from(raw.hidden_size), u64::from(raw.hc_mult)],
        "mHC width",
    )? != 16_384
        || checked_product(
            &[u64::from(raw.num_attention_heads), u64::from(raw.head_dim)],
            "attention projection width",
        )? != 32_768
        || checked_product(
            &[u64::from(raw.index_n_heads), u64::from(raw.index_head_dim)],
            "compressed index width",
        )? != 8_192
        || checked_product(
            &[
                u64::from(raw.quantization_config.weight_block_size[0]),
                u64::from(raw.quantization_config.weight_block_size[1]),
            ],
            "quantization block elements",
        )? != 16_384
    {
        return Err(invalid("checked config dimension product differs"));
    }

    let compression = raw
        .compress_ratios
        .iter()
        .map(|ratio| match ratio {
            0 => Ok(DeepSeekV4Compression::Uncompressed),
            4 => Ok(DeepSeekV4Compression::Csa4To1),
            128 => Ok(DeepSeekV4Compression::Hca128To1),
            _ => Err(invalid("unsupported compression ratio")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DeepSeekV4Config {
        hidden_size: raw.hidden_size,
        layer_count: raw.num_hidden_layers,
        hash_layer_count: raw.num_hash_layers,
        attention_heads: raw.num_attention_heads,
        kv_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        index_heads: raw.index_n_heads,
        index_head_dim: raw.index_head_dim,
        index_top_k: raw.index_topk,
        expert_count: raw.n_routed_experts,
        selected_expert_count: raw.num_experts_per_tok,
        shared_expert_count: raw.n_shared_experts,
        expert_intermediate_size: raw.moe_intermediate_size,
        max_position_embeddings: raw.max_position_embeddings,
        vocab_size: raw.vocab_size,
        sliding_window: raw.sliding_window,
        hc_multiplier: raw.hc_mult,
        q_lora_rank: raw.q_lora_rank,
        o_lora_rank: raw.o_lora_rank,
        o_groups: raw.o_groups,
        compression,
        dspark_block_size: raw.dspark_block_size,
        dspark_noise_token_id: raw.dspark_noise_token_id,
        dspark_target_layer_ids: raw.dspark_target_layer_ids,
        dspark_markov_rank: raw.dspark_markov_rank,
        next_token_prediction_layers: raw.num_nextn_predict_layers,
        rope: DeepSeekV4YarnRope {
            factor: raw.rope_scaling.factor,
            original_max_position_embeddings: raw.rope_scaling.original_max_position_embeddings,
            beta_fast: raw.rope_scaling.beta_fast,
            beta_slow: raw.rope_scaling.beta_slow,
            theta: raw.rope_theta,
            compressed_theta: raw.compress_rope_theta,
        },
        quantization: DeepSeekV4Quantization {
            activation_dynamic: true,
            value_format: "E4M3",
            scale_format: "UE8M0",
            block_shape: quant.weight_block_size,
        },
    })
}

pub fn validate_deepseek_v4_config(bytes: &[u8]) -> Result<DeepSeekV4Config, DeepSeekV4ModelError> {
    if bytes.len() != DEEPSEEK_V4_CONFIG_BYTES || sha256(bytes) != DEEPSEEK_V4_CONFIG_SHA256 {
        return Err(invalid("config SHA-256 differs"));
    }
    let raw =
        serde_json::from_slice(bytes).map_err(|error| invalid(format!("config JSON: {error}")))?;
    validate_config_document(raw)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}

struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueWeightMapVisitor;

        impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor-name to shard-name object without duplicate keys")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some((name, shard)) = entries.next_entry::<String, String>()? {
                    if map.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name: {name}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(map))
            }
        }

        deserializer.deserialize_map(UniqueWeightMapVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexDocument {
    metadata: IndexMetadata,
    weight_map: UniqueWeightMap,
}

fn validate_tensor_name(name: &str) -> Result<(), DeepSeekV4ModelError> {
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

fn scoped_layer(name: &str, scope: &str) -> Result<u32, DeepSeekV4ModelError> {
    let mut fields = name.split('.');
    if fields.next() != Some(scope) {
        return Err(invalid(format!("tensor scope differs: {name}")));
    }
    let layer = fields
        .next()
        .ok_or_else(|| invalid(format!("tensor layer is absent: {name}")))?
        .parse::<u32>()
        .map_err(|_| invalid(format!("tensor layer is invalid: {name}")))?;
    if fields.next().is_none() {
        return Err(invalid(format!("tensor suffix is absent: {name}")));
    }
    Ok(layer)
}

pub fn classify_deepseek_v4_tensor(
    name: &str,
) -> Result<DeepSeekV4TensorClass, DeepSeekV4ModelError> {
    validate_tensor_name(name)?;
    if name.starts_with("layers.") {
        return match scoped_layer(name, "layers")? {
            0..=2 => Ok(DeepSeekV4TensorClass::HashRoutedLayer),
            3..=39 => Ok(DeepSeekV4TensorClass::NextTokenLayer),
            40..=42 => Ok(DeepSeekV4TensorClass::DsparkTargetLayer),
            layer => Err(invalid(format!(
                "main tensor layer is out of range: {layer}"
            ))),
        };
    }
    if name.starts_with("mtp.") {
        return match scoped_layer(name, "mtp")? {
            0..=2 => Ok(DeepSeekV4TensorClass::DsparkStage),
            layer => Err(invalid(format!("DSpark stage is out of range: {layer}"))),
        };
    }
    match name {
        "embed.weight" | "norm.weight" | "head.weight" | "hc_head_base" | "hc_head_fn"
        | "hc_head_scale" => Ok(DeepSeekV4TensorClass::NextTokenRoot),
        _ => Err(invalid(format!("unknown tensor family: {name}"))),
    }
}

pub fn deepseek_v4_locked_shard(file_name: &str) -> Option<DeepSeekV4ShardIdentity> {
    DEEPSEEK_V4_SHARDS
        .iter()
        .copied()
        .find(|identity| identity.file_name == file_name)
}

pub fn validate_deepseek_v4_shard_lfs_identity(
    file_name: &str,
    size: u64,
    lfs_sha256: &str,
) -> Result<DeepSeekV4ShardIdentity, DeepSeekV4ModelError> {
    if file_name.contains('/') || file_name.contains('\\') || file_name.contains("..") {
        return Err(invalid("unsafe shard path"));
    }
    let expected = deepseek_v4_locked_shard(file_name)
        .ok_or_else(|| invalid(format!("unknown shard: {file_name}")))?;
    if size != expected.size || lfs_sha256 != expected.lfs_sha256 {
        return Err(invalid(format!("shard LFS identity differs: {file_name}")));
    }
    Ok(expected)
}

pub fn checked_deepseek_v4_shard_file_bytes() -> Result<u64, DeepSeekV4ModelError> {
    let total = DEEPSEEK_V4_SHARDS.iter().try_fold(0_u64, |sum, shard| {
        sum.checked_add(shard.size)
            .ok_or_else(|| invalid("shard file byte total overflowed"))
    })?;
    if total != DEEPSEEK_V4_SHARD_FILE_BYTES {
        return Err(invalid("shard file byte total differs"));
    }
    Ok(total)
}

fn catalog_sha256(weight_map: &BTreeMap<String, String>) -> Result<String, DeepSeekV4ModelError> {
    let mut hasher = Sha256::new();
    for (name, shard) in weight_map {
        let row = serde_json::to_string(&(name, shard))
            .map_err(|error| invalid(format!("catalog row JSON: {error}")))?;
        hasher.update(row.as_bytes());
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_index_document(
    document: IndexDocument,
) -> Result<DeepSeekV4Index, DeepSeekV4ModelError> {
    let weight_map = document.weight_map.0;
    if document.metadata.total_size != DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES {
        return Err(invalid("index total_size differs"));
    }
    if weight_map.len() != DEEPSEEK_V4_TENSOR_COUNT {
        return Err(invalid("index tensor count differs"));
    }

    let mut summary = DeepSeekV4TensorSummary::default();
    let mut shard_counts = BTreeMap::<&str, usize>::new();
    let mut main_layers = BTreeSet::new();
    let mut dspark_stages = BTreeSet::new();
    for (name, shard) in &weight_map {
        let class = classify_deepseek_v4_tensor(name)?;
        summary.checked_increment(class)?;
        let identity = deepseek_v4_locked_shard(shard)
            .ok_or_else(|| invalid(format!("index references unknown shard: {shard}")))?;
        if shard.contains('/') || shard.contains('\\') || shard.contains("..") {
            return Err(invalid(format!("unsafe index shard path: {shard}")));
        }
        let count = shard_counts.entry(identity.file_name).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("per-shard tensor count overflowed"))?;
        match class {
            DeepSeekV4TensorClass::HashRoutedLayer
            | DeepSeekV4TensorClass::NextTokenLayer
            | DeepSeekV4TensorClass::DsparkTargetLayer => {
                main_layers.insert(scoped_layer(name, "layers")?);
            }
            DeepSeekV4TensorClass::DsparkStage => {
                dspark_stages.insert(scoped_layer(name, "mtp")?);
            }
            DeepSeekV4TensorClass::NextTokenRoot => {}
        }
    }
    if summary
        != (DeepSeekV4TensorSummary {
            next_token_root: ROOT_TENSOR_COUNT,
            hash_routed_layers: HASH_LAYER_TENSOR_COUNT,
            next_token_layers: NEXT_TOKEN_LAYER_TENSOR_COUNT,
            dspark_target_layers: DSPARK_TARGET_LAYER_TENSOR_COUNT,
            dspark_stages: DSPARK_STAGE_TENSOR_COUNT,
        })
        || summary.checked_total()? != DEEPSEEK_V4_TENSOR_COUNT
    {
        return Err(invalid("tensor classification summary differs"));
    }
    if main_layers != (0_u32..43).collect() || dspark_stages != (0_u32..3).collect() {
        return Err(invalid("tensor layer coverage differs"));
    }
    if DEEPSEEK_V4_SHARDS.iter().any(|identity| {
        shard_counts.get(identity.file_name).copied() != Some(identity.indexed_tensor_count)
    }) || shard_counts.len() != DEEPSEEK_V4_SHARD_COUNT
    {
        return Err(invalid("index shard coverage differs"));
    }
    checked_deepseek_v4_shard_file_bytes()?;
    let catalog_sha256 = catalog_sha256(&weight_map)?;
    if catalog_sha256 != DEEPSEEK_V4_CATALOG_SHA256 {
        return Err(invalid("tensor catalog SHA-256 differs"));
    }
    Ok(DeepSeekV4Index {
        total_size: document.metadata.total_size,
        catalog_sha256,
        summary,
        weight_map,
    })
}

pub fn validate_deepseek_v4_index(bytes: &[u8]) -> Result<DeepSeekV4Index, DeepSeekV4ModelError> {
    if bytes.len() != DEEPSEEK_V4_INDEX_BYTES || sha256(bytes) != DEEPSEEK_V4_INDEX_SHA256 {
        return Err(invalid("safetensors index SHA-256 differs"));
    }
    let document = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("safetensors index JSON: {error}")))?;
    validate_index_document(document)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICIAL_CONFIG: &str = r#"{
  "architectures": [
    "DeepseekV4ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "expert_dtype": "fp4",
  "hc_eps": 1e-06,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "head_dim": 512,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "initializer_range": 0.02,
  "max_position_embeddings": 1048576,
  "model_type": "deepseek_v4",
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 43,
  "num_hash_layers": 3,
  "num_key_value_heads": 1,
  "num_nextn_predict_layers": 1,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "sliding_window": 128,
  "swiglu_limit": 10.0,
  "tie_word_embeddings": false,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 129280,
  "compress_rope_theta": 160000,
  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0],
  "dspark_block_size": 5,
  "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [40, 41, 42],
  "dspark_markov_rank": 256
}
"#;

    fn parse_config(bytes: &[u8]) -> Result<DeepSeekV4Config, DeepSeekV4ModelError> {
        let raw = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("config JSON: {error}")))?;
        validate_config_document(raw)
    }

    #[test]
    fn exact_official_config_contract() {
        assert_eq!(OFFICIAL_CONFIG.len(), DEEPSEEK_V4_CONFIG_BYTES);
        assert_eq!(
            sha256(OFFICIAL_CONFIG.as_bytes()),
            DEEPSEEK_V4_CONFIG_SHA256
        );
        let config = validate_deepseek_v4_config(OFFICIAL_CONFIG.as_bytes()).unwrap();
        assert_eq!(config.hidden_size, 4_096);
        assert_eq!(config.compression.len(), 46);
        assert_eq!(config.compression[2], DeepSeekV4Compression::Csa4To1);
        assert_eq!(config.compression[3], DeepSeekV4Compression::Hca128To1);
        assert_eq!(config.dspark_target_layer_ids, [40, 41, 42]);
        assert_eq!(config.quantization.block_shape, [128, 128]);
    }

    #[test]
    fn config_rejects_missing_extra_and_boundary_changes() {
        let mut value: serde_json::Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        value.as_object_mut().unwrap().remove("hc_mult");
        assert!(parse_config(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut value: serde_json::Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        value["unreviewed"] = serde_json::json!(true);
        assert!(parse_config(&serde_json::to_vec(&value).unwrap()).is_err());

        for layer in [39, 43] {
            let mut value: serde_json::Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
            value["dspark_target_layer_ids"][2] = serde_json::json!(layer);
            assert!(parse_config(&serde_json::to_vec(&value).unwrap()).is_err());
        }
    }

    #[test]
    fn tensor_classes_cover_non_aligned_and_boundary_layers() {
        for (name, expected) in [
            (
                "layers.0.ffn.gate.tid2eid",
                DeepSeekV4TensorClass::HashRoutedLayer,
            ),
            (
                "layers.2.attn.wq_b.weight",
                DeepSeekV4TensorClass::HashRoutedLayer,
            ),
            (
                "layers.3.ffn.experts.17.w1.scale",
                DeepSeekV4TensorClass::NextTokenLayer,
            ),
            (
                "layers.39.attn.wkv.weight",
                DeepSeekV4TensorClass::NextTokenLayer,
            ),
            (
                "layers.40.ffn.gate.weight",
                DeepSeekV4TensorClass::DsparkTargetLayer,
            ),
            (
                "layers.42.attn.wo_b.scale",
                DeepSeekV4TensorClass::DsparkTargetLayer,
            ),
            ("mtp.0.main_proj.weight", DeepSeekV4TensorClass::DsparkStage),
            (
                "mtp.2.confidence_head.proj.weight",
                DeepSeekV4TensorClass::DsparkStage,
            ),
        ] {
            assert_eq!(classify_deepseek_v4_tensor(name).unwrap(), expected);
        }
        for name in [
            "layers.43.attn.wq_a.weight",
            "mtp.3.norm.weight",
            "layers/2/weight",
            "layers..2.weight",
            "../model.safetensors",
            "unknown.weight",
        ] {
            assert!(classify_deepseek_v4_tensor(name).is_err(), "{name}");
        }
    }

    #[test]
    fn shard_manifest_and_checked_byte_accounting_are_exact() {
        assert_eq!(
            checked_deepseek_v4_shard_file_bytes().unwrap(),
            166_886_535_336
        );
        let first = DEEPSEEK_V4_SHARDS[0];
        assert_eq!(
            validate_deepseek_v4_shard_lfs_identity(first.file_name, first.size, first.lfs_sha256)
                .unwrap(),
            first
        );
        assert!(
            validate_deepseek_v4_shard_lfs_identity(
                first.file_name,
                first.size + 1,
                first.lfs_sha256
            )
            .is_err()
        );
        assert!(
            validate_deepseek_v4_shard_lfs_identity(
                "../model-00001-of-00048.safetensors",
                first.size,
                first.lfs_sha256
            )
            .is_err()
        );

        let index = DeepSeekV4Index {
            total_size: DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES,
            catalog_sha256: String::new(),
            summary: DeepSeekV4TensorSummary::default(),
            weight_map: BTreeMap::new(),
        };
        assert_eq!(
            index.checked_resident_bytes(1, 257).unwrap(),
            DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES + 257
        );
        assert!(index.checked_resident_bytes(0, 0).is_err());
        assert!(index.checked_resident_bytes(u64::MAX, 0).is_err());
        assert!(index.checked_resident_bytes(1, u64::MAX).is_err());
    }

    #[test]
    fn duplicate_index_keys_are_rejected_before_contract_validation() {
        let duplicate = br#"{"metadata":{"total_size":1},"weight_map":{"a":"x","a":"y"}}"#;
        assert!(serde_json::from_slice::<IndexDocument>(duplicate).is_err());
    }

    #[test]
    fn catalog_digest_format_is_stable() {
        let mut map = BTreeMap::new();
        map.insert(
            "layers.17.norm.weight".to_owned(),
            "model-00003-of-00048.safetensors".to_owned(),
        );
        map.insert(
            "mtp.2.norm.weight".to_owned(),
            "model-00048-of-00048.safetensors".to_owned(),
        );
        assert_eq!(
            catalog_sha256(&map).unwrap(),
            "41fadfdad3be5eb481cb6a14ea246b634ae0d73733498522abc223397df69fd9"
        );
    }

    #[test]
    #[ignore = "requires the reviewed official config and safetensors index files"]
    fn official_metadata_files_match_reviewed_identity() {
        let root = std::env::var_os("SLLM_DEEPSEEK_V4_METADATA_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SLLM_DEEPSEEK_V4_METADATA_DIR to the official metadata directory");
        let config = std::fs::read(root.join("config.json")).expect("read official config.json");
        let index = std::fs::read(root.join("model.safetensors.index.json"))
            .expect("read official model.safetensors.index.json");

        let config = validate_deepseek_v4_config(&config).expect("validate official config");
        let index = validate_deepseek_v4_index(&index).expect("validate official index");

        assert_eq!(config.layer_count, DEEPSEEK_V4_MAIN_LAYER_COUNT);
        assert_eq!(config.hash_layer_count, DEEPSEEK_V4_HASH_ROUTED_LAYER_COUNT);
        assert_eq!(
            config.next_token_prediction_layers,
            DEEPSEEK_V4_CONFIG_NEXTN_PREDICT_LAYER_COUNT
        );
        assert_eq!(index.tensor_count(), DEEPSEEK_V4_TENSOR_COUNT);
        assert_eq!(index.total_size(), DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES);
        assert_eq!(
            index.summary().checked_total().unwrap(),
            DEEPSEEK_V4_TENSOR_COUNT
        );
        assert_eq!(index.catalog_sha256(), DEEPSEEK_V4_CATALOG_SHA256);
    }
}
