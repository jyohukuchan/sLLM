//! Model-free OCP MXFP8 E4M3 W8A8 and MXFP6 E3M2 W6A6 GPU oracle.

use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, DispatchEvidence, Encoding,
    ExecutionSessionRequest, ExecutionState, MxElementFormat, QuantizedMx, SemanticOpDescriptor,
    SemanticOpKind, TensorView, quantize_mxfp6_e3m2, quantize_mxfp8_e4m3,
};
use sllm_hip::{Context, HipBackend};

const WAIT: Duration = Duration::from_secs(60);
const SHUTDOWN: Duration = Duration::from_secs(16);
const PHASE63_CANDIDATE_KERNEL_ID: u32 = 31;
const PHASE63_CANDIDATE_KERNEL_SYMBOL: &str =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2";
const PHASE63_CANDIDATE_DEVICE_SYMBOL: &str =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2";
const PHASE66_CONTROL_KERNEL_ID: u32 = 36;
const PHASE66_CANDIDATE_KERNEL_ID: u32 = 37;
const PHASE66_CONTROL_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_BOTH_GFX1201";
const PHASE66_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_WMMA_N128_DIRECT_BOTH_GFX1201";
const PHASE66_CANDIDATE_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1201.wmma128x128.bdirect.v1";
const PHASE66_CANDIDATE_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1";
const PHASE67_ROW8_KERNEL_ID: u32 = 22;
const PHASE67_CONTROL_KERNEL_ID: u32 = 27;
const PHASE67_COL16_KERNEL_ID: u32 = 38;
const PHASE67_COL32_KERNEL_ID: u32 = 39;
const PHASE67_CONTROL_FORCE_ENV: &str = "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS";
const PHASE67_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_COLUMNS";
const PHASE67_COL16_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1030.mmq-col16.v1";
const PHASE67_COL16_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1";
const PHASE67_COL32_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1030.mmq-col32.v1";
const PHASE67_COL32_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1";
const PHASE69_CONTROL_KERNEL_ID: u32 = 27;
const PHASE69_REGSCALE_KERNEL_ID: u32 = 40;
const PHASE69_VECTOR32_KERNEL_ID: u32 = 41;
const PHASE69_COMBINED_KERNEL_ID: u32 = 42;
const PHASE69_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69";
const PHASE69_REGSCALE_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale.v1";
const PHASE69_REGSCALE_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1";
const PHASE69_VECTOR32_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1";
const PHASE69_VECTOR32_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1";
const PHASE69_COMBINED_KERNEL_SYMBOL: &str =
    "matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale-vector32.v1";
const PHASE69_COMBINED_DEVICE_SYMBOL: &str =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1";
const MXFP8_FORCE_ENVIRONMENTS: &[&str] = &[
    "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
    "SLLM_MXFP8_PREFILL_FORCE_ROW8",
    "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS",
    "SLLM_MXFP8_PREFILL_FORCE_TILED16",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_N16_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_4W_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_LDS_PAD_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_WEIGHT_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_ACTIVATION_GFX1201",
    PHASE66_CONTROL_FORCE_ENV,
    PHASE66_CANDIDATE_FORCE_ENV,
    PHASE67_CANDIDATE_FORCE_ENV,
    PHASE69_CANDIDATE_FORCE_ENV,
];
const GFX1201_WMMA_CANDIDATE_IDENTITIES: &[(u32, &str, &str, u32)] = &[
    (
        32,
        "matmul.mxfp8.w8a8.gfx1201.wmma64x64.4w.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1",
        128,
    ),
    (
        33,
        "matmul.mxfp8.w8a8.gfx1201.wmma128x64.pad33.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1",
        256,
    ),
    (
        34,
        "matmul.mxfp8.w8a8.gfx1201.wmma128x64.direct.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1",
        256,
    ),
    (
        35,
        "matmul.mxfp8.w8a8.gfx1201.wmma128x64.adirect.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1",
        256,
    ),
    (
        36,
        "matmul.mxfp8.w8a8.gfx1201.wmma128x64.bdirect.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1",
        256,
    ),
    (
        PHASE66_CANDIDATE_KERNEL_ID,
        PHASE66_CANDIDATE_KERNEL_SYMBOL,
        PHASE66_CANDIDATE_DEVICE_SYMBOL,
        256,
    ),
];
const GFX1030_MMQ_CANDIDATE_IDENTITIES: &[(u32, &str, &str, u32)] = &[
    (
        PHASE67_COL16_KERNEL_ID,
        PHASE67_COL16_KERNEL_SYMBOL,
        PHASE67_COL16_DEVICE_SYMBOL,
        256,
    ),
    (
        PHASE67_COL32_KERNEL_ID,
        PHASE67_COL32_KERNEL_SYMBOL,
        PHASE67_COL32_DEVICE_SYMBOL,
        256,
    ),
    (
        PHASE69_REGSCALE_KERNEL_ID,
        PHASE69_REGSCALE_KERNEL_SYMBOL,
        PHASE69_REGSCALE_DEVICE_SYMBOL,
        256,
    ),
    (
        PHASE69_VECTOR32_KERNEL_ID,
        PHASE69_VECTOR32_KERNEL_SYMBOL,
        PHASE69_VECTOR32_DEVICE_SYMBOL,
        256,
    ),
    (
        PHASE69_COMBINED_KERNEL_ID,
        PHASE69_COMBINED_KERNEL_SYMBOL,
        PHASE69_COMBINED_DEVICE_SYMBOL,
        256,
    ),
];
const MAX_ABSOLUTE_ERROR: f32 = 0.5;
const MAX_RELATIVE_ERROR: f32 = 0.02;
const PHASE63_SPECIAL_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (63, 1023),
    (64, 1024),
    (127, 2047),
];
const PHASE63_PRODUCTION_M127_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 9215),
    (62, 4607),
    (63, 4608),
    (125, 9214),
    (126, 9215),
];
const PHASE63_PRODUCTION_M128_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 9215),
    (63, 4607),
    (64, 4608),
    (126, 9214),
    (127, 9215),
];
const PHASE63_PRODUCTION_M129_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 9215),
    (64, 4607),
    (65, 4608),
    (127, 9214),
    (128, 9215),
];
const PHASE64_PRODUCTION_M128_N12288_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 12287),
    (63, 6143),
    (64, 6144),
    (126, 12286),
    (127, 12287),
];
const PHASE64_PRODUCTION_M128_N6144_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 6143),
    (63, 3071),
    (64, 3072),
    (126, 6142),
    (127, 6143),
];
const PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 4095),
    (63, 2047),
    (64, 2048),
    (126, 4094),
    (127, 4095),
];
const PHASE63_PRODUCTION_M128_N2560_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 2559),
    (63, 1279),
    (64, 1280),
    (126, 2558),
    (127, 2559),
];
const PHASE63_PRODUCTION_M128_N1024_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 1023),
    (63, 511),
    (64, 512),
    (126, 1022),
    (127, 1023),
];
const PHASE65_PRODUCTION_M128_N512_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 511),
    (63, 255),
    (64, 256),
    (126, 510),
    (127, 511),
];
const PHASE65_PRODUCTION_M128_N256_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 31),
    (0, 32),
    (0, 33),
    (1, 0),
    (1, 255),
    (63, 127),
    (64, 128),
    (126, 254),
    (127, 255),
];
const PHASE65_PRODUCTION_M128_N64_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 15),
    (0, 31),
    (0, 32),
    (1, 0),
    (1, 63),
    (63, 31),
    (64, 32),
    (126, 62),
    (127, 63),
];
const PHASE67_M17_N9216_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 9215),
    (7, 4607),
    (8, 4608),
    (15, 9214),
    (16, 9215),
];
const PHASE67_M512_N9216_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 9215),
    (127, 4607),
    (128, 4608),
    (255, 9214),
    (256, 0),
    (510, 9214),
    (511, 9215),
];
const PHASE67_M512_N2560_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 2559),
    (127, 1279),
    (128, 1280),
    (255, 2558),
    (256, 0),
    (510, 2558),
    (511, 2559),
];
const PHASE69_M512_N4096_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 4095),
    (127, 2047),
    (128, 2048),
    (255, 4094),
    (256, 0),
    (510, 4094),
    (511, 4095),
];
const PHASE67_M512_N8192_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 8191),
    (127, 4095),
    (128, 4096),
    (255, 8190),
    (256, 0),
    (510, 8190),
    (511, 8191),
];
const PHASE67_M512_N1024_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 1023),
    (127, 511),
    (128, 512),
    (255, 1022),
    (256, 0),
    (510, 1022),
    (511, 1023),
];
const PHASE67_M2048_N9216_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 9215),
    (511, 4607),
    (512, 4608),
    (1023, 9214),
    (1024, 0),
    (2046, 9214),
    (2047, 9215),
];
const PHASE67_M2048_N8192_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 8191),
    (511, 4095),
    (512, 4096),
    (1023, 8190),
    (1024, 0),
    (2046, 8190),
    (2047, 8191),
];
const PHASE67_M2048_N4096_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 4095),
    (511, 2047),
    (512, 2048),
    (1023, 4094),
    (1024, 0),
    (2046, 4094),
    (2047, 4095),
];
const PHASE67_M2048_N2560_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 2559),
    (511, 1279),
    (512, 1280),
    (1023, 2558),
    (1024, 0),
    (2046, 2558),
    (2047, 2559),
];
const PHASE67_M2048_N1024_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 32),
    (1, 1023),
    (511, 511),
    (512, 512),
    (1023, 1022),
    (1024, 0),
    (2046, 1022),
    (2047, 1023),
];
const PHASE63_PRODUCTION_M128_N32_ORACLE_POINTS: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 15),
    (0, 16),
    (0, 31),
    (1, 0),
    (1, 31),
    (63, 15),
    (64, 16),
    (126, 30),
    (127, 31),
];

#[derive(Clone, Copy, Eq, PartialEq)]
enum Format {
    Mxfp8,
    Mxfp6,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Mxfp8 => "mxfp8-e4m3-w8a8",
            Self::Mxfp6 => "mxfp6-e3m2-w6a6",
        }
    }

    fn quantize(self, values: &[f32], rows: usize, columns: usize) -> Result<QuantizedMx, String> {
        match self {
            Self::Mxfp8 => quantize_mxfp8_e4m3(values, rows, columns),
            Self::Mxfp6 => quantize_mxfp6_e3m2(values, rows, columns),
        }
        .map_err(|error| error.to_string())
    }

    fn view(self, n: usize, k: usize) -> Result<TensorView, String> {
        let (dtype, encoding) = match self {
            Self::Mxfp8 => (
                DType::F8E4M3Fn,
                Encoding::Mxfp8W8A8 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
            Self::Mxfp6 => (
                DType::U8,
                Encoding::Mxfp6W6A6 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
        };
        TensorView::with_encoding(dtype, encoding, &[n, k]).map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy)]
enum OracleSelection {
    Full,
    FixedSample(&'static [(usize, usize)]),
}

impl OracleSelection {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::FixedSample(_) => "fixed-sample",
        }
    }

    fn output_indices(self, m: usize, n: usize) -> Result<Vec<usize>, String> {
        match self {
            Self::Full => Ok((0..m * n).collect()),
            Self::FixedSample(points) => {
                let mut indices = Vec::with_capacity(points.len());
                for &(row, column) in points {
                    if row >= m || column >= n {
                        return Err(format!(
                            "fixed oracle point ({row},{column}) exceeds [{m},{n}]"
                        ));
                    }
                    indices.push(row * n + column);
                }
                indices.sort_unstable();
                indices.dedup();
                Ok(indices)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Phase66Provider {
    Control,
    Candidate,
}

impl Phase66Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "id36-control",
            Self::Candidate => "id37-candidate",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Control => PHASE66_CONTROL_FORCE_ENV,
            Self::Candidate => PHASE66_CANDIDATE_FORCE_ENV,
        }
    }
}

#[derive(Clone, Copy)]
enum Phase67Provider {
    Row8,
    Control,
    Col16,
    Col32,
}

impl Phase67Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Row8 => "id22-row8-control",
            Self::Control => "id27-col8-control",
            Self::Col16 => "id38-col16-candidate",
            Self::Col32 => "id39-col32-candidate",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Row8 => "SLLM_MXFP8_PREFILL_FORCE_ROW8",
            Self::Control => PHASE67_CONTROL_FORCE_ENV,
            Self::Col16 | Self::Col32 => PHASE67_CANDIDATE_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Row8 => "1",
            Self::Control => "8",
            Self::Col16 => "16",
            Self::Col32 => "32",
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Row8 => PHASE67_ROW8_KERNEL_ID,
            Self::Control => PHASE67_CONTROL_KERNEL_ID,
            Self::Col16 => PHASE67_COL16_KERNEL_ID,
            Self::Col32 => PHASE67_COL32_KERNEL_ID,
        }
    }
}

#[derive(Clone, Copy)]
enum Phase69Provider {
    Control,
    Regscale,
    Vector32,
    Combined,
}

impl Phase69Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "id27-col8-control",
            Self::Regscale => "id40-regscale-candidate",
            Self::Vector32 => "id41-vector32-candidate",
            Self::Combined => "id42-regscale-vector32-candidate",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Control => PHASE67_CONTROL_FORCE_ENV,
            Self::Regscale | Self::Vector32 | Self::Combined => PHASE69_CANDIDATE_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Control => "8",
            Self::Regscale => "regscale",
            Self::Vector32 => "vector32",
            Self::Combined => "combined",
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Control => PHASE69_CONTROL_KERNEL_ID,
            Self::Regscale => PHASE69_REGSCALE_KERNEL_ID,
            Self::Vector32 => PHASE69_VECTOR32_KERNEL_ID,
            Self::Combined => PHASE69_COMBINED_KERNEL_ID,
        }
    }
}

#[derive(Clone, Copy)]
enum EvidenceMode {
    Phase62 {
        production_shape: bool,
    },
    Phase63 {
        repeats: usize,
        production_shape: bool,
        require_candidate: bool,
    },
    Phase66 {
        repeats: usize,
        provider: Phase66Provider,
    },
    Phase67 {
        repeats: usize,
        provider: Phase67Provider,
    },
    Phase69 {
        repeats: usize,
        provider: Phase69Provider,
    },
}

impl EvidenceMode {
    fn repeats(self) -> usize {
        match self {
            Self::Phase62 { .. } => 1,
            Self::Phase63 { repeats, .. } => repeats,
            Self::Phase66 { repeats, .. } => repeats,
            Self::Phase67 { repeats, .. } => repeats,
            Self::Phase69 { repeats, .. } => repeats,
        }
    }

    fn schema_version(self) -> &'static str {
        match self {
            Self::Phase62 { .. } => "sllm-ocp-mxfp8-mxfp6-wa-gpu-v1",
            Self::Phase63 { .. } => "sllm-phase63-mxfp8-matrix-operator-gpu-v1",
            Self::Phase66 { .. } => "sllm-phase66-mxfp8-wide-n-provider-gpu-v1",
            Self::Phase67 { .. } => "sllm-phase67-gfx1030-mxfp8-tile-provider-gpu-v1",
            Self::Phase69 { .. } => "sllm-phase69-gfx1030-mxfp8-software-mmq-provider-gpu-v1",
        }
    }

    fn warmup_count(self) -> usize {
        match self {
            Self::Phase69 { .. } => 2,
            _ => 0,
        }
    }

    fn report_mode(self) -> Option<&'static str> {
        match self {
            Self::Phase62 { .. } => None,
            Self::Phase63 { .. } => Some("phase63"),
            Self::Phase66 { .. } => Some("phase66-provider"),
            Self::Phase67 { .. } => Some("phase67-provider"),
            Self::Phase69 { .. } => Some("phase69-provider"),
        }
    }

    fn phase66_provider(self) -> Option<Phase66Provider> {
        match self {
            Self::Phase66 { provider, .. } => Some(provider),
            _ => None,
        }
    }

    fn phase67_provider(self) -> Option<Phase67Provider> {
        match self {
            Self::Phase67 { provider, .. } => Some(provider),
            _ => None,
        }
    }

    fn phase69_provider(self) -> Option<Phase69Provider> {
        match self {
            Self::Phase69 { provider, .. } => Some(provider),
            _ => None,
        }
    }
}

#[derive(Serialize)]
struct CaseReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    case_id: Option<&'static str>,
    format: &'static str,
    m: usize,
    k: usize,
    n: usize,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    kernel_elapsed_ns: u64,
    weight_value_sha256: String,
    weight_scale_sha256: String,
    output_bf16_sha256: String,
    max_abs_error: f32,
    max_relative_error: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_value_classes: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_encoding_contract_validated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oracle_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oracle_point_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oracle_sampled_output_indices: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_error_output_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_relative_error_output_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_nonfinite_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    actual_nonfinite_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nonfinite_mismatch_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_kernel_elapsed_ns: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_output_bf16_sha256: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase63_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase66_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase66_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase67_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase67_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase69_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase69_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    actual_dispatch_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workgroup_size_x: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grid_size_x: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_dispatch_ids: Option<Vec<u64>>,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_role: Option<&'static str>,
    target: String,
    device_index: u32,
    block_size: usize,
    scale: &'static str,
    rounding: &'static str,
    accumulation: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_kernel_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_case_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_submission_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production_shape_included: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    absolute_error_limit: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_error_limit: Option<f32>,
    cases: Vec<CaseReport>,
    retryable_cleanup: usize,
    durable_quarantine: usize,
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn matrix(rows: usize, columns: usize, phase: usize) -> Vec<u16> {
    if matches!(phase, 100 | 111) {
        return special_matrix(rows, columns, phase);
    }
    (0..rows * columns)
        .map(|index| {
            let base = (((index * 37 + phase * 19) % 257) as i32 - 128) as f32 / 17.0;
            bf16(if index % 29 == 0 { base * 17.0 } else { base })
        })
        .collect()
}

fn special_matrix(rows: usize, columns: usize, _phase: usize) -> Vec<u16> {
    let mut values = vec![bf16(1.0); rows * columns];
    let tiny_bf16 = 0x0001_u16;
    let max_bf16 = 0x7f7f_u16;
    let set_block = |values: &mut [u16], row: usize, block: usize, value: u16| {
        if row < rows && (block + 1) * 32 <= columns {
            let start = row * columns + block * 32;
            values[start..start + 32].fill(value);
        }
    };
    set_block(&mut values, 0, 0, bf16(f32::NAN));
    set_block(&mut values, 1, 0, bf16(f32::INFINITY));
    set_block(&mut values, 2, 0, tiny_bf16);
    set_block(&mut values, 3, 0, max_bf16);
    set_block(&mut values, 4, 0, bf16(-0.0));
    set_block(&mut values, 5, 0, bf16(448.0));
    set_block(&mut values, 6, 0, bf16(-448.0));
    if rows > 7 && columns >= 32 {
        values[7 * columns] = bf16(2.0_f32.powi(-17));
        values[7 * columns + 1] = bf16(1.0625);
    }
    values
}

fn validate_special_encoding(
    activation_words: &[u16],
    activation: &QuantizedMx,
    weight_words: &[u16],
    weight: &QuantizedMx,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let validate = |label: &str,
                    source: &[u16],
                    quantized: &QuantizedMx,
                    rows: usize|
     -> Result<(), String> {
        if rows < 8 || k < 32 {
            return Err(format!("{label} special fixture is too small"));
        }
        let blocks_per_row = k / 32;
        let scale = |row: usize| quantized.scales()[row * blocks_per_row];
        let value = |row: usize, lane: usize| quantized.values()[row * k + lane];
        let valid = scale(0) == 0xff
            && (0..32).all(|lane| value(0, lane) == 0)
            && scale(1) == 0x7f
            && (0..32).all(|lane| value(1, lane) == 0x7e)
            && scale(2) == 0x00
            && (0..32).all(|lane| value(2, lane) == 0x08)
            && scale(3) == 0xf6
            && (0..32).all(|lane| value(3, lane) == 0x7e)
            && scale(4) == 0x7f
            && (0..32).all(|lane| value(4, lane) == 0x80)
            && scale(5) == 0x7f
            && (0..32).all(|lane| value(5, lane) == 0x7e)
            && scale(6) == 0x7f
            && (0..32).all(|lane| value(6, lane) == 0xfe)
            && source[7 * k] == bf16(2.0_f32.powi(-17))
            && value(7, 0) == 0x01
            && source[7 * k + 1] == bf16(1.0625)
            && scale(7) == 0x77
            && value(7, 1) == 0x78;
        if !valid {
            return Err(format!(
                "{label} special E4M3/E8M0 encoding bytes differ: scales={:?}, row7_prefix={:?}",
                (0..8).map(scale).collect::<Vec<_>>(),
                &quantized.values()[7 * k..7 * k + 2]
            ));
        }
        Ok(())
    };
    validate("activation", activation_words, activation, m)?;
    validate("weight", weight_words, weight, n)
}

fn words_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn wait_ok(
    state: Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    match state.map_err(|error| format!("{label}: {error}"))? {
        ExecutionState::Success => Ok(()),
        other => Err(format!("{label}: unexpected state {other:?}")),
    }
}

#[derive(Clone, Copy)]
struct CaseSpec {
    case_id: Option<&'static str>,
    format: Format,
    m: usize,
    k: usize,
    n: usize,
    phase: usize,
    oracle: OracleSelection,
}

#[derive(Default)]
struct OracleStats {
    max_abs_error: f32,
    max_relative_error: f32,
    max_abs_error_output_index: Option<usize>,
    max_relative_error_output_index: Option<usize>,
    expected_nonfinite_count: usize,
    actual_nonfinite_count: usize,
    nonfinite_mismatch_count: usize,
}

fn validate_actual_dispatch(
    format: Format,
    m: usize,
    k: usize,
    n: usize,
    target: &str,
    dispatch: &DispatchEvidence,
) -> Result<(), String> {
    let normalized_size = m
        .checked_mul(n)
        .ok_or_else(|| "output element count overflowed usize".to_owned())?;
    let valid_kernel = match (format, m) {
        (Format::Mxfp8, 1) => dispatch.kernel_id == 18,
        (Format::Mxfp6, 1) => dispatch.kernel_id == 20,
        (Format::Mxfp8, _) => matches!(
            dispatch.kernel_id,
            19 | 22 | 24 | 26 | 27 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42
        ),
        (Format::Mxfp6, _) => matches!(dispatch.kernel_id, 21 | 23 | 25 | 28 | 29),
    };
    let format_fragment = match format {
        Format::Mxfp8 => "mxfp8",
        Format::Mxfp6 => "mxfp6",
    };
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 2
        || !valid_kernel
        || dispatch.row_count != m as u64
        || dispatch.normalized_size != normalized_size as u64
        || dispatch.backend != 1
        || !dispatch.kernel_symbol.contains(format_fragment)
        || !dispatch.device_symbol.contains(format_fragment)
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
    {
        return Err(format!(
            "unexpected {} actual dispatch: {dispatch:?}",
            format.name(),
        ));
    }
    if dispatch.kernel_id == PHASE63_CANDIDATE_KERNEL_ID
        && (format != Format::Mxfp8
            || target != "gfx1201"
            || m == 1
            || dispatch.kernel_symbol != PHASE63_CANDIDATE_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE63_CANDIDATE_DEVICE_SYMBOL)
    {
        return Err(format!(
            "Phase 63 candidate kernel {} escaped its exact gfx1201 MXFP8 prefill scope",
            dispatch.kernel_id
        ));
    }
    if dispatch.kernel_id == PHASE66_CANDIDATE_KERNEL_ID
        && (format != Format::Mxfp8
            || target != "gfx1201"
            || m == 0
            || m % 128 != 0
            || k == 0
            || k % 32 != 0
            || n == 0
            || n % 128 != 0
            || dispatch.kernel_symbol != PHASE66_CANDIDATE_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE66_CANDIDATE_DEVICE_SYMBOL)
    {
        return Err(format!(
            "Phase 66 candidate kernel {} escaped its exact gfx1201 aligned MXFP8 scope",
            dispatch.kernel_id
        ));
    }
    if let Some((_, kernel_symbol, device_symbol, workgroup_size)) =
        GFX1201_WMMA_CANDIDATE_IDENTITIES
            .iter()
            .find(|(kernel_id, _, _, _)| *kernel_id == dispatch.kernel_id)
    {
        if format != Format::Mxfp8
            || target != "gfx1201"
            || m == 1
            || dispatch.kernel_symbol != *kernel_symbol
            || dispatch.device_symbol != *device_symbol
            || dispatch.workgroup_size_x != *workgroup_size
        {
            return Err(format!(
                "gfx1201 WMMA candidate kernel {} has a truncated or mismatched launch identity",
                dispatch.kernel_id
            ));
        }
    }
    if let Some((_, kernel_symbol, device_symbol, workgroup_size)) =
        GFX1030_MMQ_CANDIDATE_IDENTITIES
            .iter()
            .find(|(kernel_id, _, _, _)| *kernel_id == dispatch.kernel_id)
    {
        if format != Format::Mxfp8
            || target != "gfx1030"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != *kernel_symbol
            || dispatch.device_symbol != *device_symbol
            || dispatch.workgroup_size_x != *workgroup_size
        {
            return Err(format!(
                "gfx1030 MMQ candidate kernel {} escaped its exact MXFP8 prefill scope",
                dispatch.kernel_id
            ));
        }
    }
    Ok(())
}

fn compare_oracle(
    spec: CaseSpec,
    activation: &[f32],
    weight: &[f32],
    output: &[u8],
) -> Result<(OracleStats, Vec<usize>), String> {
    let indices = spec.oracle.output_indices(spec.m, spec.n)?;
    let mut stats = OracleStats::default();
    for &output_index in &indices {
        let row = output_index / spec.n;
        let column = output_index - row * spec.n;
        let expected = (0..spec.k)
            .map(|inner| activation[row * spec.k + inner] * weight[column * spec.k + inner])
            .sum::<f32>();
        let byte_index = output_index * 2;
        let actual = from_bf16(u16::from_le_bytes([
            output[byte_index],
            output[byte_index + 1],
        ]));
        if !expected.is_finite() {
            stats.expected_nonfinite_count += 1;
        }
        if !actual.is_finite() {
            stats.actual_nonfinite_count += 1;
        }
        if !expected.is_finite() || !actual.is_finite() {
            let matching_nonfinite = (expected.is_nan() && actual.is_nan())
                || (expected.is_infinite()
                    && actual.is_infinite()
                    && expected.is_sign_negative() == actual.is_sign_negative());
            if !matching_nonfinite {
                stats.nonfinite_mismatch_count += 1;
            }
            continue;
        }
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1.0);
        if stats.max_abs_error_output_index.is_none() || absolute > stats.max_abs_error {
            stats.max_abs_error = absolute;
            stats.max_abs_error_output_index = Some(output_index);
        }
        if stats.max_relative_error_output_index.is_none() || relative > stats.max_relative_error {
            stats.max_relative_error = relative;
            stats.max_relative_error_output_index = Some(output_index);
        }
        if absolute > MAX_ABSOLUTE_ERROR && relative > MAX_RELATIVE_ERROR {
            return Err(format!(
                "{} mismatch output_index={output_index} row={row} column={column}: expected={expected} actual={actual} absolute={absolute} limit={MAX_ABSOLUTE_ERROR} relative={relative} limit={MAX_RELATIVE_ERROR}",
                spec.format.name()
            ));
        }
    }
    if stats.nonfinite_mismatch_count != 0 {
        return Err(format!(
            "{} has {} nonfinite oracle mismatches",
            spec.format.name(),
            stats.nonfinite_mismatch_count
        ));
    }
    Ok((stats, indices))
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    target: &str,
    spec: CaseSpec,
    mode: EvidenceMode,
) -> Result<CaseReport, String> {
    let CaseSpec {
        case_id,
        format,
        m,
        k,
        n,
        phase,
        oracle,
    } = spec;
    let activation_words = matrix(m, k, phase);
    let weight_words = matrix(n, k, phase + 11);
    let activation_source: Vec<_> = activation_words.iter().copied().map(from_bf16).collect();
    let weight_source: Vec<_> = weight_words.iter().copied().map(from_bf16).collect();
    let activation_quantized = format.quantize(&activation_source, m, k)?;
    let weight_quantized = format.quantize(&weight_source, n, k)?;
    if phase == 100 {
        validate_special_encoding(
            &activation_words,
            &activation_quantized,
            &weight_words,
            &weight_quantized,
            m,
            k,
            n,
        )?;
    }
    if (format == Format::Mxfp8 && weight_quantized.format() != MxElementFormat::E4M3Fn)
        || (format == Format::Mxfp6 && weight_quantized.format() != MxElementFormat::E3M2)
    {
        return Err("host MX format identity differs".to_owned());
    }
    let activation_decoded = activation_quantized
        .dequantize()
        .map_err(|e| e.to_string())?;
    let weight_decoded = weight_quantized.dequantize().map_err(|e| e.to_string())?;
    let mut resident = weight_quantized.values().to_vec();
    resident.extend_from_slice(weight_quantized.scales());
    let activation_bytes = words_bytes(&activation_words);
    let output_len = m * n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let weight_buffer = session
        .allocate(resident.len() as u64)
        .map_err(|error| error.to_string())?;
    let output_buffer = session
        .allocate(output_len as u64)
        .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        (
            "activation upload",
            &activation_buffer,
            activation_bytes.as_slice(),
        ),
        ("weight upload", &weight_buffer, resident.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|e| e.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| error.to_string())?;
        wait_ok(upload.wait(WAIT), label)?;
    }
    let activation_view =
        TensorView::contiguous(DType::Bf16, &[m, k]).map_err(|e| e.to_string())?;
    let weight_view = format.view(n, k)?;
    let output_view = TensorView::contiguous(DType::Bf16, &[m, n]).map_err(|e| e.to_string())?;
    let semantic = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            semantic,
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|e| e.to_string())?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|e| e.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|e| e.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let repeats = mode.repeats();
    if repeats == 0 {
        return Err("repeat count must be nonzero".to_owned());
    }
    let mut first_kernel_id = None;
    let mut first_kernel_symbol = None;
    let mut first_device_symbol = None;
    let mut first_grid_size_x = None;
    let mut first_workgroup_size_x = None;
    let mut first_dispatch_count = None;
    let mut first_backend = None;
    let mut elapsed_repeats = Vec::with_capacity(repeats);
    let mut output_digests = Vec::with_capacity(repeats);
    let mut dispatch_ids = Vec::with_capacity(repeats);
    let mut first_output = None;
    for warmup in 0..mode.warmup_count() {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        wait_ok(submission.wait(WAIT), "Phase 69 warmup")?;
        validate_actual_dispatch(format, m, k, n, target, &dispatch)
            .map_err(|error| format!("warmup {warmup}: {error}"))?;
    }
    for repeat in 0..repeats {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        wait_ok(submission.wait(WAIT), format.name())?;
        validate_actual_dispatch(format, m, k, n, target, &dispatch)?;
        if dispatch.workgroup_size_x == 0 || dispatch.grid_size_x == 0 {
            return Err(format!(
                "{} actual dispatch has a zero launch dimension: {dispatch:?}",
                format.name()
            ));
        }
        if dispatch_ids.contains(&dispatch.dispatch_id) {
            return Err(format!(
                "{} repeat {repeat} reused dispatch id {}",
                format.name(),
                dispatch.dispatch_id
            ));
        }
        dispatch_ids.push(dispatch.dispatch_id);
        if let Some(kernel_id) = first_kernel_id {
            if dispatch.kernel_id != kernel_id
                || first_kernel_symbol.as_deref() != Some(dispatch.kernel_symbol.as_str())
                || first_device_symbol.as_deref() != Some(dispatch.device_symbol.as_str())
                || first_grid_size_x != Some(dispatch.grid_size_x)
                || first_workgroup_size_x != Some(dispatch.workgroup_size_x)
                || first_dispatch_count != Some(dispatch.dispatch_count)
                || first_backend != Some(dispatch.backend)
            {
                return Err(format!(
                    "{} repeat {repeat} changed provider or launch identity: {dispatch:?}",
                    format.name()
                ));
            }
        } else {
            first_kernel_id = Some(dispatch.kernel_id);
            first_kernel_symbol = Some(dispatch.kernel_symbol.clone());
            first_device_symbol = Some(dispatch.device_symbol.clone());
            first_grid_size_x = Some(dispatch.grid_size_x);
            first_workgroup_size_x = Some(dispatch.workgroup_size_x);
            first_dispatch_count = Some(dispatch.dispatch_count);
            first_backend = Some(dispatch.backend);
        }
        elapsed_repeats.push(
            submission
                .kernel_elapsed_ns()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "GPU timing is absent".to_owned())?,
        );
        let mut readback = submission
            .start_output_readback(0)
            .map_err(|e| e.to_string())?;
        wait_ok(readback.wait(WAIT), "output readback")?;
        let mut output = vec![0_u8; output_len];
        readback.read_into(&mut output).map_err(|e| e.to_string())?;
        let output_digest = digest(&output);
        if let Some(first_digest) = output_digests.first() {
            if first_digest != &output_digest {
                return Err(format!(
                    "{} repeat {repeat} output digest differs: first={first_digest} actual={output_digest}",
                    format.name()
                ));
            }
        }
        output_digests.push(output_digest);
        if first_output.is_none() {
            first_output = Some(output);
        }
    }
    let kernel_id = first_kernel_id.ok_or("dispatch identity is absent")?;
    let kernel_symbol = first_kernel_symbol.ok_or("kernel symbol is absent")?;
    let device_symbol = first_device_symbol.ok_or("device symbol is absent")?;
    let workgroup_size_x = first_workgroup_size_x.ok_or("workgroup size is absent")?;
    let grid_size_x = first_grid_size_x.ok_or("grid size is absent")?;
    let dispatch_count = first_dispatch_count.ok_or("dispatch count is absent")?;
    let output = first_output.ok_or("GPU output is absent")?;
    let (oracle_stats, oracle_indices) =
        compare_oracle(spec, &activation_decoded, &weight_decoded, &output)?;
    let detailed = mode.report_mode().is_some();
    let phase63 = matches!(mode, EvidenceMode::Phase63 { .. });
    let phase66_provider = mode.phase66_provider();
    let phase67_provider = mode.phase67_provider();
    let phase69_provider = mode.phase69_provider();
    let sampled_indices = match oracle {
        OracleSelection::Full => None,
        OracleSelection::FixedSample(_) => Some(oracle_indices.clone()),
    };
    Ok(CaseReport {
        case_id,
        format: format.name(),
        m,
        k,
        n,
        kernel_id,
        kernel_symbol,
        device_symbol,
        kernel_elapsed_ns: elapsed_repeats[0],
        weight_value_sha256: digest(weight_quantized.values()),
        weight_scale_sha256: digest(weight_quantized.scales()),
        output_bf16_sha256: output_digests[0].clone(),
        max_abs_error: oracle_stats.max_abs_error,
        max_relative_error: oracle_stats.max_relative_error,
        special_value_classes: detailed
            .then_some((phase == 100).then_some(
                "E4M3 subnormal/tie/max/saturation, E8M0 minimum/finite/NaN scale, signed zero, Inf/NaN",
            ))
            .flatten(),
        special_encoding_contract_validated: detailed
            .then_some(phase == 100)
            .filter(|value| *value),
        oracle_mode: detailed.then_some(oracle.name()),
        oracle_point_count: detailed.then_some(oracle_indices.len()),
        oracle_sampled_output_indices: detailed.then_some(sampled_indices).flatten(),
        max_abs_error_output_index: detailed
            .then_some(oracle_stats.max_abs_error_output_index)
            .flatten(),
        max_relative_error_output_index: detailed
            .then_some(oracle_stats.max_relative_error_output_index)
            .flatten(),
        expected_nonfinite_count: detailed.then_some(oracle_stats.expected_nonfinite_count),
        actual_nonfinite_count: detailed.then_some(oracle_stats.actual_nonfinite_count),
        nonfinite_mismatch_count: detailed.then_some(oracle_stats.nonfinite_mismatch_count),
        repeat_count: detailed.then_some(repeats),
        repeat_kernel_elapsed_ns: detailed.then_some(elapsed_repeats),
        repeat_output_bf16_sha256: detailed.then_some(output_digests),
        phase63_candidate: phase63.then_some(kernel_id == PHASE63_CANDIDATE_KERNEL_ID),
        phase66_provider: phase66_provider.map(Phase66Provider::name),
        phase66_candidate: phase66_provider
            .map(|_| kernel_id == PHASE66_CANDIDATE_KERNEL_ID),
        phase67_provider: phase67_provider.map(Phase67Provider::name),
        phase67_candidate: phase67_provider.map(|provider| kernel_id == provider.kernel_id()),
        phase69_provider: phase69_provider.map(Phase69Provider::name),
        phase69_candidate: phase69_provider.map(|provider| kernel_id == provider.kernel_id()),
        actual_dispatch_count: detailed.then_some(dispatch_count),
        workgroup_size_x: detailed.then_some(workgroup_size_x),
        grid_size_x: detailed.then_some(grid_size_x),
        repeat_dispatch_ids: detailed.then_some(dispatch_ids),
    })
}

fn phase62_cases(production_shape: bool) -> Vec<CaseSpec> {
    let mut cases = Vec::new();
    for (format, phase) in [(Format::Mxfp8, 0), (Format::Mxfp6, 7)] {
        cases.push(CaseSpec {
            case_id: None,
            format,
            m: 1,
            k: 32,
            n: 7,
            phase,
            oracle: OracleSelection::Full,
        });
        cases.push(CaseSpec {
            case_id: None,
            format,
            m: 3,
            k: 64,
            n: 5,
            phase: phase + 1,
            oracle: OracleSelection::Full,
        });
        // Qwen3.5-4B's GDN in_proj_b shape overlaps the gfx1030 BF16
        // short-mixed selector. Keep this regression in the Phase 62 mode.
        cases.push(CaseSpec {
            case_id: None,
            format,
            m: 17,
            k: 2560,
            n: 32,
            phase: phase + 2,
            oracle: OracleSelection::Full,
        });
        if production_shape {
            cases.push(CaseSpec {
                case_id: None,
                format,
                m: 17,
                k: 2560,
                n: 9216,
                phase: phase + 3,
                oracle: OracleSelection::Full,
            });
        }
    }
    cases
}

fn phase63_cases(production_shape: bool) -> Vec<CaseSpec> {
    let mut cases = vec![
        CaseSpec {
            case_id: Some("decode-nonselection-m1"),
            format: Format::Mxfp8,
            m: 1,
            k: 32,
            n: 7,
            phase: 0,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("odd-tail-m3-k64-n5"),
            format: Format::Mxfp8,
            m: 3,
            k: 64,
            n: 5,
            phase: 1,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("odd-tail-m17-k96-n7"),
            format: Format::Mxfp8,
            m: 17,
            k: 96,
            n: 7,
            phase: 2,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m127-k224-n15"),
            format: Format::Mxfp8,
            m: 127,
            k: 224,
            n: 15,
            phase: 3,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m128-k256-n16"),
            format: Format::Mxfp8,
            m: 128,
            k: 256,
            n: 16,
            phase: 4,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m129-k288-n17"),
            format: Format::Mxfp8,
            m: 129,
            k: 288,
            n: 17,
            phase: 5,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m511-k32-n3"),
            format: Format::Mxfp8,
            m: 511,
            k: 32,
            n: 3,
            phase: 6,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m512-k64-n5"),
            format: Format::Mxfp8,
            m: 512,
            k: 64,
            n: 5,
            phase: 7,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m513-k96-n7"),
            format: Format::Mxfp8,
            m: 513,
            k: 96,
            n: 7,
            phase: 8,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m1023-k32-n3"),
            format: Format::Mxfp8,
            m: 1023,
            k: 32,
            n: 3,
            phase: 9,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m1024-k64-n5"),
            format: Format::Mxfp8,
            m: 1024,
            k: 64,
            n: 5,
            phase: 10,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m1025-k96-n7"),
            format: Format::Mxfp8,
            m: 1025,
            k: 96,
            n: 7,
            phase: 11,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m2047-k32-n3"),
            format: Format::Mxfp8,
            m: 2047,
            k: 32,
            n: 3,
            phase: 12,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m2048-k64-n5"),
            format: Format::Mxfp8,
            m: 2048,
            k: 64,
            n: 5,
            phase: 13,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("boundary-m2049-k96-n7"),
            format: Format::Mxfp8,
            m: 2049,
            k: 96,
            n: 7,
            phase: 14,
            oracle: OracleSelection::Full,
        },
    ];
    if production_shape {
        cases.extend([
            CaseSpec {
                case_id: Some("qwen-wide-m127-k2560-n9216"),
                format: Format::Mxfp8,
                m: 127,
                k: 2560,
                n: 9216,
                phase: 20,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M127_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-wide-m128-k2560-n9216"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 9216,
                phase: 21,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-wide-m129-k2560-n9216"),
                format: Format::Mxfp8,
                m: 129,
                k: 2560,
                n: 9216,
                phase: 22,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M129_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-wide-m128-k2560-n4096"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 4096,
                phase: 23,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen2b-wide-m128-k2048-n6144"),
                format: Format::Mxfp8,
                m: 128,
                k: 2048,
                n: 6144,
                phase: 33,
                oracle: OracleSelection::FixedSample(PHASE64_PRODUCTION_M128_N6144_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen9b-wide-m128-k4096-n12288"),
                format: Format::Mxfp8,
                m: 128,
                k: 4096,
                n: 12288,
                phase: 28,
                oracle: OracleSelection::FixedSample(PHASE64_PRODUCTION_M128_N12288_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen9b-down-m128-k12288-n4096"),
                format: Format::Mxfp8,
                m: 128,
                k: 12288,
                n: 4096,
                phase: 29,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase64-down-crossover-m128-k9216-n4096"),
                format: Format::Mxfp8,
                m: 128,
                k: 9216,
                n: 4096,
                phase: 30,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase64-down-crossover-m128-k10240-n4096"),
                format: Format::Mxfp8,
                m: 128,
                k: 10240,
                n: 4096,
                phase: 31,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase64-down-crossover-m128-k11264-n4096"),
                format: Format::Mxfp8,
                m: 128,
                k: 11264,
                n: 4096,
                phase: 32,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-down-m128-k9216-n2560"),
                format: Format::Mxfp8,
                m: 128,
                k: 9216,
                n: 2560,
                phase: 24,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N2560_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-output-m128-k4096-n2560"),
                format: Format::Mxfp8,
                m: 128,
                k: 4096,
                n: 2560,
                phase: 25,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N2560_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-narrow-m128-k2560-n1024"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 1024,
                phase: 26,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N1024_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase65-small-n-m128-k2560-n512"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 512,
                phase: 34,
                oracle: OracleSelection::FixedSample(PHASE65_PRODUCTION_M128_N512_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase65-small-n-m128-k2560-n256"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 256,
                phase: 35,
                oracle: OracleSelection::FixedSample(PHASE65_PRODUCTION_M128_N256_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase65-small-n-m128-k2560-n64"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 64,
                phase: 36,
                oracle: OracleSelection::FixedSample(PHASE65_PRODUCTION_M128_N64_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("qwen-narrow-m128-k2560-n32"),
                format: Format::Mxfp8,
                m: 128,
                k: 2560,
                n: 32,
                phase: 27,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N32_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("special-m128-k2048-n2048"),
                format: Format::Mxfp8,
                m: 128,
                k: 2048,
                n: 2048,
                phase: 100,
                oracle: OracleSelection::FixedSample(PHASE63_SPECIAL_ORACLE_POINTS),
            },
        ]);
    }
    cases
}

fn phase66_cases() -> Vec<CaseSpec> {
    let mut cases = Vec::new();
    for (case_id, m, n, phase) in [
        ("phase66-boundary-m127-k32-n128", 127, 128, 66),
        ("phase66-boundary-m128-k32-n64", 128, 64, 67),
        ("phase66-boundary-m128-k32-n127", 128, 127, 68),
        ("phase66-boundary-m128-k32-n128", 128, 128, 69),
        ("phase66-boundary-m128-k32-n129", 128, 129, 70),
        ("phase66-boundary-m128-k32-n256", 128, 256, 71),
        ("phase66-boundary-m128-k32-n512", 128, 512, 72),
        ("phase66-boundary-m128-k32-n1024", 128, 1024, 73),
        ("phase66-boundary-m129-k32-n128", 129, 128, 74),
    ] {
        cases.push(CaseSpec {
            case_id: Some(case_id),
            format: Format::Mxfp8,
            m,
            k: 32,
            n,
            phase,
            oracle: OracleSelection::Full,
        });
    }
    cases.extend([
        CaseSpec {
            case_id: Some("phase66-wide-m128-k2560-n9216"),
            format: Format::Mxfp8,
            m: 128,
            k: 2560,
            n: 9216,
            phase: 75,
            oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase66-down-m128-k9216-n2560"),
            format: Format::Mxfp8,
            m: 128,
            k: 9216,
            n: 2560,
            phase: 76,
            oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N2560_ORACLE_POINTS),
        },
    ]);
    cases
}

fn phase67_cases() -> Vec<CaseSpec> {
    let mut cases = phase66_cases();
    cases.extend([
        CaseSpec {
            case_id: Some("phase67-short-m17-k2560-n32"),
            format: Format::Mxfp8,
            m: 17,
            k: 2560,
            n: 32,
            phase: 77,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("phase67-short-m17-k2560-n9216"),
            format: Format::Mxfp8,
            m: 17,
            k: 2560,
            n: 9216,
            phase: 78,
            oracle: OracleSelection::FixedSample(PHASE67_M17_N9216_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-qkv-m128-k2560-n4096"),
            format: Format::Mxfp8,
            m: 128,
            k: 2560,
            n: 4096,
            phase: 79,
            oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N4096_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-output-m128-k4096-n2560"),
            format: Format::Mxfp8,
            m: 128,
            k: 4096,
            n: 2560,
            phase: 80,
            oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N2560_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-narrow-m128-k2560-n1024"),
            format: Format::Mxfp8,
            m: 128,
            k: 2560,
            n: 1024,
            phase: 81,
            oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_N1024_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-wide-m512-k2560-n9216"),
            format: Format::Mxfp8,
            m: 512,
            k: 2560,
            n: 9216,
            phase: 82,
            oracle: OracleSelection::FixedSample(PHASE67_M512_N9216_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-down-m512-k9216-n2560"),
            format: Format::Mxfp8,
            m: 512,
            k: 9216,
            n: 2560,
            phase: 83,
            oracle: OracleSelection::FixedSample(PHASE67_M512_N2560_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-wide-m512-k2560-n8192"),
            format: Format::Mxfp8,
            m: 512,
            k: 2560,
            n: 8192,
            phase: 84,
            oracle: OracleSelection::FixedSample(PHASE67_M512_N8192_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-narrow-m512-k2560-n1024"),
            format: Format::Mxfp8,
            m: 512,
            k: 2560,
            n: 1024,
            phase: 91,
            oracle: OracleSelection::FixedSample(PHASE67_M512_N1024_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-wide-m2048-k2560-n9216"),
            format: Format::Mxfp8,
            m: 2048,
            k: 2560,
            n: 9216,
            phase: 85,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N9216_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-wide-m2048-k2560-n8192"),
            format: Format::Mxfp8,
            m: 2048,
            k: 2560,
            n: 8192,
            phase: 86,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N8192_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-qkv-m2048-k2560-n4096"),
            format: Format::Mxfp8,
            m: 2048,
            k: 2560,
            n: 4096,
            phase: 87,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N4096_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-output-m2048-k4096-n2560"),
            format: Format::Mxfp8,
            m: 2048,
            k: 4096,
            n: 2560,
            phase: 88,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N2560_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-down-m2048-k9216-n2560"),
            format: Format::Mxfp8,
            m: 2048,
            k: 9216,
            n: 2560,
            phase: 89,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N2560_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase67-narrow-m2048-k2560-n1024"),
            format: Format::Mxfp8,
            m: 2048,
            k: 2560,
            n: 1024,
            phase: 90,
            oracle: OracleSelection::FixedSample(PHASE67_M2048_N1024_ORACLE_POINTS),
        },
    ]);
    cases
}

fn phase69_cases() -> Vec<CaseSpec> {
    let mut cases = phase67_cases();
    cases.extend([
        CaseSpec {
            case_id: Some("phase69-qkv-m512-k2560-n4096"),
            format: Format::Mxfp8,
            m: 512,
            k: 2560,
            n: 4096,
            phase: 92,
            oracle: OracleSelection::FixedSample(PHASE69_M512_N4096_ORACLE_POINTS),
        },
        CaseSpec {
            case_id: Some("phase69-output-m512-k4096-n2560"),
            format: Format::Mxfp8,
            m: 512,
            k: 4096,
            n: 2560,
            phase: 93,
            oracle: OracleSelection::FixedSample(PHASE67_M512_N2560_ORACLE_POINTS),
        },
    ]);
    cases
}

fn run(device_index: u32, target: String, mode: EvidenceMode) -> Result<Report, String> {
    match mode {
        EvidenceMode::Phase62 { .. } if !matches!(target.as_str(), "gfx1030" | "gfx1201") => {
            return Err("target must be exactly gfx1030 or gfx1201".to_owned());
        }
        EvidenceMode::Phase63 { .. } if target != "gfx1201" => {
            return Err("Phase 63 mode requires exact gfx1201".to_owned());
        }
        EvidenceMode::Phase66 { provider, .. } if target != "gfx1201" => {
            return Err(format!(
                "Phase 66 {} mode requires exact gfx1201",
                provider.name()
            ));
        }
        EvidenceMode::Phase67 { provider, .. } if target != "gfx1030" => {
            return Err(format!(
                "Phase 67 {} mode requires exact gfx1030",
                provider.name()
            ));
        }
        EvidenceMode::Phase69 { provider, .. } if target != "gfx1030" => {
            return Err(format!(
                "Phase 69 {} mode requires exact gfx1030",
                provider.name()
            ));
        }
        _ => {}
    }
    if let EvidenceMode::Phase66 { provider, .. } = mode {
        for environment in MXFP8_FORCE_ENVIRONMENTS {
            let value = std::env::var(environment).ok();
            let expected = (*environment == provider.force_environment()).then_some("1");
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 66 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    if let EvidenceMode::Phase67 { provider, .. } = mode {
        for environment in MXFP8_FORCE_ENVIRONMENTS {
            let value = std::env::var(environment).ok();
            let expected =
                (*environment == provider.force_environment()).then_some(provider.force_value());
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 67 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    if let EvidenceMode::Phase69 { provider, .. } = mode {
        for environment in MXFP8_FORCE_ENVIRONMENTS {
            let value = std::env::var(environment).ok();
            let expected =
                (*environment == provider.force_environment()).then_some(provider.force_value());
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 69 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    let device = Context::query_device(device_index).map_err(|error| error.to_string())?;
    if device.gcn_arch_name != target {
        return Err(format!(
            "device {device_index} is {}, requested {target}",
            device.gcn_arch_name
        ));
    }
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let result = (|| {
        let queue = session.create_queue().map_err(|error| error.to_string())?;
        let specs = match mode {
            EvidenceMode::Phase62 { production_shape } => phase62_cases(production_shape),
            EvidenceMode::Phase63 {
                production_shape, ..
            } => phase63_cases(production_shape),
            EvidenceMode::Phase66 { .. } => phase66_cases(),
            EvidenceMode::Phase67 { .. } => phase67_cases(),
            EvidenceMode::Phase69 { .. } => phase69_cases(),
        };
        let mut cases = Vec::with_capacity(specs.len());
        for spec in specs {
            cases.push(run_case(&session, &queue, &target, spec, mode)?);
        }
        Ok::<_, String>(cases)
    })();
    let cleanup = session
        .shutdown(SHUTDOWN)
        .map_err(|error| error.to_string())?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("nonzero cleanup state".to_owned());
    }
    let expected_candidate_kernel_id = match mode {
        EvidenceMode::Phase62 { .. } => None,
        EvidenceMode::Phase63 { .. } => Some(PHASE63_CANDIDATE_KERNEL_ID),
        EvidenceMode::Phase66 {
            provider: Phase66Provider::Control,
            ..
        } => Some(PHASE66_CONTROL_KERNEL_ID),
        EvidenceMode::Phase66 {
            provider: Phase66Provider::Candidate,
            ..
        } => Some(PHASE66_CANDIDATE_KERNEL_ID),
        EvidenceMode::Phase67 { provider, .. } => Some(provider.kernel_id()),
        EvidenceMode::Phase69 { provider, .. } => Some(provider.kernel_id()),
    };
    let candidate_case_count = cases
        .iter()
        .filter(|case| Some(case.kernel_id) == expected_candidate_kernel_id)
        .count();
    let candidate_submission_count = candidate_case_count * mode.repeats();
    if matches!(
        mode,
        EvidenceMode::Phase63 {
            require_candidate: true,
            ..
        }
    ) && candidate_submission_count == 0
    {
        return Err(format!(
            "Phase 63 candidate kernel {PHASE63_CANDIDATE_KERNEL_ID} was required but never dispatched"
        ));
    }
    if matches!(mode, EvidenceMode::Phase66 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 66 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    if matches!(mode, EvidenceMode::Phase67 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 67 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    if matches!(mode, EvidenceMode::Phase69 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 69 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    let detailed = mode.report_mode().is_some();
    let (production_shape_included, candidate_required) = match mode {
        EvidenceMode::Phase62 { .. } => (None, None),
        EvidenceMode::Phase63 {
            production_shape,
            require_candidate,
            ..
        } => (Some(production_shape), Some(require_candidate)),
        EvidenceMode::Phase66 { .. } => (Some(true), Some(true)),
        EvidenceMode::Phase67 { .. } => (Some(true), Some(true)),
        EvidenceMode::Phase69 { .. } => (Some(true), Some(true)),
    };
    Ok(Report {
        schema_version: mode.schema_version(),
        state: "PASS",
        evidence_mode: mode.report_mode(),
        provider_role: mode
            .phase66_provider()
            .map(Phase66Provider::name)
            .or_else(|| mode.phase67_provider().map(Phase67Provider::name))
            .or_else(|| mode.phase69_provider().map(Phase69Provider::name)),
        target,
        device_index,
        block_size: 32,
        scale: "E8M0",
        rounding: "roundTiesToEven-saturate",
        accumulation: "FP32",
        fallback_allowed: false,
        fallback_used: false,
        repeat_count: detailed.then_some(mode.repeats()),
        warmup_count: detailed.then_some(mode.warmup_count()),
        candidate_kernel_id: expected_candidate_kernel_id,
        candidate_case_count: detailed.then_some(candidate_case_count),
        candidate_submission_count: detailed.then_some(candidate_submission_count),
        candidate_required,
        production_shape_included,
        absolute_error_limit: detailed.then_some(MAX_ABSOLUTE_ERROR),
        relative_error_limit: detailed.then_some(MAX_RELATIVE_ERROR),
        cases,
        retryable_cleanup: cleanup.retryable_cleanup,
        durable_quarantine: cleanup.durable_quarantine,
    })
}

#[derive(Serialize)]
struct Phase66KBoundaryReport {
    k: usize,
    expected: &'static str,
    observed: &'static str,
    rejection_stage: &'static str,
}

#[derive(Serialize)]
struct Phase66ProviderComparison {
    case_id: String,
    m: u64,
    k: u64,
    n: u64,
    control_kernel_id: u64,
    candidate_kernel_id: u64,
    candidate_selected: bool,
    output_digest_equal: bool,
    control_median_kernel_elapsed_ns: u64,
    candidate_median_kernel_elapsed_ns: u64,
    candidate_over_control_elapsed_ratio: f64,
}

#[derive(Serialize)]
struct Phase66ComparisonReport {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    binary_sha256: String,
    control_force_environment: &'static str,
    candidate_force_environment: &'static str,
    candidate_scope: &'static str,
    arithmetic_contract: &'static str,
    k_boundary: Vec<Phase66KBoundaryReport>,
    comparisons: Vec<Phase66ProviderComparison>,
    control: serde_json::Value,
    candidate: serde_json::Value,
}

fn phase66_provider_child(
    device_index: u32,
    target: &str,
    repeats: usize,
    provider: Phase66Provider,
) -> Result<serde_json::Value, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("resolve Phase 66 evidence binary: {error}"))?;
    let mut command = Command::new(&executable);
    command
        .arg(device_index.to_string())
        .arg(target)
        .arg("phase66-provider")
        .arg(provider.name())
        .arg("--repeats")
        .arg(repeats.to_string());
    for environment in MXFP8_FORCE_ENVIRONMENTS {
        command.env_remove(environment);
    }
    command.env(provider.force_environment(), "1");
    let output = command
        .output()
        .map_err(|error| format!("start Phase 66 {} child: {error}", provider.name()))?;
    if !output.status.success() {
        return Err(format!(
            "Phase 66 {} child failed with {}: {}",
            provider.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse Phase 66 {} child JSON: {error}", provider.name()))?;
    if report.get("state").and_then(serde_json::Value::as_str) != Some("PASS")
        || report
            .get("provider_role")
            .and_then(serde_json::Value::as_str)
            != Some(provider.name())
        || report.get("target").and_then(serde_json::Value::as_str) != Some(target)
    {
        return Err(format!(
            "Phase 66 {} child returned a mismatched report identity",
            provider.name()
        ));
    }
    Ok(report)
}

fn json_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("Phase 66 case is missing integer {field}"))
}

fn median_elapsed_ns(case: &serde_json::Value) -> Result<u64, String> {
    let mut values = case
        .get("repeat_kernel_elapsed_ns")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Phase 66 case is missing repeat timings".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| "Phase 66 repeat timing is not an integer".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("Phase 66 repeat timing array is empty".to_owned());
    }
    values.sort_unstable();
    Ok(values[values.len() / 2])
}

fn phase66_comparisons(
    control: &serde_json::Value,
    candidate: &serde_json::Value,
) -> Result<Vec<Phase66ProviderComparison>, String> {
    let control_cases = control
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Phase 66 control cases are absent".to_owned())?;
    let candidate_cases = candidate
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Phase 66 candidate cases are absent".to_owned())?;
    if control_cases.len() != candidate_cases.len() {
        return Err("Phase 66 provider reports have different case counts".to_owned());
    }
    let mut comparisons = Vec::with_capacity(control_cases.len());
    for (control_case, candidate_case) in control_cases.iter().zip(candidate_cases) {
        let case_id = control_case
            .get("case_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "Phase 66 control case_id is absent".to_owned())?;
        if candidate_case
            .get("case_id")
            .and_then(serde_json::Value::as_str)
            != Some(case_id)
        {
            return Err(format!("Phase 66 provider case order differs at {case_id}"));
        }
        let m = json_u64(control_case, "m")?;
        let k = json_u64(control_case, "k")?;
        let n = json_u64(control_case, "n")?;
        if json_u64(candidate_case, "m")? != m
            || json_u64(candidate_case, "k")? != k
            || json_u64(candidate_case, "n")? != n
        {
            return Err(format!("Phase 66 provider dimensions differ at {case_id}"));
        }
        let control_kernel_id = json_u64(control_case, "kernel_id")?;
        let candidate_kernel_id = json_u64(candidate_case, "kernel_id")?;
        let should_select_candidate =
            m != 0 && m % 128 == 0 && k != 0 && k % 32 == 0 && n != 0 && n % 128 == 0;
        if (candidate_kernel_id == u64::from(PHASE66_CANDIDATE_KERNEL_ID))
            != should_select_candidate
        {
            return Err(format!(
                "Phase 66 candidate selection boundary differs at {case_id}: m={m} k={k} n={n} kernel={candidate_kernel_id}"
            ));
        }
        if should_select_candidate && control_kernel_id != u64::from(PHASE66_CONTROL_KERNEL_ID) {
            return Err(format!(
                "Phase 66 ID36 control was not selected for aligned case {case_id}"
            ));
        }
        let output_digest_equal =
            control_case.get("output_bf16_sha256") == candidate_case.get("output_bf16_sha256");
        if should_select_candidate && !output_digest_equal {
            return Err(format!(
                "Phase 66 ID36/ID37 arithmetic output differs at {case_id}"
            ));
        }
        let control_median = median_elapsed_ns(control_case)?;
        let candidate_median = median_elapsed_ns(candidate_case)?;
        comparisons.push(Phase66ProviderComparison {
            case_id: case_id.to_owned(),
            m,
            k,
            n,
            control_kernel_id,
            candidate_kernel_id,
            candidate_selected: should_select_candidate,
            output_digest_equal,
            control_median_kernel_elapsed_ns: control_median,
            candidate_median_kernel_elapsed_ns: candidate_median,
            candidate_over_control_elapsed_ratio: candidate_median as f64 / control_median as f64,
        });
    }
    Ok(comparisons)
}

fn phase66_k_boundaries() -> Result<Vec<Phase66KBoundaryReport>, String> {
    let mut reports = Vec::new();
    for k in [31_usize, 32, 33] {
        let source = vec![1.0_f32; k];
        let accepted = quantize_mxfp8_e4m3(&source, 1, k).is_ok();
        let expected_accepted = k == 32;
        if accepted != expected_accepted {
            return Err(format!(
                "Phase 66 MXFP8 block-32 K admission differs at K={k}: accepted={accepted}"
            ));
        }
        reports.push(Phase66KBoundaryReport {
            k,
            expected: if expected_accepted {
                "accept"
            } else {
                "reject"
            },
            observed: if accepted { "accepted" } else { "rejected" },
            rejection_stage: "host OCP block-32 encoding admission before GPU dispatch",
        });
    }
    Ok(reports)
}

fn run_phase66_comparison(
    device_index: u32,
    target: String,
    repeats: usize,
) -> Result<Phase66ComparisonReport, String> {
    if target != "gfx1201" {
        return Err("Phase 66 comparison requires exact gfx1201".to_owned());
    }
    let executable = std::env::current_exe()
        .map_err(|error| format!("resolve Phase 66 evidence binary: {error}"))?;
    let binary = std::fs::read(&executable)
        .map_err(|error| format!("read Phase 66 evidence binary: {error}"))?;
    let k_boundary = phase66_k_boundaries()?;
    let control = phase66_provider_child(device_index, &target, repeats, Phase66Provider::Control)?;
    let candidate =
        phase66_provider_child(device_index, &target, repeats, Phase66Provider::Candidate)?;
    let comparisons = phase66_comparisons(&control, &candidate)?;
    Ok(Phase66ComparisonReport {
        schema_version: "sllm-phase66-mxfp8-wide-n-comparison-gpu-v1",
        state: "PASS",
        target,
        device_index,
        binary_sha256: digest(&binary),
        control_force_environment: PHASE66_CONTROL_FORCE_ENV,
        candidate_force_environment: PHASE66_CANDIDATE_FORCE_ENV,
        candidate_scope: "exact gfx1201; M%128=0; K%32=0; N%128=0; model-independent",
        arithmetic_contract: "OCP E4M3 value + E8M0 block32 scales; Phase64/65 FP32 accumulation tree; BF16 RNE output",
        k_boundary,
        comparisons,
        control,
        candidate,
    })
}

enum RequestedMode {
    Direct(EvidenceMode),
    Phase66Comparison { repeats: usize },
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let device_index = match arguments.next().as_deref().unwrap_or("0").parse::<u32>() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid device index: {error}");
            return ExitCode::FAILURE;
        }
    };
    let target = arguments.next().unwrap_or_else(|| "gfx1030".to_owned());
    let mode = match arguments.next().as_deref() {
        None => RequestedMode::Direct(EvidenceMode::Phase62 {
            production_shape: false,
        }),
        Some("production") => {
            if arguments.next().is_some() {
                eprintln!("too many arguments");
                return ExitCode::FAILURE;
            }
            RequestedMode::Direct(EvidenceMode::Phase62 {
                production_shape: true,
            })
        }
        Some("phase63") => {
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut production_shape = false;
            let mut require_candidate = false;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--repeats" if !repeat_seen => {
                        repeat_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--repeats requires a value");
                            return ExitCode::FAILURE;
                        };
                        repeats = match value.parse::<usize>() {
                            Ok(value @ 2..=10) => value,
                            Ok(_) => {
                                eprintln!("Phase 63 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 63 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--production-shape" if !production_shape => production_shape = true,
                    "--require-candidate" if !require_candidate => require_candidate = true,
                    _ => {
                        eprintln!(
                            "invalid Phase 63 argument {argument}; expected --repeats N, --production-shape, or --require-candidate"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase63 {
                repeats,
                production_shape,
                require_candidate,
            })
        }
        Some(profile @ ("phase66" | "phase66-provider")) => {
            let provider = if profile == "phase66-provider" {
                match arguments.next().as_deref() {
                    Some("id36-control") => Some(Phase66Provider::Control),
                    Some("id37-candidate") => Some(Phase66Provider::Candidate),
                    Some(value) => {
                        eprintln!("invalid Phase 66 provider {value}");
                        return ExitCode::FAILURE;
                    }
                    None => {
                        eprintln!("phase66-provider requires a provider role");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                None
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--repeats" if !repeat_seen => {
                        repeat_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--repeats requires a value");
                            return ExitCode::FAILURE;
                        };
                        repeats = match value.parse::<usize>() {
                            Ok(value @ 2..=10) => value,
                            Ok(_) => {
                                eprintln!("Phase 66 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 66 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    _ => {
                        eprintln!("invalid Phase 66 argument {argument}; expected --repeats N");
                        return ExitCode::FAILURE;
                    }
                }
            }
            match provider {
                Some(provider) => {
                    RequestedMode::Direct(EvidenceMode::Phase66 { repeats, provider })
                }
                None => RequestedMode::Phase66Comparison { repeats },
            }
        }
        Some("phase67-provider") => {
            let provider = match arguments.next().as_deref() {
                Some("id22-row8-control") => Phase67Provider::Row8,
                Some("id27-col8-control") => Phase67Provider::Control,
                Some("id38-col16-candidate") => Phase67Provider::Col16,
                Some("id39-col32-candidate") => Phase67Provider::Col32,
                Some(value) => {
                    eprintln!("invalid Phase 67 provider {value}");
                    return ExitCode::FAILURE;
                }
                None => {
                    eprintln!("phase67-provider requires a provider role");
                    return ExitCode::FAILURE;
                }
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--repeats" if !repeat_seen => {
                        repeat_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--repeats requires a value");
                            return ExitCode::FAILURE;
                        };
                        repeats = match value.parse::<usize>() {
                            Ok(value @ 2..=10) => value,
                            Ok(_) => {
                                eprintln!("Phase 67 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 67 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    _ => {
                        eprintln!("invalid Phase 67 argument {argument}; expected --repeats N");
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase67 { repeats, provider })
        }
        Some("phase69-provider") => {
            let provider = match arguments.next().as_deref() {
                Some("id27-col8-control") => Phase69Provider::Control,
                Some("id40-regscale-candidate") => Phase69Provider::Regscale,
                Some("id41-vector32-candidate") => Phase69Provider::Vector32,
                Some("id42-regscale-vector32-candidate") => Phase69Provider::Combined,
                Some(value) => {
                    eprintln!("invalid Phase 69 provider {value}");
                    return ExitCode::FAILURE;
                }
                None => {
                    eprintln!("phase69-provider requires a provider role");
                    return ExitCode::FAILURE;
                }
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--repeats" if !repeat_seen => {
                        repeat_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--repeats requires a value");
                            return ExitCode::FAILURE;
                        };
                        repeats = match value.parse::<usize>() {
                            Ok(value @ 2..=10) => value,
                            Ok(_) => {
                                eprintln!("Phase 69 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 69 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    _ => {
                        eprintln!("invalid Phase 69 argument {argument}; expected --repeats N");
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase69 { repeats, provider })
        }
        Some(value) => {
            eprintln!(
                "invalid profile {value}; expected production, phase63, phase66, phase67-provider, or phase69-provider"
            );
            return ExitCode::FAILURE;
        }
    };
    let result = match mode {
        RequestedMode::Direct(mode) => run(device_index, target, mode)
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
        RequestedMode::Phase66Comparison { repeats } => {
            run_phase66_comparison(device_index, target, repeats)
                .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string()))
        }
    };
    match result {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("serialize evidence: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("MX W/A evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase62_case_matrix_remains_compatible() {
        let ordinary = phase62_cases(false);
        let production = phase62_cases(true);
        assert_eq!(ordinary.len(), 6);
        assert_eq!(production.len(), 8);
        assert_eq!(
            ordinary
                .iter()
                .map(|case| (case.format.name(), case.m, case.k, case.n))
                .collect::<Vec<_>>(),
            vec![
                ("mxfp8-e4m3-w8a8", 1, 32, 7),
                ("mxfp8-e4m3-w8a8", 3, 64, 5),
                ("mxfp8-e4m3-w8a8", 17, 2560, 32),
                ("mxfp6-e3m2-w6a6", 1, 32, 7),
                ("mxfp6-e3m2-w6a6", 3, 64, 5),
                ("mxfp6-e3m2-w6a6", 17, 2560, 32),
            ]
        );
    }

    #[test]
    fn phase63_case_matrix_freezes_m_and_tile_boundaries() {
        let ordinary = phase63_cases(false);
        assert_eq!(
            ordinary.iter().map(|case| case.m).collect::<Vec<_>>(),
            vec![
                1, 3, 17, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
            ]
        );
        assert_eq!(
            ordinary[3..6]
                .iter()
                .map(|case| (case.m, case.k, case.n))
                .collect::<Vec<_>>(),
            vec![(127, 224, 15), (128, 256, 16), (129, 288, 17)]
        );

        let production = phase63_cases(true);
        assert_eq!(production.len(), ordinary.len() + 18);
        assert!(production.iter().any(|case| {
            case.case_id == Some("qwen2b-wide-m128-k2048-n6144")
                && (case.m, case.k, case.n) == (128, 2048, 6144)
        }));
        assert!(production.iter().any(|case| {
            case.case_id == Some("qwen9b-wide-m128-k4096-n12288")
                && (case.m, case.k, case.n) == (128, 4096, 12288)
        }));
        assert!(production.iter().any(|case| {
            case.case_id == Some("qwen9b-down-m128-k12288-n4096")
                && (case.m, case.k, case.n) == (128, 12288, 4096)
        }));
        assert!(production.iter().any(|case| {
            case.case_id == Some("phase64-down-crossover-m128-k9216-n4096")
                && (case.m, case.k, case.n) == (128, 9216, 4096)
        }));
        assert!(production.iter().any(|case| {
            case.case_id == Some("phase64-down-crossover-m128-k11264-n4096")
                && (case.m, case.k, case.n) == (128, 11264, 4096)
        }));
        for case in &production[ordinary.len()..] {
            assert!(matches!(case.oracle, OracleSelection::FixedSample(_)));
            assert_eq!(
                case.oracle.output_indices(case.m, case.n).unwrap().len(),
                if case.case_id == Some("special-m128-k2048-n2048") {
                    13
                } else {
                    11
                }
            );
        }
    }

    #[test]
    fn candidate_dispatch_identity_is_exact_gfx1201_mxfp8_prefill() {
        let candidate = |target: &str| DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 2,
            kernel_id: PHASE63_CANDIDATE_KERNEL_ID,
            workgroup_size_x: 256,
            grid_size_x: 1,
            row_count: 128,
            normalized_size: 128 * 9216,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: PHASE63_CANDIDATE_KERNEL_SYMBOL.to_owned(),
            device_symbol: PHASE63_CANDIDATE_DEVICE_SYMBOL.to_owned(),
            target: target.to_owned(),
        };
        assert!(
            validate_actual_dispatch(
                Format::Mxfp8,
                128,
                2560,
                9216,
                "gfx1201",
                &candidate("gfx1201"),
            )
            .is_ok()
        );
        for (m, target) in [(1, "gfx1201"), (128, "gfx1030")] {
            assert!(
                validate_actual_dispatch(Format::Mxfp8, m, 2560, 9216, target, &candidate(target),)
                    .is_err()
            );
        }
    }

    #[test]
    fn phase66_case_matrix_covers_selector_boundaries_and_projection_directions() {
        let cases = phase66_cases();
        assert_eq!(cases.len(), 11);
        for m in [127, 128, 129] {
            assert!(cases.iter().any(|case| case.m == m));
        }
        for n in [64, 127, 128, 129, 256, 512, 1024] {
            assert!(cases.iter().any(|case| case.n == n));
        }
        assert!(cases.iter().any(|case| {
            case.case_id == Some("phase66-wide-m128-k2560-n9216")
                && (case.m, case.k, case.n) == (128, 2560, 9216)
        }));
        assert!(cases.iter().any(|case| {
            case.case_id == Some("phase66-down-m128-k9216-n2560")
                && (case.m, case.k, case.n) == (128, 9216, 2560)
        }));
        let k_boundary = phase66_k_boundaries().unwrap();
        assert_eq!(
            k_boundary
                .iter()
                .map(|report| (report.k, report.observed))
                .collect::<Vec<_>>(),
            vec![(31, "rejected"), (32, "accepted"), (33, "rejected")]
        );
    }

    #[test]
    fn phase66_candidate_dispatch_identity_is_exact_and_aligned() {
        let candidate = |m: u64, _k: u64, n: u64, target: &str| DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 2,
            kernel_id: PHASE66_CANDIDATE_KERNEL_ID,
            workgroup_size_x: 256,
            grid_size_x: u32::try_from(n / 128).unwrap(),
            row_count: m,
            normalized_size: m * n,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: PHASE66_CANDIDATE_KERNEL_SYMBOL.to_owned(),
            device_symbol: PHASE66_CANDIDATE_DEVICE_SYMBOL.to_owned(),
            target: target.to_owned(),
        };
        assert!(
            validate_actual_dispatch(
                Format::Mxfp8,
                128,
                32,
                128,
                "gfx1201",
                &candidate(128, 32, 128, "gfx1201"),
            )
            .is_ok()
        );
        for (m, k, n, target) in [
            (127, 32, 128, "gfx1201"),
            (128, 31, 128, "gfx1201"),
            (128, 32, 127, "gfx1201"),
            (128, 32, 128, "gfx1030"),
        ] {
            assert!(
                validate_actual_dispatch(
                    Format::Mxfp8,
                    m as usize,
                    k as usize,
                    n as usize,
                    target,
                    &candidate(m, k, n, target),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn phase67_candidate_dispatch_identity_is_exact_gfx1030_mxfp8_prefill() {
        let candidate = |kernel_id: u32,
                         kernel_symbol: &str,
                         device_symbol: &str,
                         target: &str|
         -> DispatchEvidence {
            DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id: 1,
                dispatch_count: 2,
                kernel_id,
                workgroup_size_x: 256,
                grid_size_x: 128,
                row_count: 129,
                normalized_size: 129 * 127,
                backend: 1,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: kernel_symbol.to_owned(),
                device_symbol: device_symbol.to_owned(),
                target: target.to_owned(),
            }
        };
        for (kernel_id, kernel_symbol, device_symbol, _) in GFX1030_MMQ_CANDIDATE_IDENTITIES {
            assert!(
                validate_actual_dispatch(
                    Format::Mxfp8,
                    129,
                    2560,
                    127,
                    "gfx1030",
                    &candidate(*kernel_id, kernel_symbol, device_symbol, "gfx1030"),
                )
                .is_ok()
            );
            for (format, m, k, n, target) in [
                (Format::Mxfp8, 1, 2560, 127, "gfx1030"),
                (Format::Mxfp8, 129, 2559, 127, "gfx1030"),
                (Format::Mxfp8, 129, 2560, 0, "gfx1030"),
                (Format::Mxfp8, 129, 2560, 127, "gfx1201"),
                (Format::Mxfp6, 129, 2560, 127, "gfx1030"),
            ] {
                assert!(
                    validate_actual_dispatch(
                        format,
                        m,
                        k,
                        n,
                        target,
                        &candidate(*kernel_id, kernel_symbol, device_symbol, target),
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn phase67_provider_identities_are_stable() {
        assert_eq!(Phase67Provider::Row8.kernel_id(), 22);
        assert_eq!(Phase67Provider::Row8.force_value(), "1");
        assert_eq!(Phase67Provider::Control.kernel_id(), 27);
        assert_eq!(Phase67Provider::Control.force_value(), "8");
        assert_eq!(Phase67Provider::Col16.kernel_id(), 38);
        assert_eq!(Phase67Provider::Col16.force_value(), "16");
        assert_eq!(Phase67Provider::Col32.kernel_id(), 39);
        assert_eq!(Phase67Provider::Col32.force_value(), "32");
    }

    #[test]
    fn phase69_provider_identities_are_stable() {
        assert_eq!(Phase69Provider::Control.kernel_id(), 27);
        assert_eq!(Phase69Provider::Control.force_value(), "8");
        assert_eq!(Phase69Provider::Regscale.kernel_id(), 40);
        assert_eq!(Phase69Provider::Regscale.force_value(), "regscale");
        assert_eq!(Phase69Provider::Vector32.kernel_id(), 41);
        assert_eq!(Phase69Provider::Vector32.force_value(), "vector32");
        assert_eq!(Phase69Provider::Combined.kernel_id(), 42);
        assert_eq!(Phase69Provider::Combined.force_value(), "combined");
    }

    #[test]
    fn phase67_case_matrix_covers_short_intermediate_and_large_m_shapes() {
        let cases = phase67_cases();
        assert_eq!(cases.len(), phase66_cases().len() + 15);
        for shape in [
            (17, 2560, 32),
            (17, 2560, 9216),
            (128, 2560, 4096),
            (128, 4096, 2560),
            (128, 2560, 1024),
            (512, 2560, 9216),
            (512, 9216, 2560),
            (512, 2560, 8192),
            (512, 2560, 1024),
            (2048, 2560, 9216),
            (2048, 2560, 8192),
            (2048, 2560, 4096),
            (2048, 4096, 2560),
            (2048, 9216, 2560),
            (2048, 2560, 1024),
        ] {
            assert!(cases.iter().any(|case| (case.m, case.k, case.n) == shape));
        }
    }

    #[test]
    fn phase69_case_matrix_adds_missing_m512_projection_shapes() {
        let cases = phase69_cases();
        assert_eq!(cases.len(), phase67_cases().len() + 2);
        for shape in [(512, 2560, 4096), (512, 4096, 2560)] {
            assert!(cases.iter().any(|case| (case.m, case.k, case.n) == shape));
        }
    }
}
