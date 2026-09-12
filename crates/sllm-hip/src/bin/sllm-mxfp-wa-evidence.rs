//! Model-free OCP MXFP8 E4M3 W8A8 and MXFP6 E3M2 W6A6 GPU oracle.

use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
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
const PHASE67_CONTROL_FORCE_ENV: &str = "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS";
const PHASE69_CONTROL_KERNEL_ID: u32 = 27;
const PHASE69_VECTOR32_KERNEL_ID: u32 = 41;
const PHASE69_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69";
const PHASE69_VECTOR32_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1";
const PHASE69_VECTOR32_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1";
const PHASE70_CONTROL_KERNEL_ID: u32 = 29;
const PHASE70_GFX1201_CANDIDATE_KERNEL_ID: u32 = 44;
const PHASE70_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP6_PREFILL_FORCE_PHASE70";
const PHASE70_GFX1201_CANDIDATE_KERNEL_SYMBOL: &str =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.via-e4m3.v1";
const PHASE70_GFX1201_CANDIDATE_DEVICE_SYMBOL: &str =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x64_via_e4m3_v1";
const PHASE70_GFX1201_PACK4_N64_KERNEL_ID: u32 = 45;
const PHASE70_GFX1201_PACK4_N64_KERNEL_SYMBOL: &str =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4.v2";
const PHASE70_GFX1201_PACK4_N64_DEVICE_SYMBOL: &str = "sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_v2";
const PHASE74_CONTROL_KERNEL_ID: u32 = 25;
const PHASE74_CANDIDATE_KERNEL_ID: u32 = 47;
const PHASE74_CONTROL_FORCE_ENV: &str = "SLLM_MXFP6_PREFILL_FORCE_TILED16";
const PHASE74_CANDIDATE_FORCE_ENV: &str = "SLLM_MXFP6_PREFILL_FORCE_PHASE74";
const PHASE74_CANDIDATE_FORCE_VALUE: &str = "gfx1030-half2-32x32";
const PHASE74_CANDIDATE_KERNEL_SYMBOL: &str = "matmul.mxfp6.w6a6.gfx1030.half2.32x32.v1";
const PHASE74_CANDIDATE_DEVICE_SYMBOL: &str = "sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1";
const PHASE74_GFX1201_CONTROL_KERNEL_ID: u32 = PHASE70_GFX1201_PACK4_N64_KERNEL_ID;
const PHASE74_GFX1201_CONTROL_FORCE_VALUE: &str = "gfx1201-n64-pack4";
const PHASE74_GFX1201_CONTROL_KERNEL_SYMBOL: &str = PHASE70_GFX1201_PACK4_N64_KERNEL_SYMBOL;
const PHASE74_GFX1201_CONTROL_DEVICE_SYMBOL: &str = PHASE70_GFX1201_PACK4_N64_DEVICE_SYMBOL;
const PHASE74_GFX1201_CANDIDATE_KERNEL_ID: u32 = 48;
const PHASE74_GFX1201_CANDIDATE_FORCE_VALUE: &str = "gfx1201-swar-pack4";
const PHASE74_GFX1201_CANDIDATE_KERNEL_SYMBOL: &str =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4-swar.v1";
const PHASE74_GFX1201_CANDIDATE_DEVICE_SYMBOL: &str =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1";
const PHASE75_MXFP8_FORCE_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_PHASE75";
const PHASE75_MXFP6_FORCE_ENV: &str = "SLLM_MXFP6_PREFILL_FORCE_PHASE75";
const PHASE75_MXFP8_HALF2_IDENTITIES: &[(u32, &str, &str, &str, usize, usize)] = &[(
    55,
    "id55-mxfp8-half2-128x64-k32-double",
    "matmul.mxfp8.w8a8.gfx1030.half2.128x64.k32.double.v1",
    "sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_double_v1",
    128,
    64,
)];
const PHASE75_MXFP6_HALF2_IDENTITIES: &[(u32, &str, &str, &str, usize, usize)] = &[(
    57,
    "id57-mxfp6-half2-128x64-k32-double-pack4",
    "matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.pack4.v1",
    "sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_pack4_v1",
    128,
    64,
)];
const PHASE85_MXFP8_SMALL_M_KERNEL_ID: u32 = 97;
const PHASE85_MXFP8_SMALL_M_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.mmq.rows4.col8.v1";
const PHASE85_MXFP8_SMALL_M_DEVICE_SYMBOL: &str = "sllm_mxfp8_w8a8_mmq_rows4_col8_v1";
const PHASE85_MXFP6_SMALL_M_KERNEL_ID: u32 = 98;
const PHASE85_MXFP6_SMALL_M_KERNEL_SYMBOL: &str = "matmul.mxfp6.w6a6.mmq.rows4.col8.v1";
const PHASE85_MXFP6_SMALL_M_DEVICE_SYMBOL: &str = "sllm_mxfp6_w6a6_mmq_rows4_col8_v1";
const PHASE85_SMALL_M_FORCE_ENV: &str = "SLLM_PHASE85_MX_WA_FORCE_SMALL_M";
const MXFP8_FORCE_ENVIRONMENTS: &[&str] = &[
    "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
    "SLLM_MXFP8_PREFILL_FORCE_ROW8",
    "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS",
    "SLLM_MXFP8_PREFILL_FORCE_TILED16",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_N16_GFX1201",
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_WEIGHT_GFX1201",
    PHASE66_CONTROL_FORCE_ENV,
    PHASE66_CANDIDATE_FORCE_ENV,
    PHASE69_CANDIDATE_FORCE_ENV,
    PHASE75_MXFP8_FORCE_ENV,
    PHASE85_SMALL_M_FORCE_ENV,
];
const MXFP6_FORCE_ENVIRONMENTS: &[&str] = &[
    "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
    "SLLM_MXFP6_PREFILL_FORCE_ROW8",
    PHASE74_CONTROL_FORCE_ENV,
    "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS",
    PHASE70_CANDIDATE_FORCE_ENV,
    PHASE74_CANDIDATE_FORCE_ENV,
    PHASE75_MXFP6_FORCE_ENV,
    PHASE85_SMALL_M_FORCE_ENV,
];
const GFX1201_WMMA_CANDIDATE_IDENTITIES: &[(u32, &str, &str, u32)] = &[
    (
        34,
        "matmul.mxfp8.w8a8.gfx1201.wmma128x64.direct.v1",
        "sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1",
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
const GFX1030_MMQ_CANDIDATE_IDENTITIES: &[(u32, &str, &str, u32)] = &[(
    41,
    "matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1",
    "sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1",
    256,
)];
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

const PHASE85_SCHEMA_VERSION: &str = "phase85-mxfp-shapes-v1";
const PHASE85_MAX_CASES: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase85FormatSelection {
    Mxfp8,
    Mxfp6,
    Both,
}

impl Phase85FormatSelection {
    fn accepts(self, format: Format) -> bool {
        matches!(
            (self, format),
            (Self::Mxfp8, Format::Mxfp8) | (Self::Mxfp6, Format::Mxfp6) | (Self::Both, _)
        )
    }
}

#[derive(Clone, Copy)]
enum Phase85Provider {
    Baseline,
    Forced {
        environment: &'static str,
        value: &'static str,
    },
}

impl Phase85Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline-current-selector",
            Self::Forced { environment, .. } => environment,
        }
    }

    fn force(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Baseline => None,
            Self::Forced { environment, value } => Some((environment, value)),
        }
    }
}

#[derive(Deserialize)]
struct Phase85Manifest {
    schema_version: String,
    cases: Vec<Phase85ManifestCase>,
}

#[derive(Deserialize)]
struct Phase85ManifestCase {
    case_id: String,
    format: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    role: Option<String>,
    m: usize,
    k: usize,
    n: usize,
    #[serde(default)]
    phase: Option<usize>,
    oracle: String,
}

#[derive(Clone, Copy)]
struct Phase85CaseSpec {
    case: CaseSpec,
    tags: &'static str,
    role: &'static str,
}

#[derive(Clone, Copy)]
enum OracleSelection {
    Full,
    FixedSample(&'static [(usize, usize)]),
    BoundarySample,
}

impl OracleSelection {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::FixedSample(_) => "fixed-sample",
            Self::BoundarySample => "boundary-sample",
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
            Self::BoundarySample => {
                let rows = [0, 1, m / 2, m.saturating_sub(2), m.saturating_sub(1)];
                let columns = [
                    0,
                    1,
                    31,
                    32,
                    63,
                    64,
                    n / 2,
                    n.saturating_sub(2),
                    n.saturating_sub(1),
                ];
                let mut indices = Vec::with_capacity(rows.len() * columns.len());
                for row in rows.into_iter().filter(|row| *row < m) {
                    for column in columns.into_iter().filter(|column| *column < n) {
                        indices.push(row * n + column);
                    }
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
}

impl Phase67Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Row8 => "id22-row8-control",
            Self::Control => "id27-col8-control",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Row8 => "SLLM_MXFP8_PREFILL_FORCE_ROW8",
            Self::Control => PHASE67_CONTROL_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Row8 => "1",
            Self::Control => "8",
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Row8 => PHASE67_ROW8_KERNEL_ID,
            Self::Control => PHASE67_CONTROL_KERNEL_ID,
        }
    }
}

#[derive(Clone, Copy)]
enum Phase69Provider {
    Control,
    Vector32,
}

impl Phase69Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "id27-col8-control",
            Self::Vector32 => "id41-vector32-candidate",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Control => PHASE67_CONTROL_FORCE_ENV,
            Self::Vector32 => PHASE69_CANDIDATE_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Control => "8",
            Self::Vector32 => "vector32",
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Control => PHASE69_CONTROL_KERNEL_ID,
            Self::Vector32 => PHASE69_VECTOR32_KERNEL_ID,
        }
    }
}

#[derive(Clone, Copy)]
enum Phase70Provider {
    Gfx1201Default,
    Tiled16,
    Control,
    Gfx1201Candidate,
    Gfx1201Pack4N64,
}

impl Phase70Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Gfx1201Default => "id45-gfx1201-pack4-n64-default",
            Self::Tiled16 => "id25-tiled16-control",
            Self::Control => "id29-col8-control",
            Self::Gfx1201Candidate => "id44-gfx1201-via-e4m3-n64-candidate",
            Self::Gfx1201Pack4N64 => "id45-gfx1201-pack4-n64-candidate",
        }
    }

    fn force_environment(self) -> Option<&'static str> {
        match self {
            Self::Gfx1201Default => None,
            Self::Tiled16 => Some("SLLM_MXFP6_PREFILL_FORCE_TILED16"),
            Self::Control => Some(PHASE67_CONTROL_FORCE_ENV),
            Self::Gfx1201Candidate | Self::Gfx1201Pack4N64 => Some(PHASE70_CANDIDATE_FORCE_ENV),
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Gfx1201Default => "",
            Self::Tiled16 => "1",
            Self::Control => "8",
            Self::Gfx1201Candidate => "gfx1201-n64",
            Self::Gfx1201Pack4N64 => "gfx1201-n64-pack4",
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Gfx1201Default => PHASE70_GFX1201_PACK4_N64_KERNEL_ID,
            Self::Tiled16 => 25,
            Self::Control => PHASE70_CONTROL_KERNEL_ID,
            Self::Gfx1201Candidate => PHASE70_GFX1201_CANDIDATE_KERNEL_ID,
            Self::Gfx1201Pack4N64 => PHASE70_GFX1201_PACK4_N64_KERNEL_ID,
        }
    }
}

#[derive(Clone, Copy)]
enum Phase74Provider {
    Control,
    Candidate,
    Gfx1201Control,
    Gfx1201Candidate,
}

impl Phase74Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "id25-gfx1030-control",
            Self::Candidate => "id47-gfx1030-half2-candidate",
            Self::Gfx1201Control => "id45-gfx1201-pack4-control",
            Self::Gfx1201Candidate => "id48-gfx1201-pack4-swar-candidate",
        }
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Control => PHASE74_CONTROL_FORCE_ENV,
            Self::Candidate => PHASE74_CANDIDATE_FORCE_ENV,
            Self::Gfx1201Control => PHASE70_CANDIDATE_FORCE_ENV,
            Self::Gfx1201Candidate => PHASE74_CANDIDATE_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Control => "1",
            Self::Candidate => PHASE74_CANDIDATE_FORCE_VALUE,
            Self::Gfx1201Control => PHASE74_GFX1201_CONTROL_FORCE_VALUE,
            Self::Gfx1201Candidate => PHASE74_GFX1201_CANDIDATE_FORCE_VALUE,
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Control => PHASE74_CONTROL_KERNEL_ID,
            Self::Candidate => PHASE74_CANDIDATE_KERNEL_ID,
            Self::Gfx1201Control => PHASE74_GFX1201_CONTROL_KERNEL_ID,
            Self::Gfx1201Candidate => PHASE74_GFX1201_CANDIDATE_KERNEL_ID,
        }
    }

    fn kernel_symbol(self) -> &'static str {
        match self {
            Self::Control => "matmul.mxfp6.w6a6.e3m2.block32.prefill.tiled16.v3",
            Self::Candidate => PHASE74_CANDIDATE_KERNEL_SYMBOL,
            Self::Gfx1201Control => PHASE74_GFX1201_CONTROL_KERNEL_SYMBOL,
            Self::Gfx1201Candidate => PHASE74_GFX1201_CANDIDATE_KERNEL_SYMBOL,
        }
    }

    fn device_symbol(self) -> &'static str {
        match self {
            Self::Control => "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3",
            Self::Candidate => PHASE74_CANDIDATE_DEVICE_SYMBOL,
            Self::Gfx1201Control => PHASE74_GFX1201_CONTROL_DEVICE_SYMBOL,
            Self::Gfx1201Candidate => PHASE74_GFX1201_CANDIDATE_DEVICE_SYMBOL,
        }
    }

    fn target(self) -> &'static str {
        match self {
            Self::Control | Self::Candidate => "gfx1030",
            Self::Gfx1201Control | Self::Gfx1201Candidate => "gfx1201",
        }
    }
}

#[derive(Clone, Copy)]
enum Phase75Provider {
    Mxfp8Id41Control,
    Mxfp8Half2_128x64K32Double,
    Mxfp6Id47Control,
    Mxfp6Half2_128x64K32DoublePack4,
}

impl Phase75Provider {
    fn identity(self) -> (u32, &'static str, &'static str, &'static str, usize, usize) {
        match self {
            Self::Mxfp8Id41Control => (
                PHASE69_VECTOR32_KERNEL_ID,
                "id41-mxfp8-vector32-control",
                PHASE69_VECTOR32_KERNEL_SYMBOL,
                PHASE69_VECTOR32_DEVICE_SYMBOL,
                8,
                8,
            ),
            Self::Mxfp8Half2_128x64K32Double => PHASE75_MXFP8_HALF2_IDENTITIES[0],
            Self::Mxfp6Id47Control => (
                PHASE74_CANDIDATE_KERNEL_ID,
                "id47-mxfp6-half2-32x32-control",
                PHASE74_CANDIDATE_KERNEL_SYMBOL,
                PHASE74_CANDIDATE_DEVICE_SYMBOL,
                32,
                32,
            ),
            Self::Mxfp6Half2_128x64K32DoublePack4 => PHASE75_MXFP6_HALF2_IDENTITIES[0],
        }
    }

    fn format(self) -> Format {
        match self {
            Self::Mxfp8Id41Control | Self::Mxfp8Half2_128x64K32Double => Format::Mxfp8,
            Self::Mxfp6Id47Control | Self::Mxfp6Half2_128x64K32DoublePack4 => Format::Mxfp6,
        }
    }

    fn name(self) -> &'static str {
        self.identity().1
    }

    fn kernel_id(self) -> u32 {
        self.identity().0
    }

    fn kernel_symbol(self) -> &'static str {
        self.identity().2
    }

    fn device_symbol(self) -> &'static str {
        self.identity().3
    }

    fn rows(self) -> usize {
        self.identity().4
    }

    fn columns(self) -> usize {
        self.identity().5
    }

    fn force_environment(self) -> &'static str {
        match self {
            Self::Mxfp8Id41Control => PHASE69_CANDIDATE_FORCE_ENV,
            Self::Mxfp8Half2_128x64K32Double => PHASE75_MXFP8_FORCE_ENV,
            Self::Mxfp6Id47Control => PHASE74_CANDIDATE_FORCE_ENV,
            Self::Mxfp6Half2_128x64K32DoublePack4 => PHASE75_MXFP6_FORCE_ENV,
        }
    }

    fn force_value(self) -> &'static str {
        match self {
            Self::Mxfp8Id41Control => "vector32",
            Self::Mxfp8Half2_128x64K32Double => "half2-128x64-k32-double",
            Self::Mxfp6Id47Control => PHASE74_CANDIDATE_FORCE_VALUE,
            Self::Mxfp6Half2_128x64K32DoublePack4 => "half2-128x64-k32-double-pack4",
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
    Phase70 {
        repeats: usize,
        provider: Phase70Provider,
        production_shape: bool,
        wide_n_shape: bool,
    },
    Phase74 {
        repeats: usize,
        provider: Phase74Provider,
        production_shape: bool,
    },
    Phase75 {
        repeats: usize,
        provider: Phase75Provider,
        production_shape: bool,
    },
    Phase84 {
        repeats: usize,
        format: Format,
    },
    Phase85 {
        repeats: usize,
        specs: &'static [Phase85CaseSpec],
        manifest: &'static str,
        provider: Phase85Provider,
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
            Self::Phase70 { repeats, .. } => repeats,
            Self::Phase74 { repeats, .. } => repeats,
            Self::Phase75 { repeats, .. } => repeats,
            Self::Phase84 { repeats, .. } => repeats,
            Self::Phase85 { repeats, .. } => repeats,
        }
    }

    fn schema_version(self) -> &'static str {
        match self {
            Self::Phase62 { .. } => "sllm-ocp-mxfp8-mxfp6-wa-gpu-v1",
            Self::Phase63 { .. } => "sllm-phase63-mxfp8-matrix-operator-gpu-v1",
            Self::Phase66 { .. } => "sllm-phase66-mxfp8-wide-n-provider-gpu-v1",
            Self::Phase67 { .. } => "sllm-phase67-gfx1030-mxfp8-tile-provider-gpu-v1",
            Self::Phase69 { .. } => "sllm-phase69-gfx1030-mxfp8-software-mmq-provider-gpu-v1",
            Self::Phase70 { .. } => "sllm-phase70-rdna-mxfp6-via-e4m3-provider-gpu-v1",
            Self::Phase74 { .. } => "sllm-phase74-rdna-mxfp6-provider-gpu-v1",
            Self::Phase75 { .. } => "sllm-phase75-gfx1030-shared-half2-provider-gpu-v1",
            Self::Phase84 { .. } => "sllm-phase84-qwen38-mtp-mx-provider-gpu-v1",
            Self::Phase85 { .. } => PHASE85_SCHEMA_VERSION,
        }
    }

    fn warmup_count(self) -> usize {
        match self {
            Self::Phase69 { .. } => 2,
            Self::Phase70 { .. } => 1,
            Self::Phase74 { .. } => 1,
            Self::Phase75 { .. } => 1,
            Self::Phase84 { .. } => 1,
            Self::Phase85 { .. } => 3,
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
            Self::Phase70 { .. } => Some("phase70-provider"),
            Self::Phase74 { .. } => Some("phase74-provider"),
            Self::Phase75 { .. } => Some("phase75-provider"),
            Self::Phase84 { .. } => Some("phase84"),
            Self::Phase85 { .. } => Some("phase85-shapes"),
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

    fn phase70_provider(self) -> Option<Phase70Provider> {
        match self {
            Self::Phase70 { provider, .. } => Some(provider),
            _ => None,
        }
    }

    fn phase74_provider(self) -> Option<Phase74Provider> {
        match self {
            Self::Phase74 { provider, .. } => Some(provider),
            _ => None,
        }
    }

    fn phase75_provider(self) -> Option<Phase75Provider> {
        match self {
            Self::Phase75 { provider, .. } => Some(provider),
            _ => None,
        }
    }

    fn phase85_provider(self) -> Option<Phase85Provider> {
        match self {
            Self::Phase85 { provider, .. } => Some(provider),
            _ => None,
        }
    }
}

#[derive(Serialize)]
struct CaseReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    case_id: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase85_tags: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase85_role: Option<&'static str>,
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
    sampled_row_top1: Option<Vec<RowTop1>>,
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
    phase70_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase70_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase74_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase74_candidate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase75_provider: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase75_candidate: Option<bool>,
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
struct RowTop1 {
    row: usize,
    column: usize,
    value: f32,
    margin_to_second: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase85_manifest: Option<&'static str>,
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
    wide_n_shape_included: Option<bool>,
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
    if matches!(phase, 1840 | 1841 | 1851 | 1852) {
        return phase84_m1_range_boundary_matrix(rows, columns);
    }
    if phase >= 120 {
        return (0..rows * columns)
            .map(|index| {
                let mut bits =
                    (index as u64).wrapping_add((phase as u64).wrapping_mul(0x9e3779b97f4a7c15));
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                let unit = ((bits >> 40) as u32) as f32 / 16_777_215.0;
                bf16((unit * 2.0 - 1.0) * 4.0)
            })
            .collect();
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

fn phase84_m1_range_boundary_matrix(rows: usize, columns: usize) -> Vec<u16> {
    let mut values = vec![bf16(1.0); rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            values[row * columns + column] = match column / 32 {
                // 65536 is finite in BF16 but exceeds IEEE FP16's max finite
                // value; it exercises the scalar FP32 decode contract.
                0 => bf16(65_536.0),
                // This block drives a nonzero minimum-scale path.
                1 => bf16(2.0_f32.powi(-17)),
                _ => bf16(1.0625),
            };
        }
    }
    values
}

fn validate_phase84_m1_range_boundary(
    source: &[u16],
    quantized: &QuantizedMx,
    rows: usize,
    k: usize,
) -> Result<(), String> {
    if rows == 0 || k < 32 || source.len() != rows * k {
        return Err("Phase 84 M=1 range fixture has an invalid shape".to_owned());
    }
    if source[0] != bf16(65_536.0) {
        return Err("Phase 84 M=1 range fixture lost the >FP16 finite input".to_owned());
    }
    let decoded = quantized.dequantize().map_err(|error| error.to_string())?;
    for row in 0..rows {
        let large = decoded[row * k].abs();
        if !large.is_finite() || large <= 65_504.0 {
            return Err(format!(
                "Phase 84 M=1 range fixture large value was not preserved: row={row} value={large}"
            ));
        }
        if k >= 64 {
            let tiny = decoded[row * k + 32].abs();
            if !tiny.is_finite() || tiny == 0.0 || tiny >= 1.0e-3 {
                return Err(format!(
                    "Phase 84 M=1 range fixture small-scale value was not preserved: row={row} value={tiny}"
                ));
            }
        }
    }
    Ok(())
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

fn phase85_small_m_shape(m: usize, k: usize, n: usize) -> bool {
    (2..=4).contains(&m) && k >= 2048 && k % 32 == 0 && (1024..=32768).contains(&n)
}

fn phase85_small_m_forced_shape(m: usize, k: usize, n: usize) -> bool {
    (2..=4).contains(&m) && k != 0 && k % 32 == 0 && n != 0
}

fn phase85_legacy_prefill_shape(m: usize, k: usize, n: usize) -> bool {
    m > 1 && k != 0 && k % 32 == 0 && n != 0
}

fn phase85_small_m_adopted_shape(
    format: Format,
    m: usize,
    k: usize,
    n: usize,
    target: &str,
) -> bool {
    if !phase85_small_m_shape(m, k, n) {
        return false;
    }
    match format {
        Format::Mxfp8 => match target {
            "gfx1030" => k / n != 1,
            "gfx1201" => true,
            _ => false,
        },
        Format::Mxfp6 => target == "gfx1030" || target == "gfx1201",
    }
}

fn phase85_small_m_identity(format: Format) -> (u32, &'static str, &'static str) {
    match format {
        Format::Mxfp8 => (
            PHASE85_MXFP8_SMALL_M_KERNEL_ID,
            PHASE85_MXFP8_SMALL_M_KERNEL_SYMBOL,
            PHASE85_MXFP8_SMALL_M_DEVICE_SYMBOL,
        ),
        Format::Mxfp6 => (
            PHASE85_MXFP6_SMALL_M_KERNEL_ID,
            PHASE85_MXFP6_SMALL_M_KERNEL_SYMBOL,
            PHASE85_MXFP6_SMALL_M_DEVICE_SYMBOL,
        ),
    }
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
            19 | 22 | 24 | 26 | 27 | 30 | 31 | 34 | 36 | 37 | 41 | 55 | 97
        ),
        (Format::Mxfp6, _) => matches!(
            dispatch.kernel_id,
            21 | 23 | 25 | 28 | 29 | 44 | 45 | 47 | 48 | 57 | 98
        ),
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
    if dispatch.kernel_id == PHASE85_MXFP8_SMALL_M_KERNEL_ID
        || dispatch.kernel_id == PHASE85_MXFP6_SMALL_M_KERNEL_ID
    {
        let (expected_id, expected_kernel_symbol, expected_device_symbol) =
            phase85_small_m_identity(format);
        let expected_grid_size_x = m
            .div_ceil(4)
            .checked_mul(n.div_ceil(8))
            .ok_or_else(|| "Phase 85 small-M grid size overflowed usize".to_owned())?;
        let expected_grid_size_x = u32::try_from(expected_grid_size_x)
            .map_err(|_| "Phase 85 small-M grid size exceeds u32".to_owned())?;
        if dispatch.kernel_id != expected_id
            || !phase85_small_m_adopted_shape(format, m, k, n, target)
            || dispatch.kernel_symbol != expected_kernel_symbol
            || dispatch.device_symbol != expected_device_symbol
            || dispatch.workgroup_size_x != 128
            || dispatch.grid_size_x != expected_grid_size_x
        {
            return Err(format!(
                "Phase 85 small-M kernel {} escaped its adopted scope: {dispatch:?}",
                dispatch.kernel_id
            ));
        }
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
    if dispatch.kernel_id == PHASE70_GFX1201_PACK4_N64_KERNEL_ID
        && (format != Format::Mxfp6
            || target != "gfx1201"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != PHASE70_GFX1201_PACK4_N64_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE70_GFX1201_PACK4_N64_DEVICE_SYMBOL
            || dispatch.workgroup_size_x != 256)
    {
        return Err(format!(
            "Phase 70 gfx1201 packed N64 kernel {} escaped its exact MXFP6 prefill scope",
            dispatch.kernel_id
        ));
    }
    if dispatch.kernel_id == PHASE70_GFX1201_CANDIDATE_KERNEL_ID
        && (format != Format::Mxfp6
            || target != "gfx1201"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != PHASE70_GFX1201_CANDIDATE_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE70_GFX1201_CANDIDATE_DEVICE_SYMBOL
            || dispatch.workgroup_size_x != 256)
    {
        return Err(format!(
            "Phase 70 gfx1201 candidate kernel {} escaped its exact MXFP6 prefill scope",
            dispatch.kernel_id
        ));
    }
    if dispatch.kernel_id == PHASE74_CANDIDATE_KERNEL_ID {
        let expected_grid_size_x = m
            .div_ceil(32)
            .checked_mul(n.div_ceil(32))
            .ok_or_else(|| "Phase 74 candidate grid size overflowed usize".to_owned())?;
        let expected_grid_size_x = u32::try_from(expected_grid_size_x)
            .map_err(|_| "Phase 74 candidate grid size exceeds u32".to_owned())?;
        if format != Format::Mxfp6
            || target != "gfx1030"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != PHASE74_CANDIDATE_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE74_CANDIDATE_DEVICE_SYMBOL
            || dispatch.workgroup_size_x != 256
            || dispatch.grid_size_x != expected_grid_size_x
        {
            return Err(format!(
                "Phase 74 candidate kernel {} escaped its exact gfx1030 MXFP6 half2 32x32 scope",
                dispatch.kernel_id
            ));
        }
    }
    if dispatch.kernel_id == PHASE74_GFX1201_CANDIDATE_KERNEL_ID {
        let expected_grid_size_x = u32::try_from(n.div_ceil(64))
            .map_err(|_| "Phase 74 gfx1201 candidate grid size exceeds u32".to_owned())?;
        if format != Format::Mxfp6
            || target != "gfx1201"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != PHASE74_GFX1201_CANDIDATE_KERNEL_SYMBOL
            || dispatch.device_symbol != PHASE74_GFX1201_CANDIDATE_DEVICE_SYMBOL
            || dispatch.workgroup_size_x != 256
            || dispatch.grid_size_x != expected_grid_size_x
        {
            return Err(format!(
                "Phase 74 candidate kernel {} escaped its exact gfx1201 MXFP6 WMMA 128x64 pack4 SWAR scope",
                dispatch.kernel_id
            ));
        }
    }
    if let Some((_, _, kernel_symbol, device_symbol, rows, columns)) =
        PHASE75_MXFP8_HALF2_IDENTITIES
            .iter()
            .find(|(kernel_id, _, _, _, _, _)| *kernel_id == dispatch.kernel_id)
    {
        let expected_grid_size_x = m
            .div_ceil(*rows)
            .checked_mul(n.div_ceil(*columns))
            .ok_or_else(|| "Phase 75 candidate grid size overflowed usize".to_owned())?;
        let expected_grid_size_x = u32::try_from(expected_grid_size_x)
            .map_err(|_| "Phase 75 candidate grid size exceeds u32".to_owned())?;
        if format != Format::Mxfp8
            || target != "gfx1030"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != *kernel_symbol
            || dispatch.device_symbol != *device_symbol
            || dispatch.workgroup_size_x != 256
            || dispatch.grid_size_x != expected_grid_size_x
        {
            return Err(format!(
                "Phase 75 candidate kernel {} escaped its exact gfx1030 MXFP8 half2 scope",
                dispatch.kernel_id
            ));
        }
    }
    if let Some((_, _, kernel_symbol, device_symbol, rows, columns)) =
        PHASE75_MXFP6_HALF2_IDENTITIES
            .iter()
            .find(|(kernel_id, _, _, _, _, _)| *kernel_id == dispatch.kernel_id)
    {
        let expected_grid_size_x = m
            .div_ceil(*rows)
            .checked_mul(n.div_ceil(*columns))
            .ok_or_else(|| "Phase 75 MXFP6 candidate grid size overflowed usize".to_owned())?;
        let expected_grid_size_x = u32::try_from(expected_grid_size_x)
            .map_err(|_| "Phase 75 MXFP6 candidate grid size exceeds u32".to_owned())?;
        if format != Format::Mxfp6
            || target != "gfx1030"
            || m <= 1
            || k == 0
            || k % 32 != 0
            || n == 0
            || dispatch.kernel_symbol != *kernel_symbol
            || dispatch.device_symbol != *device_symbol
            || dispatch.workgroup_size_x != 256
            || dispatch.grid_size_x != expected_grid_size_x
        {
            return Err(format!(
                "Phase 75 candidate kernel {} escaped its exact gfx1030 MXFP6 half2 scope",
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

fn validate_phase74_provider_dispatch(
    provider: Phase74Provider,
    m: usize,
    n: usize,
    target: &str,
    dispatch: &DispatchEvidence,
) -> Result<(), String> {
    let expected_grid_size_x = match provider {
        Phase74Provider::Control => n.div_ceil(16),
        Phase74Provider::Candidate => m
            .div_ceil(32)
            .checked_mul(n.div_ceil(32))
            .ok_or_else(|| "Phase 74 provider grid size overflowed usize".to_owned())?,
        Phase74Provider::Gfx1201Control | Phase74Provider::Gfx1201Candidate => n.div_ceil(64),
    };
    let expected_grid_size_x = u32::try_from(expected_grid_size_x)
        .map_err(|_| "Phase 74 provider grid size exceeds u32".to_owned())?;
    if target != provider.target()
        || dispatch.target != provider.target()
        || dispatch.kernel_id != provider.kernel_id()
        || dispatch.kernel_symbol != provider.kernel_symbol()
        || dispatch.device_symbol != provider.device_symbol()
        || dispatch.workgroup_size_x != 256
        || dispatch.grid_size_x != expected_grid_size_x
    {
        return Err(format!(
            "Phase 74 {} expected kernel {} dispatch but observed {dispatch:?}",
            provider.name(),
            provider.kernel_id()
        ));
    }
    Ok(())
}

fn validate_phase75_provider_dispatch(
    provider: Phase75Provider,
    m: usize,
    n: usize,
    target: &str,
    dispatch: &DispatchEvidence,
) -> Result<(), String> {
    let expected_grid_size_x = if matches!(provider, Phase75Provider::Mxfp8Id41Control) {
        m.div_ceil(8)
            .checked_mul(n.div_ceil(8))
            .ok_or_else(|| "Phase 75 control grid size overflowed usize".to_owned())?
    } else {
        m.div_ceil(provider.rows())
            .checked_mul(n.div_ceil(provider.columns()))
            .ok_or_else(|| "Phase 75 candidate grid size overflowed usize".to_owned())?
    };
    let expected_grid_size_x = u32::try_from(expected_grid_size_x)
        .map_err(|_| "Phase 75 provider grid size exceeds u32".to_owned())?;
    if target != "gfx1030"
        || dispatch.target != "gfx1030"
        || dispatch.kernel_id != provider.kernel_id()
        || dispatch.kernel_symbol != provider.kernel_symbol()
        || dispatch.device_symbol != provider.device_symbol()
        || dispatch.workgroup_size_x != 256
        || dispatch.grid_size_x != expected_grid_size_x
    {
        return Err(format!(
            "Phase 75 {} expected kernel {} dispatch but observed {dispatch:?}",
            provider.name(),
            provider.kernel_id()
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Phase85Grid {
    Linear,
    ColumnsOnly { columns: usize },
    Tiled { rows: usize, columns: usize },
}

#[derive(Clone, Copy)]
struct Phase85ForcedContract {
    kernel_id: u32,
    kernel_symbol: &'static str,
    device_symbol: &'static str,
    workgroup_size_x: u32,
    grid: Phase85Grid,
    target: Option<&'static str>,
}

fn phase85_forced_contract(
    format: Format,
    environment: &str,
    value: &str,
) -> Result<Phase85ForcedContract, String> {
    let contract = match (format, environment, value) {
        (Format::Mxfp8, PHASE85_SMALL_M_FORCE_ENV, "1") => Phase85ForcedContract {
            kernel_id: PHASE85_MXFP8_SMALL_M_KERNEL_ID,
            kernel_symbol: PHASE85_MXFP8_SMALL_M_KERNEL_SYMBOL,
            device_symbol: PHASE85_MXFP8_SMALL_M_DEVICE_SYMBOL,
            workgroup_size_x: 128,
            grid: Phase85Grid::Tiled {
                rows: 4,
                columns: 8,
            },
            target: None,
        },
        (Format::Mxfp6, PHASE85_SMALL_M_FORCE_ENV, "1") => Phase85ForcedContract {
            kernel_id: PHASE85_MXFP6_SMALL_M_KERNEL_ID,
            kernel_symbol: PHASE85_MXFP6_SMALL_M_KERNEL_SYMBOL,
            device_symbol: PHASE85_MXFP6_SMALL_M_DEVICE_SYMBOL,
            workgroup_size_x: 128,
            grid: Phase85Grid::Tiled {
                rows: 4,
                columns: 8,
            },
            target: None,
        },
        (Format::Mxfp8, "SLLM_MX_WA_PREFILL_FORCE_BASELINE", "1") => Phase85ForcedContract {
            kernel_id: 19,
            kernel_symbol: "matmul.mxfp8.w8a8.e4m3.block32.prefill.v1",
            device_symbol: "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1",
            workgroup_size_x: 256,
            grid: Phase85Grid::Linear,
            target: None,
        },
        (Format::Mxfp6, "SLLM_MX_WA_PREFILL_FORCE_BASELINE", "1") => Phase85ForcedContract {
            kernel_id: 21,
            kernel_symbol: "matmul.mxfp6.w6a6.e3m2.block32.prefill.v1",
            device_symbol: "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1",
            workgroup_size_x: 256,
            grid: Phase85Grid::Linear,
            target: None,
        },
        (Format::Mxfp8, "SLLM_MXFP8_PREFILL_FORCE_ROW8", "1") => Phase85ForcedContract {
            kernel_id: 22,
            kernel_symbol: "matmul.mxfp8.w8a8.e4m3.block32.prefill.row8.v2",
            device_symbol: "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 1,
            },
            target: None,
        },
        (Format::Mxfp6, "SLLM_MXFP6_PREFILL_FORCE_ROW8", "1") => Phase85ForcedContract {
            kernel_id: 23,
            kernel_symbol: "matmul.mxfp6.w6a6.e3m2.block32.prefill.row8.v2",
            device_symbol: "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 1,
            },
            target: None,
        },
        (Format::Mxfp8, "SLLM_MXFP8_PREFILL_FORCE_TILED16", "1") => Phase85ForcedContract {
            kernel_id: 24,
            kernel_symbol: "matmul.mxfp8.w8a8.e4m3.block32.prefill.tiled16.v3",
            device_symbol: "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3",
            workgroup_size_x: 256,
            grid: Phase85Grid::ColumnsOnly { columns: 16 },
            target: None,
        },
        (Format::Mxfp6, "SLLM_MXFP6_PREFILL_FORCE_TILED16", "1") => Phase85ForcedContract {
            kernel_id: 25,
            kernel_symbol: "matmul.mxfp6.w6a6.e3m2.block32.prefill.tiled16.v3",
            device_symbol: "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3",
            workgroup_size_x: 256,
            grid: Phase85Grid::ColumnsOnly { columns: 16 },
            target: None,
        },
        (Format::Mxfp8, "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS", "4") => Phase85ForcedContract {
            kernel_id: 26,
            kernel_symbol: "matmul.mxfp8.w8a8.e4m3.block32.prefill.mmq-col4.v4",
            device_symbol: "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 4,
            },
            target: None,
        },
        (Format::Mxfp8, "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS", "8") => Phase85ForcedContract {
            kernel_id: 27,
            kernel_symbol: "matmul.mxfp8.w8a8.e4m3.block32.prefill.mmq-col8.v4",
            device_symbol: "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 8,
            },
            target: None,
        },
        (Format::Mxfp6, "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS", "4") => Phase85ForcedContract {
            kernel_id: 28,
            kernel_symbol: "matmul.mxfp6.w6a6.e3m2.block32.prefill.mmq-col4.v4",
            device_symbol: "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 4,
            },
            target: None,
        },
        (Format::Mxfp6, "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS", "8") => Phase85ForcedContract {
            kernel_id: 29,
            kernel_symbol: "matmul.mxfp6.w6a6.e3m2.block32.prefill.mmq-col8.v4",
            device_symbol: "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4",
            workgroup_size_x: 256,
            grid: Phase85Grid::Tiled {
                rows: 8,
                columns: 8,
            },
            target: None,
        },
        _ => {
            return Err(format!(
                "unsupported Phase 85 forced provider {environment}={value} for {}",
                format.name()
            ));
        }
    };
    Ok(contract)
}

fn validate_phase85_dispatch(
    provider: Phase85Provider,
    format: Format,
    m: usize,
    k: usize,
    n: usize,
    target: &str,
    dispatch: &DispatchEvidence,
) -> Result<(), String> {
    let normalized_size = m
        .checked_mul(n)
        .ok_or_else(|| "Phase 85 output element count overflowed usize".to_owned())?;
    let Phase85Provider::Forced { environment, value } = provider else {
        return Err("Phase 85 dispatch validation requires a forced provider".to_owned());
    };
    let contract = phase85_forced_contract(format, environment, value)?;
    let expected_grid_size_x = match contract.grid {
        Phase85Grid::Linear => m
            .checked_mul(n)
            .ok_or_else(|| "Phase 85 output grid size overflowed usize".to_owned())?,
        Phase85Grid::ColumnsOnly { columns } => n.div_ceil(columns),
        Phase85Grid::Tiled { rows, columns } => {
            m.div_ceil(rows)
                .checked_mul(n.div_ceil(columns))
                .ok_or_else(|| "Phase 85 output grid size overflowed usize".to_owned())?
        }
    };
    let expected_grid_size_x = u32::try_from(expected_grid_size_x)
        .map_err(|_| "Phase 85 output grid size exceeds u32".to_owned())?;
    let format_fragment = match format {
        Format::Mxfp8 => "mxfp8",
        Format::Mxfp6 => "mxfp6",
    };
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 2
        || dispatch.kernel_id != contract.kernel_id
        || if environment == PHASE85_SMALL_M_FORCE_ENV {
            !phase85_small_m_forced_shape(m, k, n)
        } else {
            !phase85_legacy_prefill_shape(m, k, n)
        }
        || (target != "gfx1030" && target != "gfx1201")
        || dispatch.row_count != m as u64
        || dispatch.normalized_size != normalized_size as u64
        || dispatch.backend != 1
        || !dispatch.kernel_symbol.contains(format_fragment)
        || !dispatch.device_symbol.contains(format_fragment)
        || dispatch.kernel_symbol != contract.kernel_symbol
        || dispatch.device_symbol != contract.device_symbol
        || dispatch.workgroup_size_x != contract.workgroup_size_x
        || dispatch.grid_size_x != expected_grid_size_x
        || contract.target.is_some_and(|expected| expected != target)
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
    {
        return Err(format!(
            "unexpected {} Phase 85 candidate dispatch: {dispatch:?}",
            format.name()
        ));
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

fn sampled_row_top1(output: &[u8], n: usize, indices: &[usize]) -> Vec<RowTop1> {
    let mut rows: Vec<_> = indices.iter().map(|index| index / n).collect();
    rows.sort_unstable();
    rows.dedup();
    rows.into_iter()
        .map(|row| {
            let mut best = (usize::MAX, f32::NEG_INFINITY);
            let mut second = f32::NEG_INFINITY;
            for column in 0..n {
                let byte_index = (row * n + column) * 2;
                let value = from_bf16(u16::from_le_bytes([
                    output[byte_index],
                    output[byte_index + 1],
                ]));
                if value > best.1 {
                    second = best.1;
                    best = (column, value);
                } else if value > second {
                    second = value;
                }
            }
            RowTop1 {
                row,
                column: best.0,
                value: best.1,
                margin_to_second: best.1 - second,
            }
        })
        .collect()
}

fn upload_in_transfer_chunks(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    buffer: &sllm_core::ExecutionBuffer,
    bytes: &[u8],
    label: &str,
) -> Result<(), String> {
    let max_transfer = usize::try_from(
        session
            .max_transfer_bytes()
            .map_err(|error| format!("{label}: query transfer limit: {error}"))?,
    )
    .map_err(|_| format!("{label}: transfer limit does not fit host usize"))?;
    if max_transfer == 0 {
        return Err(format!("{label}: backend transfer limit is zero"));
    }
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let chunk_len = (bytes.len() - offset).min(max_transfer);
        let end = offset
            .checked_add(chunk_len)
            .ok_or_else(|| format!("{label}: chunk range overflow"))?;
        let range = buffer
            .range(offset as u64, chunk_len as u64)
            .map_err(|error| format!("{label} range {offset}..{end}: {error}"))?;
        let mut upload = session
            .upload(queue, range, Arc::<[u8]>::from(&bytes[offset..end]))
            .map_err(|error| format!("{label} chunk {offset}..{end}: {error}"))?;
        wait_ok(upload.wait(WAIT), &format!("{label} chunk {offset}..{end}"))?;
        offset = end;
    }
    Ok(())
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
    if matches!(phase, 1840 | 1841) {
        validate_phase84_m1_range_boundary(&activation_words, &activation_quantized, m, k)?;
        validate_phase84_m1_range_boundary(&weight_words, &weight_quantized, n, k)?;
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
    upload_in_transfer_chunks(
        session,
        queue,
        &activation_buffer,
        &activation_bytes,
        "activation upload",
    )?;
    upload_in_transfer_chunks(session, queue, &weight_buffer, &resident, "weight upload")?;
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
    let mut first_output: Option<Vec<u8>> = None;
    for warmup in 0..mode.warmup_count() {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        let warmup_label = if matches!(mode, EvidenceMode::Phase75 { .. }) {
            "Phase 75 warmup"
        } else if matches!(mode, EvidenceMode::Phase74 { .. }) {
            "Phase 74 warmup"
        } else if matches!(mode, EvidenceMode::Phase84 { .. }) {
            "Phase 84 warmup"
        } else {
            "Phase 69 warmup"
        };
        wait_ok(submission.wait(WAIT), warmup_label)?;
        if let EvidenceMode::Phase85 {
            provider: forced_provider @ Phase85Provider::Forced { .. },
            ..
        } = mode
        {
            validate_phase85_dispatch(forced_provider, format, m, k, n, target, &dispatch)
                .map_err(|error| format!("warmup {warmup}: {error}"))?;
        } else {
            validate_actual_dispatch(format, m, k, n, target, &dispatch)
                .map_err(|error| format!("warmup {warmup}: {error}"))?;
        }
        if let EvidenceMode::Phase74 { provider, .. } = mode {
            validate_phase74_provider_dispatch(provider, m, n, target, &dispatch)
                .map_err(|error| format!("warmup {warmup}: {error}"))?;
        }
        if let EvidenceMode::Phase75 { provider, .. } = mode {
            validate_phase75_provider_dispatch(provider, m, n, target, &dispatch)
                .map_err(|error| format!("warmup {warmup}: {error}"))?;
        }
    }
    for repeat in 0..repeats {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        wait_ok(submission.wait(WAIT), format.name())?;
        if let EvidenceMode::Phase85 {
            provider: forced_provider @ Phase85Provider::Forced { .. },
            ..
        } = mode
        {
            validate_phase85_dispatch(forced_provider, format, m, k, n, target, &dispatch)?;
        } else {
            validate_actual_dispatch(format, m, k, n, target, &dispatch)?;
        }
        if let EvidenceMode::Phase74 { provider, .. } = mode {
            validate_phase74_provider_dispatch(provider, m, n, target, &dispatch)?;
        }
        if let EvidenceMode::Phase75 { provider, .. } = mode {
            validate_phase75_provider_dispatch(provider, m, n, target, &dispatch)?;
        }
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
                let first = first_output
                    .as_ref()
                    .expect("first digest has output bytes");
                let differences = first
                    .chunks_exact(2)
                    .zip(output.chunks_exact(2))
                    .enumerate()
                    .filter(|(_, (left, right))| left != right)
                    .map(|(index, (left, right))| {
                        (
                            index,
                            u16::from_le_bytes([left[0], left[1]]),
                            u16::from_le_bytes([right[0], right[1]]),
                        )
                    });
                let changed_elements = differences.clone().count();
                let first_differences: Vec<_> = differences.take(16).collect();
                // A second read without another kernel submission distinguishes
                // a persistent device output change from a readback anomaly.
                drop(readback);
                let reread = (|| -> Result<String, String> {
                    let mut task = submission
                        .start_output_readback(0)
                        .map_err(|e| e.to_string())?;
                    wait_ok(task.wait(WAIT), "mismatch diagnostic readback")?;
                    let mut bytes = vec![0_u8; output_len];
                    task.read_into(&mut bytes).map_err(|e| e.to_string())?;
                    Ok(digest(&bytes))
                })();
                return Err(format!(
                    "{} repeat {repeat} output digest differs: first={first_digest} actual={output_digest}; changed_elements={changed_elements} first_differences={first_differences:?} reread={reread:?}",
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
    let phase70_provider = mode.phase70_provider();
    let phase74_provider = mode.phase74_provider();
    let phase75_provider = mode.phase75_provider();
    let phase74_candidate = phase74_provider.map(|provider| {
        kernel_id == provider.kernel_id()
            && kernel_symbol == provider.kernel_symbol()
            && device_symbol == provider.device_symbol()
    });
    let sampled_indices = match oracle {
        OracleSelection::Full => None,
        OracleSelection::FixedSample(_) | OracleSelection::BoundarySample => {
            Some(oracle_indices.clone())
        }
    };
    let sampled_row_top1 = matches!(oracle, OracleSelection::BoundarySample)
        .then(|| sampled_row_top1(&output, n, &oracle_indices));
    let special_value_classes = match phase {
        100 => Some(
            "E4M3 subnormal/tie/max/saturation, E8M0 minimum/finite/NaN scale, signed zero, Inf/NaN",
        ),
        1840 | 1841 => {
            Some("M=1 finite >FP16 range, block32 scale boundary, minimum-scale finite value")
        }
        _ => None,
    };
    Ok(CaseReport {
        case_id,
        phase85_tags: None,
        phase85_role: None,
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
        special_value_classes: detailed.then_some(special_value_classes).flatten(),
        special_encoding_contract_validated: detailed
            .then_some(matches!(phase, 100 | 1840 | 1841))
            .filter(|value| *value),
        oracle_mode: detailed.then_some(oracle.name()),
        oracle_point_count: detailed.then_some(oracle_indices.len()),
        oracle_sampled_output_indices: detailed.then_some(sampled_indices).flatten(),
        sampled_row_top1: detailed.then_some(sampled_row_top1).flatten(),
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
        phase66_candidate: phase66_provider.map(|_| kernel_id == PHASE66_CANDIDATE_KERNEL_ID),
        phase67_provider: phase67_provider.map(Phase67Provider::name),
        phase67_candidate: phase67_provider.map(|provider| kernel_id == provider.kernel_id()),
        phase69_provider: phase69_provider.map(Phase69Provider::name),
        phase69_candidate: phase69_provider.map(|provider| kernel_id == provider.kernel_id()),
        phase70_provider: phase70_provider.map(Phase70Provider::name),
        phase70_candidate: phase70_provider.map(|provider| kernel_id == provider.kernel_id()),
        phase74_provider: phase74_provider.map(Phase74Provider::name),
        phase74_candidate,
        phase75_provider: phase75_provider.map(Phase75Provider::name),
        phase75_candidate: phase75_provider.map(|provider| kernel_id == provider.kernel_id()),
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

fn phase70_wide_n_cases() -> Vec<CaseSpec> {
    [
        ("phase72-lower-control-m17-k2048-n16384", 17, 2048, 16384),
        ("phase72-lower-tail-m17-k2048-n16385", 17, 2048, 16385),
        ("phase72-qwen27b-m128-k5120-n17408", 128, 5120, 17408),
        ("phase72-qwen27b-tail-m129-k5120-n17409", 129, 5120, 17409),
        ("phase72-wide-m512-k4096-n24576", 512, 4096, 24576),
        ("phase72-vocab-m128-k4096-n32000", 128, 4096, 32000),
        ("phase72-upper-tail-m17-k2048-n32767", 17, 2048, 32767),
        ("phase72-upper-m128-k4096-n32768", 128, 4096, 32768),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (case_id, m, k, n))| CaseSpec {
        case_id: Some(case_id),
        format: Format::Mxfp6,
        m,
        k,
        n,
        phase: 120 + index,
        oracle: OracleSelection::BoundarySample,
    })
    .collect()
}

fn phase70_cases(production_shape: bool, wide_n_shape: bool) -> Vec<CaseSpec> {
    if wide_n_shape {
        return phase70_wide_n_cases();
    }
    let mut cases = vec![
        CaseSpec {
            case_id: Some("phase70-odd-tail-m3-k64-n5"),
            format: Format::Mxfp6,
            m: 3,
            k: 64,
            n: 5,
            phase: 101,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("phase70-odd-tail-m17-k96-n7"),
            format: Format::Mxfp6,
            m: 17,
            k: 96,
            n: 7,
            phase: 102,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("phase70-boundary-m127-k224-n63"),
            format: Format::Mxfp6,
            m: 127,
            k: 224,
            n: 63,
            phase: 103,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("phase70-boundary-m128-k256-n64"),
            format: Format::Mxfp6,
            m: 128,
            k: 256,
            n: 64,
            phase: 104,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("phase70-boundary-m129-k288-n65"),
            format: Format::Mxfp6,
            m: 129,
            k: 288,
            n: 65,
            phase: 105,
            oracle: OracleSelection::Full,
        },
    ];
    if production_shape {
        cases.extend([
            CaseSpec {
                case_id: Some("phase70-production-m17-k2560-n9216"),
                format: Format::Mxfp6,
                m: 17,
                k: 2560,
                n: 9216,
                phase: 109,
                oracle: OracleSelection::FixedSample(PHASE67_M17_N9216_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase70-production-m127-k2560-n9216"),
                format: Format::Mxfp6,
                m: 127,
                k: 2560,
                n: 9216,
                phase: 110,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M127_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase70-production-m128-k2560-n9216"),
                format: Format::Mxfp6,
                m: 128,
                k: 2560,
                n: 9216,
                phase: 106,
                oracle: OracleSelection::FixedSample(PHASE63_PRODUCTION_M128_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase70-production-m512-k2560-n9216"),
                format: Format::Mxfp6,
                m: 512,
                k: 2560,
                n: 9216,
                phase: 107,
                oracle: OracleSelection::FixedSample(PHASE67_M512_N9216_ORACLE_POINTS),
            },
            CaseSpec {
                case_id: Some("phase70-production-m2048-k2560-n9216"),
                format: Format::Mxfp6,
                m: 2048,
                k: 2560,
                n: 9216,
                phase: 108,
                oracle: OracleSelection::FixedSample(PHASE67_M2048_N9216_ORACLE_POINTS),
            },
        ]);
    }
    cases
}

fn phase74_cases(production_shape: bool) -> Vec<CaseSpec> {
    phase70_cases(production_shape, false)
}

fn phase75_cases(production_shape: bool, provider: Phase75Provider) -> Vec<CaseSpec> {
    match (provider.format(), production_shape) {
        (Format::Mxfp8, true) => phase69_cases(),
        (Format::Mxfp8, false) => phase66_cases(),
        (Format::Mxfp6, _) => phase74_cases(production_shape),
    }
}

fn phase84_cases(format: Format) -> Vec<CaseSpec> {
    [
        // The eight Qwen3.8 MTP matrices reduce to six unique M=1 shapes;
        // K/V and gate/up share dimensions but remain separate role labels.
        ("mtp-m1-fc", 1, 10240, 5120),
        ("mtp-m1-q", 1, 5120, 12288),
        ("mtp-m1-k", 1, 5120, 1024),
        ("mtp-m1-v", 1, 5120, 1024),
        ("mtp-m1-o", 1, 6144, 5120),
        ("mtp-m1-gate-up", 1, 5120, 17408),
        ("mtp-m1-down", 1, 17408, 5120),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (case_id, m, k, n))| CaseSpec {
        case_id: Some(case_id),
        format,
        m,
        k,
        n,
        phase: 140 + index,
        // M=1 keeps the full output oracle tractable (the largest matrix is
        // about 90M FP32 dot products) and covers every output column.
        oracle: OracleSelection::Full,
    })
    .chain(
        [
            // Normal M=1 odd-tail/range fixtures. The first covers N-tail
            // launch mapping; the second crosses a block-scale boundary and
            // includes finite >FP16 plus a minimum-scale block.
            ("mtp-m1-k32-n5", 1, 32, 5),
            ("mtp-m1-k96-n9", 1, 96, 9),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (case_id, m, k, n))| CaseSpec {
            case_id: Some(case_id),
            format,
            m,
            k,
            n,
            phase: 1840 + index,
            oracle: OracleSelection::Full,
        }),
    )
    .chain(
        [
            ("mtp-prefix-m127-n1024", 127, 5120, 1024),
            ("mtp-prefix-m128-n1024", 128, 5120, 1024),
            ("mtp-prefix-m129-n1024", 129, 5120, 1024),
            ("mtp-prefix-gate-up-m128-n17408", 128, 5120, 17408),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (case_id, m, k, n))| CaseSpec {
            case_id: Some(case_id),
            format,
            m,
            k,
            n,
            phase: 150 + index,
            oracle: OracleSelection::BoundarySample,
        }),
    )
    .chain([
        CaseSpec {
            case_id: Some("mtp-odd-tail-m3-k64-n5"),
            format,
            m: 3,
            k: 64,
            n: 5,
            phase: 155,
            oracle: OracleSelection::Full,
        },
        CaseSpec {
            case_id: Some("mtp-scale-boundary-m17-k96-n9"),
            format,
            m: 17,
            k: 96,
            n: 9,
            // The shared special fixture is the existing E4M3 scale/value
            // boundary oracle. MXFP6 keeps the same odd shape with its
            // ordinary deterministic fixture because its E3M2 byte contract
            // has a different expected-value table.
            phase: if format == Format::Mxfp8 { 100 } else { 102 },
            oracle: OracleSelection::Full,
        },
    ])
    .collect()
}

fn leak_phase85_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn parse_phase85_format(value: &str) -> Result<Format, String> {
    match value {
        "mxfp8" | "mx8" | "mxfp8-e4m3-w8a8" => Ok(Format::Mxfp8),
        "mxfp6" | "mx6" | "mxfp6-e3m2-w6a6" => Ok(Format::Mxfp6),
        _ => Err(format!(
            "invalid Phase 85 format {value}; expected mxfp8 or mxfp6"
        )),
    }
}

fn parse_phase85_oracle(value: &str) -> Result<OracleSelection, String> {
    match value {
        "full" => Ok(OracleSelection::Full),
        "boundary" | "boundary-sample" => Ok(OracleSelection::BoundarySample),
        _ => Err(format!(
            "invalid Phase 85 oracle {value}; expected full or boundary"
        )),
    }
}

fn load_phase85_manifest(
    path: &PathBuf,
    format_selection: Phase85FormatSelection,
    filters: &[String],
    case_ids: &[String],
) -> Result<(&'static [Phase85CaseSpec], &'static str), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read Phase 85 manifest {}: {error}", path.display()))?;
    let manifest: Phase85Manifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse Phase 85 manifest {}: {error}", path.display()))?;
    if manifest.schema_version != PHASE85_SCHEMA_VERSION {
        return Err(format!(
            "Phase 85 manifest schema differs: expected {PHASE85_SCHEMA_VERSION}, observed {}",
            manifest.schema_version
        ));
    }
    if manifest.cases.is_empty() || manifest.cases.len() > PHASE85_MAX_CASES {
        return Err(format!(
            "Phase 85 manifest must contain 1..={PHASE85_MAX_CASES} cases"
        ));
    }
    let mut seen_ids = std::collections::HashSet::with_capacity(manifest.cases.len());
    let mut selected = Vec::with_capacity(manifest.cases.len());
    for (index, entry) in manifest.cases.into_iter().enumerate() {
        if entry.case_id.is_empty() || !seen_ids.insert(entry.case_id.clone()) {
            return Err(format!(
                "Phase 85 manifest case_id is empty or duplicated: {}",
                entry.case_id
            ));
        }
        if entry.m == 0 || entry.k == 0 || entry.n == 0 {
            return Err(format!(
                "Phase 85 case {} has a zero matrix dimension",
                entry.case_id
            ));
        }
        if entry.k % 32 != 0 {
            return Err(format!(
                "Phase 85 case {} has K={} which is not a block-32 shape",
                entry.case_id, entry.k
            ));
        }
        let format = parse_phase85_format(&entry.format)?;
        let tags = entry.tags.iter().map(String::as_str).collect::<Vec<_>>();
        let role = entry.role.as_deref().unwrap_or("matmul");
        let filter_match = filters.is_empty()
            || filters
                .iter()
                .any(|filter| filter == role || entry.tags.iter().any(|tag| tag == filter));
        let case_match = case_ids.is_empty() || case_ids.iter().any(|id| id == &entry.case_id);
        if !format_selection.accepts(format) || !filter_match || !case_match {
            continue;
        }
        let tags = leak_phase85_string(tags.join("|"));
        let role = leak_phase85_string(role.to_owned());
        let case_id = leak_phase85_string(entry.case_id);
        selected.push(Phase85CaseSpec {
            case: CaseSpec {
                case_id: Some(case_id),
                format,
                m: entry.m,
                k: entry.k,
                n: entry.n,
                phase: entry.phase.unwrap_or(2000 + index),
                oracle: parse_phase85_oracle(&entry.oracle)?,
            },
            tags,
            role,
        });
    }
    if selected.is_empty() {
        return Err("Phase 85 selection matched no manifest cases".to_owned());
    }
    let manifest_name = leak_phase85_string(path.display().to_string());
    Ok((Box::leak(selected.into_boxed_slice()), manifest_name))
}

fn configure_phase85_provider(provider: Phase85Provider) {
    let mut environments = MXFP8_FORCE_ENVIRONMENTS.to_vec();
    environments.extend_from_slice(MXFP6_FORCE_ENVIRONMENTS);
    environments.sort_unstable();
    environments.dedup();
    for environment in environments {
        unsafe { std::env::remove_var(environment) };
    }
    if let Some((environment, value)) = provider.force() {
        unsafe { std::env::set_var(environment, value) };
    }
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
        EvidenceMode::Phase70 { provider, .. }
            if matches!(
                provider,
                Phase70Provider::Gfx1201Default
                    | Phase70Provider::Gfx1201Pack4N64
                    | Phase70Provider::Gfx1201Candidate
            ) && target != "gfx1201" =>
        {
            return Err("Phase 70 gfx1201 provider requires exact gfx1201".to_owned());
        }
        EvidenceMode::Phase70 { .. } if !matches!(target.as_str(), "gfx1030" | "gfx1201") => {
            return Err("Phase 70 control requires exact gfx1030 or gfx1201".to_owned());
        }
        EvidenceMode::Phase74 { provider, .. } if target != provider.target() => {
            return Err(format!(
                "Phase 74 {} mode requires exact {}",
                provider.name(),
                provider.target()
            ));
        }
        EvidenceMode::Phase75 { provider, .. } if target != "gfx1030" => {
            return Err(format!(
                "Phase 75 {} mode requires exact gfx1030",
                provider.name()
            ));
        }
        EvidenceMode::Phase84 { .. } if !matches!(target.as_str(), "gfx1030" | "gfx1201") => {
            return Err("Phase 84 mode requires exact gfx1030 or gfx1201".to_owned());
        }
        EvidenceMode::Phase85 { .. } if !matches!(target.as_str(), "gfx1030" | "gfx1201") => {
            return Err("Phase 85 mode requires exact gfx1030 or gfx1201".to_owned());
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
    if let EvidenceMode::Phase70 { provider, .. } = mode {
        for environment in MXFP6_FORCE_ENVIRONMENTS {
            let value = std::env::var(environment).ok();
            let expected = provider
                .force_environment()
                .filter(|expected| *expected == *environment)
                .map(|_| provider.force_value());
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 70 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    if let EvidenceMode::Phase74 { provider, .. } = mode {
        for environment in MXFP6_FORCE_ENVIRONMENTS {
            let value = std::env::var(environment).ok();
            let expected =
                (*environment == provider.force_environment()).then_some(provider.force_value());
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 74 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    if let EvidenceMode::Phase75 { provider, .. } = mode {
        let force_environments = match provider.format() {
            Format::Mxfp8 => MXFP8_FORCE_ENVIRONMENTS,
            Format::Mxfp6 => MXFP6_FORCE_ENVIRONMENTS,
        };
        for environment in force_environments {
            let value = std::env::var(environment).ok();
            let expected =
                (*environment == provider.force_environment()).then_some(provider.force_value());
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 75 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
                    provider.name()
                ));
            }
        }
    }
    if let EvidenceMode::Phase85 { provider, .. } = mode {
        let mut environments = MXFP8_FORCE_ENVIRONMENTS.to_vec();
        environments.extend_from_slice(MXFP6_FORCE_ENVIRONMENTS);
        environments.sort_unstable();
        environments.dedup();
        for environment in environments {
            let value = std::env::var(environment).ok();
            let expected = provider
                .force()
                .filter(|(expected, _)| *expected == environment)
                .map(|(_, value)| value);
            if value.as_deref() != expected {
                return Err(format!(
                    "Phase 85 {} environment isolation failed for {environment}: expected={expected:?} actual={value:?}",
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
            EvidenceMode::Phase70 {
                production_shape,
                wide_n_shape,
                ..
            } => phase70_cases(production_shape, wide_n_shape),
            EvidenceMode::Phase74 {
                production_shape, ..
            } => phase74_cases(production_shape),
            EvidenceMode::Phase75 {
                production_shape,
                provider,
                ..
            } => phase75_cases(production_shape, provider),
            EvidenceMode::Phase84 { format, .. } => phase84_cases(format),
            EvidenceMode::Phase85 { .. } => Vec::new(),
        };
        let mut cases = Vec::with_capacity(match mode {
            EvidenceMode::Phase85 { specs, .. } => specs.len(),
            _ => specs.len(),
        });
        if let EvidenceMode::Phase85 { specs, .. } = mode {
            for phase85_spec in specs {
                let mut case = run_case(&session, &queue, &target, phase85_spec.case, mode)?;
                case.phase85_tags = Some(phase85_spec.tags);
                case.phase85_role = Some(phase85_spec.role);
                cases.push(case);
            }
        } else {
            for spec in specs {
                cases.push(run_case(&session, &queue, &target, spec, mode)?);
            }
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
        EvidenceMode::Phase70 { provider, .. } => Some(provider.kernel_id()),
        EvidenceMode::Phase74 { provider, .. } => Some(provider.kernel_id()),
        EvidenceMode::Phase75 { provider, .. } => Some(provider.kernel_id()),
        EvidenceMode::Phase84 { .. } => None,
        EvidenceMode::Phase85 { .. } => None,
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
    if matches!(mode, EvidenceMode::Phase70 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 70 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    if matches!(mode, EvidenceMode::Phase74 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 74 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    if matches!(mode, EvidenceMode::Phase75 { .. }) && candidate_submission_count == 0 {
        return Err(format!(
            "Phase 75 expected kernel {} was never dispatched",
            expected_candidate_kernel_id.unwrap_or_default()
        ));
    }
    let detailed = mode.report_mode().is_some();
    let (production_shape_included, wide_n_shape_included, candidate_required) = match mode {
        EvidenceMode::Phase62 { .. } => (None, None, None),
        EvidenceMode::Phase63 {
            production_shape,
            require_candidate,
            ..
        } => (Some(production_shape), None, Some(require_candidate)),
        EvidenceMode::Phase66 { .. } => (Some(true), None, Some(true)),
        EvidenceMode::Phase67 { .. } => (Some(true), None, Some(true)),
        EvidenceMode::Phase69 { .. } => (Some(true), None, Some(true)),
        EvidenceMode::Phase70 {
            production_shape,
            wide_n_shape,
            ..
        } => (Some(production_shape), Some(wide_n_shape), Some(true)),
        EvidenceMode::Phase74 {
            production_shape, ..
        } => (Some(production_shape), None, Some(true)),
        EvidenceMode::Phase75 {
            production_shape, ..
        } => (Some(production_shape), None, Some(true)),
        EvidenceMode::Phase84 { .. } => (Some(true), None, None),
        EvidenceMode::Phase85 { .. } => (Some(true), None, None),
    };
    Ok(Report {
        schema_version: mode.schema_version(),
        state: "PASS",
        evidence_mode: mode.report_mode(),
        provider_role: mode
            .phase66_provider()
            .map(Phase66Provider::name)
            .or_else(|| mode.phase67_provider().map(Phase67Provider::name))
            .or_else(|| mode.phase69_provider().map(Phase69Provider::name))
            .or_else(|| mode.phase70_provider().map(Phase70Provider::name))
            .or_else(|| mode.phase74_provider().map(Phase74Provider::name))
            .or_else(|| mode.phase75_provider().map(Phase75Provider::name))
            .or_else(|| mode.phase85_provider().map(Phase85Provider::name)),
        phase85_manifest: match mode {
            EvidenceMode::Phase85 { manifest, .. } => Some(manifest),
            _ => None,
        },
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
        wide_n_shape_included,
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
    Phase66Comparison {
        repeats: usize,
    },
    Phase85 {
        repeats: usize,
        manifest: PathBuf,
        format: Phase85FormatSelection,
        filters: Vec<String>,
        case_ids: Vec<String>,
        provider: Phase85Provider,
    },
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
                Some("id41-vector32-candidate") => Phase69Provider::Vector32,
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
        Some("phase74-provider") => {
            let provider = match arguments.next().as_deref() {
                Some("id25-gfx1030-control") => Phase74Provider::Control,
                Some("id47-gfx1030-half2-candidate") => Phase74Provider::Candidate,
                Some("id45-gfx1201-pack4-control") => Phase74Provider::Gfx1201Control,
                Some("id48-gfx1201-pack4-swar-candidate") => Phase74Provider::Gfx1201Candidate,
                Some(value) => {
                    eprintln!("invalid Phase 74 provider {value}");
                    return ExitCode::FAILURE;
                }
                None => {
                    eprintln!("phase74-provider requires a provider role");
                    return ExitCode::FAILURE;
                }
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut production_shape = false;
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
                                eprintln!("Phase 74 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 74 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--production-shape" if !production_shape => production_shape = true,
                    _ => {
                        eprintln!(
                            "invalid Phase 74 argument {argument}; expected --repeats N or --production-shape"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase74 {
                repeats,
                provider,
                production_shape,
            })
        }
        Some("phase75-provider") => {
            let provider = match arguments.next().as_deref() {
                Some("id41-mxfp8-vector32-control") => Phase75Provider::Mxfp8Id41Control,
                Some("id55-mxfp8-half2-128x64-k32-double") => {
                    Phase75Provider::Mxfp8Half2_128x64K32Double
                }
                Some("id47-mxfp6-half2-32x32-control") => Phase75Provider::Mxfp6Id47Control,
                Some("id57-mxfp6-half2-128x64-k32-double-pack4") => {
                    Phase75Provider::Mxfp6Half2_128x64K32DoublePack4
                }
                Some(value) => {
                    eprintln!("invalid Phase 75 provider {value}");
                    return ExitCode::FAILURE;
                }
                None => {
                    eprintln!("phase75-provider requires a provider role");
                    return ExitCode::FAILURE;
                }
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut production_shape = false;
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
                                eprintln!("Phase 75 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 75 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--production-shape" if !production_shape => production_shape = true,
                    _ => {
                        eprintln!(
                            "invalid Phase 75 argument {argument}; expected --repeats N or --production-shape"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase75 {
                repeats,
                provider,
                production_shape,
            })
        }
        Some("phase84") => {
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut format = Format::Mxfp8;
            let mut format_seen = false;
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
                                eprintln!("Phase 84 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 84 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--format" if !format_seen => {
                        format_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--format requires mxfp8 or mxfp6");
                            return ExitCode::FAILURE;
                        };
                        format = match value.as_str() {
                            "mxfp8" | "mx8" => Format::Mxfp8,
                            "mxfp6" | "mx6" => Format::Mxfp6,
                            _ => {
                                eprintln!(
                                    "invalid Phase 84 format {value}; expected mxfp8 or mxfp6"
                                );
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    _ => {
                        eprintln!(
                            "invalid Phase 84 argument {argument}; expected --format mxfp8|mxfp6 and optionally --repeats N"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Direct(EvidenceMode::Phase84 { repeats, format })
        }
        Some("phase85") => {
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut manifest = PathBuf::from("ci/matrix/phase85-mxfp-shapes-v1.json");
            let mut manifest_seen = false;
            let mut format = Phase85FormatSelection::Both;
            let mut format_seen = false;
            let mut filters = Vec::new();
            let mut case_ids = Vec::new();
            let mut provider = Phase85Provider::Baseline;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--repeats" if !repeat_seen => {
                        repeat_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--repeats requires a value");
                            return ExitCode::FAILURE;
                        };
                        repeats = match value.parse::<usize>() {
                            Ok(value @ 2..=32) => value,
                            Ok(_) => {
                                eprintln!("Phase 85 repeats must be between 2 and 32");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 85 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--manifest" if !manifest_seen => {
                        manifest_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--manifest requires a path");
                            return ExitCode::FAILURE;
                        };
                        manifest = PathBuf::from(value);
                    }
                    "--format" if !format_seen => {
                        format_seen = true;
                        let Some(value) = arguments.next() else {
                            eprintln!("--format requires mxfp8, mxfp6, or both");
                            return ExitCode::FAILURE;
                        };
                        format = match value.as_str() {
                            "mxfp8" | "mx8" => Phase85FormatSelection::Mxfp8,
                            "mxfp6" | "mx6" => Phase85FormatSelection::Mxfp6,
                            "both" | "all" => Phase85FormatSelection::Both,
                            _ => {
                                eprintln!(
                                    "invalid Phase 85 format {value}; expected mxfp8, mxfp6, or both"
                                );
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--filter" => {
                        let Some(value) = arguments.next() else {
                            eprintln!("--filter requires a tag or role");
                            return ExitCode::FAILURE;
                        };
                        if value.is_empty() {
                            eprintln!("--filter must not be empty");
                            return ExitCode::FAILURE;
                        }
                        filters.push(value);
                    }
                    "--case" => {
                        let Some(value) = arguments.next() else {
                            eprintln!("--case requires a case_id");
                            return ExitCode::FAILURE;
                        };
                        if value.is_empty() {
                            eprintln!("--case must not be empty");
                            return ExitCode::FAILURE;
                        }
                        case_ids.push(value);
                    }
                    "--force-env" => {
                        if !matches!(provider, Phase85Provider::Baseline) {
                            eprintln!("Phase 85 accepts at most one --force-env");
                            return ExitCode::FAILURE;
                        }
                        let Some(value) = arguments.next() else {
                            eprintln!("--force-env requires NAME=VALUE");
                            return ExitCode::FAILURE;
                        };
                        let Some((name, force_value)) = value.split_once('=') else {
                            eprintln!("--force-env requires NAME=VALUE");
                            return ExitCode::FAILURE;
                        };
                        if name.is_empty() {
                            eprintln!("--force-env name must not be empty");
                            return ExitCode::FAILURE;
                        }
                        provider = Phase85Provider::Forced {
                            environment: leak_phase85_string(name.to_owned()),
                            value: leak_phase85_string(force_value.to_owned()),
                        };
                    }
                    _ => {
                        eprintln!(
                            "invalid Phase 85 argument {argument}; expected --manifest PATH, --format mxfp8|mxfp6|both, --filter TAG, --case ID, --force-env NAME=VALUE, or --repeats N"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            RequestedMode::Phase85 {
                repeats,
                manifest,
                format,
                filters,
                case_ids,
                provider,
            }
        }
        Some("phase70-provider") => {
            let provider = match arguments.next().as_deref() {
                Some("id45-gfx1201-pack4-n64-default") => Phase70Provider::Gfx1201Default,
                Some("id25-tiled16-control") => Phase70Provider::Tiled16,
                Some("id29-col8-control") => Phase70Provider::Control,
                Some("id44-gfx1201-via-e4m3-n64-candidate") => Phase70Provider::Gfx1201Candidate,
                Some("id45-gfx1201-pack4-n64-candidate") => Phase70Provider::Gfx1201Pack4N64,
                Some(value) => {
                    eprintln!("invalid Phase 70 provider {value}");
                    return ExitCode::FAILURE;
                }
                None => {
                    eprintln!("phase70-provider requires a provider role");
                    return ExitCode::FAILURE;
                }
            };
            let mut repeats = 3_usize;
            let mut repeat_seen = false;
            let mut production_shape = false;
            let mut wide_n_shape = false;
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
                                eprintln!("Phase 70 repeats must be between 2 and 10");
                                return ExitCode::FAILURE;
                            }
                            Err(error) => {
                                eprintln!("invalid Phase 70 repeat count: {error}");
                                return ExitCode::FAILURE;
                            }
                        };
                    }
                    "--production-shape" if !production_shape => production_shape = true,
                    "--wide-n" if !wide_n_shape => wide_n_shape = true,
                    _ => {
                        eprintln!(
                            "invalid Phase 70 argument {argument}; expected --repeats N, --production-shape, or --wide-n"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            if production_shape && wide_n_shape {
                eprintln!("--production-shape and --wide-n are mutually exclusive");
                return ExitCode::FAILURE;
            }
            RequestedMode::Direct(EvidenceMode::Phase70 {
                repeats,
                provider,
                production_shape,
                wide_n_shape,
            })
        }
        Some(value) => {
            eprintln!(
                "invalid profile {value}; expected production, phase63, phase66, phase67-provider, phase69-provider, phase70-provider, phase74-provider, phase75-provider, phase84, or phase85"
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
        RequestedMode::Phase85 {
            repeats,
            manifest,
            format,
            filters,
            case_ids,
            provider,
        } => {
            configure_phase85_provider(provider);
            load_phase85_manifest(&manifest, format, &filters, &case_ids).and_then(
                |(specs, manifest_name)| {
                    run(
                        device_index,
                        target,
                        EvidenceMode::Phase85 {
                            repeats,
                            specs,
                            manifest: manifest_name,
                            provider,
                        },
                    )
                    .and_then(|report| {
                        serde_json::to_value(report).map_err(|error| error.to_string())
                    })
                },
            )
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
    fn phase70_wide_n_matrix_covers_selector_and_vocabulary_boundaries() {
        let cases = phase70_wide_n_cases();
        assert_eq!(cases.len(), 8);
        assert_eq!(
            cases
                .iter()
                .map(|case| (case.m, case.k, case.n))
                .collect::<Vec<_>>(),
            vec![
                (17, 2048, 16384),
                (17, 2048, 16385),
                (128, 5120, 17408),
                (129, 5120, 17409),
                (512, 4096, 24576),
                (128, 4096, 32000),
                (17, 2048, 32767),
                (128, 4096, 32768),
            ]
        );
        for case in cases {
            assert!(matches!(case.oracle, OracleSelection::BoundarySample));
            let indices = case.oracle.output_indices(case.m, case.n).unwrap();
            assert!(indices.contains(&(case.n - 1)));
            assert!(indices.contains(&((case.m - 1) * case.n + case.n - 1)));
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
    fn phase85_forced_row8_rejects_small_m_candidate_misrouting() {
        let provider = Phase85Provider::Forced {
            environment: "SLLM_MXFP8_PREFILL_FORCE_ROW8",
            value: "1",
        };
        let contract =
            phase85_forced_contract(Format::Mxfp8, "SLLM_MXFP8_PREFILL_FORCE_ROW8", "1").unwrap();
        let expected = DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 2,
            kernel_id: contract.kernel_id,
            workgroup_size_x: contract.workgroup_size_x,
            grid_size_x: 1024,
            row_count: 2,
            normalized_size: 2 * 1024,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: contract.kernel_symbol.to_owned(),
            device_symbol: contract.device_symbol.to_owned(),
            target: "gfx1201".to_owned(),
        };
        assert!(validate_phase85_dispatch(
            provider,
            Format::Mxfp8,
            2,
            2048,
            1024,
            "gfx1201",
            &expected,
        )
        .is_ok());

        let misrouted = DispatchEvidence {
            kernel_id: PHASE85_MXFP8_SMALL_M_KERNEL_ID,
            workgroup_size_x: 128,
            grid_size_x: 128,
            kernel_symbol: PHASE85_MXFP8_SMALL_M_KERNEL_SYMBOL.to_owned(),
            device_symbol: PHASE85_MXFP8_SMALL_M_DEVICE_SYMBOL.to_owned(),
            ..expected
        };
        assert!(
            validate_phase85_dispatch(
                provider,
                Format::Mxfp8,
                2,
                2048,
                1024,
                "gfx1201",
                &misrouted,
            )
            .is_err()
        );
    }

    #[test]
    fn phase85_forced_grid_geometry_matches_native_contracts() {
        let check = |format, environment, value, m, n, expected_grid| {
            let provider = Phase85Provider::Forced { environment, value };
            let contract = phase85_forced_contract(format, environment, value).unwrap();
            let dispatch = DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id: 1,
                dispatch_count: 2,
                kernel_id: contract.kernel_id,
                workgroup_size_x: contract.workgroup_size_x,
                grid_size_x: expected_grid,
                row_count: m as u64,
                normalized_size: (m * n) as u64,
                backend: 1,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: contract.kernel_symbol.to_owned(),
                device_symbol: contract.device_symbol.to_owned(),
                target: "gfx1201".to_owned(),
            };
            assert!(
                validate_phase85_dispatch(provider, format, m, 2048, n, "gfx1201", &dispatch,)
                    .is_ok()
            );
        };
        for format in [Format::Mxfp8, Format::Mxfp6] {
            check(
                format,
                "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
                "1",
                3,
                1024,
                3 * 1024,
            );
            check(
                format,
                "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
                "1",
                17,
                1024,
                17 * 1024,
            );
            check(
                format,
                if format == Format::Mxfp8 {
                    "SLLM_MXFP8_PREFILL_FORCE_ROW8"
                } else {
                    "SLLM_MXFP6_PREFILL_FORCE_ROW8"
                },
                "1",
                3,
                1024,
                1024,
            );
            check(
                format,
                if format == Format::Mxfp8 {
                    "SLLM_MXFP8_PREFILL_FORCE_TILED16"
                } else {
                    "SLLM_MXFP6_PREFILL_FORCE_TILED16"
                },
                "1",
                3,
                1024,
                64,
            );
            check(
                format,
                "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS",
                "4",
                17,
                1024,
                3 * 256,
            );
            check(format, PHASE85_SMALL_M_FORCE_ENV, "1", 3, 1024, 128);
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
    fn phase69_vector32_dispatch_identity_is_exact_gfx1030_mxfp8_prefill() {
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
    }

    #[test]
    fn phase69_provider_identities_are_stable() {
        assert_eq!(Phase69Provider::Control.kernel_id(), 27);
        assert_eq!(Phase69Provider::Control.force_value(), "8");
        assert_eq!(Phase69Provider::Vector32.kernel_id(), 41);
        assert_eq!(Phase69Provider::Vector32.force_value(), "vector32");
    }

    #[test]
    fn phase74_provider_identities_and_case_reuse_are_stable() {
        assert_eq!(Phase74Provider::Control.name(), "id25-gfx1030-control");
        assert_eq!(Phase74Provider::Control.kernel_id(), 25);
        assert_eq!(
            Phase74Provider::Control.force_environment(),
            PHASE74_CONTROL_FORCE_ENV
        );
        assert_eq!(Phase74Provider::Control.force_value(), "1");
        assert_eq!(Phase74Provider::Control.target(), "gfx1030");
        assert_eq!(
            Phase74Provider::Candidate.name(),
            "id47-gfx1030-half2-candidate"
        );
        assert_eq!(Phase74Provider::Candidate.kernel_id(), 47);
        assert_eq!(
            Phase74Provider::Candidate.force_environment(),
            PHASE74_CANDIDATE_FORCE_ENV
        );
        assert_eq!(
            Phase74Provider::Candidate.force_value(),
            PHASE74_CANDIDATE_FORCE_VALUE
        );
        assert_eq!(Phase74Provider::Candidate.target(), "gfx1030");
        assert_eq!(
            Phase74Provider::Gfx1201Control.name(),
            "id45-gfx1201-pack4-control"
        );
        assert_eq!(
            Phase74Provider::Gfx1201Control.kernel_id(),
            PHASE70_GFX1201_PACK4_N64_KERNEL_ID
        );
        assert_eq!(
            Phase74Provider::Gfx1201Control.force_environment(),
            PHASE70_CANDIDATE_FORCE_ENV
        );
        assert_eq!(
            Phase74Provider::Gfx1201Control.force_value(),
            "gfx1201-n64-pack4"
        );
        assert_eq!(Phase74Provider::Gfx1201Control.target(), "gfx1201");
        assert_eq!(
            Phase74Provider::Gfx1201Control.kernel_symbol(),
            PHASE70_GFX1201_PACK4_N64_KERNEL_SYMBOL
        );
        assert_eq!(
            Phase74Provider::Gfx1201Control.device_symbol(),
            PHASE70_GFX1201_PACK4_N64_DEVICE_SYMBOL
        );
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.name(),
            "id48-gfx1201-pack4-swar-candidate"
        );
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.kernel_id(),
            PHASE74_GFX1201_CANDIDATE_KERNEL_ID
        );
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.force_environment(),
            PHASE74_CANDIDATE_FORCE_ENV
        );
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.force_value(),
            "gfx1201-swar-pack4"
        );
        assert_eq!(Phase74Provider::Gfx1201Candidate.target(), "gfx1201");
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.kernel_symbol(),
            PHASE74_GFX1201_CANDIDATE_KERNEL_SYMBOL
        );
        assert_eq!(
            Phase74Provider::Gfx1201Candidate.device_symbol(),
            PHASE74_GFX1201_CANDIDATE_DEVICE_SYMBOL
        );
        assert_eq!(
            phase74_cases(false).len(),
            phase70_cases(false, false).len()
        );
        assert_eq!(phase74_cases(true).len(), phase70_cases(true, false).len());
    }

    #[test]
    fn phase74_candidate_dispatch_identity_is_exact_gfx1030_half2_32x32() {
        let candidate = |m: usize, _k: usize, n: usize, target: &str| DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 2,
            kernel_id: PHASE74_CANDIDATE_KERNEL_ID,
            workgroup_size_x: 256,
            grid_size_x: u32::try_from(m.div_ceil(32) * n.div_ceil(32)).unwrap(),
            row_count: m as u64,
            normalized_size: (m * n) as u64,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: PHASE74_CANDIDATE_KERNEL_SYMBOL.to_owned(),
            device_symbol: PHASE74_CANDIDATE_DEVICE_SYMBOL.to_owned(),
            target: target.to_owned(),
        };
        let dispatch = candidate(33, 2560, 33, "gfx1030");
        assert!(
            validate_actual_dispatch(Format::Mxfp6, 33, 2560, 33, "gfx1030", &dispatch).is_ok()
        );
        assert!(
            validate_phase74_provider_dispatch(
                Phase74Provider::Candidate,
                33,
                33,
                "gfx1030",
                &dispatch,
            )
            .is_ok()
        );
        for (format, m, k, n, target) in [
            (Format::Mxfp6, 1, 2560, 33, "gfx1030"),
            (Format::Mxfp6, 33, 2559, 33, "gfx1030"),
            (Format::Mxfp6, 33, 2560, 33, "gfx1201"),
            (Format::Mxfp8, 33, 2560, 33, "gfx1030"),
        ] {
            assert!(
                validate_actual_dispatch(format, m, k, n, target, &candidate(m, k, n, target),)
                    .is_err()
            );
        }
    }

    #[test]
    fn phase74_gfx1201_candidate_dispatch_identity_is_exact_pack4_swar() {
        let candidate = |m: usize, _k: usize, n: usize, target: &str| DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 2,
            kernel_id: PHASE74_GFX1201_CANDIDATE_KERNEL_ID,
            workgroup_size_x: 256,
            grid_size_x: u32::try_from(n.div_ceil(64)).unwrap(),
            row_count: m as u64,
            normalized_size: (m * n) as u64,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: PHASE74_GFX1201_CANDIDATE_KERNEL_SYMBOL.to_owned(),
            device_symbol: PHASE74_GFX1201_CANDIDATE_DEVICE_SYMBOL.to_owned(),
            target: target.to_owned(),
        };
        let dispatch = candidate(17, 2048, 65, "gfx1201");
        assert!(
            validate_actual_dispatch(Format::Mxfp6, 17, 2048, 65, "gfx1201", &dispatch).is_ok()
        );
        assert!(
            validate_phase74_provider_dispatch(
                Phase74Provider::Gfx1201Candidate,
                17,
                65,
                "gfx1201",
                &dispatch,
            )
            .is_ok()
        );

        let mut wrong_grid = dispatch.clone();
        wrong_grid.grid_size_x = 1;
        assert!(
            validate_actual_dispatch(Format::Mxfp6, 17, 2048, 65, "gfx1201", &wrong_grid).is_err()
        );
        let mut wrong_workgroup = dispatch.clone();
        wrong_workgroup.workgroup_size_x = 128;
        assert!(
            validate_actual_dispatch(Format::Mxfp6, 17, 2048, 65, "gfx1201", &wrong_workgroup,)
                .is_err()
        );
        let mut wrong_symbol = dispatch.clone();
        wrong_symbol.kernel_symbol.push_str("-wrong");
        assert!(
            validate_actual_dispatch(Format::Mxfp6, 17, 2048, 65, "gfx1201", &wrong_symbol)
                .is_err()
        );
        for (format, m, k, n, target) in [
            (Format::Mxfp6, 1, 2048, 65, "gfx1201"),
            (Format::Mxfp6, 17, 2015, 65, "gfx1201"),
            (Format::Mxfp6, 17, 2048, 0, "gfx1201"),
            (Format::Mxfp6, 17, 2048, 65, "gfx1030"),
            (Format::Mxfp8, 17, 2048, 65, "gfx1201"),
        ] {
            assert!(
                validate_actual_dispatch(format, m, k, n, target, &candidate(m, k, n, target),)
                    .is_err()
            );
        }
        assert!(
            validate_phase74_provider_dispatch(
                Phase74Provider::Gfx1201Candidate,
                17,
                65,
                "gfx1030",
                &dispatch,
            )
            .is_err()
        );
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

    #[test]
    fn phase75_provider_identities_and_case_formats_are_stable() {
        let providers = [
            Phase75Provider::Mxfp8Id41Control,
            Phase75Provider::Mxfp8Half2_128x64K32Double,
            Phase75Provider::Mxfp6Id47Control,
            Phase75Provider::Mxfp6Half2_128x64K32DoublePack4,
        ];
        assert_eq!(providers.map(Phase75Provider::kernel_id), [41, 55, 47, 57]);
        for provider in providers {
            assert_eq!(
                provider.force_environment(),
                match provider.format() {
                    Format::Mxfp8 if provider.kernel_id() == 41 => PHASE69_CANDIDATE_FORCE_ENV,
                    Format::Mxfp8 => PHASE75_MXFP8_FORCE_ENV,
                    Format::Mxfp6 if provider.kernel_id() == 47 => PHASE74_CANDIDATE_FORCE_ENV,
                    Format::Mxfp6 => PHASE75_MXFP6_FORCE_ENV,
                }
            );
            let cases = phase75_cases(true, provider);
            assert!(!cases.is_empty());
            assert!(cases.iter().all(|case| case.format == provider.format()));
        }
    }

    #[test]
    fn phase75_mxfp6_candidate_dispatch_identity_is_exact() {
        for provider in [Phase75Provider::Mxfp6Half2_128x64K32DoublePack4] {
            let dispatch = DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id: 1,
                dispatch_count: 2,
                kernel_id: provider.kernel_id(),
                workgroup_size_x: 256,
                grid_size_x: u32::try_from(129_usize.div_ceil(128) * 65_usize.div_ceil(64))
                    .unwrap(),
                row_count: 129,
                normalized_size: 129 * 65,
                backend: 1,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: provider.kernel_symbol().to_owned(),
                device_symbol: provider.device_symbol().to_owned(),
                target: "gfx1030".to_owned(),
            };
            assert!(
                validate_actual_dispatch(Format::Mxfp6, 129, 2080, 65, "gfx1030", &dispatch)
                    .is_ok()
            );
            assert!(
                validate_phase75_provider_dispatch(provider, 129, 65, "gfx1030", &dispatch).is_ok()
            );
            for (format, m, k, n, target) in [
                (Format::Mxfp6, 1, 2080, 65, "gfx1030"),
                (Format::Mxfp6, 129, 2079, 65, "gfx1030"),
                (Format::Mxfp6, 129, 2080, 0, "gfx1030"),
                (Format::Mxfp6, 129, 2080, 65, "gfx1201"),
                (Format::Mxfp8, 129, 2080, 65, "gfx1030"),
            ] {
                let mut invalid = dispatch.clone();
                invalid.row_count = m as u64;
                invalid.normalized_size = (m * n) as u64;
                invalid.target = target.to_owned();
                assert!(validate_actual_dispatch(format, m, k, n, target, &invalid).is_err());
            }
        }
    }

    #[test]
    fn phase84_case_matrix_covers_mtp_roles_and_selector_boundaries() {
        for format in [Format::Mxfp8, Format::Mxfp6] {
            let cases = phase84_cases(format);
            assert_eq!(cases.len(), 15);
            assert!(cases.iter().all(|case| case.format == format));
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case.m == 1)
                    .map(|case| (case.k, case.n))
                    .collect::<Vec<_>>(),
                vec![
                    (10240, 5120),
                    (5120, 12288),
                    (5120, 1024),
                    (5120, 1024),
                    (6144, 5120),
                    (5120, 17408),
                    (17408, 5120),
                    (32, 5),
                    (96, 9),
                ]
            );
            assert!(
                cases[..7]
                    .iter()
                    .all(|case| matches!(case.oracle, OracleSelection::Full))
            );
            for m in [127, 128, 129] {
                let case = cases
                    .iter()
                    .find(|case| case.m == m && case.n == 1024)
                    .expect("prefix selector boundary case");
                assert!(matches!(case.oracle, OracleSelection::BoundarySample));
            }
            assert!(cases.iter().any(|case| {
                case.m == 128
                    && case.k == 5120
                    && case.n == 17408
                    && matches!(case.oracle, OracleSelection::BoundarySample)
            }));
            let scale_case = cases
                .iter()
                .find(|case| case.case_id == Some("mtp-scale-boundary-m17-k96-n9"))
                .expect("scale boundary case");
            assert_eq!(
                scale_case.phase,
                if format == Format::Mxfp8 { 100 } else { 102 }
            );
        }
    }

    #[test]
    fn phase84_m1_range_boundary_fixture_is_preserved_by_host_codecs() {
        for format in [Format::Mxfp8, Format::Mxfp6] {
            for (index, k) in [32_usize, 96].into_iter().enumerate() {
                for (rows, phase) in [(1, 1840 + index), (5, 1851 + index)] {
                    let source_words = matrix(rows, k, phase);
                    let source: Vec<_> = source_words.iter().copied().map(from_bf16).collect();
                    let quantized = format
                        .quantize(&source, rows, k)
                        .expect("range-boundary fixture quantizes");
                    validate_phase84_m1_range_boundary(&source_words, &quantized, rows, k)
                        .expect("activation and weight fixtures survive host codec");
                }
            }
        }
    }

    #[test]
    fn phase85_manifest_is_finite_and_covers_both_formats_and_roles() {
        let manifest: Phase85Manifest = serde_json::from_str(include_str!(
            "../../../../ci/matrix/phase85-mxfp-shapes-v1.json"
        ))
        .expect("Phase 85 manifest parses");
        assert_eq!(manifest.schema_version, PHASE85_SCHEMA_VERSION);
        assert!(manifest.cases.len() <= PHASE85_MAX_CASES);
        assert!(manifest.cases.iter().any(|case| case.format == "mxfp8"));
        assert!(manifest.cases.iter().any(|case| case.format == "mxfp6"));
        for tag in [
            "decode",
            "small-m",
            "large-m",
            "rectangular",
            "selector-boundary",
            "model",
            "mtp",
        ] {
            assert!(
                manifest
                    .cases
                    .iter()
                    .any(|case| case.tags.iter().any(|candidate| candidate == tag)),
                "manifest tag {tag}"
            );
        }
        assert!(manifest.cases.iter().any(|case| {
            case.format == "mxfp8" && case.m == 1 && case.k == 5120 && case.n == 248320
        }));
        assert!(
            manifest
                .cases
                .iter()
                .all(|case| { case.m > 0 && case.k > 0 && case.n > 0 && case.k % 32 == 0 })
        );
        assert!(
            manifest
                .cases
                .iter()
                .filter(|case| case.oracle == "full")
                .all(|case| case.m <= 16 && case.n <= 2560)
        );
    }
}
