// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::format_id::FORMAT_SQ8_0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendRequest<'a> {
    pub operation: &'a str,
    pub phase: &'a str,
    pub format_id: Option<&'a str>,
    pub model_arch: Option<&'a str>,
    pub gpu_arch: Option<&'a str>,
    pub gpu_name: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendImplementation<'a> {
    pub id: &'a str,
    pub operation: &'a str,
    pub phase: &'a str,
    pub format_id: Option<&'a str>,
    pub model_arch: Option<&'a str>,
    pub gpu_arch: Option<&'a str>,
    pub gpu_name: Option<&'a str>,
    pub priority: i32,
}

pub fn select_backend<'a>(
    request: &BackendRequest<'_>,
    implementations: &'a [BackendImplementation<'a>],
) -> Option<&'a BackendImplementation<'a>> {
    implementations
        .iter()
        .filter(|implementation| implementation.matches(request))
        .max_by_key(|implementation| (implementation.specificity(), implementation.priority))
}

impl BackendImplementation<'_> {
    fn matches(&self, request: &BackendRequest<'_>) -> bool {
        self.operation == request.operation
            && self.phase == request.phase
            && optional_match(self.format_id, request.format_id)
            && optional_match(self.model_arch, request.model_arch)
            && optional_match(self.gpu_arch, request.gpu_arch)
            && optional_match(self.gpu_name, request.gpu_name)
    }

    fn specificity(&self) -> i32 {
        [
            self.format_id,
            self.model_arch,
            self.gpu_arch,
            self.gpu_name,
        ]
        .into_iter()
        .filter(|value| value.is_some())
        .count() as i32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sq8ProjectionMatvecOperation {
    Single,
    Batch,
    Pair,
    Triple,
}

impl Sq8ProjectionMatvecOperation {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Single => SQ8_0_MATVEC_OPERATION,
            Self::Batch => SQ8_0_MATVEC_BATCH_OPERATION,
            Self::Pair => SQ8_0_MATVEC_PAIR_OPERATION,
            Self::Triple => SQ8_0_MATVEC_TRIPLE_OPERATION,
        }
    }

    pub const fn operation_token(self) -> &'static str {
        match self {
            Self::Single => "matvec",
            Self::Batch => "matvec_batch",
            Self::Pair => "matvec_pair",
            Self::Triple => "matvec_triple",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Batch => "batch",
            Self::Pair => "pair",
            Self::Triple => "triple",
        }
    }

    pub const fn all() -> &'static [Self; 4] {
        &[Self::Single, Self::Batch, Self::Pair, Self::Triple]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sq8ProjectionFusedOperation {
    SelfAttnQkv,
    SelfAttnO,
    MlpGateUp,
    MlpDown,
    LinearAttnQkv,
    LinearAttnOut,
}

impl Sq8ProjectionFusedOperation {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::SelfAttnQkv => SQ8_0_SELF_ATTN_QKV_OPERATION,
            Self::SelfAttnO => SQ8_0_SELF_ATTN_O_OPERATION,
            Self::MlpGateUp => SQ8_0_MLP_GATE_UP_OPERATION,
            Self::MlpDown => SQ8_0_MLP_DOWN_OPERATION,
            Self::LinearAttnQkv => SQ8_0_LINEAR_ATTN_QKV_OPERATION,
            Self::LinearAttnOut => SQ8_0_LINEAR_ATTN_OUT_OPERATION,
        }
    }

    pub const fn operation_token(self) -> &'static str {
        match self {
            Self::SelfAttnQkv => "self_attn_qkv",
            Self::SelfAttnO => "self_attn_o",
            Self::MlpGateUp => "mlp_gate_up",
            Self::MlpDown => "mlp_down",
            Self::LinearAttnQkv => "linear_attn_qkv",
            Self::LinearAttnOut => "linear_attn_out",
        }
    }

    pub const fn all() -> &'static [Self; 6] {
        &[
            Self::SelfAttnQkv,
            Self::SelfAttnO,
            Self::MlpGateUp,
            Self::MlpDown,
            Self::LinearAttnQkv,
            Self::LinearAttnOut,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sq8ProjectionTarget {
    Generic,
    Rdna4,
}

impl Sq8ProjectionTarget {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Rdna4 => "rdna4",
        }
    }

    pub const fn gpu_arch(self) -> Option<&'static str> {
        match self {
            Self::Generic => None,
            Self::Rdna4 => Some("RDNA4"),
        }
    }

    pub const fn all() -> &'static [Self; 2] {
        &[Self::Generic, Self::Rdna4]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sq8ProjectionFamily {
    Direct,
}

impl Sq8ProjectionFamily {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Direct => "direct",
        }
    }

    pub const fn all() -> &'static [Self; 1] {
        &[Self::Direct]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sq8ProjectionFusedFamily {
    V0,
}

impl Sq8ProjectionFusedFamily {
    pub const fn id(self) -> &'static str {
        match self {
            Self::V0 => "v0",
        }
    }

    pub const fn all() -> &'static [Self; 1] {
        &[Self::V0]
    }
}

pub const SQ8_0_PROJECTION_DISPATCH_PHASE: &str = "component";

pub const SQ8_0_MATVEC_OPERATION: &str = "sq8_0_matvec";
pub const SQ8_0_MATVEC_BATCH_OPERATION: &str = "sq8_0_matvec_batch";
pub const SQ8_0_MATVEC_PAIR_OPERATION: &str = "sq8_0_matvec_pair";
pub const SQ8_0_MATVEC_TRIPLE_OPERATION: &str = "sq8_0_matvec_triple";
pub const SQ8_0_SELF_ATTN_QKV_OPERATION: &str = "sq8_0_self_attn_qkv";
pub const SQ8_0_SELF_ATTN_O_OPERATION: &str = "sq8_0_self_attn_o";
pub const SQ8_0_MLP_GATE_UP_OPERATION: &str = "sq8_0_mlp_gate_up";
pub const SQ8_0_MLP_DOWN_OPERATION: &str = "sq8_0_mlp_down";
pub const SQ8_0_LINEAR_ATTN_QKV_OPERATION: &str = "sq8_0_linear_attn_qkv";
pub const SQ8_0_LINEAR_ATTN_OUT_OPERATION: &str = "sq8_0_linear_attn_out";

// Naming convention for SQ8_0 projection descriptors:
// sq8_0_<operation>_<target>_<family>
pub const SQ8_0_PROJECTION_IMPLEMENTATION_NAMING_TEMPLATE: &str =
    "sq8_0_<operation>_<target>_<family>";
pub const SQ8_0_PROJECTION_UNRESOLVED_ID: &str = "sq8_0_projection_unresolved";

pub const fn sq8_0_projection_descriptor_id(
    operation: Sq8ProjectionMatvecOperation,
    target: Sq8ProjectionTarget,
    family: Sq8ProjectionFamily,
) -> &'static str {
    match (operation, target, family) {
        (
            Sq8ProjectionMatvecOperation::Single,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_generic_direct",
        (
            Sq8ProjectionMatvecOperation::Single,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_rdna4_direct",
        (
            Sq8ProjectionMatvecOperation::Batch,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_batch_generic_direct",
        (
            Sq8ProjectionMatvecOperation::Batch,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_batch_rdna4_direct",
        (
            Sq8ProjectionMatvecOperation::Pair,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_pair_generic_direct",
        (
            Sq8ProjectionMatvecOperation::Pair,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_pair_rdna4_direct",
        (
            Sq8ProjectionMatvecOperation::Triple,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_triple_generic_direct",
        (
            Sq8ProjectionMatvecOperation::Triple,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFamily::Direct,
        ) => "sq8_0_matvec_triple_rdna4_direct",
    }
}

pub const fn sq8_0_fused_projection_descriptor_id(
    operation: Sq8ProjectionFusedOperation,
    target: Sq8ProjectionTarget,
    family: Sq8ProjectionFusedFamily,
) -> &'static str {
    match (operation, target, family) {
        (
            Sq8ProjectionFusedOperation::SelfAttnQkv,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_self_attn_qkv_generic_v0",
        (
            Sq8ProjectionFusedOperation::SelfAttnQkv,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_self_attn_qkv_rdna4_v0",
        (
            Sq8ProjectionFusedOperation::SelfAttnO,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_self_attn_o_generic_v0",
        (
            Sq8ProjectionFusedOperation::SelfAttnO,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_self_attn_o_rdna4_v0",
        (
            Sq8ProjectionFusedOperation::MlpGateUp,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_mlp_gate_up_generic_v0",
        (
            Sq8ProjectionFusedOperation::MlpGateUp,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_mlp_gate_up_rdna4_v0",
        (
            Sq8ProjectionFusedOperation::MlpDown,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_mlp_down_generic_v0",
        (
            Sq8ProjectionFusedOperation::MlpDown,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_mlp_down_rdna4_v0",
        (
            Sq8ProjectionFusedOperation::LinearAttnQkv,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_linear_attn_qkv_generic_v0",
        (
            Sq8ProjectionFusedOperation::LinearAttnQkv,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_linear_attn_qkv_rdna4_v0",
        (
            Sq8ProjectionFusedOperation::LinearAttnOut,
            Sq8ProjectionTarget::Generic,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_linear_attn_out_generic_v0",
        (
            Sq8ProjectionFusedOperation::LinearAttnOut,
            Sq8ProjectionTarget::Rdna4,
            Sq8ProjectionFusedFamily::V0,
        ) => "sq8_0_linear_attn_out_rdna4_v0",
    }
}

pub const SQ8_0_MATVEC_GENERIC_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Single,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_RDNA4_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Single,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_BATCH_GENERIC_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Batch,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_BATCH_RDNA4_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Batch,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_PAIR_GENERIC_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Pair,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_PAIR_RDNA4_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Pair,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_TRIPLE_GENERIC_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Triple,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFamily::Direct,
);
pub const SQ8_0_MATVEC_TRIPLE_RDNA4_DIRECT_ID: &str = sq8_0_projection_descriptor_id(
    Sq8ProjectionMatvecOperation::Triple,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFamily::Direct,
);

pub const SQ8_0_SELF_ATTN_QKV_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::SelfAttnQkv,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_SELF_ATTN_QKV_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::SelfAttnQkv,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_SELF_ATTN_O_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::SelfAttnO,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_SELF_ATTN_O_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::SelfAttnO,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_MLP_GATE_UP_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::MlpGateUp,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_MLP_GATE_UP_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::MlpGateUp,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_MLP_DOWN_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::MlpDown,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_MLP_DOWN_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::MlpDown,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_LINEAR_ATTN_QKV_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::LinearAttnQkv,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_LINEAR_ATTN_QKV_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::LinearAttnQkv,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_LINEAR_ATTN_OUT_GENERIC_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::LinearAttnOut,
    Sq8ProjectionTarget::Generic,
    Sq8ProjectionFusedFamily::V0,
);
pub const SQ8_0_LINEAR_ATTN_OUT_RDNA4_V0_ID: &str = sq8_0_fused_projection_descriptor_id(
    Sq8ProjectionFusedOperation::LinearAttnOut,
    Sq8ProjectionTarget::Rdna4,
    Sq8ProjectionFusedFamily::V0,
);

pub const SQ8_0_PROJECTION_DISPATCH_IMPLEMENTATIONS: &[BackendImplementation<'static>] = &[
    BackendImplementation {
        id: SQ8_0_MATVEC_GENERIC_DIRECT_ID,
        operation: SQ8_0_MATVEC_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_RDNA4_DIRECT_ID,
        operation: SQ8_0_MATVEC_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_BATCH_GENERIC_DIRECT_ID,
        operation: SQ8_0_MATVEC_BATCH_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_BATCH_RDNA4_DIRECT_ID,
        operation: SQ8_0_MATVEC_BATCH_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_PAIR_GENERIC_DIRECT_ID,
        operation: SQ8_0_MATVEC_PAIR_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_PAIR_RDNA4_DIRECT_ID,
        operation: SQ8_0_MATVEC_PAIR_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_TRIPLE_GENERIC_DIRECT_ID,
        operation: SQ8_0_MATVEC_TRIPLE_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MATVEC_TRIPLE_RDNA4_DIRECT_ID,
        operation: SQ8_0_MATVEC_TRIPLE_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
];

// Catalog of planned higher-level fused projection descriptor entries.
// These are intentionally kept separate from active dispatch implementations until kernels are ready.
pub const SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG: &[BackendImplementation<'static>] = &[
    BackendImplementation {
        id: SQ8_0_SELF_ATTN_QKV_GENERIC_V0_ID,
        operation: SQ8_0_SELF_ATTN_QKV_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_SELF_ATTN_QKV_RDNA4_V0_ID,
        operation: SQ8_0_SELF_ATTN_QKV_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_SELF_ATTN_O_GENERIC_V0_ID,
        operation: SQ8_0_SELF_ATTN_O_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_SELF_ATTN_O_RDNA4_V0_ID,
        operation: SQ8_0_SELF_ATTN_O_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_MLP_GATE_UP_GENERIC_V0_ID,
        operation: SQ8_0_MLP_GATE_UP_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MLP_GATE_UP_RDNA4_V0_ID,
        operation: SQ8_0_MLP_GATE_UP_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_MLP_DOWN_GENERIC_V0_ID,
        operation: SQ8_0_MLP_DOWN_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_MLP_DOWN_RDNA4_V0_ID,
        operation: SQ8_0_MLP_DOWN_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_LINEAR_ATTN_QKV_GENERIC_V0_ID,
        operation: SQ8_0_LINEAR_ATTN_QKV_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_LINEAR_ATTN_QKV_RDNA4_V0_ID,
        operation: SQ8_0_LINEAR_ATTN_QKV_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
    BackendImplementation {
        id: SQ8_0_LINEAR_ATTN_OUT_GENERIC_V0_ID,
        operation: SQ8_0_LINEAR_ATTN_OUT_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Generic.gpu_arch(),
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: SQ8_0_LINEAR_ATTN_OUT_RDNA4_V0_ID,
        operation: SQ8_0_LINEAR_ATTN_OUT_OPERATION,
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Sq8ProjectionTarget::Rdna4.gpu_arch(),
        gpu_name: None,
        priority: 10,
    },
];

pub fn select_sq8_projection_implementation(
    request: &BackendRequest<'_>,
) -> Option<&'static BackendImplementation<'static>> {
    select_backend(request, SQ8_0_PROJECTION_DISPATCH_IMPLEMENTATIONS)
}

pub fn select_sq8_projection_implementation_id(request: &BackendRequest<'_>) -> &'static str {
    select_sq8_projection_implementation(request)
        .map(|implementation| implementation.id)
        .unwrap_or(SQ8_0_PROJECTION_UNRESOLVED_ID)
}

fn optional_match(expected: Option<&str>, actual: Option<&str>) -> bool {
    match expected {
        Some(expected) => actual == Some(expected),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_gpu_decode_overrides_arch_decode() {
        let implementations = [
            BackendImplementation {
                id: "decode_Ampere",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "decode_A100",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: Some("A100_80GB"),
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "decode",
            format_id: None,
            model_arch: Some("Qwen3"),
            gpu_arch: Some("Ampere"),
            gpu_name: Some("A100_80GB"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "decode_A100");
    }

    #[test]
    fn arch_prefill_is_used_when_gpu_specific_prefill_is_absent() {
        let implementations = [
            BackendImplementation {
                id: "prefill_default",
                operation: "attention",
                phase: "prefill",
                format_id: None,
                model_arch: None,
                gpu_arch: None,
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "prefill_Ampere",
                operation: "attention",
                phase: "prefill",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: None,
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "prefill",
            format_id: None,
            model_arch: Some("Qwen3"),
            gpu_arch: Some("Ampere"),
            gpu_name: Some("A100_80GB"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "prefill_Ampere");
    }

    #[test]
    fn format_specific_implementation_beats_generic_arch_match() {
        let implementations = [
            BackendImplementation {
                id: "decode_RDNA4_generic",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("RDNA4"),
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "decode_RDNA4_SQ8_0",
                operation: "attention",
                phase: "decode",
                format_id: Some("SQ8_0"),
                model_arch: None,
                gpu_arch: Some("RDNA4"),
                gpu_name: None,
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "decode",
            format_id: Some("SQ8_0"),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("RDNA4"),
            gpu_name: Some("R9700"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "decode_RDNA4_SQ8_0");
    }

    #[test]
    fn sq8_projection_rdna4_request_selects_rdna4_direct() {
        let request = BackendRequest {
            operation: SQ8_0_MATVEC_OPERATION,
            phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
            format_id: Some(FORMAT_SQ8_0),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("RDNA4"),
            gpu_name: Some("R9700"),
        };
        let selected = select_sq8_projection_implementation(&request).unwrap();
        assert_eq!(selected.id, SQ8_0_MATVEC_RDNA4_DIRECT_ID);
    }

    #[test]
    fn sq8_projection_generic_request_selects_generic_direct() {
        let request = BackendRequest {
            operation: SQ8_0_MATVEC_BATCH_OPERATION,
            phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
            format_id: Some(FORMAT_SQ8_0),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("Ampere"),
            gpu_name: Some("A100"),
        };
        let selected = select_sq8_projection_implementation(&request).unwrap();
        assert_eq!(selected.id, SQ8_0_MATVEC_BATCH_GENERIC_DIRECT_ID);
    }

    #[test]
    fn sq8_projection_registry_covers_single_batch_pair_and_triple() {
        assert_eq!(
            SQ8_0_PROJECTION_DISPATCH_IMPLEMENTATIONS.len(),
            Sq8ProjectionMatvecOperation::all().len()
                * Sq8ProjectionTarget::all().len()
                * Sq8ProjectionFamily::all().len()
        );

        for operation in Sq8ProjectionMatvecOperation::all().iter() {
            let variants: Vec<_> = SQ8_0_PROJECTION_DISPATCH_IMPLEMENTATIONS
                .iter()
                .filter(|implementation| implementation.operation == operation.operation_id())
                .collect();
            assert_eq!(variants.len(), 2);

            for target in Sq8ProjectionTarget::all().iter() {
                let expected = sq8_0_projection_descriptor_id(
                    *operation,
                    *target,
                    Sq8ProjectionFamily::Direct,
                );
                let implementation = variants
                    .iter()
                    .find(|candidate| candidate.id == expected)
                    .unwrap();
                assert_eq!(implementation.phase, SQ8_0_PROJECTION_DISPATCH_PHASE);
                assert_eq!(implementation.format_id, Some(FORMAT_SQ8_0));
                assert_eq!(implementation.gpu_arch, target.gpu_arch());
            }
        }
    }

    #[test]
    fn sq8_projection_descriptor_ids_follow_naming_convention() {
        for operation in Sq8ProjectionMatvecOperation::all().iter() {
            for target in Sq8ProjectionTarget::all().iter() {
                let expected = sq8_0_projection_descriptor_id(
                    *operation,
                    *target,
                    Sq8ProjectionFamily::Direct,
                );
                let mut parts: Vec<&str> = expected.split('_').collect();
                assert!(parts.len() >= 3);
                let family = parts.pop().unwrap();
                let target_part = parts.pop().unwrap();
                let operation_part = parts[2..].join("_");
                assert_eq!(parts[0], "sq8");
                assert_eq!(parts[1], "0");
                assert_eq!(operation_part, operation.operation_token());
                assert_eq!(target_part, target.id());
                assert_eq!(family, Sq8ProjectionFamily::Direct.id());
            }
        }
    }

    #[test]
    fn sq8_fused_projection_descriptor_ids_follow_naming_convention() {
        for operation in Sq8ProjectionFusedOperation::all().iter() {
            for target in Sq8ProjectionTarget::all().iter() {
                for family in Sq8ProjectionFusedFamily::all().iter() {
                    let expected =
                        sq8_0_fused_projection_descriptor_id(*operation, *target, *family);
                    assert_eq!(
                        expected,
                        format!(
                            "sq8_0_{}_{}_{}",
                            operation.operation_token(),
                            target.id(),
                            family.id()
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn sq8_fused_projection_registry_covers_expected_entries() {
        assert_eq!(
            SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG.len(),
            Sq8ProjectionFusedOperation::all().len()
                * Sq8ProjectionTarget::all().len()
                * Sq8ProjectionFusedFamily::all().len()
        );

        for operation in Sq8ProjectionFusedOperation::all().iter() {
            let variants: Vec<_> = SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG
                .iter()
                .filter(|implementation| implementation.operation == operation.operation_id())
                .collect();
            assert_eq!(variants.len(), 2);

            for target in Sq8ProjectionTarget::all().iter() {
                for family in Sq8ProjectionFusedFamily::all().iter() {
                    let expected =
                        sq8_0_fused_projection_descriptor_id(*operation, *target, *family);
                    let implementation = variants
                        .iter()
                        .find(|candidate| candidate.id == expected)
                        .unwrap();
                    assert_eq!(implementation.phase, SQ8_0_PROJECTION_DISPATCH_PHASE);
                    assert_eq!(implementation.format_id, Some(FORMAT_SQ8_0));
                    assert_eq!(implementation.gpu_arch, target.gpu_arch());
                    assert_eq!(
                        implementation.priority,
                        match target {
                            Sq8ProjectionTarget::Generic => 0,
                            Sq8ProjectionTarget::Rdna4 => 10,
                        }
                    );
                    let _ = family;
                }
            }
        }
    }

    #[test]
    fn sq8_fused_projection_catalog_is_not_in_active_projection_dispatch_list() {
        for fused in SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG {
            assert!(
                SQ8_0_PROJECTION_DISPATCH_IMPLEMENTATIONS
                    .iter()
                    .all(|entry| entry.id != fused.id),
                "fused projection entry unexpectedly active: {}",
                fused.id
            );
        }
    }

    #[test]
    fn sq8_fused_projection_request_is_unresolved_until_runtime_selection_is_enabled() {
        let request = BackendRequest {
            operation: SQ8_0_SELF_ATTN_QKV_OPERATION,
            phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
            format_id: Some(FORMAT_SQ8_0),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("RDNA4"),
            gpu_name: Some("R9700"),
        };
        assert_eq!(
            select_sq8_projection_implementation_id(&request),
            SQ8_0_PROJECTION_UNRESOLVED_ID
        );
    }

    #[test]
    fn sq8_projection_selector_reports_unresolved_for_wrong_phase() {
        let request = BackendRequest {
            operation: SQ8_0_MATVEC_OPERATION,
            phase: "decode",
            format_id: Some(FORMAT_SQ8_0),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("RDNA4"),
            gpu_name: Some("R9700"),
        };
        assert_eq!(
            select_sq8_projection_implementation_id(&request),
            SQ8_0_PROJECTION_UNRESOLVED_ID
        );
    }
}
