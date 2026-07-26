// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Loader-independent, correctness-first MoE primitives.
//
// The CPU routines below are the semantic reference for the C ABI and the
// gfx1201 kernels. They deliberately retain explicit gather, grouped GEMM,
// and scatter stages so a caller can inspect every boundary while bringing a
// new architecture up.

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeWeightDtype {
    F32 = 0,
    Bf16 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeRouting {
    pub selected_expert_ids: Vec<i32>,
    pub routing_scores: Vec<f32>,
    /// One value per token. A nonzero value means HF's selected boundary had
    /// an exact tie, whose ordering has no stable `torch.topk` contract.
    pub boundary_tie_flags: Vec<u32>,
}

impl MoeRouting {
    pub fn has_boundary_tie(&self) -> bool {
        self.boundary_tie_flags.iter().any(|&flag| flag != 0)
    }
}

pub struct MoeF32Weights<'a> {
    /// `[num_experts, hidden_size]`, matching `mlp.gate.weight`.
    pub router: &'a [f32],
    /// `[num_experts, 2 * intermediate_size, hidden_size]`.
    pub expert_gate_up: &'a [f32],
    /// `[num_experts, hidden_size, intermediate_size]`.
    pub expert_down: &'a [f32],
    /// `[shared_intermediate_size, hidden_size]`.
    pub shared_gate: &'a [f32],
    /// `[shared_intermediate_size, hidden_size]`.
    pub shared_up: &'a [f32],
    /// `[hidden_size, shared_intermediate_size]`.
    pub shared_down: &'a [f32],
    /// `[hidden_size]`.
    pub shared_expert_gate: &'a [f32],
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| format!("{label} overflows usize"))
}

fn validate_shape(shape: MoeShape) -> Result<(), String> {
    if shape.tokens == 0
        || shape.hidden_size == 0
        || shape.num_experts == 0
        || shape.top_k == 0
        || shape.top_k > shape.num_experts
        || shape.intermediate_size == 0
        || shape.shared_intermediate_size == 0
    {
        return Err("MoE shape requires nonzero dimensions and top_k <= num_experts".to_string());
    }
    Ok(())
}

fn expect_len(actual: usize, expected: usize, label: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("{label} length is {actual}, expected {expected}"));
    }
    Ok(())
}

fn selected_elements(shape: MoeShape) -> Result<usize, String> {
    checked_mul(shape.tokens, shape.top_k, "MoE selected assignment count")
}

fn f32_to_bf16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    let rounded = bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1);
    f32::from_bits(rounded & 0xffff_0000)
}

/// Reference implementation of Qwen3.5's FP32-softmax → top-k → selected
/// weight renormalization. For BF16 source routers it emulates the HF BF16
/// linear boundary (activation cast and output-logit/score roundtrip) while
/// retaining F32 host storage. Exact ties are reported, not given a made-up
/// model meaning; the returned order is only a deterministic diagnostic order.
pub fn moe_route_reference_with_weight_dtype_f32(
    shape: MoeShape,
    hidden: &[f32],
    router: &[f32],
    router_weight_dtype: MoeWeightDtype,
) -> Result<MoeRouting, String> {
    validate_shape(shape)?;
    let hidden_elements = checked_mul(shape.tokens, shape.hidden_size, "MoE hidden elements")?;
    let router_elements = checked_mul(shape.num_experts, shape.hidden_size, "MoE router elements")?;
    expect_len(hidden.len(), hidden_elements, "MoE hidden")?;
    expect_len(router.len(), router_elements, "MoE router")?;

    let selected = selected_elements(shape)?;
    let mut selected_expert_ids = vec![0_i32; selected];
    let mut routing_scores = vec![0.0_f32; selected];
    let mut boundary_tie_flags = vec![0_u32; shape.tokens];
    let mut probabilities = vec![0.0_f32; shape.num_experts];
    let mut order = vec![0_usize; shape.num_experts];

    for token in 0..shape.tokens {
        let hidden_base = token * shape.hidden_size;
        let mut maximum = f32::NEG_INFINITY;
        for expert in 0..shape.num_experts {
            let weight_base = expert * shape.hidden_size;
            let mut logit = 0.0_f32;
            for column in 0..shape.hidden_size {
                let activation = match router_weight_dtype {
                    MoeWeightDtype::F32 => hidden[hidden_base + column],
                    MoeWeightDtype::Bf16 => f32_to_bf16_roundtrip(hidden[hidden_base + column]),
                };
                logit += activation * router[weight_base + column];
            }
            if router_weight_dtype == MoeWeightDtype::Bf16 {
                logit = f32_to_bf16_roundtrip(logit);
            }
            probabilities[expert] = logit;
            maximum = maximum.max(logit);
            order[expert] = expert;
        }
        let mut denominator = 0.0_f32;
        for probability in &mut probabilities {
            *probability = (*probability - maximum).exp();
            denominator += *probability;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(format!(
                "MoE router softmax denominator is invalid for token {token}"
            ));
        }
        for probability in &mut probabilities {
            *probability /= denominator;
        }
        order.sort_by(|lhs, rhs| {
            probabilities[*rhs]
                .total_cmp(&probabilities[*lhs])
                .then_with(|| rhs.cmp(lhs))
        });
        let selected_sum = order[..shape.top_k]
            .iter()
            .map(|&expert| probabilities[expert])
            .sum::<f32>();
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return Err(format!(
                "MoE selected routing mass is invalid for token {token}"
            ));
        }
        let boundary = probabilities[order[shape.top_k - 1]];
        boundary_tie_flags[token] = order[shape.top_k..]
            .iter()
            .any(|&expert| probabilities[expert] == boundary)
            as u32;
        let selected_base = token * shape.top_k;
        for rank in 0..shape.top_k {
            let expert = order[rank];
            selected_expert_ids[selected_base + rank] = i32::try_from(expert)
                .map_err(|_| "MoE expert index does not fit i32".to_string())?;
            let score = probabilities[expert] / selected_sum;
            routing_scores[selected_base + rank] = if router_weight_dtype == MoeWeightDtype::Bf16 {
                f32_to_bf16_roundtrip(score)
            } else {
                score
            };
        }
    }

    Ok(MoeRouting {
        selected_expert_ids,
        routing_scores,
        boundary_tie_flags,
    })
}

/// F32-weight reference route used by synthetic correctness tests and callers
/// whose router is explicitly F32 rather than a checkpoint BF16 tensor.
pub fn moe_route_reference_f32(
    shape: MoeShape,
    hidden: &[f32],
    router: &[f32],
) -> Result<MoeRouting, String> {
    moe_route_reference_with_weight_dtype_f32(shape, hidden, router, MoeWeightDtype::F32)
}

/// Materializes assignment-major `[tokens * top_k, hidden_size]` rows.
pub fn moe_gather_reference_f32(
    hidden: &[f32],
    tokens: usize,
    hidden_size: usize,
    top_k: usize,
) -> Result<Vec<f32>, String> {
    if tokens == 0 || hidden_size == 0 || top_k == 0 {
        return Err("MoE gather dimensions must be nonzero".to_string());
    }
    let input_elements = checked_mul(tokens, hidden_size, "MoE gather input elements")?;
    expect_len(hidden.len(), input_elements, "MoE gather hidden")?;
    let assignments = checked_mul(tokens, top_k, "MoE gather assignments")?;
    let output_elements = checked_mul(assignments, hidden_size, "MoE gather output elements")?;
    let mut output = vec![0.0_f32; output_elements];
    for assignment in 0..assignments {
        let token = assignment / top_k;
        output[assignment * hidden_size..(assignment + 1) * hidden_size]
            .copy_from_slice(&hidden[token * hidden_size..(token + 1) * hidden_size]);
    }
    Ok(output)
}

/// Reference grouped GEMM with row-major `[expert, row, column]` weight
/// storage. `expert_ids` may name groups with arbitrary (including zero) hit
/// counts, but every assignment ID must be in range.
pub fn moe_grouped_gemm_reference_f32(
    weights: &[f32],
    expert_ids: &[i32],
    input: &[f32],
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if assignments == 0 || num_experts == 0 || rows_per_expert == 0 || cols == 0 {
        return Err("MoE grouped GEMM dimensions must be nonzero".to_string());
    }
    let weight_rows = checked_mul(num_experts, rows_per_expert, "MoE grouped GEMM rows")?;
    let weight_elements = checked_mul(weight_rows, cols, "MoE grouped GEMM weights")?;
    let input_elements = checked_mul(assignments, cols, "MoE grouped GEMM input")?;
    let output_elements = checked_mul(assignments, rows_per_expert, "MoE grouped GEMM output")?;
    expect_len(weights.len(), weight_elements, "MoE grouped GEMM weights")?;
    expect_len(expert_ids.len(), assignments, "MoE grouped GEMM IDs")?;
    expect_len(input.len(), input_elements, "MoE grouped GEMM input")?;
    let mut output = vec![0.0_f32; output_elements];
    for assignment in 0..assignments {
        let expert = usize::try_from(expert_ids[assignment]).map_err(|_| {
            format!("MoE grouped GEMM has negative expert ID at assignment {assignment}")
        })?;
        if expert >= num_experts {
            return Err(format!(
                "MoE grouped GEMM expert ID {expert} is out of range"
            ));
        }
        for row in 0..rows_per_expert {
            let weight_base = (expert * rows_per_expert + row) * cols;
            let input_base = assignment * cols;
            let mut value = 0.0_f32;
            for column in 0..cols {
                value += weights[weight_base + column] * input[input_base + column];
            }
            output[assignment * rows_per_expert + row] = value;
        }
    }
    Ok(output)
}

/// Reference grouped GEMM for the raw weight storage accepted by the runtime
/// ABI. F32 values use native F32 bytes; BF16 values are raw IEEE BF16 words,
/// exactly as they occur in the Qwen3.5 safetensors checkpoint. Activations
/// and results remain F32.
pub fn moe_grouped_gemm_reference_raw_f32(
    weights: &[u8],
    weight_dtype: MoeWeightDtype,
    expert_ids: &[i32],
    input: &[f32],
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if assignments == 0 || num_experts == 0 || rows_per_expert == 0 || cols == 0 {
        return Err("MoE grouped GEMM dimensions must be nonzero".to_string());
    }
    let weight_elements = checked_mul(
        checked_mul(num_experts, rows_per_expert, "MoE grouped GEMM rows")?,
        cols,
        "MoE grouped GEMM weights",
    )?;
    let element_bytes = match weight_dtype {
        MoeWeightDtype::F32 => std::mem::size_of::<f32>(),
        MoeWeightDtype::Bf16 => std::mem::size_of::<u16>(),
    };
    let expected_bytes = checked_mul(
        weight_elements,
        element_bytes,
        "MoE grouped GEMM weight bytes",
    )?;
    expect_len(
        weights.len(),
        expected_bytes,
        "MoE grouped GEMM raw weights",
    )?;

    let decoded = match weight_dtype {
        MoeWeightDtype::F32 => weights
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("F32-sized chunk")))
            .collect::<Vec<_>>(),
        MoeWeightDtype::Bf16 => weights
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| {
                let bits = u16::from_ne_bytes(chunk.try_into().expect("BF16-sized chunk"));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect::<Vec<_>>(),
    };
    moe_grouped_gemm_reference_f32(
        &decoded,
        expert_ids,
        input,
        assignments,
        num_experts,
        rows_per_expert,
        cols,
    )
}

pub fn moe_gated_silu_reference_f32(
    gate_up: &[f32],
    assignments: usize,
    intermediate_size: usize,
) -> Result<Vec<f32>, String> {
    if assignments == 0 || intermediate_size == 0 {
        return Err("MoE gated SiLU dimensions must be nonzero".to_string());
    }
    let output_elements = checked_mul(assignments, intermediate_size, "MoE gated SiLU output")?;
    let gate_elements = checked_mul(output_elements, 2, "MoE gated SiLU gate/up")?;
    expect_len(gate_up.len(), gate_elements, "MoE gated SiLU input")?;
    let mut output = vec![0.0_f32; output_elements];
    for assignment in 0..assignments {
        let base = assignment * 2 * intermediate_size;
        for channel in 0..intermediate_size {
            let gate = gate_up[base + channel];
            let up = gate_up[base + intermediate_size + channel];
            output[assignment * intermediate_size + channel] = gate / (1.0 + (-gate).exp()) * up;
        }
    }
    Ok(output)
}

pub fn moe_scatter_weighted_reference_f32(
    expert_output: &[f32],
    routing_scores: &[f32],
    tokens: usize,
    top_k: usize,
    hidden_size: usize,
) -> Result<Vec<f32>, String> {
    if tokens == 0 || top_k == 0 || hidden_size == 0 {
        return Err("MoE scatter dimensions must be nonzero".to_string());
    }
    let assignments = checked_mul(tokens, top_k, "MoE scatter assignments")?;
    let expert_elements = checked_mul(assignments, hidden_size, "MoE scatter expert output")?;
    let output_elements = checked_mul(tokens, hidden_size, "MoE scatter output")?;
    expect_len(
        expert_output.len(),
        expert_elements,
        "MoE scatter expert output",
    )?;
    expect_len(
        routing_scores.len(),
        assignments,
        "MoE scatter routing scores",
    )?;
    let mut output = vec![0.0_f32; output_elements];
    for token in 0..tokens {
        let assignment_base = token * top_k;
        for hidden in 0..hidden_size {
            let mut value = 0.0_f32;
            for rank in 0..top_k {
                value += routing_scores[assignment_base + rank]
                    * expert_output[(assignment_base + rank) * hidden_size + hidden];
            }
            output[token * hidden_size + hidden] = value;
        }
    }
    Ok(output)
}

pub fn moe_sigmoid_gate_reference_f32(
    gates: &[f32],
    input: &[f32],
    tokens: usize,
    hidden_size: usize,
) -> Result<Vec<f32>, String> {
    if tokens == 0 || hidden_size == 0 {
        return Err("MoE sigmoid gate dimensions must be nonzero".to_string());
    }
    let elements = checked_mul(tokens, hidden_size, "MoE sigmoid gate elements")?;
    expect_len(gates.len(), tokens, "MoE sigmoid gate values")?;
    expect_len(input.len(), elements, "MoE sigmoid gate input")?;
    let mut output = vec![0.0_f32; elements];
    for token in 0..tokens {
        let scale = 1.0_f32 / (1.0 + (-gates[token]).exp());
        for hidden in 0..hidden_size {
            output[token * hidden_size + hidden] = scale * input[token * hidden_size + hidden];
        }
    }
    Ok(output)
}

/// Full Qwen3.5 MoE MLP reference, including the shared expert. It refuses a
/// top-k boundary tie because the upstream HF implementation offers no stable
/// tie-ordering contract to reproduce.
pub fn moe_forward_reference_f32(
    shape: MoeShape,
    hidden: &[f32],
    weights: MoeF32Weights<'_>,
) -> Result<Vec<f32>, String> {
    validate_shape(shape)?;
    let hidden_elements = checked_mul(shape.tokens, shape.hidden_size, "MoE hidden elements")?;
    expect_len(hidden.len(), hidden_elements, "MoE hidden")?;
    expect_len(
        weights.router.len(),
        checked_mul(shape.num_experts, shape.hidden_size, "MoE router elements")?,
        "MoE router",
    )?;
    expect_len(
        weights.expert_gate_up.len(),
        checked_mul(
            checked_mul(
                shape.num_experts,
                2 * shape.intermediate_size,
                "MoE gate/up rows",
            )?,
            shape.hidden_size,
            "MoE gate/up elements",
        )?,
        "MoE expert gate/up",
    )?;
    expect_len(
        weights.expert_down.len(),
        checked_mul(
            checked_mul(shape.num_experts, shape.hidden_size, "MoE down rows")?,
            shape.intermediate_size,
            "MoE down elements",
        )?,
        "MoE expert down",
    )?;
    expect_len(
        weights.shared_gate.len(),
        checked_mul(
            shape.shared_intermediate_size,
            shape.hidden_size,
            "MoE shared gate elements",
        )?,
        "MoE shared gate",
    )?;
    expect_len(
        weights.shared_up.len(),
        checked_mul(
            shape.shared_intermediate_size,
            shape.hidden_size,
            "MoE shared up elements",
        )?,
        "MoE shared up",
    )?;
    expect_len(
        weights.shared_down.len(),
        checked_mul(
            shape.hidden_size,
            shape.shared_intermediate_size,
            "MoE shared down elements",
        )?,
        "MoE shared down",
    )?;
    expect_len(
        weights.shared_expert_gate.len(),
        shape.hidden_size,
        "MoE shared expert gate",
    )?;

    let routing = moe_route_reference_f32(shape, hidden, weights.router)?;
    if routing.has_boundary_tie() {
        return Err(
            "MoE router has a top-k boundary tie; HF does not define a stable tie order"
                .to_string(),
        );
    }
    let assignments = selected_elements(shape)?;
    let gathered = moe_gather_reference_f32(hidden, shape.tokens, shape.hidden_size, shape.top_k)?;
    let gate_up = moe_grouped_gemm_reference_f32(
        weights.expert_gate_up,
        &routing.selected_expert_ids,
        &gathered,
        assignments,
        shape.num_experts,
        2 * shape.intermediate_size,
        shape.hidden_size,
    )?;
    let activated = moe_gated_silu_reference_f32(&gate_up, assignments, shape.intermediate_size)?;
    let expert_output = moe_grouped_gemm_reference_f32(
        weights.expert_down,
        &routing.selected_expert_ids,
        &activated,
        assignments,
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let routed = moe_scatter_weighted_reference_f32(
        &expert_output,
        &routing.routing_scores,
        shape.tokens,
        shape.top_k,
        shape.hidden_size,
    )?;

    let mut shared_gate_up = vec![0.0_f32; shape.tokens * 2 * shape.shared_intermediate_size];
    for token in 0..shape.tokens {
        for row in 0..shape.shared_intermediate_size {
            let input_base = token * shape.hidden_size;
            let weight_base = row * shape.hidden_size;
            let mut gate = 0.0_f32;
            let mut up = 0.0_f32;
            for column in 0..shape.hidden_size {
                gate += weights.shared_gate[weight_base + column] * hidden[input_base + column];
                up += weights.shared_up[weight_base + column] * hidden[input_base + column];
            }
            shared_gate_up[token * 2 * shape.shared_intermediate_size + row] = gate;
            shared_gate_up[token * 2 * shape.shared_intermediate_size
                + shape.shared_intermediate_size
                + row] = up;
        }
    }
    let shared_active = moe_gated_silu_reference_f32(
        &shared_gate_up,
        shape.tokens,
        shape.shared_intermediate_size,
    )?;
    let shared_ids = vec![0_i32; shape.tokens];
    let shared_output = moe_grouped_gemm_reference_f32(
        weights.shared_down,
        &shared_ids,
        &shared_active,
        shape.tokens,
        1,
        shape.hidden_size,
        shape.shared_intermediate_size,
    )?;
    let mut shared_gate_values = vec![0.0_f32; shape.tokens];
    for token in 0..shape.tokens {
        let mut value = 0.0_f32;
        for column in 0..shape.hidden_size {
            value +=
                weights.shared_expert_gate[column] * hidden[token * shape.hidden_size + column];
        }
        shared_gate_values[token] = value;
    }
    let gated_shared = moe_sigmoid_gate_reference_f32(
        &shared_gate_values,
        &shared_output,
        shape.tokens,
        shape.hidden_size,
    )?;
    Ok(routed
        .into_iter()
        .zip(gated_shared)
        .map(|(routed, shared)| routed + shared)
        .collect())
}

unsafe extern "C" {
    fn ullm_runtime_moe_route_f32(
        hidden_buffer: *const RawRuntimeBuffer,
        router_weight_buffer: *const RawRuntimeBuffer,
        router_weight_dtype: c_int,
        tokens: usize,
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        routing_scores_buffer: *mut RawRuntimeBuffer,
        selected_expert_ids_buffer: *mut RawRuntimeBuffer,
        boundary_tie_flags_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_moe_gather_f32(
        hidden_buffer: *const RawRuntimeBuffer,
        tokens: usize,
        hidden_size: usize,
        top_k: usize,
        gathered_hidden_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_moe_grouped_gemm_f32(
        weight_buffer: *const RawRuntimeBuffer,
        weight_dtype: c_int,
        expert_ids_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        assignments: usize,
        num_experts: usize,
        rows_per_expert: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_moe_gated_silu_f32(
        gate_up_buffer: *const RawRuntimeBuffer,
        assignments: usize,
        intermediate_size: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_moe_scatter_weighted_f32(
        expert_output_buffer: *const RawRuntimeBuffer,
        routing_scores_buffer: *const RawRuntimeBuffer,
        tokens: usize,
        top_k: usize,
        hidden_size: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_moe_sigmoid_gate_f32(
        gate_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        tokens: usize,
        hidden_size: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
}

fn f32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    checked_mul(elements, std::mem::size_of::<f32>(), label)
}

fn i32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    checked_mul(elements, std::mem::size_of::<i32>(), label)
}

fn u32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    checked_mul(elements, std::mem::size_of::<u32>(), label)
}

pub fn moe_route_f32(
    hidden_buffer: &RuntimeBuffer,
    router_weight_buffer: &RuntimeBuffer,
    router_weight_dtype: MoeWeightDtype,
    tokens: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    routing_scores_buffer: &mut RuntimeBuffer,
    selected_expert_ids_buffer: &mut RuntimeBuffer,
    boundary_tie_flags_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if tokens == 0 || hidden_size == 0 || num_experts == 0 || top_k == 0 || top_k > num_experts {
        return Err("MoE route requires nonzero dimensions and top_k <= num_experts".to_string());
    }
    let hidden_elements = checked_mul(tokens, hidden_size, "MoE route hidden elements")?;
    let weight_elements = checked_mul(num_experts, hidden_size, "MoE route weight elements")?;
    let selected = checked_mul(tokens, top_k, "MoE route selected elements")?;
    let weight_element_bytes = match router_weight_dtype {
        MoeWeightDtype::F32 => std::mem::size_of::<f32>(),
        MoeWeightDtype::Bf16 => std::mem::size_of::<u16>(),
    };
    check_copy_range(
        0,
        f32_bytes(hidden_elements, "MoE route hidden bytes")?,
        hidden_buffer.size()?,
    )?;
    check_copy_range(
        0,
        checked_mul(
            weight_elements,
            weight_element_bytes,
            "MoE route weight bytes",
        )?,
        router_weight_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(selected, "MoE route score bytes")?,
        routing_scores_buffer.size()?,
    )?;
    check_copy_range(
        0,
        i32_bytes(selected, "MoE route ID bytes")?,
        selected_expert_ids_buffer.size()?,
    )?;
    check_copy_range(
        0,
        u32_bytes(tokens, "MoE route flag bytes")?,
        boundary_tie_flags_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_route_f32(
            hidden_buffer.raw.as_ptr(),
            router_weight_buffer.raw.as_ptr(),
            router_weight_dtype as c_int,
            tokens,
            hidden_size,
            num_experts,
            top_k,
            routing_scores_buffer.raw.as_ptr(),
            selected_expert_ids_buffer.raw.as_ptr(),
            boundary_tie_flags_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn moe_gather_f32(
    hidden_buffer: &RuntimeBuffer,
    tokens: usize,
    hidden_size: usize,
    top_k: usize,
    gathered_hidden_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if tokens == 0 || hidden_size == 0 || top_k == 0 {
        return Err("MoE gather dimensions must be nonzero".to_string());
    }
    let input = checked_mul(tokens, hidden_size, "MoE gather input elements")?;
    let assignments = checked_mul(tokens, top_k, "MoE gather assignments")?;
    let output = checked_mul(assignments, hidden_size, "MoE gather output elements")?;
    check_copy_range(
        0,
        f32_bytes(input, "MoE gather input bytes")?,
        hidden_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(output, "MoE gather output bytes")?,
        gathered_hidden_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_gather_f32(
            hidden_buffer.raw.as_ptr(),
            tokens,
            hidden_size,
            top_k,
            gathered_hidden_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_f32(
    weight_buffer: &RuntimeBuffer,
    weight_dtype: MoeWeightDtype,
    expert_ids_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if assignments == 0 || num_experts == 0 || rows_per_expert == 0 || cols == 0 {
        return Err("MoE grouped GEMM dimensions must be nonzero".to_string());
    }
    let rows = checked_mul(num_experts, rows_per_expert, "MoE grouped GEMM rows")?;
    let weight_elements = checked_mul(rows, cols, "MoE grouped GEMM weight elements")?;
    let input_elements = checked_mul(assignments, cols, "MoE grouped GEMM input elements")?;
    let output_elements = checked_mul(
        assignments,
        rows_per_expert,
        "MoE grouped GEMM output elements",
    )?;
    let weight_bytes = checked_mul(
        weight_elements,
        match weight_dtype {
            MoeWeightDtype::F32 => std::mem::size_of::<f32>(),
            MoeWeightDtype::Bf16 => std::mem::size_of::<u16>(),
        },
        "MoE grouped GEMM weight bytes",
    )?;
    check_copy_range(0, weight_bytes, weight_buffer.size()?)?;
    check_copy_range(
        0,
        i32_bytes(assignments, "MoE grouped GEMM ID bytes")?,
        expert_ids_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(input_elements, "MoE grouped GEMM input bytes")?,
        input_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(output_elements, "MoE grouped GEMM output bytes")?,
        output_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_grouped_gemm_f32(
            weight_buffer.raw.as_ptr(),
            weight_dtype as c_int,
            expert_ids_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            assignments,
            num_experts,
            rows_per_expert,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn moe_gated_silu_f32(
    gate_up_buffer: &RuntimeBuffer,
    assignments: usize,
    intermediate_size: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if assignments == 0 || intermediate_size == 0 {
        return Err("MoE gated SiLU dimensions must be nonzero".to_string());
    }
    let output = checked_mul(
        assignments,
        intermediate_size,
        "MoE gated SiLU output elements",
    )?;
    let gate_up = checked_mul(output, 2, "MoE gated SiLU gate/up elements")?;
    check_copy_range(
        0,
        f32_bytes(gate_up, "MoE gated SiLU input bytes")?,
        gate_up_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(output, "MoE gated SiLU output bytes")?,
        output_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_gated_silu_f32(
            gate_up_buffer.raw.as_ptr(),
            assignments,
            intermediate_size,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn moe_scatter_weighted_f32(
    expert_output_buffer: &RuntimeBuffer,
    routing_scores_buffer: &RuntimeBuffer,
    tokens: usize,
    top_k: usize,
    hidden_size: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if tokens == 0 || top_k == 0 || hidden_size == 0 {
        return Err("MoE scatter dimensions must be nonzero".to_string());
    }
    let assignments = checked_mul(tokens, top_k, "MoE scatter assignments")?;
    let expert = checked_mul(assignments, hidden_size, "MoE scatter expert elements")?;
    let output = checked_mul(tokens, hidden_size, "MoE scatter output elements")?;
    check_copy_range(
        0,
        f32_bytes(expert, "MoE scatter expert bytes")?,
        expert_output_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(assignments, "MoE scatter score bytes")?,
        routing_scores_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(output, "MoE scatter output bytes")?,
        output_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_scatter_weighted_f32(
            expert_output_buffer.raw.as_ptr(),
            routing_scores_buffer.raw.as_ptr(),
            tokens,
            top_k,
            hidden_size,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn moe_sigmoid_gate_f32(
    gate_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    tokens: usize,
    hidden_size: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if tokens == 0 || hidden_size == 0 {
        return Err("MoE sigmoid gate dimensions must be nonzero".to_string());
    }
    let elements = checked_mul(tokens, hidden_size, "MoE sigmoid gate elements")?;
    check_copy_range(
        0,
        f32_bytes(tokens, "MoE sigmoid gate values bytes")?,
        gate_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(elements, "MoE sigmoid gate input bytes")?,
        input_buffer.size()?,
    )?;
    check_copy_range(
        0,
        f32_bytes(elements, "MoE sigmoid gate output bytes")?,
        output_buffer.size()?,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_moe_sigmoid_gate_f32(
            gate_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            tokens,
            hidden_size,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[cfg(test)]
mod moe_tests {
    use super::*;

    fn values(count: usize, phase: f32) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as f32 + 1.0) * 0.173 + phase).sin() * 0.37)
            .collect()
    }

    #[test]
    fn reference_pipeline_is_finite_and_normalizes_routing() {
        let shape = MoeShape {
            tokens: 3,
            hidden_size: 5,
            num_experts: 4,
            top_k: 2,
            intermediate_size: 3,
            shared_intermediate_size: 2,
        };
        let hidden = values(shape.tokens * shape.hidden_size, 0.1);
        let router = values(shape.num_experts * shape.hidden_size, 0.2);
        let expert_gate_up = values(
            shape.num_experts * 2 * shape.intermediate_size * shape.hidden_size,
            0.3,
        );
        let expert_down = values(
            shape.num_experts * shape.hidden_size * shape.intermediate_size,
            0.4,
        );
        let shared_gate = values(shape.shared_intermediate_size * shape.hidden_size, 0.5);
        let shared_up = values(shape.shared_intermediate_size * shape.hidden_size, 0.6);
        let shared_down = values(shape.hidden_size * shape.shared_intermediate_size, 0.7);
        let shared_expert_gate = values(shape.hidden_size, 0.8);
        let routing = moe_route_reference_f32(shape, &hidden, &router).unwrap();
        assert!(!routing.has_boundary_tie());
        for token in 0..shape.tokens {
            let sum = routing.routing_scores[token * shape.top_k..(token + 1) * shape.top_k]
                .iter()
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 1.0e-6);
        }
        let output = moe_forward_reference_f32(
            shape,
            &hidden,
            MoeF32Weights {
                router: &router,
                expert_gate_up: &expert_gate_up,
                expert_down: &expert_down,
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_expert_gate: &shared_expert_gate,
            },
        )
        .unwrap();
        assert_eq!(output.len(), shape.tokens * shape.hidden_size);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn topk_boundary_tie_is_observable_not_silently_semantic() {
        let shape = MoeShape {
            tokens: 1,
            hidden_size: 2,
            num_experts: 4,
            top_k: 2,
            intermediate_size: 1,
            shared_intermediate_size: 1,
        };
        let routing = moe_route_reference_f32(shape, &[1.0, -1.0], &[0.0; 8]).unwrap();
        assert!(routing.has_boundary_tie());
    }

    #[test]
    fn raw_bf16_grouped_gemm_decodes_checkpoint_storage() {
        // Two expert matrices, each [rows=2, columns=3]. The selected order
        // deliberately visits expert one before expert zero.
        let f32_weights = [
            1.0_f32, -2.0, 0.5, 0.25, 0.75, -1.0, // expert 0
            -0.5, 1.5, 2.0, 3.0, -1.25, 0.125, // expert 1
        ];
        let mut bf16_weights = Vec::with_capacity(f32_weights.len() * 2);
        for value in f32_weights {
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1);
            bf16_weights.extend_from_slice(&((rounded >> 16) as u16).to_ne_bytes());
        }
        let ids = [1_i32, 0_i32];
        let input = [2.0_f32, -1.0, 0.5, -0.25, 0.75, 1.0];
        let raw = moe_grouped_gemm_reference_raw_f32(
            &bf16_weights,
            MoeWeightDtype::Bf16,
            &ids,
            &input,
            2,
            2,
            2,
            3,
        )
        .unwrap();
        let rounded_weights = f32_weights
            .iter()
            .map(|value| f32_to_bf16_roundtrip(*value))
            .collect::<Vec<_>>();
        let expected =
            moe_grouped_gemm_reference_f32(&rounded_weights, &ids, &input, 2, 2, 2, 3).unwrap();
        assert_eq!(raw, expected);
    }
}
