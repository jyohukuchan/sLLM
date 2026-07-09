fn golden_fixture_default_layer_range(
    fixture: &GoldenTensorFixture,
) -> Result<(usize, usize), String> {
    if let (Some(start), Some(end_exclusive)) = (
        fixture.metadata().layer_start,
        fixture.metadata().layer_end_exclusive,
    ) {
        if end_exclusive <= start {
            return Err(format!(
                "golden fixture metadata has invalid layer range: start={start}, end_exclusive={end_exclusive}"
            ));
        }
        return Ok((start, end_exclusive));
    }

    let min_layer = fixture
        .layers()
        .iter()
        .map(|layer| layer.layer_index)
        .min()
        .ok_or_else(|| "golden fixture has no layer entries".to_string())?;
    let max_layer = fixture
        .layers()
        .iter()
        .map(|layer| layer.layer_index)
        .max()
        .ok_or_else(|| "golden fixture has no layer entries".to_string())?;
    let end_exclusive = max_layer
        .checked_add(1)
        .ok_or_else(|| "golden fixture max layer index overflows".to_string())?;
    Ok((min_layer, end_exclusive))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageDecoderLayerKind {
    SelfAttention,
    LinearAttention,
}

impl PackageDecoderLayerKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::SelfAttention => "self_attention",
            Self::LinearAttention => "linear_attention",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackageManifestLayerEntry {
    layer_index: usize,
    kind: PackageDecoderLayerKind,
}

fn package_manifest_layer_entries(path: &str) -> Result<Vec<PackageManifestLayerEntry>, String> {
    let bundles = list_tensor_payload_bundles(path)?;
    let mut layers =
        std::collections::BTreeMap::<usize, std::collections::BTreeSet<&'static str>>::new();
    for bundle in bundles {
        if let Some(layer_index) = parse_language_model_layer_tensor_suffix(
            &bundle.tensor_name,
            ".self_attn.q_proj.weight",
        ) {
            layers
                .entry(layer_index)
                .or_default()
                .insert("self_attention");
        }
        if let Some(layer_index) = parse_language_model_layer_tensor_suffix(
            &bundle.tensor_name,
            ".linear_attn.in_proj_qkv.weight",
        ) {
            layers
                .entry(layer_index)
                .or_default()
                .insert("linear_attention");
        }
    }
    if layers.is_empty() {
        return Err(format!(
            "package {path} has no supported self_attn or linear_attn layer tensors"
        ));
    }

    let mut entries = Vec::with_capacity(layers.len());
    for (layer_index, kinds) in layers {
        let kind = if kinds.len() == 1 && kinds.contains("self_attention") {
            PackageDecoderLayerKind::SelfAttention
        } else if kinds.len() == 1 && kinds.contains("linear_attention") {
            PackageDecoderLayerKind::LinearAttention
        } else {
            return Err(format!(
                "package {path} layer {layer_index} has ambiguous layer kinds: {:?}",
                kinds
            ));
        };
        entries.push(PackageManifestLayerEntry { layer_index, kind });
    }
    Ok(entries)
}

fn package_layer_entries_for_indices(
    path: &str,
    layer_indices: &[usize],
) -> Result<Vec<PackageManifestLayerEntry>, String> {
    layer_indices
        .iter()
        .copied()
        .map(|layer_index| {
            package_decoder_layer_kind(path, layer_index)
                .map(|kind| PackageManifestLayerEntry { layer_index, kind })
        })
        .collect()
}

fn package_layer_entries_are_contiguous(entries: &[PackageManifestLayerEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| window[0].layer_index + 1 == window[1].layer_index)
}

fn package_decoder_layer_kind(
    path: &str,
    layer_index: usize,
) -> Result<PackageDecoderLayerKind, String> {
    let self_attn_q = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    if select_tensor_payload_bundle(path, &TensorSelector::Name(self_attn_q)).is_ok() {
        return Ok(PackageDecoderLayerKind::SelfAttention);
    }

    let linear_qkv =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    if select_tensor_payload_bundle(path, &TensorSelector::Name(linear_qkv)).is_ok() {
        return Ok(PackageDecoderLayerKind::LinearAttention);
    }

    Err(format!(
        "package layer {layer_index} has neither supported self_attn nor linear_attn package tensors"
    ))
}

fn package_linear_attn_candidate_ids(path: &str, layer_index: usize) -> Vec<String> {
    [
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight"),
    ]
    .iter()
    .map(|tensor_name| {
        select_tensor_payload_bundle(path, &TensorSelector::Name(tensor_name.clone()))
            .ok()
            .and_then(|bundle| bundle.candidate_id)
            .unwrap_or_else(|| "unknown".to_string())
    })
    .collect()
}

fn insert_json_detail<T: serde::Serialize>(
    details: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: T,
) {
    details.insert(key.to_string(), serde_json::json!(value));
}

fn package_runtime_line_metrics(line: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut metrics = serde_json::Map::new();
    for token in line.split_whitespace() {
        let Some((key, raw_value)) = token.split_once('=') else {
            continue;
        };
        if !package_runtime_line_metric_key(key) {
            continue;
        }
        let value = raw_value.trim_matches('"').trim_end_matches(',');
        if value == "true" || value == "false" {
            insert_json_detail(&mut metrics, key, value == "true");
        } else if let Ok(value) = value.parse::<i64>() {
            insert_json_detail(&mut metrics, key, value);
        } else if let Ok(value) = value.parse::<f64>() {
            insert_json_detail(&mut metrics, key, value);
        }
    }
    metrics
}

fn package_runtime_line_metric_key(key: &str) -> bool {
    matches!(
        key,
        "hidden"
            | "key_heads"
            | "value_heads"
            | "key_dim"
            | "value_dim"
            | "sequence_len"
            | "kernel_size"
            | "q_scale"
            | "device_index"
            | "verified"
    ) || key.ends_with("_max_abs_diff")
        || key.ends_with("_mse")
        || key.ends_with("_mean_abs_diff")
        || key.ends_with("_cosine_similarity")
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_runtime_diagnostic(
    prepared_attention_output: &[f32],
    layer_attention_output: &[f32],
    layer_attention_projection_input: &[f32],
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    q_gate: Option<&[f32]>,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let attention_width = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "self-attn diagnostic attention width overflows".to_string())?;
    let expected_attention_elements = sequence_len
        .checked_mul(attention_width)
        .ok_or_else(|| "self-attn diagnostic attention element count overflows".to_string())?;
    let expected_q_elements = sequence_len
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn diagnostic q element count overflows".to_string())?;
    let expected_k_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn diagnostic k element count overflows".to_string())?;
    let expected_v_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "self-attn diagnostic v element count overflows".to_string())?;
    for (label, values, expected) in [
        (
            "prepared_attention_output",
            prepared_attention_output,
            expected_attention_elements,
        ),
        (
            "layer_attention_output",
            layer_attention_output,
            expected_attention_elements,
        ),
        (
            "layer_attention_projection_input",
            layer_attention_projection_input,
            expected_attention_elements,
        ),
        ("q_rope", q_rope, expected_q_elements),
        ("k_rope", k_rope, expected_k_elements),
        ("v_projected", v_projected, expected_v_elements),
    ] {
        if values.len() != expected {
            return Err(format!(
                "self-attn diagnostic {label} length mismatch: got {} expected {expected}",
                values.len()
            ));
        }
    }
    if let Some(gate) = q_gate {
        if gate.len() != expected_attention_elements {
            return Err(format!(
                "self-attn diagnostic q_gate length mismatch: got {} expected {expected_attention_elements}",
                gate.len()
            ));
        }
    }

    let host_attention_output = runtime_host_causal_attn_f32(
        q_rope,
        k_rope,
        v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    if host_attention_output.len() != expected_attention_elements {
        return Err(format!(
            "self-attn diagnostic host attention length mismatch: got {} expected {expected_attention_elements}",
            host_attention_output.len()
        ));
    }

    let prepared_projection_input = package_attention_projection_input_from_gate(
        q_gate,
        prepared_attention_output,
        expected_attention_elements,
        "prepared",
    )?;
    let layer_projection_input_from_attention = package_attention_projection_input_from_gate(
        q_gate,
        layer_attention_output,
        expected_attention_elements,
        "layer",
    )?;
    let host_projection_input = package_attention_projection_input_from_gate(
        q_gate,
        &host_attention_output,
        expected_attention_elements,
        "host",
    )?;

    let mut diagnostic = serde_json::Map::new();
    insert_json_detail(&mut diagnostic, "sequence_len", sequence_len);
    insert_json_detail(&mut diagnostic, "attention_width", attention_width);
    insert_json_detail(&mut diagnostic, "q_heads", q_heads);
    insert_json_detail(&mut diagnostic, "kv_heads", kv_heads);
    insert_json_detail(&mut diagnostic, "head_dim", head_dim);
    insert_json_detail(&mut diagnostic, "value_dim", value_dim);
    insert_json_detail(&mut diagnostic, "softmax_scale", softmax_scale);
    insert_json_detail(&mut diagnostic, "has_q_gate", q_gate.is_some());
    insert_json_detail(
        &mut diagnostic,
        "prepared_attention_vs_host_causal",
        package_hidden_distribution(
            prepared_attention_output,
            &host_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_attention_vs_host_causal",
        package_hidden_distribution(
            layer_attention_output,
            &host_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_attention_vs_prepared_attention",
        package_hidden_distribution(
            layer_attention_output,
            prepared_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_host_projection_input",
        package_hidden_distribution(
            layer_attention_projection_input,
            &host_projection_input,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_prepared_projection_input",
        package_hidden_distribution(
            layer_attention_projection_input,
            &prepared_projection_input,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_layer_attention_gate_replay",
        package_hidden_distribution(
            layer_attention_projection_input,
            &layer_projection_input_from_attention,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "sample_locations",
        package_self_attn_causal_attention_sample_locations(
            prepared_attention_output,
            layer_attention_output,
            &host_attention_output,
            &prepared_projection_input,
            layer_attention_projection_input,
            &host_projection_input,
            q_rope,
            k_rope,
            v_projected,
            q_gate,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            attention_width,
        ),
    );
    Ok(diagnostic)
}

fn package_attention_projection_input_from_gate(
    q_gate: Option<&[f32]>,
    attention_output: &[f32],
    expected_attention_elements: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    if attention_output.len() != expected_attention_elements {
        return Err(format!(
            "self-attn diagnostic {label} attention length mismatch: got {} expected {expected_attention_elements}",
            attention_output.len()
        ));
    }
    match q_gate {
        Some(gate) => {
            if gate.len() != expected_attention_elements {
                return Err(format!(
                    "self-attn diagnostic {label} gate length mismatch: got {} expected {expected_attention_elements}",
                    gate.len()
                ));
            }
            let gated = runtime_host_sigmoid_mul_f32(gate, attention_output);
            if gated.len() != expected_attention_elements {
                return Err(format!(
                    "self-attn diagnostic {label} gated output length mismatch: got {} expected {expected_attention_elements}",
                    gated.len()
                ));
            }
            Ok(gated)
        }
        None => Ok(attention_output.to_vec()),
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_sample_locations(
    prepared_attention_output: &[f32],
    layer_attention_output: &[f32],
    host_attention_output: &[f32],
    prepared_projection_input: &[f32],
    layer_projection_input: &[f32],
    host_projection_input: &[f32],
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    q_gate: Option<&[f32]>,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    attention_width: usize,
) -> Vec<serde_json::Value> {
    let sample_targets = [(8_usize, 503_usize)];
    sample_targets
        .into_iter()
        .filter_map(|(token_index, feature_index)| {
            if feature_index >= attention_width {
                return None;
            }
            let flat_index = token_index.checked_mul(attention_width)?.checked_add(feature_index)?;
            let prepared_attention = *prepared_attention_output.get(flat_index)?;
            let layer_attention = *layer_attention_output.get(flat_index)?;
            let host_attention = *host_attention_output.get(flat_index)?;
            let prepared_projection = *prepared_projection_input.get(flat_index)?;
            let layer_projection = *layer_projection_input.get(flat_index)?;
            let host_projection = *host_projection_input.get(flat_index)?;
            let q_gate_value = q_gate.and_then(|gate| gate.get(flat_index)).copied();
            let q_gate_sigmoid =
                q_gate_value.map(|gate| 1.0_f32 / (1.0_f32 + (-gate).exp()));
            let attention_breakdown = package_self_attn_causal_attention_breakdown(
                q_rope,
                k_rope,
                v_projected,
                token_index,
                feature_index,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            );
            Some(serde_json::json!({
                "token_index": token_index,
                "feature_index": feature_index,
                "flat_index": flat_index,
                "prepared_attention_output": prepared_attention,
                "layer_attention_output": layer_attention,
                "host_attention_output": host_attention,
                "layer_attention_minus_host_attention": layer_attention - host_attention,
                "prepared_attention_minus_host_attention": prepared_attention - host_attention,
                "layer_attention_minus_prepared_attention": layer_attention - prepared_attention,
                "q_gate": q_gate_value,
                "q_gate_sigmoid": q_gate_sigmoid,
                "prepared_projection_input": prepared_projection,
                "layer_projection_input": layer_projection,
                "host_projection_input": host_projection,
                "layer_projection_minus_host_projection": layer_projection - host_projection,
                "layer_projection_minus_prepared_projection": layer_projection - prepared_projection,
                "attention_breakdown": attention_breakdown,
            }))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_breakdown(
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    token_index: usize,
    feature_index: usize,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Option<serde_json::Value> {
    if sequence_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || token_index >= sequence_len
        || !q_heads.is_multiple_of(kv_heads)
    {
        return None;
    }
    let q_head = feature_index / value_dim;
    let value_offset = feature_index % value_dim;
    if q_head >= q_heads {
        return None;
    }
    let q_per_kv = q_heads / kv_heads;
    let kv_head = q_head / q_per_kv;
    if kv_head >= kv_heads {
        return None;
    }
    let expected_q_elements = sequence_len.checked_mul(q_heads)?.checked_mul(head_dim)?;
    let expected_k_elements = sequence_len.checked_mul(kv_heads)?.checked_mul(head_dim)?;
    let expected_v_elements = sequence_len.checked_mul(kv_heads)?.checked_mul(value_dim)?;
    if q_rope.len() != expected_q_elements
        || k_rope.len() != expected_k_elements
        || v_projected.len() != expected_v_elements
    {
        return None;
    }

    let q_base = token_index
        .checked_mul(q_heads)?
        .checked_add(q_head)?
        .checked_mul(head_dim)?;
    let mut scores = Vec::with_capacity(token_index + 1);
    let mut dots = Vec::with_capacity(token_index + 1);
    for source_token in 0..=token_index {
        let k_base = source_token
            .checked_mul(kv_heads)?
            .checked_add(kv_head)?
            .checked_mul(head_dim)?;
        let dot = (0..head_dim)
            .map(|dim| q_rope[q_base + dim] * k_rope[k_base + dim])
            .sum::<f32>();
        dots.push(dot);
        scores.push(dot * softmax_scale);
    }
    let max_score = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, score| max.max(score));
    let weights = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let denominator = weights.iter().sum::<f32>();
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }

    let mut computed_attention_output = 0.0_f32;
    let mut source_tokens = Vec::with_capacity(token_index + 1);
    for source_token in 0..=token_index {
        let weight = weights[source_token] / denominator;
        let v_index = source_token
            .checked_mul(kv_heads)?
            .checked_add(kv_head)?
            .checked_mul(value_dim)?
            .checked_add(value_offset)?;
        let v_value = v_projected[v_index];
        let contribution = weight * v_value;
        computed_attention_output += contribution;
        source_tokens.push(serde_json::json!({
            "source_token_index": source_token,
            "dot": dots[source_token],
            "score": scores[source_token],
            "softmax_weight": weight,
            "v_value": v_value,
            "weighted_v_contribution": contribution,
        }));
    }

    Some(serde_json::json!({
        "q_head": q_head,
        "kv_head": kv_head,
        "q_per_kv": q_per_kv,
        "value_offset": value_offset,
        "softmax_max_score": max_score,
        "softmax_denominator": denominator,
        "computed_attention_output": computed_attention_output,
        "source_tokens": source_tokens,
    }))
}

#[allow(clippy::too_many_arguments)]
fn package_module_contribution_summary(
    actual_before: &[f32],
    expected_before: &[f32],
    expected_after: &[f32],
    attention_projection_input: Option<&[f32]>,
    attention_output: &[f32],
    attention_block_output: &[f32],
    post_normed: &[f32],
    mlp_activation: Option<(&[f32], usize)>,
    extra_hot_input_vectors: &[(&str, &[f32], usize)],
    mlp_output: &[f32],
    actual_after: &[f32],
    sequence_len: usize,
    hidden: usize,
    sampled_token_indices: &[usize],
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "module contribution hidden element count overflows".to_string())?;
    for (label, values) in [
        ("actual_before", actual_before),
        ("expected_before", expected_before),
        ("expected_after", expected_after),
        ("attention_output", attention_output),
        ("attention_block_output", attention_block_output),
        ("post_normed", post_normed),
        ("mlp_output", mlp_output),
        ("actual_after", actual_after),
    ] {
        if values.len() != expected_elements {
            return Err(format!(
                "module contribution {label} length mismatch: got {} expected {expected_elements}",
                values.len()
            ));
        }
    }
    if let Some(values) = attention_projection_input {
        if values.len() != expected_elements {
            return Err(format!(
                "module contribution attention_projection_input length mismatch: got {} expected {expected_elements}",
                values.len()
            ));
        }
    }
    if let Some((values, feature_dim)) = mlp_activation {
        if feature_dim == 0 {
            return Err(
                "module contribution MLP activation feature dimension must be positive".to_string(),
            );
        }
        let expected_mlp_elements = sequence_len.checked_mul(feature_dim).ok_or_else(|| {
            "module contribution MLP activation element count overflows".to_string()
        })?;
        if values.len() != expected_mlp_elements {
            return Err(format!(
                "module contribution MLP activation length mismatch: got {} expected {expected_mlp_elements}",
                values.len()
            ));
        }
    }
    for (name, values, feature_dim) in extra_hot_input_vectors {
        if *feature_dim == 0 {
            return Err(format!(
                "module contribution {name} feature dimension must be positive"
            ));
        }
        let expected_extra_elements = sequence_len
            .checked_mul(*feature_dim)
            .ok_or_else(|| format!("module contribution {name} element count overflows"))?;
        if values.len() != expected_extra_elements {
            return Err(format!(
                "module contribution {name} length mismatch: got {} expected {expected_extra_elements}",
                values.len()
            ));
        }
    }
    for token_index in sampled_token_indices {
        if *token_index >= sequence_len {
            return Err(format!(
                "module contribution sampled token index {} is outside sequence_len={sequence_len}",
                token_index
            ));
        }
    }

    let actual_delta = actual_after
        .iter()
        .zip(actual_before.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    let expected_delta = expected_after
        .iter()
        .zip(expected_before.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    let delta_diff = actual_delta
        .iter()
        .zip(expected_delta.iter())
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let residual_identity_error = actual_delta
        .iter()
        .zip(attention_output.iter())
        .zip(mlp_output.iter())
        .map(|((delta, attention), mlp)| delta - attention - mlp)
        .collect::<Vec<_>>();

    let max_output_diff_index = actual_after
        .iter()
        .zip(expected_after.iter())
        .enumerate()
        .max_by(
            |(_, (actual_left, expected_left)), (_, (actual_right, expected_right))| {
                (*actual_left - *expected_left)
                    .abs()
                    .partial_cmp(&(*actual_right - *expected_right).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            },
        )
        .map(|(index, _)| index)
        .unwrap_or(0);
    let hot_hidden_index = if hidden == 0 {
        0
    } else {
        max_output_diff_index % hidden
    };
    let hot_token_index = if hidden == 0 {
        0
    } else {
        max_output_diff_index / hidden
    };

    let point = |flat_index: usize| {
        let output_diff = actual_after[flat_index] - expected_after[flat_index];
        let delta_diff = delta_diff[flat_index];
        serde_json::json!({
            "flat_index": flat_index,
            "token_index": flat_index / hidden,
            "hidden_index": flat_index % hidden,
            "actual_input": actual_before[flat_index],
            "expected_input": expected_before[flat_index],
            "input_diff": actual_before[flat_index] - expected_before[flat_index],
            "attention_output": attention_output[flat_index],
            "attention_block_output": attention_block_output[flat_index],
            "post_normed": post_normed[flat_index],
            "mlp_output": mlp_output[flat_index],
            "actual_delta": actual_delta[flat_index],
            "expected_delta": expected_delta[flat_index],
            "delta_diff": delta_diff,
            "residual_identity_error": residual_identity_error[flat_index],
            "actual_output": actual_after[flat_index],
            "expected_output": expected_after[flat_index],
            "output_diff": output_diff,
            "abs_output_diff": output_diff.abs(),
        })
    };
    let per_token_hot_hidden = (0..sequence_len)
        .map(|token_index| point(token_index * hidden + hot_hidden_index))
        .collect::<Vec<_>>();

    let mut summary = serde_json::Map::new();
    insert_json_detail(
        &mut summary,
        "delta_distribution",
        package_hidden_distribution(&actual_delta, &expected_delta, sequence_len, hidden)?,
    );
    insert_json_detail(
        &mut summary,
        "actual_delta_stats",
        package_slice_distribution_stats(&actual_delta),
    );
    insert_json_detail(
        &mut summary,
        "expected_delta_stats",
        package_slice_distribution_stats(&expected_delta),
    );
    insert_json_detail(
        &mut summary,
        "attention_output_stats",
        package_slice_distribution_stats(attention_output),
    );
    insert_json_detail(
        &mut summary,
        "mlp_output_stats",
        package_slice_distribution_stats(mlp_output),
    );
    insert_json_detail(
        &mut summary,
        "residual_identity_error_stats",
        package_slice_distribution_stats(&residual_identity_error),
    );
    insert_json_detail(&mut summary, "hot_hidden_index", hot_hidden_index);
    insert_json_detail(
        &mut summary,
        "hot_input_vectors",
        package_hot_input_vectors(
            hot_token_index,
            hidden,
            attention_projection_input,
            mlp_activation,
            extra_hot_input_vectors,
        )?,
    );
    if !sampled_token_indices.is_empty() {
        let mut sampled = Vec::new();
        let mut deduped = sampled_token_indices.to_vec();
        deduped.sort_unstable();
        deduped.dedup();
        for token_index in deduped {
            let mut item = package_hot_input_vectors(
                token_index,
                hidden,
                attention_projection_input,
                mlp_activation,
                extra_hot_input_vectors,
            )?;
            insert_json_detail(&mut item, "token_index", token_index);
            sampled.push(serde_json::Value::Object(item));
        }
        insert_json_detail(&mut summary, "sampled_hot_input_vectors", sampled);
    }
    insert_json_detail(
        &mut summary,
        "max_output_diff_trace",
        point(max_output_diff_index),
    );
    insert_json_detail(
        &mut summary,
        "per_token_hot_hidden_trace",
        per_token_hot_hidden,
    );
    Ok(summary)
}

fn package_hot_input_vectors(
    token_index: usize,
    hidden: usize,
    attention_projection_input: Option<&[f32]>,
    mlp_activation: Option<(&[f32], usize)>,
    extra_hot_input_vectors: &[(&str, &[f32], usize)],
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let mut vectors = serde_json::Map::new();
    let mut attention_hot_feature_indices = Vec::new();
    let mut mlp_hot_feature_indices = Vec::new();
    let hidden_group_width = if hidden % 128 == 0 { Some(128) } else { None };
    if let Some(values) = attention_projection_input {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "attention projection input token offset overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "attention projection input token end overflows".to_string())?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "attention projection input token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        attention_hot_feature_indices = package_top_abs_feature_indices(slice, 8);
        insert_json_detail(
            &mut vectors,
            "attention_projection_input",
            package_vector_summary(
                token_index,
                slice,
                &attention_hot_feature_indices,
                hidden_group_width,
            ),
        );
    }
    if let Some((values, feature_dim)) = mlp_activation {
        let start = token_index
            .checked_mul(feature_dim)
            .ok_or_else(|| "MLP activation token offset overflows".to_string())?;
        let end = start
            .checked_add(feature_dim)
            .ok_or_else(|| "MLP activation token end overflows".to_string())?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "MLP activation token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        mlp_hot_feature_indices = package_top_abs_feature_indices(slice, 8);
        insert_json_detail(
            &mut vectors,
            "mlp_activation",
            package_vector_summary(token_index, slice, &mlp_hot_feature_indices, None),
        );
    }
    for (name, values, feature_dim) in extra_hot_input_vectors {
        let start = token_index
            .checked_mul(*feature_dim)
            .ok_or_else(|| format!("{name} token offset overflows"))?;
        let end = start
            .checked_add(*feature_dim)
            .ok_or_else(|| format!("{name} token end overflows"))?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "{name} token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        let sampled_feature_indices_storage =
            if name.starts_with("mlp_") && !mlp_hot_feature_indices.is_empty() {
                mlp_hot_feature_indices
                    .iter()
                    .copied()
                    .filter(|feature_index| *feature_index < *feature_dim)
                    .collect::<Vec<_>>()
            } else {
                package_mapped_hot_feature_indices(
                    slice,
                    *feature_dim,
                    hidden,
                    &attention_hot_feature_indices,
                )
            };
        let sampled_feature_indices = sampled_feature_indices_storage.as_slice();
        insert_json_detail(
            &mut vectors,
            *name,
            package_vector_summary(
                token_index,
                slice,
                sampled_feature_indices,
                if *feature_dim % 128 == 0 {
                    Some(128)
                } else {
                    None
                },
            ),
        );
    }
    Ok(vectors)
}

fn package_mapped_hot_feature_indices(
    values: &[f32],
    feature_dim: usize,
    hidden: usize,
    attention_hot_feature_indices: &[usize],
) -> Vec<usize> {
    if feature_dim == hidden {
        return attention_hot_feature_indices.to_vec();
    }
    let head_width = 128_usize;
    if hidden % head_width == 0 {
        let value_heads = hidden / head_width;
        if feature_dim == value_heads {
            let mut indices = attention_hot_feature_indices
                .iter()
                .map(|feature_index| feature_index / head_width)
                .filter(|head_index| *head_index < feature_dim)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            indices.dedup();
            if !indices.is_empty() {
                return indices;
            }
        } else if feature_dim % head_width == 0 {
            let feature_heads = feature_dim / head_width;
            if feature_heads > 0 && feature_heads <= value_heads && value_heads % feature_heads == 0
            {
                let value_heads_per_feature_head = value_heads / feature_heads;
                let mut indices = attention_hot_feature_indices
                    .iter()
                    .map(|feature_index| {
                        let value_head = feature_index / head_width;
                        let head_offset = feature_index % head_width;
                        let feature_head = value_head / value_heads_per_feature_head;
                        feature_head * head_width + head_offset
                    })
                    .filter(|feature_index| *feature_index < feature_dim)
                    .collect::<Vec<_>>();
                indices.sort_unstable();
                indices.dedup();
                if !indices.is_empty() {
                    return indices;
                }
            }
        }
        if feature_dim > hidden {
            let v_base = feature_dim - hidden;
            let mut indices = attention_hot_feature_indices
                .iter()
                .map(|feature_index| v_base + feature_index)
                .filter(|feature_index| *feature_index < feature_dim)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            indices.dedup();
            if !indices.is_empty() {
                return indices;
            }
        }
    }
    package_top_abs_feature_indices(values, 8)
}

fn package_vector_summary(
    token_index: usize,
    values: &[f32],
    sampled_feature_indices: &[usize],
    sampled_group_width: Option<usize>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut summary = serde_json::Map::new();
    insert_json_detail(&mut summary, "token_index", token_index);
    insert_json_detail(&mut summary, "feature_count", values.len());
    insert_json_detail(
        &mut summary,
        "stats",
        package_slice_distribution_stats(values),
    );
    insert_json_detail(
        &mut summary,
        "top_abs_features",
        package_top_abs_value_locations(values, 8),
    );
    if !sampled_feature_indices.is_empty() {
        insert_json_detail(
            &mut summary,
            "sampled_features",
            package_sampled_value_locations(values, sampled_feature_indices, sampled_group_width),
        );
    }
    summary
}

fn package_top_abs_feature_indices(values: &[f32], limit: usize) -> Vec<usize> {
    let mut indexed = values.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| {
        right
            .abs()
            .partial_cmp(&left.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(limit)
        .map(|(feature_index, _)| feature_index)
        .collect()
}

fn package_top_abs_value_locations(values: &[f32], limit: usize) -> Vec<serde_json::Value> {
    package_top_abs_feature_indices(values, limit)
        .into_iter()
        .filter_map(|feature_index| {
            values
                .get(feature_index)
                .map(|value| (feature_index, value))
        })
        .map(|(feature_index, value)| {
            serde_json::json!({
                "feature_index": feature_index,
                "value": *value,
                "abs_value": value.abs(),
            })
        })
        .collect()
}

fn package_sampled_value_locations(
    values: &[f32],
    sampled_feature_indices: &[usize],
    sampled_group_width: Option<usize>,
) -> Vec<serde_json::Value> {
    let mut indices = sampled_feature_indices
        .iter()
        .copied()
        .filter(|index| *index < values.len())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    indices
        .into_iter()
        .map(|feature_index| {
            let value = values[feature_index];
            let mut location = serde_json::json!({
                "feature_index": feature_index,
                "value": value,
                "abs_value": value.abs(),
            });
            if let Some(group_width) = sampled_group_width {
                if group_width > 0 {
                    let group_index = feature_index / group_width;
                    let group_start = group_index * group_width;
                    let group_end = (group_start + group_width).min(values.len());
                    if group_start < group_end {
                        if let Some(object) = location.as_object_mut() {
                            insert_json_detail(object, "group_index", group_index);
                            insert_json_detail(object, "group_offset", feature_index - group_start);
                            insert_json_detail(object, "group_width", group_end - group_start);
                            insert_json_detail(
                                object,
                                "group_stats",
                                package_slice_distribution_stats(&values[group_start..group_end]),
                            );
                        }
                    }
                }
            }
            location
        })
        .collect()
}

fn package_hidden_distribution(
    actual: &[f32],
    expected: &[f32],
    sequence_len: usize,
    hidden: usize,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "hidden distribution element count overflows".to_string())?;
    if actual.len() != expected_elements || expected.len() != expected_elements {
        return Err(format!(
            "hidden distribution length mismatch: actual={} expected={} expected_elements={expected_elements}",
            actual.len(),
            expected.len()
        ));
    }

    let diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let max_abs_diff = diff
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, value)| {
            serde_json::json!({
                "flat_index": index,
                "token_index": index / hidden,
                "hidden_index": index % hidden,
                "actual": actual[index],
                "expected": expected[index],
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .unwrap_or_else(|| serde_json::json!(null));

    let per_token = (0..sequence_len)
        .map(|token_index| {
            let start = token_index * hidden;
            let end = start + hidden;
            let token_actual = &actual[start..end];
            let token_expected = &expected[start..end];
            let token_diff = &diff[start..end];
            let metrics = compare_f32_slices(token_actual, token_expected)?;
            Ok(serde_json::json!({
                "token_index": token_index,
                "mse": metrics.mse,
                "mean_abs_diff": metrics.mean_abs_diff,
                "max_abs_diff": metrics.max_abs_diff,
                "cosine_similarity": metrics.cosine_similarity,
                "actual_rms": package_slice_rms(token_actual),
                "expected_rms": package_slice_rms(token_expected),
                "diff_rms": package_slice_rms(token_diff),
                "diff_max_abs_location": package_slice_max_abs_location(
                    token_diff,
                    Some(token_actual),
                    Some(token_expected),
                    token_index,
                ),
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut distribution = serde_json::Map::new();
    insert_json_detail(
        &mut distribution,
        "actual_stats",
        package_slice_distribution_stats(actual),
    );
    insert_json_detail(
        &mut distribution,
        "expected_stats",
        package_slice_distribution_stats(expected),
    );
    insert_json_detail(
        &mut distribution,
        "diff_stats",
        package_slice_distribution_stats(&diff),
    );
    insert_json_detail(&mut distribution, "max_abs_diff_location", max_abs_diff);
    insert_json_detail(
        &mut distribution,
        "top_abs_diff_locations",
        package_top_abs_diff_locations(&diff, actual, expected, hidden, 8),
    );
    insert_json_detail(&mut distribution, "per_token", per_token);
    Ok(distribution)
}

fn package_slice_rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let square_sum = values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();
    (square_sum / values.len() as f64).sqrt()
}

fn package_slice_distribution_stats(values: &[f32]) -> serde_json::Map<String, serde_json::Value> {
    let mut finite_count = 0_usize;
    let mut nonfinite_count = 0_usize;
    let mut sum = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0_f64;
    let mut max_abs_index = 0_usize;

    for (index, value) in values.iter().enumerate() {
        let value = f64::from(*value);
        if !value.is_finite() {
            nonfinite_count += 1;
            continue;
        }
        finite_count += 1;
        sum += value;
        square_sum += value * value;
        abs_sum += value.abs();
        min = min.min(value);
        max = max.max(value);
        if value.abs() > max_abs {
            max_abs = value.abs();
            max_abs_index = index;
        }
    }

    let mean = if finite_count == 0 {
        0.0
    } else {
        sum / finite_count as f64
    };
    let mean_square = if finite_count == 0 {
        0.0
    } else {
        square_sum / finite_count as f64
    };
    let variance = (mean_square - mean * mean).max(0.0);
    let mut stats = serde_json::Map::new();
    insert_json_detail(&mut stats, "count", values.len());
    insert_json_detail(&mut stats, "finite_count", finite_count);
    insert_json_detail(&mut stats, "nonfinite_count", nonfinite_count);
    insert_json_detail(&mut stats, "mean", mean);
    insert_json_detail(
        &mut stats,
        "abs_mean",
        if finite_count == 0 {
            0.0
        } else {
            abs_sum / finite_count as f64
        },
    );
    insert_json_detail(&mut stats, "variance", variance);
    insert_json_detail(&mut stats, "stddev", variance.sqrt());
    insert_json_detail(&mut stats, "rms", mean_square.sqrt());
    insert_json_detail(&mut stats, "l2_norm", square_sum.sqrt());
    insert_json_detail(&mut stats, "min", if finite_count == 0 { 0.0 } else { min });
    insert_json_detail(&mut stats, "max", if finite_count == 0 { 0.0 } else { max });
    insert_json_detail(&mut stats, "max_abs", max_abs);
    insert_json_detail(&mut stats, "max_abs_index", max_abs_index);
    stats
}

fn package_slice_max_abs_location(
    diff: &[f32],
    actual: Option<&[f32]>,
    expected: Option<&[f32]>,
    token_index: usize,
) -> serde_json::Value {
    diff.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(hidden_index, value)| {
            serde_json::json!({
                "token_index": token_index,
                "hidden_index": hidden_index,
                "actual": actual.and_then(|values| values.get(hidden_index)).copied(),
                "expected": expected.and_then(|values| values.get(hidden_index)).copied(),
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .unwrap_or_else(|| serde_json::json!(null))
}

fn package_top_abs_diff_locations(
    diff: &[f32],
    actual: &[f32],
    expected: &[f32],
    hidden: usize,
    limit: usize,
) -> Vec<serde_json::Value> {
    let mut indexed = diff.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| {
        right
            .abs()
            .partial_cmp(&left.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(limit)
        .map(|(index, value)| {
            serde_json::json!({
                "flat_index": index,
                "token_index": index / hidden,
                "hidden_index": index % hidden,
                "actual": actual[index],
                "expected": expected[index],
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn append_package_golden_prefix_report_entry(
    report_entries: &mut Vec<serde_json::Value>,
    path: &str,
    fixture_path: &str,
    fixture_kind: Option<&str>,
    device_index: u32,
    backend: &str,
    device_name: &str,
    layer_position: usize,
    layer_index: usize,
    layer_kind: &str,
    layer_start: usize,
    layer_end_exclusive: usize,
    sequence_len: usize,
    hidden: usize,
    run_mode: PackageGoldenPrefixRunMode,
    input_metrics: &ullm_engine::golden::GoldenComparisonMetrics,
    input_failure_class: &str,
    input_expected_preview: Vec<f32>,
    input_actual_preview: Vec<f32>,
    input_diff_preview: Vec<f32>,
    metrics: &ullm_engine::golden::GoldenComparisonMetrics,
    failure_class: &str,
    expected_preview: Vec<f32>,
    actual_preview: Vec<f32>,
    diff_preview: Vec<f32>,
    details: serde_json::Map<String, serde_json::Value>,
) {
    let mut entry = serde_json::Map::new();
    insert_json_detail(&mut entry, "command", "package-golden-prefix-smoke");
    insert_json_detail(&mut entry, "package", path);
    insert_json_detail(&mut entry, "fixture", fixture_path);
    insert_json_detail(&mut entry, "fixture_kind", fixture_kind);
    insert_json_detail(&mut entry, "device_index", device_index);
    insert_json_detail(&mut entry, "backend", backend);
    insert_json_detail(&mut entry, "device_name", device_name);
    insert_json_detail(&mut entry, "layer_position", layer_position);
    insert_json_detail(&mut entry, "layer_index", layer_index);
    insert_json_detail(&mut entry, "layer_kind", layer_kind);
    insert_json_detail(&mut entry, "layer_start", layer_start);
    insert_json_detail(&mut entry, "layer_end_exclusive", layer_end_exclusive);
    insert_json_detail(&mut entry, "sequence_len", sequence_len);
    insert_json_detail(&mut entry, "hidden_size", hidden);
    insert_json_detail(&mut entry, "run_mode", run_mode.as_str());
    insert_json_detail(&mut entry, "input_mse", input_metrics.mse);
    insert_json_detail(
        &mut entry,
        "input_mean_abs_diff",
        input_metrics.mean_abs_diff,
    );
    insert_json_detail(&mut entry, "input_max_abs_diff", input_metrics.max_abs_diff);
    insert_json_detail(
        &mut entry,
        "input_cosine_similarity",
        input_metrics.cosine_similarity,
    );
    insert_json_detail(&mut entry, "input_failure_class", input_failure_class);
    insert_json_detail(&mut entry, "input_expected_preview", input_expected_preview);
    insert_json_detail(&mut entry, "input_actual_preview", input_actual_preview);
    insert_json_detail(&mut entry, "input_diff_preview", input_diff_preview);
    insert_json_detail(&mut entry, "mse", metrics.mse);
    insert_json_detail(&mut entry, "mean_abs_diff", metrics.mean_abs_diff);
    insert_json_detail(&mut entry, "max_abs_diff", metrics.max_abs_diff);
    insert_json_detail(&mut entry, "cosine_similarity", metrics.cosine_similarity);
    insert_json_detail(&mut entry, "failure_class", failure_class);
    insert_json_detail(&mut entry, "expected_preview", expected_preview);
    insert_json_detail(&mut entry, "actual_preview", actual_preview);
    insert_json_detail(&mut entry, "diff_preview", diff_preview);
    insert_json_detail(&mut entry, "verified", true);
    entry.extend(details);
    report_entries.push(serde_json::Value::Object(entry));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageGoldenPrefixRunMode {
    ActualPrefix,
    GoldenBeforeEachLayer,
}

impl PackageGoldenPrefixRunMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::ActualPrefix => "actual_prefix",
            Self::GoldenBeforeEachLayer => "golden_before_each_layer",
        }
    }
}

fn parse_package_golden_prefix_run_mode(
    run_mode: Option<&str>,
) -> Result<PackageGoldenPrefixRunMode, ExitCode> {
    match run_mode {
        Some(raw) => match raw {
            "actual_prefix" => Ok(PackageGoldenPrefixRunMode::ActualPrefix),
            "golden_before_each_layer" => Ok(PackageGoldenPrefixRunMode::GoldenBeforeEachLayer),
            _ => {
                eprintln!(
                    "invalid run_mode: {raw}; expected actual_prefix or golden_before_each_layer"
                );
                Err(ExitCode::from(2))
            }
        },
        None => Ok(PackageGoldenPrefixRunMode::ActualPrefix),
    }
}

fn package_golden_prefix_failure_class(
    metrics: &ullm_engine::golden::GoldenComparisonMetrics,
) -> &'static str {
    if !metrics.mse.is_finite()
        || !metrics.mean_abs_diff.is_finite()
        || !metrics.max_abs_diff.is_finite()
        || !metrics.cosine_similarity.is_finite()
    {
        "numeric_drift"
    } else if metrics.cosine_similarity < 0.5 || metrics.mse > 0.1 {
        "numeric_drift"
    } else if metrics.max_abs_diff > 0.0 {
        "possible_quantization_error"
    } else {
        "ok"
    }
}

fn write_jsonl_report(path: &str, entries: &[serde_json::Value]) -> Result<(), String> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    let mut file = File::create(path)
        .map_err(|err| format!("failed to create report {}: {err}", path.display()))?;
    for entry in entries {
        serde_json::to_writer(&mut file, entry)
            .map_err(|err| format!("failed to write report {}: {err}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|err| format!("failed to write report {}: {err}", path.display()))?;
    }
    Ok(())
}

fn write_package_prefix_input_dump(
    dump_dir: &str,
    layer_index: usize,
    run_mode: PackageGoldenPrefixRunMode,
    sequence_len: usize,
    hidden: usize,
    values: &[f32],
) -> Result<String, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "prefix input dump element count overflows".to_string())?;
    if values.len() != expected_elements {
        return Err(format!(
            "prefix input dump layer {layer_index} length mismatch: got {} expected {expected_elements}",
            values.len()
        ));
    }

    let dir = std::path::Path::new(dump_dir);
    fs::create_dir_all(dir).map_err(|err| {
        format!(
            "failed to create input dump directory {}: {err}",
            dir.display()
        )
    })?;
    let file_name = format!("layer-{layer_index:04}-input.f32");
    let path = dir.join(&file_name);
    let mut file = File::create(&path)
        .map_err(|err| format!("failed to create input dump {}: {err}", path.display()))?;
    for chunk in values.chunks(4096) {
        let mut bytes = Vec::with_capacity(chunk.len() * std::mem::size_of::<f32>());
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)
            .map_err(|err| format!("failed to write input dump {}: {err}", path.display()))?;
    }

    let metadata_name = format!("layer-{layer_index:04}-input.json");
    let metadata_path = dir.join(metadata_name);
    let metadata = serde_json::json!({
        "schema_version": "package-golden-prefix-input-dump-v0.1",
        "layer_index": layer_index,
        "run_mode": run_mode.as_str(),
        "dtype": "float32",
        "shape": [1, sequence_len, hidden],
        "file": file_name,
    });
    let mut metadata_file = File::create(&metadata_path).map_err(|err| {
        format!(
            "failed to create input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    serde_json::to_writer_pretty(&mut metadata_file, &metadata).map_err(|err| {
        format!(
            "failed to write input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    metadata_file.write_all(b"\n").map_err(|err| {
        format!(
            "failed to finish input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;

    Ok(path.to_string_lossy().into_owned())
}

#[allow(clippy::too_many_arguments)]
fn package_layer_golden_smoke_impl(
    path: &str,
    fixture_path: &str,
    fixture: GoldenTensorFixture,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    let golden_layer = fixture.select_layer(layer_index)?;
    let sequence_len = fixture.metadata().sequence_len;
    let hidden = fixture.metadata().hidden_size;
    if sequence_len == 0 || hidden == 0 {
        return Err(format!(
            "golden fixture has invalid sequence_len={sequence_len} hidden_size={hidden}"
        ));
    }
    validate_golden_hidden_shape(
        &golden_layer.before_shape,
        sequence_len,
        hidden,
        "golden before hidden",
    )?;
    validate_golden_hidden_shape(
        &golden_layer.after_shape,
        sequence_len,
        hidden,
        "golden after hidden",
    )?;
    validate_golden_position_ids(
        &fixture.metadata().position_ids,
        sequence_len,
        position_offset,
    )?;

    let before = fixture.read_layer_before_f32(layer_index)?;
    let after = fixture.read_layer_after_f32(layer_index)?;
    let expected_hidden_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "golden hidden element count overflows".to_string())?;
    if before.len() != expected_hidden_elements || after.len() != expected_hidden_elements {
        return Err(format!(
            "golden fixture payload element mismatch: before={} after={} expected={expected_hidden_elements}",
            before.len(),
            after.len()
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let layer = qwen3_package_decoder_layer_runtime_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        layer_index,
    )?;
    if layer.runtime_shape.hidden != hidden {
        return Err(format!(
            "golden hidden_size {hidden} does not match package layer hidden {}",
            layer.runtime_shape.hidden
        ));
    }
    let rotary_dim =
        parse_package_layer_golden_rotary_dim(layer.runtime_shape.head_dim, rotary_dim)?;
    let block_size = sequence_len;
    let cache_blocks = 1_usize;
    let block_table = vec![0_u32];
    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        &mut context,
        &mut stream,
        &layer.weights.self_attn,
        before,
        sequence_len,
        &layer.q_norm.values,
        &layer.k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        &block_table,
        block_size,
        cache_blocks,
    )?;
    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query: _,
                k_projected: _,
                q_normed: _,
                k_normed: _,
                q_rope,
                k_rope,
                v_projected,
                q_gate,
                attention_output: _,
                shape,
                softmax_scale,
                q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: _,
        paged_v_cache: _,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    } = prepared;
    let decode_shape = PagedDecodeShape {
        block_size: paged_block_size,
        cache_blocks: paged_cache_blocks,
        q_heads: shape.q_heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        value_dim: shape.value_dim,
    };
    let mlp_epsilon = 1e-5_f32;
    let layer_output = qwen3_decoder_layer_sequence_to_host_f32(
        &layer.weights,
        &mut context,
        &mut stream,
        decode_shape,
        &paged_block_table,
        softmax_scale,
        mlp_epsilon,
        &q_rope,
        &k_rope,
        &v_projected,
        q_gate.as_deref(),
        &residual_sequence,
        sequence_len,
    )?;
    let metrics = compare_f32_slices(&layer_output.layer_output, &after)?;
    let preview_len = 8.min(after.len()).min(layer_output.layer_output.len());
    let diff_preview = layer_output
        .layer_output
        .iter()
        .zip(after.iter())
        .take(preview_len)
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let candidate_ids = package_layer_candidate_ids(path, &layer);

    Ok(format!(
        "package-layer-golden-smoke package={} fixture={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" candidate_ids={:?} sequence_len={} hidden={} before_shape={:?} after_shape={:?} block_size={} cache_blocks={} block_table={:?} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} mlp_epsilon={:.9} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" mse={:.12} mean_abs_diff={:.9} max_abs_diff={:.9} cosine_similarity={:.9} expected_preview={} actual_preview={} diff_preview={} verified=true",
        path,
        fixture_path,
        layer_index,
        layer.q_tensor,
        layer.k_tensor,
        layer.v_tensor,
        layer.o_tensor,
        layer.gate_tensor,
        layer.up_tensor,
        layer.down_tensor,
        candidate_ids,
        sequence_len,
        hidden,
        &golden_layer.before_shape,
        &golden_layer.after_shape,
        paged_block_size,
        paged_cache_blocks,
        paged_block_table,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        softmax_scale,
        mlp_epsilon,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        layer.q_norm.dtype,
        layer.k_norm.dtype,
        layer.post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        metrics.mse,
        metrics.mean_abs_diff,
        metrics.max_abs_diff,
        metrics.cosine_similarity,
        format_f32_preview(&after[..preview_len]),
        format_f32_preview(&layer_output.layer_output[..preview_len]),
        format_f32_preview(&diff_preview),
    ))
}

fn parse_package_layer_golden_rotary_dim(
    head_dim: usize,
    rotary_dim: Option<String>,
) -> Result<usize, String> {
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        return Err(format!(
            "default rotary_dim is zero for head_dim={head_dim}"
        ));
    }
    let rotary_dim = match rotary_dim {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid rotary dim {raw:?}: {err}"))?,
        None => default_rotary_dim,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        ));
    }
    Ok(rotary_dim)
}

fn validate_golden_hidden_shape(
    shape: &[usize],
    sequence_len: usize,
    hidden: usize,
    label: &str,
) -> Result<(), String> {
    let mut elements = 1_usize;
    for dim in shape {
        if *dim == 0 {
            return Err(format!("{label} shape contains zero: {shape:?}"));
        }
        elements = elements
            .checked_mul(*dim)
            .ok_or_else(|| format!("{label} shape element count overflows: {shape:?}"))?;
    }
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| format!("{label} expected element count overflows"))?;
    if elements != expected_elements {
        return Err(format!(
            "{label} shape {shape:?} has {elements} elements, expected {expected_elements}"
        ));
    }
    match shape {
        [seq, width] if *seq == sequence_len && *width == hidden => Ok(()),
        [batch, seq, width] if *batch == 1 && *seq == sequence_len && *width == hidden => Ok(()),
        _ => Err(format!(
            "{label} shape {shape:?} must be [sequence_len, hidden] or [1, sequence_len, hidden] with sequence_len={sequence_len} hidden={hidden}"
        )),
    }
}

fn validate_golden_position_ids(
    position_ids: &[u64],
    sequence_len: usize,
    position_offset: usize,
) -> Result<(), String> {
    if position_ids.len() != sequence_len {
        return Err(format!(
            "golden position_ids length {} does not match sequence_len={sequence_len}",
            position_ids.len()
        ));
    }
    for (index, position_id) in position_ids.iter().enumerate() {
        let expected = position_offset
            .checked_add(index)
            .ok_or_else(|| "golden position id expectation overflows".to_string())?;
        let expected = u64::try_from(expected)
            .map_err(|_| "golden expected position id does not fit u64".to_string())?;
        if *position_id != expected {
            return Err(format!(
                "golden position_ids are not contiguous from position_offset={position_offset}: index={index} expected={expected} got={position_id}"
            ));
        }
    }
    Ok(())
}

fn package_layer_candidate_ids(
    path: &str,
    layer: &ullm_engine::qwen3_loader::Qwen3PackageDecoderLayerRuntime,
) -> Vec<String> {
    [
        &layer.q_tensor,
        &layer.k_tensor,
        &layer.v_tensor,
        &layer.o_tensor,
        &layer.gate_tensor,
        &layer.up_tensor,
        &layer.down_tensor,
    ]
    .iter()
    .map(|tensor_name| {
        select_tensor_payload_bundle(path, &TensorSelector::Name((*tensor_name).clone()))
            .ok()
            .and_then(|bundle| bundle.candidate_id)
            .unwrap_or_else(|| "unknown".to_string())
    })
    .collect()
}

fn package_self_attn_mlp_block_model_loop_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    second_layer_or_sequence_len: Option<String>,
    sequence_len_or_rotary_dim: Option<String>,
    rotary_dim_or_rope_base: Option<String>,
    rope_base_or_position_offset: Option<String>,
    position_offset_or_extra: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-mlp-block-model-loop-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let cli_tail = match parse_package_model_loop_cli_tail(
        layer_indices,
        second_layer_or_sequence_len,
        sequence_len_or_rotary_dim,
        rotary_dim_or_rope_base,
        rope_base_or_position_offset,
        position_offset_or_extra,
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(cli_tail.sequence_len, 3, "sequence length") {
        Ok(value) if value >= 3 => value,
        Ok(_) => {
            eprintln!("sequence length must be at least three for model-loop smoke");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(cli_tail.rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(cli_tail.position_offset, 3, "position offset")
    {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_self_attn_mlp_block_model_loop_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        cli_tail.layer_indices,
        sequence_len,
        cli_tail.rotary_dim,
        rope_base,
        position_offset,
    ) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

struct PackageModelLoopCliTail {
    layer_indices: Vec<usize>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
}

fn parse_package_model_loop_cli_tail(
    layer_indices: Option<String>,
    second_layer_or_sequence_len: Option<String>,
    sequence_len_or_rotary_dim: Option<String>,
    rotary_dim_or_rope_base: Option<String>,
    rope_base_or_position_offset: Option<String>,
    position_offset_or_extra: Option<String>,
) -> Result<PackageModelLoopCliTail, ExitCode> {
    let Some(first) = layer_indices else {
        return Ok(PackageModelLoopCliTail {
            layer_indices: vec![3, 7],
            sequence_len: None,
            rotary_dim: None,
            rope_base: None,
            position_offset: None,
        });
    };

    if first.contains(',') {
        if position_offset_or_extra.is_some() {
            eprintln!(
                "too many model-loop arguments for comma-separated layer list; expected LAYERS_CSV [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
            );
            return Err(ExitCode::from(2));
        }
        return Ok(PackageModelLoopCliTail {
            layer_indices: parse_usize_csv(&first, "layer list")?,
            sequence_len: second_layer_or_sequence_len,
            rotary_dim: sequence_len_or_rotary_dim,
            rope_base: rotary_dim_or_rope_base,
            position_offset: rope_base_or_position_offset,
        });
    }

    let first_layer_index = parse_usize_value(&first, "first layer index")?;
    if let Some(raw) = second_layer_or_sequence_len
        .as_deref()
        .filter(|raw| raw.contains(','))
    {
        let mut layer_indices = Vec::new();
        layer_indices.push(first_layer_index);
        layer_indices.extend(parse_usize_csv(raw, "second layer list")?);
        return Ok(PackageModelLoopCliTail {
            layer_indices,
            sequence_len: sequence_len_or_rotary_dim,
            rotary_dim: rotary_dim_or_rope_base,
            rope_base: rope_base_or_position_offset,
            position_offset: position_offset_or_extra,
        });
    }

    let second_layer_index = match second_layer_or_sequence_len {
        Some(raw) => parse_usize_value(&raw, "second layer index")?,
        None => 7,
    };
    Ok(PackageModelLoopCliTail {
        layer_indices: vec![first_layer_index, second_layer_index],
        sequence_len: sequence_len_or_rotary_dim,
        rotary_dim: rotary_dim_or_rope_base,
        rope_base: rope_base_or_position_offset,
        position_offset: position_offset_or_extra,
    })
}

fn parse_usize_csv(value: &str, label: &str) -> Result<Vec<usize>, ExitCode> {
    let mut parsed = Vec::new();
    for raw in value.split(',') {
        let entry = raw.trim();
        if entry.is_empty() {
            eprintln!("invalid {label}: empty entry in {value:?}");
            return Err(ExitCode::from(2));
        }
        parsed.push(parse_usize_value(entry, label)?);
    }
    if parsed.is_empty() {
        eprintln!("invalid {label}: expected at least one entry");
        return Err(ExitCode::from(2));
    }
    Ok(parsed)
}

fn parse_usize_value(value: &str, label: &str) -> Result<usize, ExitCode> {
    value.parse::<usize>().map_err(|err| {
        eprintln!("invalid {label}: {err}");
        ExitCode::from(2)
    })
}

#[cfg(test)]
mod package_model_loop_cli_tail_tests {
    use super::*;

    fn parse_tail(
        args: [Option<&str>; 6],
    ) -> Result<PackageModelLoopCliTail, std::process::ExitCode> {
        parse_package_model_loop_cli_tail(
            args[0].map(str::to_string),
            args[1].map(str::to_string),
            args[2].map(str::to_string),
            args[3].map(str::to_string),
            args[4].map(str::to_string),
            args[5].map(str::to_string),
        )
    }

    #[test]
    fn package_model_loop_cli_tail_defaults_to_two_layers() {
        let tail = parse_tail([None, None, None, None, None, None]).unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7]);
        assert_eq!(tail.sequence_len, None);
        assert_eq!(tail.rotary_dim, None);
        assert_eq!(tail.rope_base, None);
        assert_eq!(tail.position_offset, None);
    }

    #[test]
    fn package_model_loop_cli_tail_keeps_legacy_two_layer_layout() {
        let tail = parse_tail([
            Some("3"),
            Some("7"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_accepts_first_argument_layer_csv() {
        let tail = parse_tail([
            Some("3,7,11"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
            None,
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7, 11]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_accepts_second_argument_layer_csv() {
        let tail = parse_tail([
            Some("3"),
            Some("7,11"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7, 11]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_rejects_empty_layer_csv_entry() {
        assert!(parse_tail([Some("3,,7"), None, None, None, None, None]).is_err());
    }

    #[test]
    fn package_model_loop_self_attn_layers_from_manifest_are_sorted() {
        let root = std::env::temp_dir().join(format!(
            "ullm-model-loop-self-attn-layers-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("passthrough")).unwrap();
        for name in ["q3.raw", "k3.raw", "q7.raw", "k7.raw", "linear.raw"] {
            std::fs::write(root.join("passthrough").join(name), [0_u8; 4]).unwrap();
        }
        std::fs::write(
            root.join("manifest.json"),
            r#"{
              "schema_version": "test",
              "passthrough_tensors": [
                {
                  "name": "model.language_model.layers.7.self_attn.q_norm.weight",
                  "dtype": "F32",
                  "shape": [1],
                  "elements": 1,
                  "payload_bytes": 4,
                  "payload_file": "passthrough/q7.raw"
                },
                {
                  "name": "model.language_model.layers.3.self_attn.q_norm.weight",
                  "dtype": "F32",
                  "shape": [1],
                  "elements": 1,
                  "payload_bytes": 4,
                  "payload_file": "passthrough/q3.raw"
                },
                {
                  "name": "model.language_model.layers.7.self_attn.k_norm.weight",
                  "dtype": "F32",
                  "shape": [1],
                  "elements": 1,
                  "payload_bytes": 4,
                  "payload_file": "passthrough/k7.raw"
                },
                {
                  "name": "model.language_model.layers.3.self_attn.k_norm.weight",
                  "dtype": "F32",
                  "shape": [1],
                  "elements": 1,
                  "payload_bytes": 4,
                  "payload_file": "passthrough/k3.raw"
                },
                {
                  "name": "model.language_model.layers.0.linear_attn.norm.weight",
                  "dtype": "F32",
                  "shape": [1],
                  "elements": 1,
                  "payload_bytes": 4,
                  "payload_file": "passthrough/linear.raw"
                }
              ]
            }"#,
        )
        .unwrap();

        let layers = package_model_loop_self_attn_layer_indices(root.to_str().unwrap()).unwrap();
        assert_eq!(layers, vec![3, 7]);

        std::fs::remove_dir_all(root).unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_mlp_block_model_loop_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    sequence_len: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let mut smoke_run = PackageModelLoopSmokeRun::new(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &layer_indices,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
    )?;
    smoke_run.execute(&mut context, &mut stream)?;
    smoke_run.format_output(path, device_index, &info)
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_model_loop_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids_batch: Option<String>,
    generated_tokens_batch: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-token-ids-model-loop-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices = match parse_package_token_ids_model_loop_layer_indices(&path, layer_indices)
    {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids_batch = match parse_package_prompt_token_ids_batch(
        prompt_token_ids_batch.or_else(|| Some("len:3x3".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens_batch = match parse_package_generated_tokens_batch(
        generated_tokens_batch,
        prompt_token_ids_batch.len(),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_token_ids_model_loop_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
    ) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sq_fp8_token_ids_model_loop_smoke(
    path: Option<String>,
    artifact_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids_batch: Option<String>,
    generated_tokens_batch: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("sq-fp8-token-ids-model-loop-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(artifact_path) = artifact_path else {
        eprintln!("sq-fp8-token-ids-model-loop-smoke requires an SQ FP8 artifact path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices = match parse_package_token_ids_model_loop_layer_indices(&path, layer_indices)
    {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids_batch = match parse_package_prompt_token_ids_batch(
        prompt_token_ids_batch.or_else(|| Some("len:3x3".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens_batch = match parse_package_generated_tokens_batch(
        generated_tokens_batch,
        prompt_token_ids_batch.len(),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let artifact = match read_sq_fp8_artifact(&artifact_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read SQ FP8 artifact: {err}");
            return ExitCode::from(1);
        }
    };

    match package_token_ids_model_loop_smoke_impl_with_sq_overlay(
        "sq-fp8-token-ids-model-loop-smoke",
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        Some(&artifact),
    ) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn parse_package_token_ids_model_loop_layer_indices(
    path: &str,
    value: Option<String>,
) -> Result<Vec<usize>, ExitCode> {
    let Some(raw) = value else {
        return Ok(vec![3, 7]);
    };
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("default") {
        return Ok(vec![3, 7]);
    }
    if matches!(
        raw.to_ascii_lowercase().as_str(),
        "self-attn" | "self_attn" | "all-self-attn" | "all_self_attn" | "manifest-self-attn"
    ) {
        return package_model_loop_self_attn_layer_indices(path).map_err(|err| {
            eprintln!("{err}");
            ExitCode::from(2)
        });
    }
    if raw.eq_ignore_ascii_case("all") {
        eprintln!(
            "package-token-ids-model-loop-smoke cannot infer full mixed-attention layer order from all; use all-self-attn for manifest self-attention layers or a CSV such as 3,7"
        );
        return Err(ExitCode::from(2));
    }
    parse_usize_csv(raw, "model-loop token-id layer list")
}

fn package_model_loop_self_attn_layer_indices(path: &str) -> Result<Vec<usize>, String> {
    let bundles = list_passthrough_payload_bundles(path)?;
    let mut q_norm_layers = std::collections::BTreeSet::new();
    let mut k_norm_layers = std::collections::BTreeSet::new();
    for bundle in bundles {
        if let Some(layer_index) = parse_language_model_layer_tensor_suffix(
            &bundle.tensor_name,
            ".self_attn.q_norm.weight",
        ) {
            q_norm_layers.insert(layer_index);
        }
        if let Some(layer_index) = parse_language_model_layer_tensor_suffix(
            &bundle.tensor_name,
            ".self_attn.k_norm.weight",
        ) {
            k_norm_layers.insert(layer_index);
        }
    }
    if q_norm_layers.is_empty() {
        return Err(format!(
            "package {path} has no manifest self-attention q_norm layers"
        ));
    }
    if q_norm_layers != k_norm_layers {
        return Err(format!(
            "package {path} has mismatched self-attention q_norm/k_norm layer sets: q_norm={:?} k_norm={:?}",
            q_norm_layers, k_norm_layers
        ));
    }
    Ok(q_norm_layers.into_iter().collect())
}

fn parse_language_model_layer_tensor_suffix(tensor_name: &str, suffix: &str) -> Option<usize> {
    let layer = tensor_name
        .strip_prefix("model.language_model.layers.")?
        .strip_suffix(suffix)?;
    layer.parse::<usize>().ok()
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_model_loop_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids_batch: Vec<Vec<usize>>,
    generated_tokens_batch: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    package_token_ids_model_loop_smoke_impl_with_sq_overlay(
        "package-token-ids-model-loop-smoke",
        path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_model_loop_smoke_impl_with_sq_overlay(
    command_name: &'static str,
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids_batch: Vec<Vec<usize>>,
    generated_tokens_batch: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let mut smoke_run = PackageModelLoopSmokeRun::new_from_token_ids(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        command_name,
        sq_artifact,
    )?;
    smoke_run.execute(&mut context, &mut stream)?;
    smoke_run.compute_final_top_logits(
        path,
        &mut context,
        &mut stream,
        chunk_bytes,
        top_k,
        lm_head_chunk_rows,
    )?;
    smoke_run.format_output(path, device_index, &info)
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_mixed_request_state_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids_batch: Option<String>,
    generated_tokens_batch: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-token-ids-mixed-request-state-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices = match parse_package_token_ids_layer_indices_for_package(
        &path,
        layer_indices.or_else(|| Some("manifest-all".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids_batch = match parse_package_prompt_token_ids_batch(
        prompt_token_ids_batch.or_else(|| Some("len:2x2".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens_batch = match parse_package_generated_tokens_batch(
        generated_tokens_batch,
        prompt_token_ids_batch.len(),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 1, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_token_ids_mixed_request_state_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
    ) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sq_fp8_token_ids_mixed_request_state_smoke(
    path: Option<String>,
    artifact_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids_batch: Option<String>,
    generated_tokens_batch: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("sq-fp8-token-ids-mixed-request-state-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(artifact_path) = artifact_path else {
        eprintln!("sq-fp8-token-ids-mixed-request-state-smoke requires an SQ FP8 artifact path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices = match parse_package_token_ids_layer_indices_for_package(
        &path,
        layer_indices.or_else(|| Some("manifest-all".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids_batch = match parse_package_prompt_token_ids_batch(
        prompt_token_ids_batch.or_else(|| Some("len:2x2".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens_batch = match parse_package_generated_tokens_batch(
        generated_tokens_batch,
        prompt_token_ids_batch.len(),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 1, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let artifact = match read_sq_fp8_artifact(&artifact_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read SQ FP8 artifact: {err}");
            return ExitCode::from(1);
        }
    };

    match package_token_ids_mixed_request_state_smoke_impl_with_sq_overlay(
        "sq-fp8-token-ids-mixed-request-state-smoke",
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        Some(&artifact),
    ) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_mixed_request_state_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids_batch: Vec<Vec<usize>>,
    generated_tokens_batch: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    package_token_ids_mixed_request_state_smoke_impl_with_sq_overlay(
        "package-token-ids-mixed-request-state-smoke",
        path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_mixed_request_state_smoke_impl_with_sq_overlay(
    command_name: &'static str,
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids_batch: Vec<Vec<usize>>,
    generated_tokens_batch: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    if layer_indices.is_empty() {
        return Err("mixed request-state smoke requires at least one layer".to_string());
    }
    if prompt_token_ids_batch.is_empty() {
        return Err("mixed request-state smoke requires at least one request".to_string());
    }
    if prompt_token_ids_batch.len() != generated_tokens_batch.len() {
        return Err(format!(
            "mixed request-state prompt request count {} does not match generated token count {}",
            prompt_token_ids_batch.len(),
            generated_tokens_batch.len()
        ));
    }

    let run_started = Instant::now();
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create mixed request-state context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query mixed request-state device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create mixed request-state stream: {err}"))?;
    let sq_row_chunk = 256_usize;
    let sq_overlay = sq_artifact.map(|artifact| Qwen3PackageSqOverlay {
        artifact,
        row_chunk: sq_row_chunk,
    });
    let sq_overlay_info =
        sq_artifact.map(|artifact| package_model_loop_sq_overlay_info(artifact, sq_row_chunk));

    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if hidden == 0 {
        return Err("mixed request-state embedding hidden size is zero".to_string());
    }
    let max_total_tokens = prompt_token_ids_batch
        .iter()
        .zip(generated_tokens_batch.iter())
        .map(|(prompt, generated)| prompt.len().saturating_add(*generated))
        .max()
        .unwrap_or(1)
        .max(1);
    let block_size = 256_usize.min(max_total_tokens);
    let request_plan = PackageModelLoopRequestPlan::from_token_id_batches(
        path,
        prompt_token_ids_batch,
        generated_tokens_batch,
        hidden,
        block_size,
    )?;
    let request_ids = request_plan
        .requests
        .iter()
        .map(|request| request.id)
        .collect::<Vec<_>>();

    let layer_load_started = Instant::now();
    let mut layers = Vec::with_capacity(layer_indices.len());
    let mut layer_kinds = Vec::with_capacity(layer_indices.len());
    let mut self_attn_shapes = Vec::new();
    for &layer_index in &layer_indices {
        let layer_kind = package_decoder_layer_kind(path, layer_index).map_err(|err| {
            format!("failed to identify mixed request-state layer {layer_index}: {err}")
        })?;
        layer_kinds.push(layer_kind.as_str());
        match layer_kind {
            PackageDecoderLayerKind::LinearAttention => {
                let layer = PackageLinearAttnResidentStepBatchLayer::load(
                    &mut context,
                    &mut stream,
                    path,
                    chunk_bytes,
                    layer_index,
                    request_ids.clone(),
                    sq_overlay.as_ref(),
                )
                .map_err(|err| {
                    format!(
                        "failed to load mixed request-state linear-attn layer {layer_index}: {err}"
                    )
                })?;
                if layer.hidden() != hidden {
                    return Err(format!(
                        "mixed request-state linear-attn layer {layer_index} hidden mismatch: layer_hidden={} embedding_hidden={hidden}",
                        layer.hidden()
                    ));
                }
                layers.push(PackageMixedRequestStateLayer::LinearAttention(layer));
            }
            PackageDecoderLayerKind::SelfAttention => {
                let layer = PackageSelfAttnResidentStepBatchLayer::load(
                    &mut context,
                    &mut stream,
                    path,
                    chunk_bytes,
                    layer_index,
                    request_ids.clone(),
                    request_plan.block_size,
                    request_plan.cache_blocks,
                    sq_overlay.as_ref(),
                )
                .map_err(|err| {
                    format!(
                        "failed to load mixed request-state self-attn layer {layer_index}: {err}"
                    )
                })?;
                if layer.hidden() != hidden {
                    return Err(format!(
                        "mixed request-state self-attn layer {layer_index} hidden mismatch: layer_hidden={} embedding_hidden={hidden}",
                        layer.hidden()
                    ));
                }
                self_attn_shapes.push(serde_json::json!({
                    "layer_index": layer.layer_index(),
                    "q_heads": layer.q_heads(),
                    "kv_heads": layer.kv_heads(),
                    "head_dim": layer.head_dim(),
                    "value_dim": layer.value_dim(),
                    "block_size": layer.block_size(),
                    "cache_blocks": layer.cache_blocks(),
                }));
                layers.push(PackageMixedRequestStateLayer::SelfAttention(layer));
            }
        }
    }
    let layer_load_ms = layer_load_started.elapsed().as_secs_f64() * 1000.0;
    reset_sq_fp8_projection_telemetry();

    let rotary_dim_value = if let Some(head_dim) = layers
        .iter()
        .find_map(PackageMixedRequestStateLayer::self_attn_head_dim)
    {
        parse_package_token_ids_rotary_dim(head_dim, rotary_dim.as_deref())?
    } else {
        0
    };

    let prefill_started = Instant::now();
    let mut prefill_batch_request_counts = Vec::new();
    for timestep in 0..request_plan
        .prompt_tokens
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
    {
        let mut batch_items = Vec::new();
        for (request_index, request) in request_plan.requests.iter().enumerate() {
            if timestep >= request.prompt_tokens {
                continue;
            }
            let request_position_base = position_offset
                .checked_add(
                    request_index
                        .checked_mul(request_plan.position_stride)
                        .ok_or_else(|| {
                            "mixed request-state request position stride overflows".to_string()
                        })?,
                )
                .ok_or_else(|| "mixed request-state request position base overflows".to_string())?;
            let rope_position = request_position_base
                .checked_add(timestep)
                .ok_or_else(|| "mixed request-state prefill position overflows".to_string())?;
            let residual =
                mixed_request_state_residual_slice(&request_plan, request_index, timestep, hidden)?
                    .to_vec();
            batch_items.push(MixedRequestStateBatchStepItem {
                request_id: request.id,
                residual,
                rope_position,
                cache_position: timestep,
            });
        }
        let count = package_mixed_request_state_layers_batch_step(
            &mut stream,
            &mut layers,
            &batch_items,
            hidden,
            rotary_dim_value,
            rope_base,
            &format!("mixed request-state prefill batch timestep={timestep}"),
        )?;
        if count > 0 {
            prefill_batch_request_counts.push(count);
        }
    }
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize mixed request-state prefill: {err}"))?;
    let prefill_wall_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;

    let decode_started = Instant::now();
    let mut decode_batch_request_counts = Vec::new();
    for decode_index in 0..request_plan
        .max_new_tokens
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
    {
        let mut batch_items = Vec::new();
        for (request_index, request) in request_plan.requests.iter().enumerate() {
            if decode_index >= request.max_new_tokens {
                continue;
            }
            let token_index = request
                .prompt_tokens
                .checked_add(decode_index)
                .ok_or_else(|| "mixed request-state decode token index overflows".to_string())?;
            let request_position_base = position_offset
                .checked_add(
                    request_index
                        .checked_mul(request_plan.position_stride)
                        .ok_or_else(|| {
                            "mixed request-state request position stride overflows".to_string()
                        })?,
                )
                .ok_or_else(|| "mixed request-state request position base overflows".to_string())?;
            let rope_position = request_position_base
                .checked_add(token_index)
                .ok_or_else(|| "mixed request-state decode position overflows".to_string())?;
            let residual = mixed_request_state_residual_slice(
                &request_plan,
                request_index,
                token_index,
                hidden,
            )?
            .to_vec();
            batch_items.push(MixedRequestStateBatchStepItem {
                request_id: request.id,
                residual,
                rope_position,
                cache_position: token_index,
            });
        }
        let count = package_mixed_request_state_layers_batch_step(
            &mut stream,
            &mut layers,
            &batch_items,
            hidden,
            rotary_dim_value,
            rope_base,
            &format!("mixed request-state decode batch decode_index={decode_index}"),
        )?;
        if count > 0 {
            decode_batch_request_counts.push(count);
        }
    }
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize mixed request-state decode: {err}"))?;
    let decode_wall_ms = decode_started.elapsed().as_secs_f64() * 1000.0;

    let final_norm = read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, chunk_bytes)
        .map_err(|err| format!("failed to read mixed request-state final RMSNorm tensor: {err}"))?;
    let final_norm_values =
        effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
    if final_norm_values.len() != hidden {
        return Err(format!(
            "mixed request-state final RMSNorm length mismatch: len={} hidden={hidden}",
            final_norm_values.len()
        ));
    }

    let mut resident_lm_head_runtime = None;
    let mut final_top1_tokens = Vec::with_capacity(request_plan.request_count());
    let mut final_top1_logits = Vec::with_capacity(request_plan.request_count());
    let mut final_topk_tokens = Vec::with_capacity(request_plan.request_count());
    let mut final_topk_logits = Vec::with_capacity(request_plan.request_count());
    let final_logits_started = Instant::now();
    for request in &request_plan.requests {
        let final_layer = layers
            .last()
            .ok_or_else(|| "mixed request-state has no final layer".to_string())?;
        let final_hidden = final_layer.read_output(&mut stream, request.id)?;
        if final_hidden.len() != hidden {
            return Err(format!(
                "mixed request-state final hidden length mismatch for request {}: got {} expected {hidden}",
                request.id.0,
                final_hidden.len()
            ));
        }
        let final_normed = runtime_host_rmsnorm_f32(&final_hidden, &final_norm_values, 1e-6_f32);
        if final_normed.len() != hidden || final_normed.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "mixed request-state final normalized hidden for request {} contains invalid values",
                request.id.0
            ));
        }
        let top_logits = match package_lm_head_top_k_from_rows(
            path,
            &final_normed,
            top_k,
            lm_head_chunk_rows,
        ) {
            Ok((_, _, _, top_logits)) => top_logits,
            Err(cpu_err) => {
                if resident_lm_head_runtime.is_none() {
                    resident_lm_head_runtime = Some(
                            PackageLmHeadRuntime::load(
                                PackageLmHeadMode::GpuResidentF32,
                                &mut context,
                                &mut stream,
                                path,
                                chunk_bytes,
                                hidden,
                                lm_head_chunk_rows,
                            )
                            .map_err(|resident_err| {
                                format!(
                                    "failed to load mixed request-state resident lm_head: cpu_chunked_error={cpu_err}; resident_error={resident_err}"
                                )
                            })?,
                        );
                }
                resident_lm_head_runtime
                    .as_mut()
                    .ok_or_else(|| "mixed request-state resident lm_head disappeared".to_string())?
                    .top_logits(path, &mut stream, &final_normed, top_k)?
            }
        };
        let top1 = top_logits.first().ok_or_else(|| {
            format!(
                "mixed request-state request {} produced no logits",
                request.id.0
            )
        })?;
        final_top1_tokens.push(top1.token_id);
        final_top1_logits.push(format!("{:.9}", top1.logit));
        final_topk_tokens.push(
            top_logits
                .iter()
                .map(|entry| entry.token_id.to_string())
                .collect::<Vec<_>>()
                .join(":"),
        );
        final_topk_logits.push(
            top_logits
                .iter()
                .map(|entry| format!("{:.9}", entry.logit))
                .collect::<Vec<_>>()
                .join(":"),
        );
    }
    let final_logits_wall_ms = final_logits_started.elapsed().as_secs_f64() * 1000.0;
    let sq_fp8_projection_telemetry = snapshot_sq_fp8_projection_telemetry();
    let sq_projection_boundary = sq_fp8_projection_boundary(sq_fp8_projection_telemetry);
    let total_wall_ms = prefill_wall_ms + decode_wall_ms + final_logits_wall_ms;
    let outer_wall_ms = run_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_total_input_tokens = request_plan.prompt_tokens.iter().sum::<usize>();
    let decode_total_generated_tokens = request_plan.max_new_tokens.iter().sum::<usize>();
    let end_to_end_total_tokens = prefill_total_input_tokens
        .checked_add(decode_total_generated_tokens)
        .ok_or_else(|| "mixed request-state total token count overflows".to_string())?;
    let self_attn_weight_bundle_shared = request_plan.request_count() > 1
        && layer_kinds.iter().any(|kind| *kind == "self_attention");
    let linear_attn_weight_bundle_shared = request_plan.request_count() > 1
        && layer_kinds.iter().any(|kind| *kind == "linear_attention");
    let prefill_executor_request_parallelism = prefill_batch_request_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let decode_executor_request_parallelism = decode_batch_request_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let prefill_real_batch = prefill_executor_request_parallelism > 1;
    let decode_real_batch = decode_executor_request_parallelism > 1;
    let batching_mode = if prefill_real_batch && decode_real_batch {
        "real"
    } else if prefill_real_batch || decode_real_batch {
        "hybrid"
    } else {
        "single"
    };
    let sq_overlay_enabled = sq_overlay_info.is_some();
    let sq_candidate = sq_overlay_info
        .as_ref()
        .map(|info| info.candidate.as_str())
        .unwrap_or("none");
    let sq_artifact_path = sq_overlay_info
        .as_ref()
        .map(|info| info.artifact.as_str())
        .unwrap_or("none");
    let sq_schema_version = sq_overlay_info
        .as_ref()
        .map(|info| info.schema_version.as_str())
        .unwrap_or("none");
    let sq_fp8_tensor_count = sq_overlay_info
        .as_ref()
        .map(|info| info.fp8_tensor_count)
        .unwrap_or(0);
    let sq_passthrough_tensor_count = sq_overlay_info
        .as_ref()
        .map(|info| info.passthrough_tensor_count)
        .unwrap_or(0);
    let sq_row_chunk_value = sq_overlay_info
        .as_ref()
        .map(|info| info.row_chunk)
        .unwrap_or(0);
    let sq_execution_mode = if sq_overlay_enabled {
        "direct_fp8_dequant_matvec"
    } else {
        "none"
    };

    Ok(format!(
        "{} package={} layers={:?} layers_csv={} layer_kinds={:?} input_source={} prefill_mode=token_id_full_mixed_request_state full_mixed_request_state=true request_state_dispatch=true request_batch_executor=true fused_request_batch=false throughput_row=true load_excluded_from_total=true final_logits_in_total=true sq_overlay={} sq_candidate={} sq_artifact={} sq_schema_version={} sq_fp8_tensor_count={} sq_passthrough_tensor_count={} sq_row_chunk={} sq_execution_mode={} sq_projection_boundary={} sq_fp8_single_matvec_count={} sq_fp8_batch_matvec_count={} sq_fp8_pair_matvec_count={} sq_fp8_triple_matvec_count={} batching_mode={} prefill_executor=mixed_request_state_layer_batch_step decode_executor=mixed_request_state_layer_batch_step prefill_real_batch={} decode_real_batch={} prefill_executor_request_parallelism={} decode_executor_request_parallelism={} prompt_token_ids_by_request={:?} decode_token_ids_by_request={:?} final_lm_head_guard=true lm_head_top_k={} lm_head_chunk_rows={} final_top1_tokens={:?} final_top1_tokens_csv={} final_top1_logits_csv={} final_topk_tokens_csv={} final_topk_logits_csv={} sequence_len={} request_count={} concurrent_requests={} request_ids={:?} prompt_tokens={:?} prompt_tokens_csv={} max_new_tokens={:?} max_new_tokens_csv={} total_tokens={:?} total_tokens_csv={} prefill_total_input_tokens={} decode_total_generated_tokens={} end_to_end_total_tokens={} prefill_wall_ms={:.6} decode_wall_ms={:.6} final_logits_wall_ms={:.6} layer_load_ms={:.6} total_wall_ms={:.6} outer_wall_ms={:.6} prefill_total_input_tps={} decode_total_generated_tps={} end_to_end_total_tps={} paged_block_size={} paged_cache_blocks={} per_request_cache_buffers=true slot_aq4_payload_registry_shared=true slot_aq4_scale_values_shared=true slot_passthrough_weight_buffers_shared=true self_attn_weight_bundle_shared={} linear_attn_weight_bundle_shared={} shared_paged_cache=false block_tables={:?} prefill_batch_request_counts={:?} prefill_batch_request_counts_csv={} decode_batch_request_counts={:?} decode_batch_request_counts_csv={} hidden={} embedding_vocab={} self_attn_shapes={} rotary_dim={} position_offset={} rope_base={} backend={} device_index={} name=\"{}\" verified=true",
        command_name,
        path,
        layer_indices,
        usize_csv(&layer_indices),
        layer_kinds,
        request_plan.input_source,
        sq_overlay_enabled,
        sq_candidate,
        sq_artifact_path,
        sq_schema_version,
        sq_fp8_tensor_count,
        sq_passthrough_tensor_count,
        sq_row_chunk_value,
        sq_execution_mode,
        sq_projection_boundary,
        sq_fp8_projection_telemetry.single_matvec_count,
        sq_fp8_projection_telemetry.batch_matvec_count,
        sq_fp8_projection_telemetry.pair_matvec_count,
        sq_fp8_projection_telemetry.triple_matvec_count,
        batching_mode,
        prefill_real_batch,
        decode_real_batch,
        prefill_executor_request_parallelism,
        decode_executor_request_parallelism,
        request_plan.prompt_token_ids_by_request,
        request_plan.decode_token_ids_by_request,
        top_k,
        lm_head_chunk_rows,
        final_top1_tokens,
        usize_csv(&final_top1_tokens),
        final_top1_logits.join(","),
        final_topk_tokens.join(";"),
        final_topk_logits.join(";"),
        request_plan.position_stride,
        request_plan.request_count(),
        request_plan.request_count(),
        request_plan.request_ids,
        request_plan.prompt_tokens,
        usize_csv(&request_plan.prompt_tokens),
        request_plan.max_new_tokens,
        usize_csv(&request_plan.max_new_tokens),
        request_plan.total_tokens,
        usize_csv(&request_plan.total_tokens),
        prefill_total_input_tokens,
        decode_total_generated_tokens,
        end_to_end_total_tokens,
        prefill_wall_ms,
        decode_wall_ms,
        final_logits_wall_ms,
        layer_load_ms,
        total_wall_ms,
        outer_wall_ms,
        optional_f64_string(tps(prefill_total_input_tokens, prefill_wall_ms)),
        optional_f64_string(tps(decode_total_generated_tokens, decode_wall_ms)),
        optional_f64_string(tps(end_to_end_total_tokens, total_wall_ms)),
        request_plan.block_size,
        request_plan.cache_blocks,
        self_attn_weight_bundle_shared,
        linear_attn_weight_bundle_shared,
        request_plan.block_tables,
        prefill_batch_request_counts,
        usize_csv(&prefill_batch_request_counts),
        decode_batch_request_counts,
        usize_csv(&decode_batch_request_counts),
        hidden,
        embedding_vocab,
        serde_json::Value::Array(self_attn_shapes),
        rotary_dim_value,
        position_offset,
        rope_base,
        info.backend,
        device_index,
        info.name,
    ))
}

struct MixedRequestStateBatchStepItem {
    request_id: RequestId,
    residual: Vec<f32>,
    rope_position: usize,
    cache_position: usize,
}

fn usize_csv(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn optional_f64_string(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn mixed_request_state_residual_slice<'a>(
    request_plan: &'a PackageModelLoopRequestPlan,
    request_index: usize,
    token_index: usize,
    hidden: usize,
) -> Result<&'a [f32], String> {
    let request = request_plan.requests.get(request_index).ok_or_else(|| {
        format!("mixed request-state request index {request_index} is out of range")
    })?;
    let total_tokens = request_plan
        .total_tokens
        .get(request_index)
        .copied()
        .ok_or_else(|| {
            format!("mixed request-state total token count missing for request {request_index}")
        })?;
    if token_index >= total_tokens {
        return Err(format!(
            "mixed request-state token index {token_index} exceeds request {} total tokens {total_tokens}",
            request.id.0
        ));
    }
    let residual = request_plan
        .initial_residuals
        .get(request_index)
        .ok_or_else(|| {
            format!("mixed request-state residuals missing for request {request_index}")
        })?;
    let start = token_index
        .checked_mul(hidden)
        .ok_or_else(|| "mixed request-state residual start offset overflows".to_string())?;
    let end = start
        .checked_add(hidden)
        .ok_or_else(|| "mixed request-state residual end offset overflows".to_string())?;
    residual.get(start..end).ok_or_else(|| {
        format!(
            "mixed request-state residual slice out of range for request {} token {token_index}: range={start}..{end} len={}",
            request.id.0,
            residual.len()
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn package_mixed_request_state_layers_batch_step(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    layers: &mut [PackageMixedRequestStateLayer],
    items: &[MixedRequestStateBatchStepItem],
    hidden: usize,
    rotary_dim: usize,
    rope_base: f32,
    label: &str,
) -> Result<usize, String> {
    if items.is_empty() {
        return Ok(0);
    }
    if layers.is_empty() {
        return Err(format!(
            "{label} mixed request-state layer batch requires at least one layer"
        ));
    }
    for item in items {
        if item.residual.len() != hidden {
            return Err(format!(
                "{label} request {:?} residual length {} does not match hidden {hidden}",
                item.request_id,
                item.residual.len()
            ));
        }
    }

    let mut residual_device_layer: Option<usize> = None;
    for layer_position in 0..layers.len() {
        if let Some(previous_position) = residual_device_layer {
            let (previous_layers, current_layers) = layers.split_at_mut(layer_position);
            let previous = previous_layers.get(previous_position).ok_or_else(|| {
                format!("{label} previous device residual layer {previous_position} is missing")
            })?;
            let current = current_layers
                .get_mut(0)
                .ok_or_else(|| format!("{label} current layer {layer_position} is missing"))?;
            for item in items {
                let residual_buffer = previous.output_buffer(item.request_id)?;
                current.step_from_device_to_device(
                    stream,
                    item.request_id,
                    residual_buffer,
                    rotary_dim,
                    rope_base,
                    item.rope_position,
                    item.cache_position,
                    &format!(
                        "{label} layer {layer_position} request={} position={}",
                        item.request_id.0, item.rope_position
                    ),
                )?;
            }
        } else {
            let current = layers
                .get_mut(layer_position)
                .ok_or_else(|| format!("{label} current layer {layer_position} is missing"))?;
            for item in items {
                current.step_from_host_to_device(
                    stream,
                    item.request_id,
                    &item.residual,
                    rotary_dim,
                    rope_base,
                    item.rope_position,
                    item.cache_position,
                    &format!(
                        "{label} layer {layer_position} request={} position={}",
                        item.request_id.0, item.rope_position
                    ),
                )?;
            }
        }
        residual_device_layer = Some(layer_position);
    }

    Ok(items.len())
}

const QWEN3_EMBED_TOKENS_TENSOR: &str = "model.language_model.embed_tokens.weight";
const QWEN3_FINAL_NORM_TENSOR: &str = "model.language_model.norm.weight";
const QWEN3_LM_HEAD_TENSOR: &str = "lm_head.weight";
const QWEN35_9B_DEFAULT_LAYER_COUNT: usize = 32;

#[derive(Debug, Clone)]
struct PackageTokenLogit {
    token_id: usize,
    logit: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageLmHeadMode {
    CpuChunked,
    GpuResidentF32,
}

impl PackageLmHeadMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::CpuChunked => "cpu_chunked",
            Self::GpuResidentF32 => "gpu_resident_f32",
        }
    }
}

#[derive(Clone, Copy)]
enum PackageLmHeadMatrixStorage {
    F32,
    Bf16,
}

impl PackageLmHeadMatrixStorage {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::Bf16 => std::mem::size_of::<u16>(),
        }
    }
}

fn parse_package_lm_head_mode(value: Option<String>) -> Result<PackageLmHeadMode, ExitCode> {
    match value.as_deref() {
        None | Some("") | Some("cpu") | Some("cpu_chunked") => Ok(PackageLmHeadMode::CpuChunked),
        Some("gpu") | Some("gpu_resident_f32") => Ok(PackageLmHeadMode::GpuResidentF32),
        Some(raw) => {
            eprintln!("invalid lm head mode: {raw}; expected cpu_chunked or gpu_resident_f32");
            Err(ExitCode::from(2))
        }
    }
}

fn parse_package_token_ids_layer_indices(value: Option<String>) -> Result<Vec<usize>, ExitCode> {
    match value.as_deref() {
        None | Some("") | Some("all") | Some("default") => {
            Ok((0..QWEN35_9B_DEFAULT_LAYER_COUNT).collect())
        }
        Some(raw) => parse_usize_csv(raw, "layer list"),
    }
}

fn parse_package_token_ids_layer_indices_for_package(
    path: &str,
    value: Option<String>,
) -> Result<Vec<usize>, ExitCode> {
    match value.as_deref() {
        Some("manifest-all") | Some("manifest_all") | Some("all-manifest")
        | Some("all_manifest") => package_manifest_layer_entries(path)
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| entry.layer_index)
                    .collect::<Vec<_>>()
            })
            .map_err(|err| {
                eprintln!("{err}");
                ExitCode::from(2)
            }),
        _ => parse_package_token_ids_layer_indices(value),
    }
}

fn parse_package_token_ids(value: Option<String>) -> Result<Vec<usize>, ExitCode> {
    match value {
        Some(raw) => parse_usize_csv(&raw, "token IDs"),
        None => Ok(vec![1, 2, 3, 4]),
    }
}

fn package_token_ids_from_len(len: usize) -> Vec<usize> {
    (0..len)
        .map(|index| 1 + (index % 32_000))
        .collect::<Vec<_>>()
}

fn parse_package_prompt_token_ids(value: Option<String>) -> Result<Vec<usize>, ExitCode> {
    match value {
        Some(raw) => {
            if let Some(len_raw) = raw
                .strip_prefix("len:")
                .or_else(|| raw.strip_prefix("len="))
            {
                match len_raw.parse::<usize>() {
                    Ok(len) if len > 0 => Ok(package_token_ids_from_len(len)),
                    Ok(_) => {
                        eprintln!("prompt token length must be greater than zero");
                        Err(ExitCode::from(2))
                    }
                    Err(err) => {
                        eprintln!("invalid prompt token length {len_raw:?}: {err}");
                        Err(ExitCode::from(2))
                    }
                }
            } else {
                parse_usize_csv(&raw, "prompt token IDs")
            }
        }
        None => Ok(vec![1, 2, 3, 4]),
    }
}

fn parse_package_stop_token_ids(value: Option<String>) -> Result<Vec<usize>, ExitCode> {
    match value {
        Some(raw) => match raw.trim() {
            "" | "-" | "none" | "None" | "NONE" => Ok(Vec::new()),
            _ => parse_usize_csv(&raw, "stop token IDs"),
        },
        None => Ok(Vec::new()),
    }
}

fn parse_package_stop_token_sequences(value: Option<String>) -> Result<Vec<Vec<usize>>, ExitCode> {
    match value {
        Some(raw) => match raw.trim() {
            "" | "-" | "none" | "None" | "NONE" => Ok(Vec::new()),
            _ => {
                let mut sequences = Vec::new();
                for raw_sequence in raw.split(';') {
                    let sequence = raw_sequence.trim();
                    if sequence.is_empty() {
                        eprintln!("invalid stop token sequences {raw:?}: empty sequence");
                        return Err(ExitCode::from(2));
                    }
                    sequences.push(parse_usize_csv(sequence, "stop token sequence")?);
                }
                Ok(sequences)
            }
        },
        None => Ok(Vec::new()),
    }
}

fn matched_stop_token_sequence(
    generated_token_ids: &[usize],
    stop_token_sequences: &[Vec<usize>],
) -> Option<Vec<usize>> {
    for sequence in stop_token_sequences {
        if sequence.is_empty() || generated_token_ids.len() < sequence.len() {
            continue;
        }
        let start = generated_token_ids.len() - sequence.len();
        if generated_token_ids[start..] == sequence[..] {
            return Some(sequence.clone());
        }
    }
    None
}

fn parse_package_token_ids_rotary_dim(
    head_dim: usize,
    rotary_dim: Option<&str>,
) -> Result<usize, String> {
    let rotary_dim = match rotary_dim {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid rotary dim {raw:?}: {err}"))?,
        None => {
            let candidate = if head_dim >= 4 {
                head_dim / 4
            } else {
                head_dim
            };
            candidate - (candidate % 2)
        }
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        ));
    }
    Ok(rotary_dim)
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_logits_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    token_ids: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-token-ids-logits-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices =
        match parse_package_token_ids_layer_indices_for_package(&path, layer_indices) {
            Ok(value) => value,
            Err(code) => return code,
        };
    let token_ids = match parse_package_token_ids(token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_token_ids_logits_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        token_ids,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sq_fp8_token_ids_logits_smoke(
    path: Option<String>,
    artifact_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    token_ids: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("sq-fp8-token-ids-logits-smoke requires a .ullm.d package path");
        return ExitCode::from(2);
    };
    let Some(artifact_path) = artifact_path else {
        eprintln!("sq-fp8-token-ids-logits-smoke requires an SQ FP8 artifact path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices =
        match parse_package_token_ids_layer_indices_for_package(&path, layer_indices) {
            Ok(value) => value,
            Err(code) => return code,
        };
    let token_ids = match parse_package_token_ids(token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let artifact = match read_sq_fp8_artifact(&artifact_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read SQ FP8 artifact: {err}");
            return ExitCode::from(1);
        }
    };

    match package_token_ids_logits_smoke_impl_with_sq_overlay(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        token_ids,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        Some(&artifact),
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_generate_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids: Option<String>,
    generated_tokens: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
    lm_head_mode: Option<String>,
    stop_token_ids: Option<String>,
    stop_token_sequences: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-token-ids-generate-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices =
        match parse_package_token_ids_layer_indices_for_package(&path, layer_indices) {
            Ok(value) => value,
            Err(code) => return code,
        };
    let prompt_token_ids = match parse_package_prompt_token_ids(prompt_token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens = match parse_optional_usize(generated_tokens, 1, "generated tokens") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let lm_head_mode = match parse_package_lm_head_mode(lm_head_mode) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_ids = match parse_package_stop_token_ids(stop_token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_sequences = match parse_package_stop_token_sequences(stop_token_sequences) {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_token_ids_generate_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids,
        generated_tokens,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        lm_head_mode,
        stop_token_ids,
        stop_token_sequences,
        None,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sq_fp8_token_ids_generate_smoke(
    path: Option<String>,
    artifact_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids: Option<String>,
    generated_tokens: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
    lm_head_mode: Option<String>,
    stop_token_ids: Option<String>,
    stop_token_sequences: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("sq-fp8-token-ids-generate-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(artifact_path) = artifact_path else {
        eprintln!("sq-fp8-token-ids-generate-smoke requires an SQ FP8 artifact path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices =
        match parse_package_token_ids_layer_indices_for_package(&path, layer_indices) {
            Ok(value) => value,
            Err(code) => return code,
        };
    let prompt_token_ids = match parse_package_prompt_token_ids(prompt_token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens = match parse_optional_usize(generated_tokens, 1, "generated tokens") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let lm_head_mode = match parse_package_lm_head_mode(lm_head_mode) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_ids = match parse_package_stop_token_ids(stop_token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_sequences = match parse_package_stop_token_sequences(stop_token_sequences) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let artifact = match read_sq_fp8_artifact(&artifact_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read SQ FP8 artifact: {err}");
            return ExitCode::from(1);
        }
    };

    match package_token_ids_generate_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids,
        generated_tokens,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        lm_head_mode,
        stop_token_ids,
        stop_token_sequences,
        Some(&artifact),
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_batch_throughput_bench(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    prompt_token_ids_batch: Option<String>,
    generated_tokens_batch: Option<String>,
    top_k: Option<String>,
    lm_head_chunk_rows: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
    lm_head_mode: Option<String>,
    stop_token_ids: Option<String>,
    stop_token_sequences: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-batch-throughput-bench requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_indices =
        match parse_package_token_ids_layer_indices_for_package(&path, layer_indices) {
            Ok(value) => value,
            Err(code) => return code,
        };
    let prompt_token_ids_batch = match parse_package_prompt_token_ids_batch(prompt_token_ids_batch)
    {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generated_tokens_batch = match parse_package_generated_tokens_batch(
        generated_tokens_batch,
        prompt_token_ids_batch.len(),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let top_k = match parse_optional_usize(top_k, 8, "top k") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("top k must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let lm_head_chunk_rows =
        match parse_optional_usize(lm_head_chunk_rows, 1024, "lm head chunk rows") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("lm head chunk rows must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        };
    let lm_head_mode = match parse_package_lm_head_mode(lm_head_mode) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 0, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_ids = match parse_package_stop_token_ids(stop_token_ids) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let stop_token_sequences = match parse_package_stop_token_sequences(stop_token_sequences) {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_batch_throughput_bench_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_indices,
        prompt_token_ids_batch,
        generated_tokens_batch,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        lm_head_mode,
        stop_token_ids,
        stop_token_sequences,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn parse_package_prompt_token_ids_batch(
    value: Option<String>,
) -> Result<Vec<Vec<usize>>, ExitCode> {
    let Some(raw) = value else {
        return Ok(vec![vec![1, 2, 3, 4]]);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        eprintln!("prompt token batch must not be empty");
        return Err(ExitCode::from(2));
    }
    if let Some(rest) = raw
        .strip_prefix("len:")
        .or_else(|| raw.strip_prefix("len="))
    {
        if let Some((len_raw, count_raw)) = rest.split_once('x') {
            let len = parse_usize_value(len_raw.trim(), "prompt token length")?;
            let count = parse_usize_value(count_raw.trim(), "request count")?;
            if len == 0 || count == 0 {
                eprintln!("prompt token batch len:NxM requires N and M greater than zero");
                return Err(ExitCode::from(2));
            }
            return Ok((0..count)
                .map(|_| package_token_ids_from_len(len))
                .collect());
        }
    }
    if raw.contains(';') {
        let mut requests = Vec::new();
        for request in raw.split(';') {
            let request = request.trim();
            if request.is_empty() {
                eprintln!("invalid prompt token batch {raw:?}: empty request");
                return Err(ExitCode::from(2));
            }
            requests.push(parse_package_prompt_token_ids(Some(request.to_string()))?);
        }
        if requests.is_empty() {
            eprintln!("invalid prompt token batch {raw:?}: expected at least one request");
            return Err(ExitCode::from(2));
        }
        return Ok(requests);
    }
    Ok(vec![parse_package_prompt_token_ids(Some(raw.to_string()))?])
}

fn parse_package_generated_tokens_batch(
    value: Option<String>,
    request_count: usize,
) -> Result<Vec<usize>, ExitCode> {
    if request_count == 0 {
        eprintln!("request count must be greater than zero");
        return Err(ExitCode::from(2));
    }
    let Some(raw) = value else {
        return Ok(vec![1; request_count]);
    };
    if raw.contains(',') {
        let parsed = parse_usize_csv(&raw, "generated token counts")?;
        if parsed.len() != request_count {
            eprintln!(
                "generated token count list length {} does not match request count {request_count}",
                parsed.len()
            );
            return Err(ExitCode::from(2));
        }
        if parsed.contains(&0) {
            eprintln!("generated token counts must be greater than zero");
            return Err(ExitCode::from(2));
        }
        return Ok(parsed);
    }
    let value = parse_usize_value(raw.trim(), "generated tokens")?;
    if value == 0 {
        eprintln!("generated tokens must be greater than zero");
        return Err(ExitCode::from(2));
    }
    Ok(vec![value; request_count])
}

fn json_f64_path<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

fn json_usize_path(value: &serde_json::Value, path: &[&str]) -> Option<usize> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    let value = current.as_u64()?;
    usize::try_from(value).ok()
}

fn json_array_len_path(value: &serde_json::Value, path: &[&str]) -> Option<usize> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_array().map(Vec::len)
}

fn cold_prefill_attention_work_tokens_from_lengths(
    prompt_tokens_per_request: &[usize],
) -> Result<u64, String> {
    let mut total = 0_u128;
    for prompt_tokens in prompt_tokens_per_request {
        let prompt_tokens = *prompt_tokens as u128;
        let request_work =
            prompt_tokens
                .checked_mul(prompt_tokens.checked_add(1).ok_or_else(|| {
                    "cold prefill attention work token count overflows".to_string()
                })?)
                .map(|value| value / 2)
                .ok_or_else(|| "cold prefill attention work token count overflows".to_string())?;
        total = total
            .checked_add(request_work)
            .ok_or_else(|| "cold prefill attention work token count overflows".to_string())?;
    }
    u64::try_from(total)
        .map_err(|_| "cold prefill attention work token count exceeds u64".to_string())
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn percentile_f64(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let rank = ((sorted.len() as f64) * percentile).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[index])
}

#[allow(clippy::too_many_arguments)]
fn package_batch_throughput_bench_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids_batch: Vec<Vec<usize>>,
    generated_tokens_batch: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    lm_head_mode: PackageLmHeadMode,
    stop_token_ids: Vec<usize>,
    stop_token_sequences: Vec<Vec<usize>>,
) -> Result<String, String> {
    if prompt_token_ids_batch.is_empty() {
        return Err("package batch throughput bench requires at least one request".to_string());
    }
    if prompt_token_ids_batch.len() != generated_tokens_batch.len() {
        return Err(format!(
            "prompt request count {} does not match generated token count {}",
            prompt_token_ids_batch.len(),
            generated_tokens_batch.len()
        ));
    }

    let context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    drop(context);

    let batch_started = Instant::now();
    let mut request_reports = Vec::with_capacity(prompt_token_ids_batch.len());
    let mut request_latency_ms = Vec::with_capacity(prompt_token_ids_batch.len());
    let mut time_to_first_token_ms = Vec::with_capacity(prompt_token_ids_batch.len());
    let mut time_per_output_token_ms = Vec::new();
    let mut per_request_decode_tps = Vec::new();
    let mut prefill_total_input_tokens = 0_usize;
    let mut generated_tokens_total = 0_usize;
    let mut decode_total_generated_tokens = 0_usize;
    let mut prefill_wall_ms = 0.0_f64;
    let mut decode_wall_ms = 0.0_f64;
    let mut sum_report_total_wall_ms = 0.0_f64;
    let mut kv_cache_bytes_total = 0_u64;
    let mut verified_all = true;
    let mut prefill_executors = Vec::with_capacity(prompt_token_ids_batch.len());

    for (request_index, prompt_token_ids) in prompt_token_ids_batch.iter().enumerate() {
        let requested_generated_tokens = generated_tokens_batch[request_index];
        let request_started = Instant::now();
        let report_text = package_token_ids_generate_smoke_impl(
            path,
            device_index,
            chunk_bytes,
            layer_indices.clone(),
            prompt_token_ids.clone(),
            requested_generated_tokens,
            top_k,
            lm_head_chunk_rows,
            rotary_dim.clone(),
            rope_base,
            position_offset,
            lm_head_mode,
            stop_token_ids.clone(),
            stop_token_sequences.clone(),
            None,
        )?;
        let external_request_wall_ms = request_started.elapsed().as_secs_f64() * 1000.0;
        let report = serde_json::from_str::<serde_json::Value>(&report_text)
            .map_err(|err| format!("failed to decode request {request_index} report: {err}"))?;
        let request_prefill_tokens = json_usize_path(&report, &["prefill", "prompt_tokens"])
            .unwrap_or(prompt_token_ids.len());
        let request_generated_tokens =
            json_array_len_path(&report, &["generated_token_ids"]).unwrap_or(0);
        let request_decode_tokens =
            json_usize_path(&report, &["decode", "timed_incremental_steps"])
                .or_else(|| json_usize_path(&report, &["decode", "timed_recompute_steps"]))
                .unwrap_or_else(|| request_generated_tokens.saturating_sub(1));
        let request_prefill_wall_ms =
            json_f64_path(&report, &["prefill", "wall_ms"]).unwrap_or(0.0);
        let request_prefill_executor = report
            .get("prefill")
            .and_then(|prefill| prefill.get("executor"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let request_decode_wall_ms = json_f64_path(&report, &["decode", "wall_ms"]).unwrap_or(0.0);
        let request_total_wall_ms = json_f64_path(&report, &["timing_ms", "total"])
            .unwrap_or(external_request_wall_ms)
            .max(external_request_wall_ms);
        let request_decode_tps =
            json_f64_path(&report, &["decode", "timed_step_tps"]).or_else(|| {
                if request_decode_tokens > 0 {
                    tps(request_decode_tokens, request_decode_wall_ms)
                } else {
                    None
                }
            });
        let request_time_per_output_token_ms = if request_decode_tokens > 0 {
            Some(request_decode_wall_ms / request_decode_tokens as f64)
        } else {
            None
        };
        let request_verified = report
            .get("correctness")
            .and_then(|correctness| correctness.get("verified"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            && report
                .get("verified")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
        verified_all &= request_verified;

        if let Some(value) = request_decode_tps {
            per_request_decode_tps.push(value);
        }
        if let Some(value) = request_time_per_output_token_ms {
            time_per_output_token_ms.push(value);
        }
        request_latency_ms.push(request_total_wall_ms);
        time_to_first_token_ms.push(request_prefill_wall_ms);
        prefill_executors.push(request_prefill_executor);
        prefill_total_input_tokens = prefill_total_input_tokens
            .checked_add(request_prefill_tokens)
            .ok_or_else(|| "prefill total input tokens overflow".to_string())?;
        generated_tokens_total = generated_tokens_total
            .checked_add(request_generated_tokens)
            .ok_or_else(|| "generated token total overflow".to_string())?;
        decode_total_generated_tokens = decode_total_generated_tokens
            .checked_add(request_decode_tokens)
            .ok_or_else(|| "decode generated token total overflow".to_string())?;
        prefill_wall_ms += request_prefill_wall_ms;
        decode_wall_ms += request_decode_wall_ms;
        sum_report_total_wall_ms += request_total_wall_ms;
        kv_cache_bytes_total = kv_cache_bytes_total
            .checked_add(
                report
                    .get("memory")
                    .and_then(|memory| memory.get("kv_cache_bytes"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            )
            .ok_or_else(|| "KV cache byte total overflow".to_string())?;

        request_reports.push(serde_json::json!({
            "request_index": request_index,
            "prompt_tokens": request_prefill_tokens,
            "requested_generated_tokens": requested_generated_tokens,
            "generated_tokens": request_generated_tokens,
            "decode_timed_generated_tokens": request_decode_tokens,
            "prefill_wall_ms": request_prefill_wall_ms,
            "decode_wall_ms": request_decode_wall_ms,
            "request_wall_ms": request_total_wall_ms,
            "external_request_wall_ms": external_request_wall_ms,
            "time_to_first_token_ms": request_prefill_wall_ms,
            "time_per_output_token_ms": request_time_per_output_token_ms,
            "decode_tps": request_decode_tps,
            "prefill": report
                .get("prefill")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            "stop": report.get("stop").cloned().unwrap_or(serde_json::Value::Null),
            "correctness": report
                .get("correctness")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            "generated_token_ids": report
                .get("generated_token_ids")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            "last_top_logits": report
                .get("decode")
                .and_then(|decode| decode.get("last_top_logits"))
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        }));
    }

    let batch_wall_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
    let end_to_end_total_tokens = prefill_total_input_tokens
        .checked_add(generated_tokens_total)
        .ok_or_else(|| "end-to-end total token count overflows".to_string())?;
    let prefill_executor = prefill_executors
        .first()
        .filter(|first| prefill_executors.iter().all(|value| value == *first))
        .map(String::as_str)
        .unwrap_or("mixed");
    let prompt_tokens_per_request = prompt_token_ids_batch
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let cached_prefix_tokens_per_request = vec![0_usize; prompt_tokens_per_request.len()];
    let new_prefill_tokens_per_request = prompt_tokens_per_request.clone();
    let total_context_tokens_after_prefill_per_request = prompt_tokens_per_request.clone();
    let estimated_prefill_attention_work_tokens =
        cold_prefill_attention_work_tokens_from_lengths(&prompt_tokens_per_request)?;
    let total_context_tokens_after_prefill = total_context_tokens_after_prefill_per_request
        .iter()
        .try_fold(0_usize, |acc, value| acc.checked_add(*value))
        .ok_or_else(|| "total context tokens after prefill overflows".to_string())?;
    let report = serde_json::json!({
        "schema_version": "package-batch-throughput-bench-v0.1",
        "package": path,
        "git_commit": current_git_commit(),
        "backend": info.backend.to_string(),
        "device_index": device_index,
        "device_name": info.name,
        "device_total_global_mem": info.total_global_mem,
        "layers": layer_indices,
        "top_k": top_k,
        "lm_head_chunk_rows": lm_head_chunk_rows,
        "lm_head_mode": lm_head_mode.as_str(),
        "rotary_dim": rotary_dim,
        "rope_base": rope_base,
        "position_offset": position_offset,
        "workload": {
            "batch_size": prompt_token_ids_batch.len(),
            "concurrent_requests": prompt_token_ids_batch.len(),
            "prefill_mode": "cold",
            "prompt_tokens_per_request": prompt_tokens_per_request,
            "cached_prefix_tokens_per_request": cached_prefix_tokens_per_request,
            "new_prefill_tokens_per_request": new_prefill_tokens_per_request,
            "total_context_tokens_after_prefill_per_request": total_context_tokens_after_prefill_per_request,
            "generated_tokens_per_request": generated_tokens_batch,
            "fixed_decode_steps": stop_token_ids.is_empty() && stop_token_sequences.is_empty(),
        },
        "batching": {
            "mode": "logical",
            "prefill_executor": prefill_executor,
            "prefill_real_batch": false,
            "prefill_executor_token_parallelism": 1,
            "prefill_executor_request_parallelism": 1,
            "decode_executor": "sequential_package_token_ids_generate",
            "decode_real_batch": false,
            "decode_executor_request_parallelism": 1,
            "scheduler_policy": "fixed_batch",
            "runtime_reused_across_requests": false,
            "weights_reloaded_per_request": true,
        },
        "metrics": {
            "prefill_total_input_tokens": prefill_total_input_tokens,
            "cached_prefix_total_tokens": 0,
            "total_context_tokens_after_prefill": total_context_tokens_after_prefill,
            "estimated_prefill_attention_work_tokens": estimated_prefill_attention_work_tokens,
            "decode_total_generated_tokens": decode_total_generated_tokens,
            "generated_tokens_total": generated_tokens_total,
            "end_to_end_total_tokens": end_to_end_total_tokens,
            "prefill_wall_ms_sum": prefill_wall_ms,
            "decode_wall_ms_sum": decode_wall_ms,
            "request_wall_ms_sum": sum_report_total_wall_ms,
            "batch_wall_ms": batch_wall_ms,
            "prefill_total_input_tps": tps(prefill_total_input_tokens, prefill_wall_ms),
            "decode_total_generated_tps": tps(decode_total_generated_tokens, decode_wall_ms),
            "end_to_end_total_tps": tps(end_to_end_total_tokens, batch_wall_ms),
            "per_request_decode_tps_mean": mean_f64(&per_request_decode_tps),
            "time_to_first_token_ms_p50": percentile_f64(&time_to_first_token_ms, 0.50),
            "time_to_first_token_ms_p95": percentile_f64(&time_to_first_token_ms, 0.95),
            "request_latency_ms_p50": percentile_f64(&request_latency_ms, 0.50),
            "request_latency_ms_p95": percentile_f64(&request_latency_ms, 0.95),
            "time_per_output_token_ms_p50": percentile_f64(&time_per_output_token_ms, 0.50),
            "time_per_output_token_ms_p95": percentile_f64(&time_per_output_token_ms, 0.95),
        },
        "memory": {
            "vram_baseline_bytes": serde_json::Value::Null,
            "vram_peak_bytes": serde_json::Value::Null,
            "vram_consumed_bytes": serde_json::Value::Null,
            "kv_cache_bytes_total": kv_cache_bytes_total,
        },
        "requests": request_reports,
        "correctness": {
            "verified_all": verified_all,
        },
        "notes": [
            "This is a logical batch benchmark. It sequentially invokes the existing single-request package-token-ids generate path and does not prove real batch kernel throughput.",
            "Use this output for result schema, control-plane, latency, and accounting validation before real batch prefill/decode executors are added.",
            "decode_total_generated_tokens counts timed decode-loop tokens, excluding the first token produced by prefill/top-logits."
        ],
        "verified": verified_all,
    });
    serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to encode batch throughput report: {err}"))
}

fn package_prefill_rmsnorm_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-prefill-rmsnorm-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_prefill_rmsnorm_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_prefill_rmsnorm_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("prefill RMSNorm batch smoke requires at least one token".to_string());
    }
    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if hidden == 0 {
        return Err("prefill RMSNorm batch smoke hidden size is zero".to_string());
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "prefill RMSNorm batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "prefill RMSNorm batch weight length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden
        || embedding_rows.values.len() != prompt_token_ids.len() * hidden
    {
        return Err(format!(
            "prefill RMSNorm batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            prompt_token_ids.len()
        ));
    }

    let mut expected = Vec::with_capacity(embedding_rows.values.len());
    for token_index in 0..prompt_token_ids.len() {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "prefill RMSNorm expected slice start overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "prefill RMSNorm expected slice end overflows".to_string())?;
        expected.extend(runtime_host_rmsnorm_f32(
            &embedding_rows.values[start..end],
            &input_norm.values,
            1e-6_f32,
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let input_elements = prompt_token_ids
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| "prefill RMSNorm input element count overflows".to_string())?;
    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "prefill RMSNorm input",
        )?)
        .map_err(|err| format!("failed to allocate prefill RMSNorm input buffer: {err}"))?;
    let mut weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(hidden, "prefill RMSNorm weight")?)
        .map_err(|err| format!("failed to allocate prefill RMSNorm weight buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "prefill RMSNorm output",
        )?)
        .map_err(|err| format!("failed to allocate prefill RMSNorm output buffer: {err}"))?;
    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy prefill RMSNorm input: {err}"))?;
    weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy prefill RMSNorm weight: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize prefill RMSNorm setup: {err}"))?;

    ullm_runtime_sys::segmented_rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        prompt_token_ids.len(),
        hidden,
        1e-6_f32,
        &mut output_buffer,
        Some(&mut stream),
    )
    .map_err(|err| format!("failed to run warmup segmented prefill RMSNorm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize warmup segmented prefill RMSNorm: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let segmented_started = Instant::now();
        ullm_runtime_sys::segmented_rmsnorm_f32(
            &input_buffer,
            &weight_buffer,
            prompt_token_ids.len(),
            hidden,
            1e-6_f32,
            &mut output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run segmented prefill RMSNorm: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize segmented prefill RMSNorm: {err}"))?;
        measured_ms.push(segmented_started.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let output = read_runtime_buffer_f32(
        &output_buffer,
        &mut stream,
        input_elements,
        "segmented prefill RMSNorm output",
    )?;
    let max_abs_diff = verify_f32_close(
        "segmented prefill RMSNorm",
        &output,
        &expected,
        1e-4_f32,
        1e-4_f32,
    )?;
    let preview_len = output.len().min(8);
    Ok(format!(
        "package-prefill-rmsnorm-batch-smoke package={} layer={} input_norm_tensor=\"{}\" input_norm_dtype={} prompt_tokens={} hidden={} segments={} segment_size={} input_elements={} executor=segmented_rmsnorm_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} max_abs_diff={max_abs_diff:.9} preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        input_norm.dtype,
        prompt_token_ids.len(),
        hidden,
        prompt_token_ids.len(),
        hidden,
        input_elements,
        prompt_token_ids.len(),
        info.backend,
        device_index,
        info.name,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(prompt_token_ids.len(), wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        format_f32_preview(&output[..preview_len]),
    ))
}

fn package_prefill_aq4_matvec_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_name: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-prefill-aq4-matvec-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let tensor_name = tensor_name.unwrap_or_else(|| {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string()
    });
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_prefill_aq4_matvec_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        &tensor_name,
        prompt_token_ids,
        measured_repeats,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_prefill_aq4_matvec_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    tensor_name: &str,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("prefill AQ4 matvec batch smoke requires at least one token".to_string());
    }
    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "prefill AQ4 matvec batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden
        || embedding_rows.values.len() != prompt_token_ids.len() * hidden
    {
        return Err(format!(
            "prefill AQ4 matvec batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            prompt_token_ids.len()
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        tensor_name,
        chunk_bytes,
    )?;
    if matrix.cols != hidden {
        return Err(format!(
            "prefill AQ4 matvec batch tensor {tensor_name} has cols={} but embedding hidden={hidden}",
            matrix.cols
        ));
    }

    let (materialized_rows, materialized_cols, materialized_matrix) =
        materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            tensor_name,
            chunk_bytes,
        )?;
    if (materialized_rows, materialized_cols) != (matrix.rows, matrix.cols) {
        return Err(format!(
            "prefill AQ4 matvec batch materialized shape mismatch: resident=[{},{}] materialized=[{},{}]",
            matrix.rows, matrix.cols, materialized_rows, materialized_cols
        ));
    }
    let materialized_values = read_runtime_buffer_f32(
        &materialized_matrix,
        &mut stream,
        matrix
            .rows
            .checked_mul(matrix.cols)
            .ok_or_else(|| "prefill AQ4 materialized matrix element count overflows".to_string())?,
        "prefill AQ4 materialized matrix",
    )?;

    let mut expected = Vec::with_capacity(
        prompt_token_ids
            .len()
            .checked_mul(matrix.rows)
            .ok_or_else(|| "prefill AQ4 expected output element count overflows".to_string())?,
    );
    for token_index in 0..prompt_token_ids.len() {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "prefill AQ4 expected input slice start overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "prefill AQ4 expected input slice end overflows".to_string())?;
        expected.extend(runtime_host_matvec_f32(
            &materialized_values,
            &embedding_rows.values[start..end],
            matrix.rows,
            matrix.cols,
        ));
    }

    let input_elements = prompt_token_ids
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| "prefill AQ4 matvec batch input element count overflows".to_string())?;
    let output_elements = prompt_token_ids
        .len()
        .checked_mul(matrix.rows)
        .ok_or_else(|| "prefill AQ4 matvec batch output element count overflows".to_string())?;
    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "prefill AQ4 matvec batch input",
        )?)
        .map_err(|err| format!("failed to allocate prefill AQ4 matvec batch input: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            output_elements,
            "prefill AQ4 matvec batch output",
        )?)
        .map_err(|err| format!("failed to allocate prefill AQ4 matvec batch output: {err}"))?;
    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy prefill AQ4 matvec batch input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize prefill AQ4 matvec batch setup: {err}"))?;

    matrix.matvec_batch(
        &input_buffer,
        prompt_token_ids.len(),
        &mut output_buffer,
        &mut stream,
        "prefill AQ4 projection warmup",
    )?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize prefill AQ4 matvec batch warmup: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        matrix.matvec_batch(
            &input_buffer,
            prompt_token_ids.len(),
            &mut output_buffer,
            &mut stream,
            "prefill AQ4 projection",
        )?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize prefill AQ4 matvec batch projection: {err}")
        })?;
        measured_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let output = read_runtime_buffer_f32(
        &output_buffer,
        &mut stream,
        output_elements,
        "prefill AQ4 matvec batch output",
    )?;
    let max_abs_diff = verify_f32_close(
        "prefill AQ4 matvec batch",
        &output,
        &expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let preview_len = output.len().min(8);
    Ok(format!(
        "package-prefill-aq4-matvec-batch-smoke package={} tensor=\"{}\" prompt_tokens={} hidden={} rows={} cols={} input_elements={} output_elements={} executor=aq4_matvec_batch_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} element_tps_mean={} max_abs_diff={max_abs_diff:.9} preview={} verified=true",
        path,
        tensor_name,
        prompt_token_ids.len(),
        hidden,
        matrix.rows,
        matrix.cols,
        input_elements,
        output_elements,
        prompt_token_ids.len(),
        info.backend,
        device_index,
        info.name,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(prompt_token_ids.len(), wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        tps(output_elements, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        format_f32_preview(&output[..preview_len]),
    ))
}

fn package_linear_attn_proj_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-proj-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_linear_attn_proj_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_linear_attn_proj_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn projection batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn projection batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }
    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn projection batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn projection batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden
        || embedding_rows.values.len() != prompt_token_ids.len() * hidden
    {
        return Err(format!(
            "linear-attn projection batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            prompt_token_ids.len()
        ));
    }
    let mut normed_expected = Vec::with_capacity(embedding_rows.values.len());
    for token_index in 0..prompt_token_ids.len() {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "linear-attn projection batch norm input start overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "linear-attn projection batch norm input end overflows".to_string())?;
        normed_expected.extend(runtime_host_rmsnorm_f32(
            &embedding_rows.values[start..end],
            &input_norm.values,
            1e-6_f32,
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let qkv_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &qkv_tensor,
        chunk_bytes,
    )?;
    let z_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &z_tensor,
        chunk_bytes,
    )?;
    let a_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &a_tensor,
        chunk_bytes,
    )?;
    let b_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &b_tensor,
        chunk_bytes,
    )?;
    if qkv_matrix.cols != hidden
        || z_matrix.cols != hidden
        || a_matrix.cols != hidden
        || b_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn projection batch matrix cols mismatch: qkv={} z={} a={} b={} hidden={hidden}",
            qkv_matrix.cols, z_matrix.cols, a_matrix.cols, b_matrix.cols
        ));
    }
    if z_matrix.rows != hidden || a_matrix.rows != b_matrix.rows {
        return Err(format!(
            "linear-attn projection batch matrix rows mismatch: z_rows={} hidden={hidden} a_rows={} b_rows={}",
            z_matrix.rows, a_matrix.rows, b_matrix.rows
        ));
    }

    let (qkv_rows, qkv_cols, qkv_materialized) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &qkv_tensor,
        chunk_bytes,
    )?;
    let (z_rows, z_cols, z_materialized) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &z_tensor,
        chunk_bytes,
    )?;
    let (a_rows, a_cols, a_materialized) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &a_tensor,
        chunk_bytes,
    )?;
    let (b_rows, b_cols, b_materialized) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &b_tensor,
        chunk_bytes,
    )?;
    if (qkv_rows, qkv_cols) != (qkv_matrix.rows, qkv_matrix.cols)
        || (z_rows, z_cols) != (z_matrix.rows, z_matrix.cols)
        || (a_rows, a_cols) != (a_matrix.rows, a_matrix.cols)
        || (b_rows, b_cols) != (b_matrix.rows, b_matrix.cols)
    {
        return Err(
            "linear-attn projection batch resident/materialized shape mismatch".to_string(),
        );
    }
    let qkv_materialized_values = read_runtime_buffer_f32(
        &qkv_materialized,
        &mut stream,
        qkv_matrix
            .rows
            .checked_mul(qkv_matrix.cols)
            .ok_or_else(|| "linear-attn qkv materialized element count overflows".to_string())?,
        "linear-attn qkv materialized matrix",
    )?;
    let z_materialized_values = read_runtime_buffer_f32(
        &z_materialized,
        &mut stream,
        z_matrix
            .rows
            .checked_mul(z_matrix.cols)
            .ok_or_else(|| "linear-attn z materialized element count overflows".to_string())?,
        "linear-attn z materialized matrix",
    )?;
    let a_materialized_values = read_runtime_buffer_f32(
        &a_materialized,
        &mut stream,
        a_matrix
            .rows
            .checked_mul(a_matrix.cols)
            .ok_or_else(|| "linear-attn a materialized element count overflows".to_string())?,
        "linear-attn a materialized matrix",
    )?;
    let b_materialized_values = read_runtime_buffer_f32(
        &b_materialized,
        &mut stream,
        b_matrix
            .rows
            .checked_mul(b_matrix.cols)
            .ok_or_else(|| "linear-attn b materialized element count overflows".to_string())?,
        "linear-attn b materialized matrix",
    )?;

    let expected_projection = |matrix: &[f32], rows: usize| -> Result<Vec<f32>, String> {
        let mut expected =
            Vec::with_capacity(prompt_token_ids.len().checked_mul(rows).ok_or_else(|| {
                "linear-attn expected projection element count overflows".to_string()
            })?);
        for token_index in 0..prompt_token_ids.len() {
            let start = token_index.checked_mul(hidden).ok_or_else(|| {
                "linear-attn expected projection input start overflows".to_string()
            })?;
            let end = start
                .checked_add(hidden)
                .ok_or_else(|| "linear-attn expected projection input end overflows".to_string())?;
            expected.extend(runtime_host_matvec_f32(
                matrix,
                &normed_expected[start..end],
                rows,
                hidden,
            ));
        }
        Ok(expected)
    };
    let qkv_expected = expected_projection(&qkv_materialized_values, qkv_matrix.rows)?;
    let z_expected = expected_projection(&z_materialized_values, z_matrix.rows)?;
    let a_expected = expected_projection(&a_materialized_values, a_matrix.rows)?;
    let b_expected = expected_projection(&b_materialized_values, b_matrix.rows)?;

    let input_elements = prompt_token_ids
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn projection batch input element count overflows".to_string())?;
    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn projection batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn projection batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn projection batch input norm weight",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn input norm weight: {err}"))?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn projection batch input normed",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn projection batch input normed: {err}")
        })?;
    let mut qkv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            prompt_token_ids
                .len()
                .checked_mul(qkv_matrix.rows)
                .ok_or_else(|| "linear-attn qkv output element count overflows".to_string())?,
            "linear-attn qkv batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn qkv batch output: {err}"))?;
    let mut z_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            prompt_token_ids
                .len()
                .checked_mul(z_matrix.rows)
                .ok_or_else(|| "linear-attn z output element count overflows".to_string())?,
            "linear-attn z batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn z batch output: {err}"))?;
    let mut a_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            prompt_token_ids
                .len()
                .checked_mul(a_matrix.rows)
                .ok_or_else(|| "linear-attn a output element count overflows".to_string())?,
            "linear-attn a batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn a batch output: {err}"))?;
    let mut b_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            prompt_token_ids
                .len()
                .checked_mul(b_matrix.rows)
                .ok_or_else(|| "linear-attn b output element count overflows".to_string())?,
            "linear-attn b batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn b batch output: {err}"))?;
    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn projection batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn projection input norm: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn projection batch setup: {err}")
    })?;

    let mut run_projection_batch =
        |stream: &mut ullm_runtime_sys::RuntimeStream| -> Result<(), String> {
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                prompt_token_ids.len(),
                hidden,
                1e-6_f32,
                &mut input_normed_buffer,
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn projection batch input RMSNorm: {err}")
            })?;
            qkv_matrix.matvec_batch(
                &input_normed_buffer,
                prompt_token_ids.len(),
                &mut qkv_output_buffer,
                stream,
                "linear-attn qkv projection batch",
            )?;
            z_matrix.matvec_batch(
                &input_normed_buffer,
                prompt_token_ids.len(),
                &mut z_output_buffer,
                stream,
                "linear-attn z projection batch",
            )?;
            a_matrix.matvec_batch(
                &input_normed_buffer,
                prompt_token_ids.len(),
                &mut a_output_buffer,
                stream,
                "linear-attn a projection batch",
            )?;
            b_matrix.matvec_batch(
                &input_normed_buffer,
                prompt_token_ids.len(),
                &mut b_output_buffer,
                stream,
                "linear-attn b projection batch",
            )
        };

    run_projection_batch(&mut stream)?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn projection batch warmup: {err}")
    })?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        run_projection_batch(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn projection batch measured run: {err}")
        })?;
        measured_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let qkv_output = read_runtime_buffer_f32(
        &qkv_output_buffer,
        &mut stream,
        qkv_expected.len(),
        "linear-attn qkv batch output",
    )?;
    let z_output = read_runtime_buffer_f32(
        &z_output_buffer,
        &mut stream,
        z_expected.len(),
        "linear-attn z batch output",
    )?;
    let a_output = read_runtime_buffer_f32(
        &a_output_buffer,
        &mut stream,
        a_expected.len(),
        "linear-attn a batch output",
    )?;
    let b_output = read_runtime_buffer_f32(
        &b_output_buffer,
        &mut stream,
        b_expected.len(),
        "linear-attn b batch output",
    )?;
    let qkv_max_abs_diff = verify_f32_close(
        "linear-attn qkv projection batch",
        &qkv_output,
        &qkv_expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let z_max_abs_diff = verify_f32_close(
        "linear-attn z projection batch",
        &z_output,
        &z_expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let a_max_abs_diff = verify_f32_close(
        "linear-attn a projection batch",
        &a_output,
        &a_expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let b_max_abs_diff = verify_f32_close(
        "linear-attn b projection batch",
        &b_output,
        &b_expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let output_elements = qkv_output
        .len()
        .checked_add(z_output.len())
        .and_then(|value| value.checked_add(a_output.len()))
        .and_then(|value| value.checked_add(b_output.len()))
        .ok_or_else(|| "linear-attn projection batch output element count overflows".to_string())?;
    let preview_len = qkv_output.len().min(8);
    Ok(format!(
        "package-linear-attn-proj-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} qkv_rows={} z_rows={} a_rows={} b_rows={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} qkv_max_abs_diff={qkv_max_abs_diff:.9} z_max_abs_diff={z_max_abs_diff:.9} a_max_abs_diff={a_max_abs_diff:.9} b_max_abs_diff={b_max_abs_diff:.9} qkv_preview={} verified=true",
        path,
        layer_index,
        qkv_tensor,
        z_tensor,
        a_tensor,
        b_tensor,
        prompt_token_ids.len(),
        hidden,
        qkv_matrix.rows,
        z_matrix.rows,
        a_matrix.rows,
        b_matrix.rows,
        input_elements,
        output_elements,
        prompt_token_ids.len(),
        info.backend,
        device_index,
        info.name,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(prompt_token_ids.len(), wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        tps(output_elements, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        format_f32_preview(&qkv_output[..preview_len]),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelfAttnBatchSmokeStage {
    QkvRope,
    Attention,
    Block,
    Layer,
}

impl SelfAttnBatchSmokeStage {
    fn smoke_name(self) -> &'static str {
        match self {
            SelfAttnBatchSmokeStage::QkvRope => "package-self-attn-qkv-rope-batch-smoke",
            SelfAttnBatchSmokeStage::Attention => "package-self-attn-attention-batch-smoke",
            SelfAttnBatchSmokeStage::Block => "package-self-attn-block-batch-smoke",
            SelfAttnBatchSmokeStage::Layer => "package-self-attn-layer-batch-smoke",
        }
    }

    fn include_attention(self) -> bool {
        matches!(
            self,
            SelfAttnBatchSmokeStage::Attention
                | SelfAttnBatchSmokeStage::Block
                | SelfAttnBatchSmokeStage::Layer
        )
    }

    fn include_block(self) -> bool {
        matches!(
            self,
            SelfAttnBatchSmokeStage::Block | SelfAttnBatchSmokeStage::Layer
        )
    }

    fn include_layer(self) -> bool {
        self == SelfAttnBatchSmokeStage::Layer
    }
}

fn package_self_attn_qkv_rope_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-qkv-rope-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_self_attn_qkv_rope_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
        rotary_dim,
        rope_base,
        position_offset,
        SelfAttnBatchSmokeStage::QkvRope,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_self_attn_attention_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-attention-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_self_attn_qkv_rope_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
        rotary_dim,
        rope_base,
        position_offset,
        SelfAttnBatchSmokeStage::Attention,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_self_attn_block_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-block-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_self_attn_qkv_rope_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
        rotary_dim,
        rope_base,
        position_offset,
        SelfAttnBatchSmokeStage::Block,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_self_attn_layer_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-layer-batch-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let prompt_token_ids = match parse_package_prompt_token_ids(
        prompt_token_ids.or_else(|| Some("len:4".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 1, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_self_attn_qkv_rope_batch_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        prompt_token_ids,
        measured_repeats,
        rotary_dim,
        rope_base,
        position_offset,
        SelfAttnBatchSmokeStage::Layer,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_qkv_rope_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
    rotary_dim_arg: Option<String>,
    rope_base_arg: Option<String>,
    position_offset_arg: Option<String>,
    stage: SelfAttnBatchSmokeStage,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("self-attn qkv RoPE batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::SelfAttention {
        return Err(format!(
            "self-attn qkv RoPE batch smoke requires a self attention layer, got layer {layer_index}"
        ));
    }

    let sequence_len = prompt_token_ids.len();
    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "self-attn qkv RoPE batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "self-attn qkv RoPE batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let mut q_norm = read_named_passthrough_f32(path, &q_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read q RMSNorm tensor: {err}"))?;
    q_norm.values = effective_rmsnorm_weight_values(&q_norm_tensor, &q_norm.values);
    let mut k_norm = read_named_passthrough_f32(path, &k_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read k RMSNorm tensor: {err}"))?;
    k_norm.values = effective_rmsnorm_weight_values(&k_norm_tensor, &k_norm.values);
    let mut post_norm = if stage.include_layer() {
        Some(
            read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)
                .map_err(|err| format!("failed to read post RMSNorm tensor: {err}"))?,
        )
    } else {
        None
    };
    if let Some(post_norm) = post_norm.as_mut() {
        post_norm.values = effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
        if post_norm.values.len() != hidden {
            return Err(format!(
                "self-attn layer batch post norm length {} does not match hidden {hidden}",
                post_norm.values.len()
            ));
        }
    }
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        return Err(format!(
            "self-attn qkv RoPE batch q/k norm head dims must be nonzero and equal: q={} k={}",
            head_dim,
            k_norm.values.len()
        ));
    }
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        return Err(format!(
            "self-attn qkv RoPE batch default rotary_dim is zero for head_dim {head_dim}"
        ));
    }
    let rotary_dim = parse_optional_usize(rotary_dim_arg, default_rotary_dim, "rotary dim")
        .map_err(|code| format!("failed to parse rotary dim, exit_code={code:?}"))?;
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "self-attn qkv RoPE batch rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim} head_dim={head_dim}"
        ));
    }
    let rope_base = parse_optional_f32(rope_base_arg, 10_000_000.0, "rope base")
        .map_err(|code| format!("failed to parse rope base, exit_code={code:?}"))?;
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err(
            "self-attn qkv RoPE batch rope base must be finite and greater than one".into(),
        );
    }
    let position_offset = parse_optional_usize(position_offset_arg, 0, "position offset")
        .map_err(|code| format!("failed to parse position offset, exit_code={code:?}"))?;

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "self-attn qkv RoPE batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
        ));
    }
    let mut expected_input_normed = Vec::with_capacity(sequence_len * hidden);
    for token_index in 0..sequence_len {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "self-attn qkv RoPE batch norm input start overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "self-attn qkv RoPE batch norm input end overflows".to_string())?;
        expected_input_normed.extend(runtime_host_rmsnorm_f32(
            &embedding_rows.values[start..end],
            &input_norm.values,
            1e-6_f32,
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let q_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &q_tensor,
        chunk_bytes,
    )?;
    let k_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &k_tensor,
        chunk_bytes,
    )?;
    let v_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &v_tensor,
        chunk_bytes,
    )?;
    let o_matrix = if stage.include_block() {
        Some(PackageAq4ResidentMatvec::load(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &o_tensor,
            chunk_bytes,
        )?)
    } else {
        None
    };
    let gate_matrix = if stage.include_layer() {
        Some(PackageAq4ResidentMatvec::load(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &gate_tensor,
            chunk_bytes,
        )?)
    } else {
        None
    };
    let up_matrix = if stage.include_layer() {
        Some(PackageAq4ResidentMatvec::load(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &up_tensor,
            chunk_bytes,
        )?)
    } else {
        None
    };
    let down_matrix = if stage.include_layer() {
        Some(PackageAq4ResidentMatvec::load(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &down_tensor,
            chunk_bytes,
        )?)
    } else {
        None
    };
    if q_matrix.cols != hidden || k_matrix.cols != hidden || v_matrix.cols != hidden {
        return Err(format!(
            "self-attn qkv RoPE batch q/k/v hidden mismatch: q_cols={} k_cols={} v_cols={} hidden={hidden}",
            q_matrix.cols, k_matrix.cols, v_matrix.cols
        ));
    }
    let two_hidden = hidden
        .checked_mul(2)
        .ok_or_else(|| "self-attn qkv RoPE batch hidden*2 overflows".to_string())?;
    let two_head_dim = head_dim
        .checked_mul(2)
        .ok_or_else(|| "self-attn qkv RoPE batch head_dim*2 overflows".to_string())?;
    if q_matrix.rows != two_hidden || !q_matrix.rows.is_multiple_of(two_head_dim) {
        return Err(format!(
            "self-attn qkv RoPE batch requires Qwen3.5 gated q layout: q_rows={} hidden={} head_dim={head_dim}",
            q_matrix.rows, hidden
        ));
    }
    if !k_matrix.rows.is_multiple_of(head_dim) {
        return Err(format!(
            "self-attn qkv RoPE batch k rows {} are not a multiple of head_dim {head_dim}",
            k_matrix.rows
        ));
    }
    let q_heads = q_matrix.rows / two_head_dim;
    let kv_heads = k_matrix.rows / head_dim;
    if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "self-attn qkv RoPE batch invalid heads: q_heads={q_heads} kv_heads={kv_heads}"
        ));
    }
    if !v_matrix.rows.is_multiple_of(kv_heads) {
        return Err(format!(
            "self-attn qkv RoPE batch v rows {} are not compatible with kv_heads {kv_heads}",
            v_matrix.rows
        ));
    }
    let value_dim = v_matrix.rows / kv_heads;
    let attention_width = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "self-attn batch attention width overflows".to_string())?;
    if let Some(o_matrix) = o_matrix.as_ref() {
        if o_matrix.rows != hidden || o_matrix.cols != attention_width {
            return Err(format!(
                "self-attn block batch o projection shape mismatch: o=[{},{}] expected [{hidden},{attention_width}]",
                o_matrix.rows, o_matrix.cols
            ));
        }
    }
    let intermediate = if let (Some(gate_matrix), Some(up_matrix), Some(down_matrix)) = (
        gate_matrix.as_ref(),
        up_matrix.as_ref(),
        down_matrix.as_ref(),
    ) {
        if gate_matrix.rows != up_matrix.rows
            || gate_matrix.cols != up_matrix.cols
            || gate_matrix.cols != hidden
        {
            return Err(format!(
                "self-attn layer batch gate/up shape mismatch: gate=[{},{}] up=[{},{}] hidden={hidden}",
                gate_matrix.rows, gate_matrix.cols, up_matrix.rows, up_matrix.cols
            ));
        }
        let intermediate = gate_matrix.rows;
        if down_matrix.rows != hidden || down_matrix.cols != intermediate {
            return Err(format!(
                "self-attn layer batch down shape mismatch: down=[{},{}] expected [{hidden},{intermediate}]",
                down_matrix.rows, down_matrix.cols
            ));
        }
        intermediate
    } else {
        0
    };

    let input_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "self-attn qkv RoPE batch input element count overflows".to_string())?;
    let q_projected_elements = sequence_len.checked_mul(q_matrix.rows).ok_or_else(|| {
        "self-attn qkv RoPE batch q projected element count overflows".to_string()
    })?;
    let k_projected_elements = sequence_len.checked_mul(k_matrix.rows).ok_or_else(|| {
        "self-attn qkv RoPE batch k projected element count overflows".to_string()
    })?;
    let v_projected_elements = sequence_len.checked_mul(v_matrix.rows).ok_or_else(|| {
        "self-attn qkv RoPE batch v projected element count overflows".to_string()
    })?;
    let q_output_elements = sequence_len
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn qkv RoPE batch q output element count overflows".to_string())?;
    let k_output_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn qkv RoPE batch k output element count overflows".to_string())?;
    let attention_output_elements = sequence_len
        .checked_mul(attention_width)
        .ok_or_else(|| "self-attn attention batch output element count overflows".to_string())?;
    let intermediate_elements = if stage.include_layer() {
        sequence_len.checked_mul(intermediate).ok_or_else(|| {
            "self-attn layer batch intermediate element count overflows".to_string()
        })?
    } else {
        0
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "self-attn qkv RoPE batch input",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "self-attn qkv RoPE batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate self-attn qkv RoPE batch input norm weight: {err}")
        })?;
    let mut q_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_norm.values.len(),
            "self-attn qkv RoPE batch q norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate self-attn qkv RoPE batch q norm weight: {err}")
        })?;
    let mut k_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            k_norm.values.len(),
            "self-attn qkv RoPE batch k norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate self-attn qkv RoPE batch k norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "self-attn qkv RoPE batch input normed",
        )?)
        .map_err(|err| {
            format!("failed to allocate self-attn qkv RoPE batch input normed: {err}")
        })?;
    let mut q_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_projected_elements,
            "self-attn qkv RoPE batch q projected",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch q projected: {err}"))?;
    let mut k_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            k_projected_elements,
            "self-attn qkv RoPE batch k projected",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch k projected: {err}"))?;
    let mut v_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            v_projected_elements,
            "self-attn qkv RoPE batch v projected",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch v projected: {err}"))?;
    let mut q_gate_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "self-attn qkv RoPE batch q gate",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch q gate: {err}"))?;
    let mut q_rope_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "self-attn qkv RoPE batch q RoPE",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch q RoPE: {err}"))?;
    let mut k_rope_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            k_output_elements,
            "self-attn qkv RoPE batch k RoPE",
        )?)
        .map_err(|err| format!("failed to allocate self-attn qkv RoPE batch k RoPE: {err}"))?;
    let mut attention_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            attention_output_elements,
            "self-attn attention batch output",
        )?)
        .map_err(|err| format!("failed to allocate self-attn attention batch output: {err}"))?;
    let mut attention_projection_input_buffer = if stage.include_block() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    attention_output_elements,
                    "self-attn block batch attention projection input",
                )?)
                .map_err(|err| {
                    format!(
                        "failed to allocate self-attn block batch attention projection input: {err}"
                    )
                })?,
        )
    } else {
        None
    };
    let mut attn_projected_buffer = if stage.include_block() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    input_elements,
                    "self-attn block batch projected output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn block batch projected output: {err}")
                })?,
        )
    } else {
        None
    };
    let mut block_output_buffer = if stage.include_block() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    input_elements,
                    "self-attn block batch output",
                )?)
                .map_err(|err| format!("failed to allocate self-attn block batch output: {err}"))?,
        )
    } else {
        None
    };
    let mut post_norm_weight_buffer = if stage.include_layer() {
        let post_norm = post_norm
            .as_ref()
            .ok_or_else(|| "self-attn layer batch post norm missing".to_string())?;
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    post_norm.values.len(),
                    "self-attn layer batch post norm weight",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch post norm weight: {err}")
                })?,
        )
    } else {
        None
    };
    let mut post_normed_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    input_elements,
                    "self-attn layer batch post normed",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch post normed: {err}")
                })?,
        )
    } else {
        None
    };
    let mut mlp_gate_output_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    intermediate_elements,
                    "self-attn layer batch MLP gate output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch MLP gate output: {err}")
                })?,
        )
    } else {
        None
    };
    let mut mlp_up_output_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    intermediate_elements,
                    "self-attn layer batch MLP up output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch MLP up output: {err}")
                })?,
        )
    } else {
        None
    };
    let mut mlp_activation_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    intermediate_elements,
                    "self-attn layer batch MLP activation",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch MLP activation: {err}")
                })?,
        )
    } else {
        None
    };
    let mut mlp_down_output_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    input_elements,
                    "self-attn layer batch MLP down output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate self-attn layer batch MLP down output: {err}")
                })?,
        )
    } else {
        None
    };
    let mut layer_output_buffer = if stage.include_layer() {
        Some(
            context
                .alloc_buffer(checked_f32_byte_len(
                    input_elements,
                    "self-attn layer batch output",
                )?)
                .map_err(|err| format!("failed to allocate self-attn layer batch output: {err}"))?,
        )
    } else {
        None
    };

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy self-attn qkv RoPE batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy self-attn qkv RoPE batch input norm: {err}"))?;
    q_norm_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&q_norm.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy self-attn qkv RoPE batch q norm: {err}"))?;
    k_norm_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&k_norm.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy self-attn qkv RoPE batch k norm: {err}"))?;
    if let (Some(buffer), Some(post_norm)) = (post_norm_weight_buffer.as_mut(), post_norm.as_ref())
    {
        buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&post_norm.values),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy self-attn layer batch post norm: {err}"))?;
    }
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize self-attn qkv RoPE batch setup: {err}"))?;

    let mut run_qkv_rope_batch =
        |stream: &mut ullm_runtime_sys::RuntimeStream| -> Result<(), String> {
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                sequence_len,
                hidden,
                1e-6_f32,
                &mut input_normed_buffer,
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to run self-attn qkv RoPE batch input RMSNorm: {err}")
            })?;
            q_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut q_projected_buffer,
                stream,
                "self-attn q projection batch",
            )?;
            k_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut k_projected_buffer,
                stream,
                "self-attn k projection batch",
            )?;
            v_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut v_projected_buffer,
                stream,
                "self-attn v projection batch",
            )?;
            ullm_runtime_sys::qwen35_qk_norm_rope_batch_f32(
                &q_projected_buffer,
                &k_projected_buffer,
                &q_norm_weight_buffer,
                &k_norm_weight_buffer,
                q_heads,
                kv_heads,
                sequence_len,
                head_dim,
                rotary_dim,
                position_offset,
                rope_base,
                1e-5_f32,
                &mut q_gate_buffer,
                &mut q_rope_buffer,
                &mut k_rope_buffer,
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to run self-attn qkv RoPE batch Qwen3.5 QK norm RoPE: {err}")
            })?;
            if stage.include_attention() {
                ullm_runtime_sys::causal_attn_f32(
                    &q_rope_buffer,
                    &k_rope_buffer,
                    &v_projected_buffer,
                    sequence_len,
                    q_heads,
                    kv_heads,
                    head_dim,
                    value_dim,
                    softmax_scale,
                    &mut attention_output_buffer,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run self-attn attention batch causal attention: {err}")
                })?;
            }
            if stage.include_block() {
                let attention_projection_input_buffer =
                    attention_projection_input_buffer.as_mut().ok_or_else(|| {
                        "self-attn block batch projection input buffer missing".to_string()
                    })?;
                let attn_projected_buffer = attn_projected_buffer
                    .as_mut()
                    .ok_or_else(|| "self-attn block batch projected buffer missing".to_string())?;
                let block_output_buffer = block_output_buffer
                    .as_mut()
                    .ok_or_else(|| "self-attn block batch output buffer missing".to_string())?;
                let o_matrix = o_matrix
                    .as_ref()
                    .ok_or_else(|| "self-attn block batch o projection missing".to_string())?;
                ullm_runtime_sys::sigmoid_mul_f32(
                    &q_gate_buffer,
                    &attention_output_buffer,
                    attention_output_elements,
                    attention_projection_input_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run self-attn block batch output gate: {err}"))?;
                o_matrix.matvec_batch(
                    attention_projection_input_buffer,
                    sequence_len,
                    attn_projected_buffer,
                    stream,
                    "self-attn block batch o projection",
                )?;
                ullm_runtime_sys::add_f32(
                    attn_projected_buffer,
                    &input_buffer,
                    input_elements,
                    block_output_buffer,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run self-attn block batch residual add: {err}")
                })?;
            }
            if stage.include_layer() {
                let block_output_buffer = block_output_buffer.as_ref().ok_or_else(|| {
                    "self-attn layer batch block output buffer missing".to_string()
                })?;
                let post_norm_weight_buffer =
                    post_norm_weight_buffer.as_ref().ok_or_else(|| {
                        "self-attn layer batch post norm weight buffer missing".to_string()
                    })?;
                let post_normed_buffer = post_normed_buffer.as_mut().ok_or_else(|| {
                    "self-attn layer batch post normed buffer missing".to_string()
                })?;
                let mlp_gate_output_buffer = mlp_gate_output_buffer.as_mut().ok_or_else(|| {
                    "self-attn layer batch MLP gate output buffer missing".to_string()
                })?;
                let mlp_up_output_buffer = mlp_up_output_buffer.as_mut().ok_or_else(|| {
                    "self-attn layer batch MLP up output buffer missing".to_string()
                })?;
                let mlp_activation_buffer = mlp_activation_buffer.as_mut().ok_or_else(|| {
                    "self-attn layer batch MLP activation buffer missing".to_string()
                })?;
                let mlp_down_output_buffer = mlp_down_output_buffer.as_mut().ok_or_else(|| {
                    "self-attn layer batch MLP down output buffer missing".to_string()
                })?;
                let layer_output_buffer = layer_output_buffer
                    .as_mut()
                    .ok_or_else(|| "self-attn layer batch output buffer missing".to_string())?;
                let gate_matrix = gate_matrix
                    .as_ref()
                    .ok_or_else(|| "self-attn layer batch gate projection missing".to_string())?;
                let up_matrix = up_matrix
                    .as_ref()
                    .ok_or_else(|| "self-attn layer batch up projection missing".to_string())?;
                let down_matrix = down_matrix
                    .as_ref()
                    .ok_or_else(|| "self-attn layer batch down projection missing".to_string())?;
                ullm_runtime_sys::segmented_rmsnorm_f32(
                    block_output_buffer,
                    post_norm_weight_buffer,
                    sequence_len,
                    hidden,
                    1e-5_f32,
                    post_normed_buffer,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run self-attn layer batch post RMSNorm: {err}")
                })?;
                gate_matrix.matvec_batch(
                    post_normed_buffer,
                    sequence_len,
                    mlp_gate_output_buffer,
                    stream,
                    "self-attn layer batch MLP gate projection",
                )?;
                up_matrix.matvec_batch(
                    post_normed_buffer,
                    sequence_len,
                    mlp_up_output_buffer,
                    stream,
                    "self-attn layer batch MLP up projection",
                )?;
                ullm_runtime_sys::silu_mul_f32(
                    mlp_gate_output_buffer,
                    mlp_up_output_buffer,
                    intermediate_elements,
                    mlp_activation_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run self-attn layer batch SiLU-mul: {err}"))?;
                down_matrix.matvec_batch(
                    mlp_activation_buffer,
                    sequence_len,
                    mlp_down_output_buffer,
                    stream,
                    "self-attn layer batch MLP down projection",
                )?;
                ullm_runtime_sys::add_f32(
                    mlp_down_output_buffer,
                    block_output_buffer,
                    input_elements,
                    layer_output_buffer,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run self-attn layer batch residual add: {err}")
                })?;
            }
            Ok(())
        };

    run_qkv_rope_batch(&mut stream)?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize self-attn qkv RoPE batch warmup: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        run_qkv_rope_batch(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize self-attn qkv RoPE batch measured run: {err}")
        })?;
        measured_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let verification_started = Instant::now();
    let input_normed = read_runtime_buffer_f32(
        &input_normed_buffer,
        &mut stream,
        input_elements,
        "self-attn qkv RoPE batch input normed",
    )?;
    let q_projected = read_runtime_buffer_f32(
        &q_projected_buffer,
        &mut stream,
        q_projected_elements,
        "self-attn qkv RoPE batch q projected",
    )?;
    let k_projected = read_runtime_buffer_f32(
        &k_projected_buffer,
        &mut stream,
        k_projected_elements,
        "self-attn qkv RoPE batch k projected",
    )?;
    let v_projected = read_runtime_buffer_f32(
        &v_projected_buffer,
        &mut stream,
        v_projected_elements,
        "self-attn qkv RoPE batch v projected",
    )?;
    let q_gate = read_runtime_buffer_f32(
        &q_gate_buffer,
        &mut stream,
        q_output_elements,
        "self-attn qkv RoPE batch q gate",
    )?;
    let q_rope = read_runtime_buffer_f32(
        &q_rope_buffer,
        &mut stream,
        q_output_elements,
        "self-attn qkv RoPE batch q RoPE",
    )?;
    let k_rope = read_runtime_buffer_f32(
        &k_rope_buffer,
        &mut stream,
        k_output_elements,
        "self-attn qkv RoPE batch k RoPE",
    )?;
    let attention_output = if stage.include_attention() {
        Some(read_runtime_buffer_f32(
            &attention_output_buffer,
            &mut stream,
            attention_output_elements,
            "self-attn attention batch output",
        )?)
    } else {
        None
    };
    let attention_projection_input =
        if let Some(buffer) = attention_projection_input_buffer.as_ref() {
            Some(read_runtime_buffer_f32(
                buffer,
                &mut stream,
                attention_output_elements,
                "self-attn block batch attention projection input",
            )?)
        } else {
            None
        };
    let attn_projected = if let Some(buffer) = attn_projected_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            input_elements,
            "self-attn block batch projected output",
        )?)
    } else {
        None
    };
    let block_output = if let Some(buffer) = block_output_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            input_elements,
            "self-attn block batch output",
        )?)
    } else {
        None
    };
    let post_normed = if let Some(buffer) = post_normed_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            input_elements,
            "self-attn layer batch post normed",
        )?)
    } else {
        None
    };
    let mlp_gate_output = if let Some(buffer) = mlp_gate_output_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            intermediate_elements,
            "self-attn layer batch MLP gate output",
        )?)
    } else {
        None
    };
    let mlp_up_output = if let Some(buffer) = mlp_up_output_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            intermediate_elements,
            "self-attn layer batch MLP up output",
        )?)
    } else {
        None
    };
    let mlp_activation = if let Some(buffer) = mlp_activation_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            intermediate_elements,
            "self-attn layer batch MLP activation",
        )?)
    } else {
        None
    };
    let mlp_down_output = if let Some(buffer) = mlp_down_output_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            input_elements,
            "self-attn layer batch MLP down output",
        )?)
    } else {
        None
    };
    let layer_output = if let Some(buffer) = layer_output_buffer.as_ref() {
        Some(read_runtime_buffer_f32(
            buffer,
            &mut stream,
            input_elements,
            "self-attn layer batch output",
        )?)
    } else {
        None
    };

    let input_norm_max_abs_diff = verify_f32_close(
        "self-attn qkv RoPE batch input RMSNorm",
        &input_normed,
        &expected_input_normed,
        1e-4_f32,
        1e-5_f32,
    )?;
    let q_split = split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        q_matrix.rows,
        hidden,
        head_dim,
    )?;
    if q_split.layout != "qwen3.5-gated" || q_split.q_heads != q_heads {
        return Err(format!(
            "self-attn qkv RoPE batch unexpected q split layout={} q_heads={} expected q_heads={q_heads}",
            q_split.layout, q_split.q_heads
        ));
    }
    let expected_q_gate = q_split
        .gate
        .as_ref()
        .ok_or_else(|| "self-attn qkv RoPE batch expected gated q projection".to_string())?;
    let expected_q_normed = q_split
        .query
        .chunks_exact(head_dim)
        .flat_map(|segment| runtime_host_rmsnorm_f32(segment, &q_norm.values, 1e-5_f32))
        .collect::<Vec<_>>();
    let expected_k_normed = k_projected
        .chunks_exact(head_dim)
        .flat_map(|segment| runtime_host_rmsnorm_f32(segment, &k_norm.values, 1e-5_f32))
        .collect::<Vec<_>>();
    let expected_q_rope = runtime_host_rope_f32(
        &expected_q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let expected_k_rope = runtime_host_rope_f32(
        &expected_k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    if expected_q_rope.len() != q_output_elements || expected_k_rope.len() != k_output_elements {
        return Err("self-attn qkv RoPE batch failed to build RoPE references".to_string());
    }
    let q_gate_max_abs_diff = verify_f32_close(
        "self-attn qkv RoPE batch q gate",
        &q_gate,
        expected_q_gate,
        1e-5_f32,
        1e-5_f32,
    )?;
    let rope_abs_floor = self_attn_batch_rope_abs_floor(sequence_len, position_offset);
    let q_rope_max_abs_diff = verify_f32_close(
        "self-attn qkv RoPE batch q RoPE",
        &q_rope,
        &expected_q_rope,
        rope_abs_floor,
        2e-5_f32,
    )?;
    let k_rope_max_abs_diff = verify_f32_close(
        "self-attn qkv RoPE batch k RoPE",
        &k_rope,
        &expected_k_rope,
        rope_abs_floor,
        2e-5_f32,
    )?;
    let attention_verification = if let Some(attention_output) = attention_output.as_ref() {
        if self_attn_batch_use_sampled_attention_verification(sequence_len) {
            let (checked_values, max_abs_diff) = verify_causal_attention_output_sampled(
                "self-attn attention batch causal attention",
                attention_output,
                &q_rope,
                &k_rope,
                &v_projected,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                2e-4_f32,
                2e-4_f32,
            )?;
            Some(("sampled", checked_values, max_abs_diff))
        } else {
            let expected_attention = runtime_host_causal_attn_f32(
                &q_rope,
                &k_rope,
                &v_projected,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            );
            if expected_attention.len() != attention_output_elements {
                return Err(format!(
                    "self-attn attention batch reference output size mismatch: expected {} got {}",
                    attention_output_elements,
                    expected_attention.len()
                ));
            }
            Some((
                "full",
                expected_attention.len(),
                verify_f32_close(
                    "self-attn attention batch causal attention",
                    attention_output,
                    &expected_attention,
                    2e-4_f32,
                    2e-4_f32,
                )?,
            ))
        }
    } else {
        None
    };
    let block_verification = if stage.include_block() {
        let attention_output = attention_output.as_ref().ok_or_else(|| {
            "self-attn block batch attention output missing for verification".to_string()
        })?;
        let attention_projection_input = attention_projection_input.as_ref().ok_or_else(|| {
            "self-attn block batch projection input missing for verification".to_string()
        })?;
        let attn_projected = attn_projected.as_ref().ok_or_else(|| {
            "self-attn block batch projected output missing for verification".to_string()
        })?;
        let block_output = block_output
            .as_ref()
            .ok_or_else(|| "self-attn block batch output missing for verification".to_string())?;
        let o_matrix = o_matrix.as_ref().ok_or_else(|| {
            "self-attn block batch o projection missing for verification".to_string()
        })?;
        let output_gate_max_abs_diff = verify_sigmoid_mul_f32_close(
            "self-attn block batch output gate",
            &q_gate,
            attention_output,
            attention_projection_input,
            1e-5_f32,
            1e-5_f32,
        )?;
        let mut o_row_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                o_matrix.cols,
                "self-attn block batch sampled o projection row",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn block batch sampled o row: {err}")
            })?;
        let (o_proj_checked_values, o_proj_max_abs_diff) = verify_aq4_matvec_batch_output_sampled(
            "self-attn block batch o projection",
            o_matrix,
            attention_projection_input,
            attn_projected,
            sequence_len,
            &mut o_row_buffer,
            &mut stream,
            3e-3_f32,
            2e-5_f32,
        )?;
        let block_max_abs_diff = verify_add_f32_close(
            "self-attn block batch residual add",
            &attn_projected,
            &embedding_rows.values,
            block_output,
            1e-5_f32,
            1e-6_f32,
        )?;
        Some((
            output_gate_max_abs_diff,
            o_proj_checked_values,
            o_proj_max_abs_diff,
            block_max_abs_diff,
        ))
    } else {
        None
    };
    let layer_verification = if stage.include_layer() {
        let block_output = block_output.as_ref().ok_or_else(|| {
            "self-attn layer batch block output missing for verification".to_string()
        })?;
        let post_normed = post_normed.as_ref().ok_or_else(|| {
            "self-attn layer batch post normed missing for verification".to_string()
        })?;
        let mlp_gate_output = mlp_gate_output.as_ref().ok_or_else(|| {
            "self-attn layer batch MLP gate output missing for verification".to_string()
        })?;
        let mlp_up_output = mlp_up_output.as_ref().ok_or_else(|| {
            "self-attn layer batch MLP up output missing for verification".to_string()
        })?;
        let mlp_activation = mlp_activation.as_ref().ok_or_else(|| {
            "self-attn layer batch MLP activation missing for verification".to_string()
        })?;
        let mlp_down_output = mlp_down_output.as_ref().ok_or_else(|| {
            "self-attn layer batch MLP down output missing for verification".to_string()
        })?;
        let layer_output = layer_output
            .as_ref()
            .ok_or_else(|| "self-attn layer batch output missing for verification".to_string())?;
        let post_norm = post_norm.as_ref().ok_or_else(|| {
            "self-attn layer batch post norm missing for verification".to_string()
        })?;
        let gate_matrix = gate_matrix.as_ref().ok_or_else(|| {
            "self-attn layer batch gate projection missing for verification".to_string()
        })?;
        let up_matrix = up_matrix.as_ref().ok_or_else(|| {
            "self-attn layer batch up projection missing for verification".to_string()
        })?;
        let down_matrix = down_matrix.as_ref().ok_or_else(|| {
            "self-attn layer batch down projection missing for verification".to_string()
        })?;

        let mut expected_post_normed = Vec::with_capacity(input_elements);
        for token_index in 0..sequence_len {
            let start = token_index
                .checked_mul(hidden)
                .ok_or_else(|| "self-attn layer batch expected norm start overflows".to_string())?;
            let end = start
                .checked_add(hidden)
                .ok_or_else(|| "self-attn layer batch expected norm end overflows".to_string())?;
            expected_post_normed.extend(runtime_host_rmsnorm_f32(
                &block_output[start..end],
                &post_norm.values,
                1e-5_f32,
            ));
        }
        if expected_post_normed.len() != input_elements {
            return Err("failed to build self-attn layer batch post norm reference".to_string());
        }
        let post_norm_max_abs_diff = verify_f32_close(
            "self-attn layer batch post norm",
            post_normed,
            &expected_post_normed,
            1e-4_f32,
            1e-5_f32,
        )?;
        let mut gate_row_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                gate_matrix.cols,
                "self-attn layer batch sampled gate projection row",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn layer batch sampled gate row: {err}")
            })?;
        let (gate_checked_values, gate_max_abs_diff) = verify_aq4_matvec_batch_output_sampled(
            "self-attn layer batch MLP gate projection",
            gate_matrix,
            post_normed,
            mlp_gate_output,
            sequence_len,
            &mut gate_row_buffer,
            &mut stream,
            3e-3_f32,
            2e-5_f32,
        )?;
        let mut up_row_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                up_matrix.cols,
                "self-attn layer batch sampled up projection row",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn layer batch sampled up row: {err}")
            })?;
        let (up_checked_values, up_max_abs_diff) = verify_aq4_matvec_batch_output_sampled(
            "self-attn layer batch MLP up projection",
            up_matrix,
            post_normed,
            mlp_up_output,
            sequence_len,
            &mut up_row_buffer,
            &mut stream,
            3e-3_f32,
            2e-5_f32,
        )?;
        let activation_max_abs_diff = verify_silu_mul_f32_close(
            "self-attn layer batch MLP activation",
            mlp_gate_output,
            mlp_up_output,
            mlp_activation,
            1e-4_f32,
            1e-5_f32,
        )?;
        let mut down_row_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                down_matrix.cols,
                "self-attn layer batch sampled down projection row",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn layer batch sampled down row: {err}")
            })?;
        let (down_checked_values, down_max_abs_diff) = verify_aq4_matvec_batch_output_sampled(
            "self-attn layer batch MLP down projection",
            down_matrix,
            mlp_activation,
            mlp_down_output,
            sequence_len,
            &mut down_row_buffer,
            &mut stream,
            3e-3_f32,
            2e-5_f32,
        )?;
        let layer_residual_max_abs_diff = verify_add_f32_close(
            "self-attn layer batch residual add",
            mlp_down_output,
            block_output,
            layer_output,
            1e-4_f32,
            1e-5_f32,
        )?;
        Some((
            post_norm_max_abs_diff,
            gate_checked_values,
            gate_max_abs_diff,
            up_checked_values,
            up_max_abs_diff,
            activation_max_abs_diff,
            down_checked_values,
            down_max_abs_diff,
            layer_residual_max_abs_diff,
        ))
    } else {
        None
    };
    let verification_wall_ms = verification_started.elapsed().as_secs_f64() * 1000.0;

    let output_elements = input_normed
        .len()
        .checked_add(q_projected.len())
        .and_then(|value| value.checked_add(k_projected.len()))
        .and_then(|value| value.checked_add(v_projected.len()))
        .and_then(|value| value.checked_add(q_gate.len()))
        .and_then(|value| value.checked_add(q_rope.len()))
        .and_then(|value| value.checked_add(k_rope.len()))
        .and_then(|value| {
            value.checked_add(
                attention_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                attention_projection_input
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                attn_projected
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                block_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(post_normed.as_ref().map(|output| output.len()).unwrap_or(0))
        })
        .and_then(|value| {
            value.checked_add(
                mlp_gate_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                mlp_up_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                mlp_activation
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                mlp_down_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .and_then(|value| {
            value.checked_add(
                layer_output
                    .as_ref()
                    .map(|output| output.len())
                    .unwrap_or(0),
            )
        })
        .ok_or_else(|| "self-attn qkv RoPE batch output element count overflows".to_string())?;
    let preview_len = q_rope.len().min(8);
    let executor = if stage.include_layer() {
        "segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+aq4_matvec_batch_f32+add_f32+segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32"
    } else if stage.include_block() {
        "segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+aq4_matvec_batch_f32+add_f32"
    } else if stage.include_attention() {
        "segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32"
    } else {
        "segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32"
    };
    if let Some(layer_output) = layer_output.as_ref() {
        let attention_output = attention_output.as_ref().ok_or_else(|| {
            "self-attn layer batch attention output missing for report".to_string()
        })?;
        let attention_projection_input = attention_projection_input.as_ref().ok_or_else(|| {
            "self-attn layer batch projection input missing for report".to_string()
        })?;
        let attn_projected = attn_projected.as_ref().ok_or_else(|| {
            "self-attn layer batch projected output missing for report".to_string()
        })?;
        let block_output = block_output
            .as_ref()
            .ok_or_else(|| "self-attn layer batch block output missing for report".to_string())?;
        let (attention_verification_mode, attention_checked_values, attention_max_abs_diff) =
            attention_verification.ok_or_else(|| {
                "self-attn layer batch attention verification summary missing".to_string()
            })?;
        let (
            output_gate_max_abs_diff,
            o_proj_checked_values,
            o_proj_max_abs_diff,
            block_max_abs_diff,
        ) = block_verification.ok_or_else(|| {
            "self-attn layer batch block verification summary missing".to_string()
        })?;
        let (
            post_norm_max_abs_diff,
            gate_checked_values,
            gate_max_abs_diff,
            up_checked_values,
            up_max_abs_diff,
            activation_max_abs_diff,
            down_checked_values,
            down_max_abs_diff,
            layer_residual_max_abs_diff,
        ) = layer_verification
            .ok_or_else(|| "self-attn layer batch verification summary missing".to_string())?;
        return Ok(format!(
            "{} package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} intermediate={} q_rows={} k_rows={} v_rows={} o_rows={} gate_rows={} down_rows={} q_projection_layout={} q_gate_elements={} output_gate_layout=qwen3.5-sigmoid q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} input_elements={} intermediate_elements={} output_elements={} executor={} real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} verification_wall_ms={verification_wall_ms:.6} token_tps_mean={} output_element_tps_mean={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} q_gate_max_abs_diff={q_gate_max_abs_diff:.9} q_rope_abs_floor={rope_abs_floor:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_abs_floor={rope_abs_floor:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_verification={} attention_checked_values={} attention_max_abs_diff={:.9} output_gate_max_abs_diff={output_gate_max_abs_diff:.9} o_proj_verification=sampled o_proj_checked_values={} o_proj_max_abs_diff={o_proj_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} mlp_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_gate_verification=sampled mlp_gate_checked_values={} mlp_gate_max_abs_diff={gate_max_abs_diff:.9} mlp_up_verification=sampled mlp_up_checked_values={} mlp_up_max_abs_diff={up_max_abs_diff:.9} mlp_activation_max_abs_diff={activation_max_abs_diff:.9} mlp_down_verification=sampled mlp_down_checked_values={} mlp_down_max_abs_diff={down_max_abs_diff:.9} layer_residual_max_abs_diff={layer_residual_max_abs_diff:.9} q_rope_preview={} attention_preview={} projection_input_preview={} projected_preview={} block_preview={} layer_preview={} verified=true",
            stage.smoke_name(),
            path,
            layer_index,
            input_norm_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            q_norm_tensor,
            k_norm_tensor,
            post_norm_tensor,
            gate_tensor,
            up_tensor,
            down_tensor,
            sequence_len,
            hidden,
            intermediate,
            q_matrix.rows,
            k_matrix.rows,
            v_matrix.rows,
            o_matrix.as_ref().map(|matrix| matrix.rows).unwrap_or(0),
            gate_matrix.as_ref().map(|matrix| matrix.rows).unwrap_or(0),
            down_matrix.as_ref().map(|matrix| matrix.rows).unwrap_or(0),
            q_split.layout,
            expected_q_gate.len(),
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            input_elements,
            intermediate_elements,
            output_elements,
            executor,
            sequence_len,
            info.backend,
            device_index,
            info.name,
            measured_repeats,
            wall_ms,
            wall_ms_min,
            wall_ms_max,
            tps(sequence_len, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            tps(output_elements, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            attention_verification_mode,
            attention_checked_values,
            attention_max_abs_diff,
            o_proj_checked_values,
            gate_checked_values,
            up_checked_values,
            down_checked_values,
            format_f32_preview(&q_rope[..preview_len]),
            format_f32_preview(&attention_output[..attention_output.len().min(8)]),
            format_f32_preview(
                &attention_projection_input[..attention_projection_input.len().min(8)],
            ),
            format_f32_preview(&attn_projected[..attn_projected.len().min(8)]),
            format_f32_preview(&block_output[..block_output.len().min(8)]),
            format_f32_preview(&layer_output[..layer_output.len().min(8)]),
        ));
    }
    if let Some(block_output) = block_output.as_ref() {
        let attention_output = attention_output.as_ref().ok_or_else(|| {
            "self-attn block batch attention output missing for report".to_string()
        })?;
        let attention_projection_input = attention_projection_input.as_ref().ok_or_else(|| {
            "self-attn block batch projection input missing for report".to_string()
        })?;
        let attn_projected = attn_projected.as_ref().ok_or_else(|| {
            "self-attn block batch projected output missing for report".to_string()
        })?;
        let (attention_verification_mode, attention_checked_values, attention_max_abs_diff) =
            attention_verification.ok_or_else(|| {
                "self-attn block batch attention verification summary missing".to_string()
            })?;
        let (
            output_gate_max_abs_diff,
            o_proj_checked_values,
            o_proj_max_abs_diff,
            block_max_abs_diff,
        ) = block_verification
            .ok_or_else(|| "self-attn block batch verification summary missing".to_string())?;
        return Ok(format!(
            "{} package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} q_rows={} k_rows={} v_rows={} o_rows={} q_projection_layout={} q_gate_elements={} output_gate_layout=qwen3.5-sigmoid q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} input_elements={} output_elements={} executor={} real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} verification_wall_ms={verification_wall_ms:.6} token_tps_mean={} output_element_tps_mean={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} q_gate_max_abs_diff={q_gate_max_abs_diff:.9} q_rope_abs_floor={rope_abs_floor:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_abs_floor={rope_abs_floor:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_verification={} attention_checked_values={} attention_max_abs_diff={:.9} output_gate_max_abs_diff={output_gate_max_abs_diff:.9} o_proj_verification=sampled o_proj_checked_values={} o_proj_max_abs_diff={o_proj_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} q_rope_preview={} attention_preview={} projection_input_preview={} projected_preview={} block_preview={} verified=true",
            stage.smoke_name(),
            path,
            layer_index,
            input_norm_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            q_norm_tensor,
            k_norm_tensor,
            sequence_len,
            hidden,
            q_matrix.rows,
            k_matrix.rows,
            v_matrix.rows,
            o_matrix.as_ref().map(|matrix| matrix.rows).unwrap_or(0),
            q_split.layout,
            expected_q_gate.len(),
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            input_elements,
            output_elements,
            executor,
            sequence_len,
            info.backend,
            device_index,
            info.name,
            measured_repeats,
            wall_ms,
            wall_ms_min,
            wall_ms_max,
            tps(sequence_len, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            tps(output_elements, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            attention_verification_mode,
            attention_checked_values,
            attention_max_abs_diff,
            o_proj_checked_values,
            format_f32_preview(&q_rope[..preview_len]),
            format_f32_preview(&attention_output[..attention_output.len().min(8)]),
            format_f32_preview(
                &attention_projection_input[..attention_projection_input.len().min(8)],
            ),
            format_f32_preview(&attn_projected[..attn_projected.len().min(8)]),
            format_f32_preview(&block_output[..block_output.len().min(8)]),
        ));
    }
    if let Some(attention_output) = attention_output.as_ref() {
        let (attention_verification_mode, attention_checked_values, attention_max_abs_diff) =
            attention_verification.ok_or_else(|| {
                "self-attn attention batch verification summary missing".to_string()
            })?;
        return Ok(format!(
            "{} package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} q_rows={} k_rows={} v_rows={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} input_elements={} output_elements={} executor={} real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} verification_wall_ms={verification_wall_ms:.6} token_tps_mean={} output_element_tps_mean={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} q_gate_max_abs_diff={q_gate_max_abs_diff:.9} q_rope_abs_floor={rope_abs_floor:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_abs_floor={rope_abs_floor:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_verification={} attention_checked_values={} attention_max_abs_diff={:.9} q_rope_preview={} v_preview={} attention_preview={} verified=true",
            stage.smoke_name(),
            path,
            layer_index,
            input_norm_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            q_norm_tensor,
            k_norm_tensor,
            sequence_len,
            hidden,
            q_matrix.rows,
            k_matrix.rows,
            v_matrix.rows,
            q_split.layout,
            expected_q_gate.len(),
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            input_elements,
            output_elements,
            executor,
            sequence_len,
            info.backend,
            device_index,
            info.name,
            measured_repeats,
            wall_ms,
            wall_ms_min,
            wall_ms_max,
            tps(sequence_len, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            tps(output_elements, wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            attention_verification_mode,
            attention_checked_values,
            attention_max_abs_diff,
            format_f32_preview(&q_rope[..preview_len]),
            format_f32_preview(&v_projected[..v_projected.len().min(8)]),
            format_f32_preview(&attention_output[..attention_output.len().min(8)]),
        ));
    }
    Ok(format!(
        "{} package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} q_rows={} k_rows={} v_rows={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} input_elements={} output_elements={} executor={} real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} verification_wall_ms={verification_wall_ms:.6} token_tps_mean={} output_element_tps_mean={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} q_gate_max_abs_diff={q_gate_max_abs_diff:.9} q_rope_abs_floor={rope_abs_floor:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_abs_floor={rope_abs_floor:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} q_rope_preview={} v_preview={} verified=true",
        stage.smoke_name(),
        path,
        layer_index,
        input_norm_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        q_norm_tensor,
        k_norm_tensor,
        sequence_len,
        hidden,
        q_matrix.rows,
        k_matrix.rows,
        v_matrix.rows,
        q_split.layout,
        expected_q_gate.len(),
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        input_elements,
        output_elements,
        executor,
        sequence_len,
        info.backend,
        device_index,
        info.name,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(sequence_len, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        tps(output_elements, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        format_f32_preview(&q_rope[..preview_len]),
        format_f32_preview(&v_projected[..v_projected.len().min(8)]),
    ))
}
