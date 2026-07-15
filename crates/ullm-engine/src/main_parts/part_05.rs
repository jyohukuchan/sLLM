const PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION: &str = "package-row-scale-overrides-v0.1";
const PACKAGE_CELL_DELTA_OVERRIDES_SCHEMA_VERSION: &str = "package-cell-delta-overrides-v0.1";

#[derive(Debug, Clone)]
struct PackageRowScaleOverrides {
    source_path: String,
    overrides: Vec<PackageRowScaleOverride>,
}
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageRowScaleOverridesFile {
    schema_version: String,
    overrides: Vec<PackageRowScaleOverride>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageRowScaleOverride {
    layer_index: usize,
    tensor_suffix: String,
    row_index: usize,
    scale: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AppliedPackageRowScaleOverride {
    layer_index: usize,
    tensor_name: String,
    tensor_suffix: String,
    row_index: usize,
    scale: f32,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone)]
struct PackageCellDeltaOverrides {
    source_path: String,
    overrides: Vec<PackageCellDeltaOverride>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageCellDeltaOverridesFile {
    schema_version: String,
    overrides: Vec<PackageCellDeltaOverride>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageCellDeltaOverride {
    layer_index: usize,
    tensor_suffix: String,
    row_index: usize,
    col_index: usize,
    delta: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AppliedPackageCellDeltaOverride {
    layer_index: usize,
    tensor_name: String,
    tensor_suffix: String,
    row_index: usize,
    col_index: usize,
    delta: f32,
    previous_value: f32,
    new_value: f32,
    rows: usize,
    cols: usize,
}

fn load_package_row_scale_overrides(
    path: Option<&str>,
) -> Result<Option<PackageRowScaleOverrides>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_empty() || path == "none" {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read row scale overrides JSON {path}: {err}"))?;
    let parsed: PackageRowScaleOverridesFile = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse row scale overrides JSON {path}: {err}"))?;
    if parsed.schema_version != PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION {
        return Err(format!(
            "row scale overrides schema_version must be {}, got {}",
            PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION, parsed.schema_version
        ));
    }

    let mut seen = std::collections::BTreeSet::<(usize, String, usize)>::new();
    for override_entry in &parsed.overrides {
        validate_package_row_scale_override(override_entry)?;
        let key = (
            override_entry.layer_index,
            override_entry.tensor_suffix.clone(),
            override_entry.row_index,
        );
        if !seen.insert(key) {
            return Err(format!(
                "duplicate row scale override: layer={} tensor_suffix={} row={}",
                override_entry.layer_index, override_entry.tensor_suffix, override_entry.row_index
            ));
        }
    }

    Ok(Some(PackageRowScaleOverrides {
        source_path: path.to_string(),
        overrides: parsed.overrides,
    }))
}

fn load_package_cell_delta_overrides(
    path: Option<&str>,
) -> Result<Option<PackageCellDeltaOverrides>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_empty() || path == "none" {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read cell delta overrides JSON {path}: {err}"))?;
    let parsed: PackageCellDeltaOverridesFile = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse cell delta overrides JSON {path}: {err}"))?;
    if parsed.schema_version != PACKAGE_CELL_DELTA_OVERRIDES_SCHEMA_VERSION {
        return Err(format!(
            "cell delta overrides schema_version must be {}, got {}",
            PACKAGE_CELL_DELTA_OVERRIDES_SCHEMA_VERSION, parsed.schema_version
        ));
    }

    let mut seen = std::collections::BTreeSet::<(usize, String, usize, usize)>::new();
    for override_entry in &parsed.overrides {
        validate_package_cell_delta_override(override_entry)?;
        let key = (
            override_entry.layer_index,
            override_entry.tensor_suffix.clone(),
            override_entry.row_index,
            override_entry.col_index,
        );
        if !seen.insert(key) {
            return Err(format!(
                "duplicate cell delta override: layer={} tensor_suffix={} row={} col={}",
                override_entry.layer_index,
                override_entry.tensor_suffix,
                override_entry.row_index,
                override_entry.col_index
            ));
        }
    }

    Ok(Some(PackageCellDeltaOverrides {
        source_path: path.to_string(),
        overrides: parsed.overrides,
    }))
}

fn validate_package_row_scale_override(
    override_entry: &PackageRowScaleOverride,
) -> Result<(), String> {
    if !matches!(
        override_entry.tensor_suffix.as_str(),
        "linear_attn.out_proj.weight" | "self_attn.o_proj.weight" | "mlp.down_proj.weight"
    ) {
        return Err(format!(
            "unsupported row scale override tensor_suffix={}; expected linear_attn.out_proj.weight, self_attn.o_proj.weight, or mlp.down_proj.weight",
            override_entry.tensor_suffix
        ));
    }
    if !override_entry.scale.is_finite() || override_entry.scale <= 0.0 {
        return Err(format!(
            "row scale override must be finite and positive: layer={} tensor_suffix={} row={} scale={}",
            override_entry.layer_index,
            override_entry.tensor_suffix,
            override_entry.row_index,
            override_entry.scale
        ));
    }
    Ok(())
}

fn validate_package_cell_delta_override(
    override_entry: &PackageCellDeltaOverride,
) -> Result<(), String> {
    if !matches!(
        override_entry.tensor_suffix.as_str(),
        "linear_attn.out_proj.weight"
            | "self_attn.o_proj.weight"
            | "linear_attn.in_proj_qkv.weight"
            | "mlp.down_proj.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
    ) {
        return Err(format!(
            "unsupported cell delta override tensor_suffix={}; expected linear_attn.in_proj_qkv.weight, linear_attn.out_proj.weight, self_attn.o_proj.weight, mlp.down_proj.weight, mlp.gate_proj.weight, or mlp.up_proj.weight",
            override_entry.tensor_suffix
        ));
    }
    if !override_entry.delta.is_finite() {
        return Err(format!(
            "cell delta override must be finite: layer={} tensor_suffix={} row={} col={} delta={}",
            override_entry.layer_index,
            override_entry.tensor_suffix,
            override_entry.row_index,
            override_entry.col_index,
            override_entry.delta
        ));
    }
    Ok(())
}

fn matching_package_row_scale_overrides(
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    layer_index: usize,
    tensor_suffix: &str,
) -> Vec<PackageRowScaleOverride> {
    row_scale_overrides
        .map(|overrides| {
            overrides
                .overrides
                .iter()
                .filter(|override_entry| {
                    override_entry.layer_index == layer_index
                        && override_entry.tensor_suffix == tensor_suffix
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn matching_package_cell_delta_overrides(
    cell_delta_overrides: Option<&PackageCellDeltaOverrides>,
    layer_index: usize,
    tensor_suffix: &str,
) -> Vec<PackageCellDeltaOverride> {
    cell_delta_overrides
        .map(|overrides| {
            overrides
                .overrides
                .iter()
                .filter(|override_entry| {
                    override_entry.layer_index == layer_index
                        && override_entry.tensor_suffix == tensor_suffix
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn apply_package_row_scale_overrides_to_matrix(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &mut ullm_runtime_sys::RuntimeBuffer,
    rows: usize,
    cols: usize,
    tensor_name: &str,
    overrides: &[PackageRowScaleOverride],
) -> Result<Vec<AppliedPackageRowScaleOverride>, String> {
    if overrides.is_empty() {
        return Ok(Vec::new());
    }
    if rows == 0 || cols == 0 {
        return Err(format!(
            "cannot apply row scale overrides to empty matrix {tensor_name}: rows={rows} cols={cols}"
        ));
    }
    let matrix_bytes_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!("row scale override matrix byte size overflows for {tensor_name}")
        })?;
    let mut matrix_bytes = vec![0_u8; matrix_bytes_len];
    matrix
        .copy_to_host(0, &mut matrix_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {tensor_name} for row scale overrides: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {tensor_name} override copy: {err}"))?;

    let mut applied = Vec::with_capacity(overrides.len());
    for override_entry in overrides {
        if override_entry.row_index >= rows {
            return Err(format!(
                "row scale override row out of range for {tensor_name}: row={} rows={rows}",
                override_entry.row_index
            ));
        }
        let row_start = override_entry
            .row_index
            .checked_mul(cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| format!("row scale override row offset overflows for {tensor_name}"))?;
        let row_end = row_start
            .checked_add(
                cols.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        format!("row scale override row byte size overflows for {tensor_name}")
                    })?,
            )
            .ok_or_else(|| format!("row scale override row end overflows for {tensor_name}"))?;
        for offset in (row_start..row_end).step_by(std::mem::size_of::<f32>()) {
            let mut raw = [0_u8; 4];
            raw.copy_from_slice(&matrix_bytes[offset..offset + 4]);
            let scaled = f32::from_le_bytes(raw) * override_entry.scale;
            matrix_bytes[offset..offset + 4].copy_from_slice(&scaled.to_le_bytes());
        }
        applied.push(AppliedPackageRowScaleOverride {
            layer_index: override_entry.layer_index,
            tensor_name: tensor_name.to_string(),
            tensor_suffix: override_entry.tensor_suffix.clone(),
            row_index: override_entry.row_index,
            scale: override_entry.scale,
            rows,
            cols,
        });
    }

    matrix
        .copy_from_host(0, &matrix_bytes, Some(stream))
        .map_err(|err| format!("failed to copy row-scaled {tensor_name} back to runtime: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after row-scaled {tensor_name} copy back: {err}")
    })?;
    Ok(applied)
}

fn apply_package_cell_delta_overrides_to_matrix(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &mut ullm_runtime_sys::RuntimeBuffer,
    rows: usize,
    cols: usize,
    tensor_name: &str,
    overrides: &[PackageCellDeltaOverride],
) -> Result<Vec<AppliedPackageCellDeltaOverride>, String> {
    if overrides.is_empty() {
        return Ok(Vec::new());
    }
    if rows == 0 || cols == 0 {
        return Err(format!(
            "cannot apply cell delta overrides to empty matrix {tensor_name}: rows={rows} cols={cols}"
        ));
    }
    let matrix_bytes_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!("cell delta override matrix byte size overflows for {tensor_name}")
        })?;
    let mut matrix_bytes = vec![0_u8; matrix_bytes_len];
    matrix
        .copy_to_host(0, &mut matrix_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {tensor_name} for cell delta overrides: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after {tensor_name} cell override copy: {err}")
    })?;

    let mut applied = Vec::with_capacity(overrides.len());
    for override_entry in overrides {
        if override_entry.row_index >= rows {
            return Err(format!(
                "cell delta override row out of range for {tensor_name}: row={} rows={rows}",
                override_entry.row_index
            ));
        }
        if override_entry.col_index >= cols {
            return Err(format!(
                "cell delta override column out of range for {tensor_name}: col={} cols={cols}",
                override_entry.col_index
            ));
        }
        let element_index = override_entry
            .row_index
            .checked_mul(cols)
            .and_then(|value| value.checked_add(override_entry.col_index))
            .ok_or_else(|| {
                format!("cell delta override element index overflows for {tensor_name}")
            })?;
        let offset = element_index
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("cell delta override byte offset overflows for {tensor_name}")
            })?;
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(&matrix_bytes[offset..offset + 4]);
        let previous_value = f32::from_le_bytes(raw);
        let new_value = previous_value + override_entry.delta;
        if !new_value.is_finite() {
            return Err(format!(
                "cell delta override produced non-finite value for {tensor_name}: layer={} row={} col={} previous={} delta={}",
                override_entry.layer_index,
                override_entry.row_index,
                override_entry.col_index,
                previous_value,
                override_entry.delta
            ));
        }
        matrix_bytes[offset..offset + 4].copy_from_slice(&new_value.to_le_bytes());
        applied.push(AppliedPackageCellDeltaOverride {
            layer_index: override_entry.layer_index,
            tensor_name: tensor_name.to_string(),
            tensor_suffix: override_entry.tensor_suffix.clone(),
            row_index: override_entry.row_index,
            col_index: override_entry.col_index,
            delta: override_entry.delta,
            previous_value,
            new_value,
            rows,
            cols,
        });
    }

    matrix
        .copy_from_host(0, &matrix_bytes, Some(stream))
        .map_err(|err| {
            format!("failed to copy cell-delta-adjusted {tensor_name} back to runtime: {err}")
        })?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after cell-delta-adjusted {tensor_name} copy back: {err}")
    })?;
    Ok(applied)
}

struct PackageLinearAttnMlpBlockSequenceRun {
    line: String,
    applied_row_scale_overrides: Vec<AppliedPackageRowScaleOverride>,
    applied_cell_delta_overrides: Vec<AppliedPackageCellDeltaOverride>,
    attention_input_normed: Vec<f32>,
    attention_qkv_projection: Vec<f32>,
    attention_qkv_projection_dim: usize,
    attention_z_projection: Vec<f32>,
    attention_gate_silu: Vec<f32>,
    attention_a_projection: Vec<f32>,
    attention_b_projection: Vec<f32>,
    attention_gate_dim: usize,
    attention_conv_pre_silu: Vec<f32>,
    attention_conv: Vec<f32>,
    attention_recurrent_q: Vec<f32>,
    attention_recurrent_k: Vec<f32>,
    attention_recurrent_v: Vec<f32>,
    attention_recurrent_qk_dim: usize,
    attention_gate: Vec<f32>,
    attention_beta: Vec<f32>,
    attention_recurrent: Vec<f32>,
    attention_normed: Vec<f32>,
    attention_projection_input: Vec<f32>,
    attention_output: Vec<f32>,
    attention_block_output: Vec<f32>,
    post_normed: Vec<f32>,
    mlp_gate_projection: Vec<f32>,
    mlp_gate_silu: Vec<f32>,
    mlp_up_projection: Vec<f32>,
    mlp_activation: Vec<f32>,
    mlp_intermediate: usize,
    mlp_output: Vec<f32>,
    layer_output: Vec<f32>,
}

fn package_linear_attn_mlp_block_sequence_run(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    residual_sequence: Vec<f32>,
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    cell_delta_overrides: Option<&PackageCellDeltaOverrides>,
) -> Result<PackageLinearAttnMlpBlockSequenceRun, String> {
    package_linear_attn_mlp_block_sequence_run_with_diagnostic_inputs(
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        residual_sequence,
        row_scale_overrides,
        cell_delta_overrides,
        None,
        None,
    )
}

/// Diagnostic-only extension of the production sequence path.
///
/// The normal worker never calls this function.  `input_normed_override` is
/// an explicit fixed-input probe hook and `z_projection_override` is applied
/// only after the production Z matvec has completed and immediately before
/// the production SiLU-mul.  Both options are `None` for every normal path.
fn package_linear_attn_mlp_block_sequence_run_with_diagnostic_inputs(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    residual_sequence: Vec<f32>,
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    cell_delta_overrides: Option<&PackageCellDeltaOverrides>,
    input_normed_override: Option<&[f32]>,
    z_projection_override: Option<&[f32]>,
) -> Result<PackageLinearAttnMlpBlockSequenceRun, String> {
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden;
    let qkv_rows_expected = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let input_epsilon = 1e-6_f32;
    let mlp_epsilon = 1e-5_f32;

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
    if input_norm.values.len() != hidden {
        return Err(format!(
            "input RMSNorm length must match hidden={hidden}: len={}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "conv1d channel count is too large for this host".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "conv1d kernel size is too large for this host".to_string())?;
    if conv_channels != qkv_rows_expected {
        return Err(format!(
            "conv1d channels must match q/k/v layout: conv_channels={conv_channels}, expected={qkv_rows_expected}"
        ));
    }
    if conv.values.len() != conv_channels * kernel_size {
        return Err(format!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv.values.len()
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)?;
    if a_log.values.len() != value_heads {
        return Err(format!(
            "A_log length must match value_heads={value_heads}: len={}",
            a_log.values.len()
        ));
    }
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)?;
    if dt_bias.values.len() != value_heads {
        return Err(format!(
            "dt_bias length must match value_heads={value_heads}: len={}",
            dt_bias.values.len()
        ));
    }
    let attn_norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)?;
    if attn_norm.values.len() != value_dim {
        return Err(format!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            attn_norm.values.len()
        ));
    }
    let post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
    if post_norm.values.len() != hidden {
        return Err(format!(
            "post RMSNorm length must match hidden={hidden}: len={}",
            post_norm.values.len()
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

    let hidden_bytes = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "hidden byte size overflows".to_string())?;
    let hidden_sequence_bytes = hidden_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "hidden sequence byte size overflows".to_string())?;
    let qkv_step_bytes = qkv_rows_expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "qkv step byte size overflows".to_string())?;
    let qkv_sequence_bytes = qkv_step_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "qkv sequence byte size overflows".to_string())?;
    let gate_beta_step_bytes = value_heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "gate/beta step byte size overflows".to_string())?;
    let gate_beta_sequence_bytes = gate_beta_step_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "gate/beta sequence byte size overflows".to_string())?;

    if residual_sequence.len() != sequence_len * hidden {
        return Err(format!(
            "linear attention residual sequence length mismatch for layer {layer_index}: got {} expected {}",
            residual_sequence.len(),
            sequence_len * hidden
        ));
    }
    let input_norm_weight_values =
        effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    let post_norm_weight_values =
        effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    let input_norm_weight_bytes = encode_f32_to_bytes(&input_norm_weight_values);
    let conv_weight_bytes = encode_f32_to_bytes(&conv.values);
    let a_log_bytes = encode_f32_to_bytes(&a_log.values);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias.values);
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm_weight_values);
    let mut applied_row_scale_overrides = Vec::new();
    let mut applied_cell_delta_overrides = Vec::new();

    let mut input_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(input_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate input RMSNorm weight buffer: {err}"))?;
    let mut input_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input RMSNorm output buffer: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input norm weight copy: {err}"))?;

    let (input_normed, input_norm_max_abs_diff, input_normed_sequence_bytes) =
        if let Some(override_values) = input_normed_override {
            if override_values.len() != sequence_len * hidden
                || override_values.iter().any(|value| !value.is_finite())
            {
                return Err(format!(
                    "diagnostic input RMSNorm override length/finite mismatch: got {} expected {}",
                    override_values.len(),
                    sequence_len * hidden
                ));
            }
            (override_values.to_vec(), 0.0_f32, encode_f32_to_bytes(override_values))
        } else {
            let mut expected_input_normed = Vec::with_capacity(sequence_len * hidden);
            let mut input_normed_sequence_bytes = vec![0_u8; hidden_sequence_bytes];
            for timestep in 0..sequence_len {
                let residual_start = timestep * hidden;
                let residual_end = residual_start + hidden;
                let residual = &residual_sequence[residual_start..residual_end];
                let residual_bytes = encode_f32_to_bytes(residual);
                input_buffer
                    .copy_from_host(0, &residual_bytes, Some(&mut stream))
                    .map_err(|err| {
                        format!("failed to copy residual timestep {timestep} into runtime buffer: {err}")
                    })?;
                ullm_runtime_sys::rmsnorm_f32(
                    &input_buffer,
                    &input_norm_weight_buffer,
                    hidden,
                    input_epsilon,
                    &mut input_normed_buffer,
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to run input RMSNorm timestep {timestep}: {err}"))?;
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize after input RMSNorm timestep {timestep}: {err}")
                })?;
                let byte_start = timestep * hidden_bytes;
                let byte_end = byte_start + hidden_bytes;
                input_normed_buffer
                    .copy_to_host(
                        0,
                        &mut input_normed_sequence_bytes[byte_start..byte_end],
                        Some(&mut stream),
                    )
                    .map_err(|err| {
                        format!("failed to copy input RMSNorm timestep {timestep} to host: {err}")
                    })?;
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize after input RMSNorm host copy {timestep}: {err}")
                })?;
                let expected = runtime_host_rmsnorm_f32(residual, &input_norm_weight_values, input_epsilon);
                expected_input_normed.extend_from_slice(&expected);
            }
            let input_normed = decode_f32_le_values(&input_normed_sequence_bytes);
            let input_norm_max_abs_diff = verify_f32_close(
                "package-linear-attn-mlp-block-smoke input RMSNorm",
                &input_normed,
                &expected_input_normed,
                1e-4,
                1e-5,
            )?;
            (input_normed, input_norm_max_abs_diff, input_normed_sequence_bytes)
        };
    input_normed_buffer
        .copy_from_host(0, &input_normed_sequence_bytes[..hidden_bytes], Some(&mut stream))
        .map_err(|err| format!("failed to seed diagnostic input norm buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize diagnostic input norm seed: {err}"))?;

    let (
        attention_block_output,
        qkv_output,
        z_output,
        a_output,
        b_output,
        conv_output,
        conv_activated,
        recurrent_q,
        recurrent_k,
        recurrent_v,
        gate_output,
        beta_output,
        recurrent_output,
        attn_normed,
        attn_activated,
        attn_output,
        attn_block_max_abs_diff,
        conv_max_abs_diff,
        gate_beta_max_abs_diff,
        recurrent_max_abs_diff,
        attn_norm_max_abs_diff,
        attn_activation_max_abs_diff,
        attn_output_max_abs_diff,
    ) = {
        let mut registry = WeightRegistry::new();
        let (qkv_rows, qkv_cols, mut qkv_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &qkv_tensor,
            chunk_bytes,
        )?;
        let (a_rows, a_cols, a_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &a_tensor,
            chunk_bytes,
        )?;
        let (b_rows, b_cols, b_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &b_tensor,
            chunk_bytes,
        )?;
        let (z_rows, z_cols, z_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &z_tensor,
            chunk_bytes,
        )?;
        let (out_rows, out_cols, mut out_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &out_tensor,
            chunk_bytes,
        )?;
        if qkv_rows != qkv_rows_expected || qkv_cols != hidden {
            return Err(format!(
                "qkv shape must be [{qkv_rows_expected},{hidden}], got [{qkv_rows},{qkv_cols}]"
            ));
        }
        if a_rows != value_heads || b_rows != value_heads || a_cols != hidden || b_cols != hidden {
            return Err(format!(
                "a/b shape must be [{value_heads},{hidden}], got a=[{a_rows},{a_cols}] b=[{b_rows},{b_cols}]"
            ));
        }
        if z_rows != hidden || z_cols != hidden || out_rows != hidden || out_cols != hidden {
            return Err(format!(
                "z/out shape must be [{hidden},{hidden}], got z=[{z_rows},{z_cols}] out=[{out_rows},{out_cols}]"
            ));
        }
        let qkv_cell_delta_overrides = matching_package_cell_delta_overrides(
            cell_delta_overrides,
            layer_index,
            "linear_attn.in_proj_qkv.weight",
        );
        applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
            &mut stream,
            &mut qkv_matrix,
            qkv_rows,
            qkv_cols,
            &qkv_tensor,
            &qkv_cell_delta_overrides,
        )?);
        let out_row_scale_overrides = matching_package_row_scale_overrides(
            row_scale_overrides,
            layer_index,
            "linear_attn.out_proj.weight",
        );
        applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
            &mut stream,
            &mut out_matrix,
            out_rows,
            out_cols,
            &out_tensor,
            &out_row_scale_overrides,
        )?);
        let out_cell_delta_overrides = matching_package_cell_delta_overrides(
            cell_delta_overrides,
            layer_index,
            "linear_attn.out_proj.weight",
        );
        applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
            &mut stream,
            &mut out_matrix,
            out_rows,
            out_cols,
            &out_tensor,
            &out_cell_delta_overrides,
        )?);

        let mut qkv_step_buffer = context
            .alloc_buffer(qkv_step_bytes)
            .map_err(|err| format!("failed to allocate qkv step buffer: {err}"))?;
        let mut a_step_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate a step buffer: {err}"))?;
        let mut b_step_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate b step buffer: {err}"))?;
        let mut z_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate z step buffer: {err}"))?;
        let mut qkv_sequence_bytes_host = vec![0_u8; qkv_sequence_bytes];
        let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut z_sequence_bytes = vec![0_u8; hidden_sequence_bytes];
        for timestep in 0..sequence_len {
            let hidden_start = timestep * hidden_bytes;
            let hidden_end = hidden_start + hidden_bytes;
            input_normed_buffer
                .copy_from_host(
                    0,
                    &input_normed_sequence_bytes[hidden_start..hidden_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy input normed timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &qkv_matrix,
                &input_normed_buffer,
                qkv_rows,
                qkv_cols,
                &mut qkv_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run qkv matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &a_matrix,
                &input_normed_buffer,
                a_rows,
                a_cols,
                &mut a_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run a matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &b_matrix,
                &input_normed_buffer,
                b_rows,
                b_cols,
                &mut b_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run b matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &z_matrix,
                &input_normed_buffer,
                z_rows,
                z_cols,
                &mut z_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run z matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after projections timestep {timestep}: {err}")
            })?;
            let qkv_start = timestep * qkv_step_bytes;
            let qkv_end = qkv_start + qkv_step_bytes;
            qkv_step_buffer
                .copy_to_host(
                    0,
                    &mut qkv_sequence_bytes_host[qkv_start..qkv_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy qkv timestep {timestep}: {err}"))?;
            let gate_start = timestep * gate_beta_step_bytes;
            let gate_end = gate_start + gate_beta_step_bytes;
            a_step_buffer
                .copy_to_host(
                    0,
                    &mut a_sequence_bytes[gate_start..gate_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy a timestep {timestep}: {err}"))?;
            b_step_buffer
                .copy_to_host(
                    0,
                    &mut b_sequence_bytes[gate_start..gate_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy b timestep {timestep}: {err}"))?;
            z_step_buffer
                .copy_to_host(
                    0,
                    &mut z_sequence_bytes[hidden_start..hidden_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy z timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after projection copies timestep {timestep}: {err}")
            })?;
        }

        let qkv_output = decode_f32_le_values(&qkv_sequence_bytes_host);
        let a_output = decode_f32_le_values(&a_sequence_bytes);
        let b_output = decode_f32_le_values(&b_sequence_bytes);
        let mut z_output = decode_f32_le_values(&z_sequence_bytes);
        if let Some(override_values) = z_projection_override {
            if override_values.len() != sequence_len * hidden
                || override_values.iter().any(|value| !value.is_finite())
            {
                return Err(format!(
                    "diagnostic Z projection override length/finite mismatch: got {} expected {}",
                    override_values.len(),
                    sequence_len * hidden
                ));
            }
            z_output.copy_from_slice(override_values);
        }
        let z_sequence_bytes = encode_f32_to_bytes(&z_output);
        let mut qkv_sequence_buffer = context
            .alloc_buffer(qkv_sequence_bytes)
            .map_err(|err| format!("failed to allocate qkv sequence buffer: {err}"))?;
        let mut conv_weight_buffer = context
            .alloc_buffer(conv_weight_bytes.len())
            .map_err(|err| format!("failed to allocate conv1d weight buffer: {err}"))?;
        let mut conv_output_buffer = context
            .alloc_buffer(qkv_sequence_bytes)
            .map_err(|err| format!("failed to allocate conv1d output buffer: {err}"))?;
        qkv_sequence_buffer
            .copy_from_host(0, &qkv_sequence_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy qkv sequence into runtime buffer: {err}"))?;
        conv_weight_buffer
            .copy_from_host(0, &conv_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d weight into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d input copy: {err}"))?;
        ullm_runtime_sys::depthwise_conv1d_f32(
            &qkv_sequence_buffer,
            &conv_weight_buffer,
            qkv_rows,
            sequence_len,
            kernel_size,
            &mut conv_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run depthwise conv1d: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d: {err}"))?;
        let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes];
        conv_output_buffer
            .copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d host copy: {err}"))?;
        let conv_output = decode_f32_le_values(&conv_output_bytes);
        let expected_conv = runtime_host_depthwise_conv1d_f32(
            &qkv_output,
            &conv.values,
            qkv_rows,
            sequence_len,
            kernel_size,
        );
        let conv_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke conv1d",
            &conv_output,
            &expected_conv,
            1e-4,
            1e-5,
        )?;

        let mut a_sequence_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate a sequence buffer: {err}"))?;
        let mut b_sequence_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate b sequence buffer: {err}"))?;
        let mut a_log_buffer = context
            .alloc_buffer(a_log_bytes.len())
            .map_err(|err| format!("failed to allocate A_log buffer: {err}"))?;
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias_bytes.len())
            .map_err(|err| format!("failed to allocate dt_bias buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate gate output buffer: {err}"))?;
        let mut beta_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate beta output buffer: {err}"))?;
        a_sequence_buffer
            .copy_from_host(0, &a_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy a sequence into runtime buffer: {err}"))?;
        b_sequence_buffer
            .copy_from_host(0, &b_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy b sequence into runtime buffer: {err}"))?;
        a_log_buffer
            .copy_from_host(0, &a_log_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy A_log into runtime buffer: {err}"))?;
        dt_bias_buffer
            .copy_from_host(0, &dt_bias_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy dt_bias into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta aux copy: {err}"))?;
        ullm_runtime_sys::linear_attn_gate_beta_f32(
            &a_sequence_buffer,
            &b_sequence_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            value_heads,
            sequence_len,
            &mut gate_buffer,
            &mut beta_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention gate/beta: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta: {err}"))?;
        let mut gate_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut beta_bytes = vec![0_u8; gate_beta_sequence_bytes];
        gate_buffer
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy gate output to host: {err}"))?;
        beta_buffer
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy beta output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta host copy: {err}"))?;
        let gate_output = decode_f32_le_values(&gate_bytes);
        let beta_output = decode_f32_le_values(&beta_bytes);
        let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
            &a_output,
            &b_output,
            &a_log.values,
            &dt_bias.values,
            value_heads,
            sequence_len,
        );
        let gate_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke gate",
            &gate_output,
            &expected_gate,
            1e-4,
            1e-5,
        )?;
        let beta_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke beta",
            &beta_output,
            &expected_beta,
            1e-4,
            1e-5,
        )?;
        let gate_beta_max_abs_diff = gate_max_abs_diff.max(beta_max_abs_diff);

        let conv_activated = runtime_host_silu_f32(&conv_output);
        let qkv_split = split_linear_attn_qkv_for_recurrent(
            &conv_activated,
            sequence_len,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            qk_l2_norm,
            q_scale,
        )
        .map_err(|err| format!("failed to split qkv for recurrent: {err}"))?;
        let recurrent_q = qkv_split.q.clone();
        let recurrent_k = qkv_split.k.clone();
        let recurrent_v = qkv_split.v.clone();
        let state_elements = value_heads
            .checked_mul(key_dim)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| "linear attention state element count overflows".to_string())?;
        let mut expected_state = vec![0.0_f32; state_elements];
        let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
            &qkv_split.q,
            &qkv_split.k,
            &qkv_split.v,
            &expected_gate,
            &expected_beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut expected_state,
        );
        let q_bytes = encode_f32_to_bytes(&qkv_split.q);
        let k_bytes = encode_f32_to_bytes(&qkv_split.k);
        let v_bytes = encode_f32_to_bytes(&qkv_split.v);
        let state_bytes = encode_f32_to_bytes(&vec![0.0_f32; state_elements]);
        let mut q_buffer = context
            .alloc_buffer(q_bytes.len())
            .map_err(|err| format!("failed to allocate q buffer: {err}"))?;
        let mut k_buffer = context
            .alloc_buffer(k_bytes.len())
            .map_err(|err| format!("failed to allocate k buffer: {err}"))?;
        let mut v_buffer = context
            .alloc_buffer(v_bytes.len())
            .map_err(|err| format!("failed to allocate v buffer: {err}"))?;
        let mut state_buffer = context
            .alloc_buffer(state_bytes.len())
            .map_err(|err| format!("failed to allocate recurrent state buffer: {err}"))?;
        let mut recurrent_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate recurrent output buffer: {err}"))?;
        q_buffer
            .copy_from_host(0, &q_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy q into runtime buffer: {err}"))?;
        k_buffer
            .copy_from_host(0, &k_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy k into runtime buffer: {err}"))?;
        v_buffer
            .copy_from_host(0, &v_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy v into runtime buffer: {err}"))?;
        state_buffer
            .copy_from_host(0, &state_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent state into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent input copy: {err}"))?;
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut recurrent_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention recurrent: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent: {err}"))?;
        let mut recurrent_bytes = vec![0_u8; hidden_sequence_bytes];
        recurrent_buffer
            .copy_to_host(0, &mut recurrent_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent host copy: {err}"))?;
        let recurrent_output = decode_f32_le_values(&recurrent_bytes);
        let recurrent_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke recurrent",
            &recurrent_output,
            &expected_recurrent,
            1e-3,
            1e-5,
        )?;

        let mut expected_attn_normed = vec![0.0_f32; sequence_len * hidden];
        for row in 0..(sequence_len * value_heads) {
            let start = row * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(
                &expected_recurrent[start..end],
                &attn_norm.values,
                input_epsilon,
            );
            expected_attn_normed[start..end].copy_from_slice(&normed);
        }
        let mut attn_norm_weight_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm weight buffer: {err}")
            })?;
        let mut attn_norm_input_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm input buffer: {err}")
            })?;
        let mut attn_norm_output_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm output buffer: {err}")
            })?;
        attn_norm_weight_buffer
            .copy_from_host(0, &attn_norm_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention norm weight: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention norm weight copy: {err}")
        })?;
        let mut attn_normed_bytes = vec![0_u8; hidden_sequence_bytes];
        for row in 0..(sequence_len * value_heads) {
            let start = row * value_dim;
            let byte_start = start * std::mem::size_of::<f32>();
            let byte_end = byte_start + attn_norm_weight_bytes.len();
            attn_norm_input_buffer
                .copy_from_host(0, &recurrent_bytes[byte_start..byte_end], Some(&mut stream))
                .map_err(|err| format!("failed to copy linear attention norm row {row}: {err}"))?;
            ullm_runtime_sys::rmsnorm_f32(
                &attn_norm_input_buffer,
                &attn_norm_weight_buffer,
                value_dim,
                input_epsilon,
                &mut attn_norm_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run linear attention norm row {row}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm row {row}: {err}")
            })?;
            attn_norm_output_buffer
                .copy_to_host(
                    0,
                    &mut attn_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy linear attention norm row {row}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm copy row {row}: {err}")
            })?;
        }
        let attn_normed = decode_f32_le_values(&attn_normed_bytes);
        let attn_norm_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention norm",
            &attn_normed,
            &expected_attn_normed,
            1e-3,
            1e-5,
        )?;
        let expected_attn_activated = runtime_host_silu_mul_f32(&z_output, &expected_attn_normed);
        let mut z_sequence_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate z sequence buffer: {err}"))?;
        let mut attn_normed_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate linear attention normed buffer: {err}"))?;
        let mut attn_activated_buffer =
            context.alloc_buffer(hidden_sequence_bytes).map_err(|err| {
                format!("failed to allocate linear attention activated buffer: {err}")
            })?;
        z_sequence_buffer
            .copy_from_host(0, &z_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy z sequence into runtime buffer: {err}"))?;
        attn_normed_buffer
            .copy_from_host(0, &attn_normed_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention normed values: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention normed copy: {err}")
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &z_sequence_buffer,
            &attn_normed_buffer,
            sequence_len * hidden,
            &mut attn_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention SiLU-mul: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention SiLU-mul: {err}")
        })?;
        let mut attn_activated_bytes = vec![0_u8; hidden_sequence_bytes];
        attn_activated_buffer
            .copy_to_host(0, &mut attn_activated_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention activated values: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after activated host copy: {err}"))?;
        let attn_activated = decode_f32_le_values(&attn_activated_bytes);
        let attn_activation_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention activation",
            &attn_activated,
            &expected_attn_activated,
            1e-3,
            1e-5,
        )?;

        let out_matrix_bytes_len = out_rows
            .checked_mul(out_cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "out projection matrix byte size overflows".to_string())?;
        let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
        out_matrix
            .copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy out projection matrix to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after out matrix copy: {err}"))?;
        let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
        let mut expected_attn_output = Vec::with_capacity(sequence_len * hidden);
        let mut attn_activated_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention activated step buffer: {err}"))?;
        let mut attn_output_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention output buffer: {err}"))?;
        let mut residual_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate residual step buffer: {err}"))?;
        let mut attn_block_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention block step buffer: {err}"))?;
        let mut attn_output_bytes = vec![0_u8; hidden_sequence_bytes];
        let mut attn_block_bytes = vec![0_u8; hidden_sequence_bytes];
        let residual_sequence_bytes = encode_f32_to_bytes(&residual_sequence);
        for timestep in 0..sequence_len {
            let element_start = timestep * hidden;
            let element_end = element_start + hidden;
            let byte_start = timestep * hidden_bytes;
            let byte_end = byte_start + hidden_bytes;
            let expected_step = runtime_host_matvec_f32(
                &out_matrix_host,
                &expected_attn_activated[element_start..element_end],
                out_rows,
                out_cols,
            );
            expected_attn_output.extend_from_slice(&expected_step);
            attn_activated_step_buffer
                .copy_from_host(
                    0,
                    &attn_activated_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy attention activated timestep {timestep}: {err}")
                })?;
            ullm_runtime_sys::matvec_f32(
                &out_matrix,
                &attn_activated_step_buffer,
                out_rows,
                out_cols,
                &mut attn_output_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run out projection timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after out projection timestep {timestep}: {err}")
            })?;
            attn_output_step_buffer
                .copy_to_host(
                    0,
                    &mut attn_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy linear attention output timestep {timestep}: {err}")
                })?;
            residual_step_buffer
                .copy_from_host(
                    0,
                    &residual_sequence_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy residual timestep {timestep}: {err}"))?;
            ullm_runtime_sys::add_f32(
                &residual_step_buffer,
                &attn_output_step_buffer,
                hidden,
                &mut attn_block_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to run attention residual add timestep {timestep}: {err}")
            })?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after attention residual timestep {timestep}: {err}")
            })?;
            attn_block_step_buffer
                .copy_to_host(
                    0,
                    &mut attn_block_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy attention block timestep {timestep}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize after attention block host copy timestep {timestep}: {err}"
                )
            })?;
        }
        let attn_output = decode_f32_le_values(&attn_output_bytes);
        let attn_output_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke out projection",
            &attn_output,
            &expected_attn_output,
            3e-3,
            2e-5,
        )?;
        let attention_block_output = decode_f32_le_values(&attn_block_bytes);
        let expected_attention_block = runtime_host_add_f32(&residual_sequence, &attn_output);
        let attn_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke attention residual",
            &attention_block_output,
            &expected_attention_block,
            1e-5,
            1e-6,
        )?;
        (
            attention_block_output,
            qkv_output,
            z_output,
            a_output,
            b_output,
            conv_output,
            conv_activated,
            recurrent_q,
            recurrent_k,
            recurrent_v,
            gate_output,
            beta_output,
            recurrent_output,
            attn_normed,
            attn_activated,
            attn_output,
            attn_block_max_abs_diff,
            conv_max_abs_diff,
            gate_beta_max_abs_diff,
            recurrent_max_abs_diff,
            attn_norm_max_abs_diff,
            attn_activation_max_abs_diff,
            attn_output_max_abs_diff,
        )
    };

    let z_silu_output = runtime_host_silu_f32(&z_output);

    let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let expected = runtime_host_rmsnorm_f32(
            &attention_block_output[start..end],
            &post_norm_weight_values,
            mlp_epsilon,
        );
        post_normed_expected.extend_from_slice(&expected);
    }
    let mut attn_block_step_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate retained attention block buffer: {err}"))?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    let mut post_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate post RMSNorm output buffer: {err}"))?;
    let attention_block_bytes = encode_f32_to_bytes(&attention_block_output);
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm weight copy: {err}"))?;
    let mut post_normed_bytes = vec![0_u8; hidden_sequence_bytes];
    for timestep in 0..sequence_len {
        let byte_start = timestep * hidden_bytes;
        let byte_end = byte_start + hidden_bytes;
        attn_block_step_buffer
            .copy_from_host(
                0,
                &attention_block_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy attention block timestep {timestep} for post norm: {err}")
            })?;
        ullm_runtime_sys::rmsnorm_f32(
            &attn_block_step_buffer,
            &post_norm_weight_buffer,
            hidden,
            mlp_epsilon,
            &mut post_normed_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run post RMSNorm timestep {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after post RMSNorm timestep {timestep}: {err}")
        })?;
        post_normed_buffer
            .copy_to_host(
                0,
                &mut post_normed_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy post RMSNorm timestep {timestep} to host: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after post RMSNorm host copy {timestep}: {err}")
        })?;
    }
    let post_normed = decode_f32_le_values(&post_normed_bytes);
    let post_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke post RMSNorm",
        &post_normed,
        &post_normed_expected,
        1e-4,
        1e-5,
    )?;

    let (
        mlp_gate_projection,
        mlp_gate_silu,
        mlp_up_projection,
        mlp_activation,
        mlp_intermediate,
        mlp_output,
        layer_output,
        layer_block_max_abs_diff,
    ) = {
        let mut registry = WeightRegistry::new();
        let (gate_rows, gate_cols, mut gate_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &gate_tensor,
            chunk_bytes,
        )?;
        let (up_rows, up_cols, mut up_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &up_tensor,
            chunk_bytes,
        )?;
        let (down_rows, down_cols, mut down_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &down_tensor,
            chunk_bytes,
        )?;
        if gate_rows != up_rows || gate_cols != up_cols || gate_cols != hidden {
            return Err(format!(
                "MLP gate/up shape mismatch: gate=[{gate_rows},{gate_cols}] up=[{up_rows},{up_cols}] hidden={hidden}"
            ));
        }
        if down_rows != hidden || down_cols != gate_rows {
            return Err(format!(
                "MLP down shape mismatch: expected [{hidden},{gate_rows}], got [{down_rows},{down_cols}]"
            ));
        }
        let down_row_scale_overrides = matching_package_row_scale_overrides(
            row_scale_overrides,
            layer_index,
            "mlp.down_proj.weight",
        );
        applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
            &mut stream,
            &mut down_matrix,
            down_rows,
            down_cols,
            &down_tensor,
            &down_row_scale_overrides,
        )?);
        let gate_cell_delta_overrides = matching_package_cell_delta_overrides(
            cell_delta_overrides,
            layer_index,
            "mlp.gate_proj.weight",
        );
        applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
            &mut stream,
            &mut gate_matrix,
            gate_rows,
            gate_cols,
            &gate_tensor,
            &gate_cell_delta_overrides,
        )?);
        let up_cell_delta_overrides = matching_package_cell_delta_overrides(
            cell_delta_overrides,
            layer_index,
            "mlp.up_proj.weight",
        );
        applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
            &mut stream,
            &mut up_matrix,
            up_rows,
            up_cols,
            &up_tensor,
            &up_cell_delta_overrides,
        )?);
        let down_cell_delta_overrides = matching_package_cell_delta_overrides(
            cell_delta_overrides,
            layer_index,
            "mlp.down_proj.weight",
        );
        applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
            &mut stream,
            &mut down_matrix,
            down_rows,
            down_cols,
            &down_tensor,
            &down_cell_delta_overrides,
        )?);
        let intermediate = gate_rows;
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "MLP intermediate byte size overflows".to_string())?;
        let mut post_normed_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate post normed step buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP gate buffer: {err}"))?;
        let mut up_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP up buffer: {err}"))?;
        let mut mlp_activated_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP activated buffer: {err}"))?;
        let mut mlp_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate MLP output buffer: {err}"))?;
        let mut layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate layer output buffer: {err}"))?;
        let mut mlp_output_bytes = vec![0_u8; hidden_sequence_bytes];
        let mut layer_output_bytes = vec![0_u8; hidden_sequence_bytes];
        let mut mlp_activated_bytes = vec![
            0_u8;
            intermediate_bytes.checked_mul(sequence_len).ok_or_else(
                || "MLP activated sequence byte size overflows".to_string()
            )?
        ];
        let mut mlp_gate_bytes = vec![
            0_u8;
            intermediate_bytes.checked_mul(sequence_len).ok_or_else(
                || "MLP gate sequence byte size overflows".to_string()
            )?
        ];
        let mut mlp_up_bytes = vec![
            0_u8;
            intermediate_bytes.checked_mul(sequence_len).ok_or_else(
                || "MLP up sequence byte size overflows".to_string()
            )?
        ];
        for timestep in 0..sequence_len {
            let byte_start = timestep * hidden_bytes;
            let byte_end = byte_start + hidden_bytes;
            let intermediate_byte_start = timestep * intermediate_bytes;
            let intermediate_byte_end = intermediate_byte_start + intermediate_bytes;
            post_normed_step_buffer
                .copy_from_host(
                    0,
                    &post_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy post normed timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &gate_matrix,
                &post_normed_step_buffer,
                gate_rows,
                gate_cols,
                &mut gate_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP gate matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &up_matrix,
                &post_normed_step_buffer,
                up_rows,
                up_cols,
                &mut up_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP up matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP gate/up timestep {timestep}: {err}")
            })?;
            gate_buffer
                .copy_to_host(
                    0,
                    &mut mlp_gate_bytes[intermediate_byte_start..intermediate_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy MLP gate timestep {timestep}: {err}"))?;
            up_buffer
                .copy_to_host(
                    0,
                    &mut mlp_up_bytes[intermediate_byte_start..intermediate_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy MLP up timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP gate/up host copy {timestep}: {err}")
            })?;
            ullm_runtime_sys::silu_mul_f32(
                &gate_buffer,
                &up_buffer,
                intermediate,
                &mut mlp_activated_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP SiLU-mul timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP SiLU-mul timestep {timestep}: {err}")
            })?;
            mlp_activated_buffer
                .copy_to_host(
                    0,
                    &mut mlp_activated_bytes[intermediate_byte_start..intermediate_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy MLP activated timestep {timestep}: {err}")
                })?;
            ullm_runtime_sys::matvec_f32(
                &down_matrix,
                &mlp_activated_buffer,
                down_rows,
                down_cols,
                &mut mlp_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP down matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP down timestep {timestep}: {err}")
            })?;
            mlp_output_buffer
                .copy_to_host(
                    0,
                    &mut mlp_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy MLP output timestep {timestep}: {err}"))?;
            attn_block_step_buffer
                .copy_from_host(
                    0,
                    &attention_block_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!(
                        "failed to copy attention block timestep {timestep} for MLP residual: {err}"
                    )
                })?;
            ullm_runtime_sys::add_f32(
                &attn_block_step_buffer,
                &mlp_output_buffer,
                hidden,
                &mut layer_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP residual add timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP residual timestep {timestep}: {err}")
            })?;
            layer_output_buffer
                .copy_to_host(
                    0,
                    &mut layer_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy layer output timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after layer output copy timestep {timestep}: {err}")
            })?;
        }
        let mlp_gate_projection = decode_f32_le_values(&mlp_gate_bytes);
        let mlp_gate_silu = runtime_host_silu_f32(&mlp_gate_projection);
        let mlp_up_projection = decode_f32_le_values(&mlp_up_bytes);
        let mlp_activation = decode_f32_le_values(&mlp_activated_bytes);
        let mlp_output = decode_f32_le_values(&mlp_output_bytes);
        let layer_output = decode_f32_le_values(&layer_output_bytes);
        let expected_layer_output = runtime_host_add_f32(&attention_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (
            mlp_gate_projection,
            mlp_gate_silu,
            mlp_up_projection,
            mlp_activation,
            intermediate,
            mlp_output,
            layer_output,
            layer_block_max_abs_diff,
        )
    };

    let line = format!(
        "package-linear-attn-mlp-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} post_norm_dtype={} row_scale_overrides={} cell_delta_overrides={} backend={} device_index={} name=\"{}\" residual_preview={} attention_output_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} attn_norm_max_abs_diff={attn_norm_max_abs_diff:.9} attn_activation_max_abs_diff={attn_activation_max_abs_diff:.9} attn_output_max_abs_diff={attn_output_max_abs_diff:.9} attn_block_max_abs_diff={attn_block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} layer_block_max_abs_diff={layer_block_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        input_norm.dtype,
        conv.dtype,
        a_log.dtype,
        dt_bias.dtype,
        attn_norm.dtype,
        post_norm.dtype,
        applied_row_scale_overrides.len(),
        applied_cell_delta_overrides.len(),
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&residual_sequence[..8.min(residual_sequence.len())]),
        format_f32_preview(&attn_output[..8.min(attn_output.len())]),
        format_f32_preview(&attention_block_output[..8.min(attention_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
    );
    Ok(PackageLinearAttnMlpBlockSequenceRun {
        line,
        applied_row_scale_overrides,
        applied_cell_delta_overrides,
        attention_input_normed: input_normed,
        attention_qkv_projection: qkv_output,
        attention_qkv_projection_dim: qkv_rows_expected,
        attention_z_projection: z_output,
        attention_gate_silu: z_silu_output,
        attention_a_projection: a_output,
        attention_b_projection: b_output,
        attention_gate_dim: value_heads,
        attention_conv_pre_silu: conv_output,
        attention_conv: conv_activated,
        attention_recurrent_q: recurrent_q,
        attention_recurrent_k: recurrent_k,
        attention_recurrent_v: recurrent_v,
        attention_recurrent_qk_dim: key_heads * key_dim,
        attention_gate: gate_output,
        attention_beta: beta_output,
        attention_recurrent: recurrent_output,
        attention_normed: attn_normed,
        attention_projection_input: attn_activated,
        attention_output: attn_output,
        attention_block_output,
        post_normed,
        mlp_gate_projection,
        mlp_gate_silu,
        mlp_up_projection,
        mlp_activation,
        mlp_intermediate,
        mlp_output,
        layer_output,
    })
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ZHybridInputHeader {
    kind: String,
    schema_version: String,
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct ZHybridInputCase {
    kind: String,
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    input_sha256: String,
    values: Vec<f32>,
}

#[derive(Debug, serde::Serialize)]
struct ZHybridMetric {
    rows: usize,
    elements_per_row: usize,
    max_abs: f64,
    relative_l2: f64,
    cosine: Option<f64>,
    nonfinite: bool,
}

fn z_hybrid_metric(actual: &[f32], reference: &[f32], rows: usize) -> Result<ZHybridMetric, String> {
    if rows == 0 || actual.len() != reference.len() || actual.len() % rows != 0 {
        return Err("Z hybrid metric geometry differs".to_string());
    }
    let elements_per_row = actual.len() / rows;
    let mut max_abs = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    let mut actual_sq = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut nonfinite = false;
    for (&lhs, &rhs) in actual.iter().zip(reference) {
        if !lhs.is_finite() || !rhs.is_finite() {
            nonfinite = true;
            continue;
        }
        let lhs = lhs as f64;
        let rhs = rhs as f64;
        let diff = lhs - rhs;
        max_abs = max_abs.max(diff.abs());
        diff_sq += diff * diff;
        ref_sq += rhs * rhs;
        actual_sq += lhs * lhs;
        dot += lhs * rhs;
    }
    let relative_l2 = if ref_sq > 0.0 {
        diff_sq.sqrt() / ref_sq.sqrt()
    } else if diff_sq == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine = if actual_sq > 0.0 && ref_sq > 0.0 {
        Some(dot / (actual_sq * ref_sq).sqrt())
    } else {
        None
    };
    Ok(ZHybridMetric {
        rows,
        elements_per_row,
        max_abs,
        relative_l2,
        cosine,
        nonfinite,
    })
}

fn z_hybrid_per_step_metrics(
    actual: &[f32],
    reference: &[f32],
    rows: usize,
) -> Result<Vec<ZHybridMetric>, String> {
    if rows == 0 || actual.len() != reference.len() || actual.len() % rows != 0 {
        return Err("Z hybrid per-step metric geometry differs".to_string());
    }
    let width = actual.len() / rows;
    (0..rows)
        .map(|row| z_hybrid_metric(&actual[row * width..(row + 1) * width], &reference[row * width..(row + 1) * width], 1))
        .collect()
}

fn z_hybrid_write_f32(path: &std::path::Path, values: &[f32]) -> Result<String, String> {
    if path.exists() {
        return Err(format!("refusing to overwrite Z hybrid sidecar: {}", path.display()));
    }
    let bytes = encode_f32_to_bytes(values);
    fs::write(path, &bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    Ok(z_hybrid_sha256(&bytes))
}

fn z_hybrid_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn z_hybrid_read_input(
    path: &str,
    expected_sequence_len: usize,
) -> Result<(Vec<ZHybridInputCase>, Vec<f32>, String), String> {
    let raw = fs::read(path).map_err(|err| format!("failed to read diagnostic input {path}: {err}"))?;
    let input_sha256 = z_hybrid_sha256(&raw);
    let mut lines = raw.split(|byte| *byte == b'\n').filter(|line| !line.is_empty());
    let header_line = lines.next().ok_or_else(|| "diagnostic input is empty".to_string())?;
    let header: ZHybridInputHeader = serde_json::from_slice(header_line)
        .map_err(|err| format!("failed to parse diagnostic input header: {err}"))?;
    if header.kind != "header"
        || header.schema_version != "ullm.aq4_layer0_input_normed_jsonl.v1"
        || header.tensor_name != "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        || header.dtype != "f32"
        || header.shape != [4096]
    {
        return Err("diagnostic input header contract differs".to_string());
    }
    let mut cases = Vec::new();
    let mut values = Vec::new();
    for line in lines {
        let case: ZHybridInputCase = serde_json::from_slice(line)
            .map_err(|err| format!("failed to parse diagnostic input case: {err}"))?;
        if case.kind != "case" || case.values.len() != 4096 || case.values.iter().any(|value| !value.is_finite()) {
            return Err(format!("diagnostic input case {} is invalid", case.case_id));
        }
        let encoded = encode_f32_to_bytes(&case.values);
        if z_hybrid_sha256(&encoded) != case.input_sha256 {
            return Err(format!("diagnostic input case {} hash differs", case.case_id));
        }
        values.extend_from_slice(&case.values);
        cases.push(case);
    }
    if cases.len() != expected_sequence_len {
        return Err(format!(
            "diagnostic input row count differs: got {} expected {}",
            cases.len(), expected_sequence_len
        ));
    }
    Ok((cases, values, input_sha256))
}

fn package_linear_attn_z_hybrid_diagnostic(
    path: Option<String>,
    input: Option<String>,
    source_z: Option<String>,
    output: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-z-hybrid-diagnostic requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(input) = input else {
        eprintln!("package-linear-attn-z-hybrid-diagnostic requires a fixed input JSONL");
        return ExitCode::from(2);
    };
    let Some(source_z) = source_z else {
        eprintln!("package-linear-attn-z-hybrid-diagnostic requires a source Z f32le sidecar");
        return ExitCode::from(2);
    };
    let Some(output) = output else {
        eprintln!("package-linear-attn-z-hybrid-diagnostic requires a new output directory");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    if device_index != 0 {
        eprintln!("package-linear-attn-z-hybrid-diagnostic is CPU-only and requires device index 0");
        return ExitCode::from(2);
    }
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 3, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    if std::path::Path::new(&output).exists() {
        eprintln!("refusing to overwrite diagnostic output directory: {output}");
        return ExitCode::from(2);
    }
    let result = (|| -> Result<serde_json::Value, String> {
        let (cases, input_values, input_sha256) = z_hybrid_read_input(&input, sequence_len)?;
        let source_z_bytes = fs::read(&source_z)
            .map_err(|err| format!("failed to read source Z sidecar {source_z}: {err}"))?;
        let hidden = 32 * 128;
        let expected_source_bytes = sequence_len
            .checked_mul(hidden)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "source Z sidecar size overflows".to_string())?;
        if source_z_bytes.len() != expected_source_bytes {
            return Err(format!(
                "source Z sidecar size differs: got {} expected {}",
                source_z_bytes.len(), expected_source_bytes
            ));
        }
        let source_z_values = decode_f32_le_values(&source_z_bytes);
        if source_z_values.iter().any(|value| !value.is_finite()) {
            return Err("source Z sidecar contains non-finite values".to_string());
        }
        fs::create_dir_all(&output).map_err(|err| format!("failed to create diagnostic output: {err}"))?;
        let residual_base = deterministic_f32_vector(hidden);
        let mut residual_sequence = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            residual_sequence.extend(linear_attn_step_input(&residual_base, timestep));
        }
        let baseline = package_linear_attn_mlp_block_sequence_run_with_diagnostic_inputs(
            &path,
            device_index,
            chunk_bytes,
            0,
            sequence_len,
            residual_sequence.clone(),
            None,
            None,
            Some(&input_values),
            None,
        )?;
        let hybrid = package_linear_attn_mlp_block_sequence_run_with_diagnostic_inputs(
            &path,
            device_index,
            chunk_bytes,
            0,
            sequence_len,
            residual_sequence,
            None,
            None,
            Some(&input_values),
            Some(&source_z_values),
        )?;
        let output_dir = std::path::Path::new(&output);
        let baseline_z_sha = z_hybrid_write_f32(&output_dir.join("baseline-z.f32le"), &baseline.attention_z_projection)?;
        let hybrid_z_sha = z_hybrid_write_f32(&output_dir.join("hybrid-z.f32le"), &hybrid.attention_z_projection)?;
        let baseline_block_sha = z_hybrid_write_f32(&output_dir.join("baseline-attention-block.f32le"), &baseline.attention_block_output)?;
        let hybrid_block_sha = z_hybrid_write_f32(&output_dir.join("hybrid-attention-block.f32le"), &hybrid.attention_block_output)?;
        let baseline_layer_sha = z_hybrid_write_f32(&output_dir.join("baseline-layer-output.f32le"), &baseline.layer_output)?;
        let hybrid_layer_sha = z_hybrid_write_f32(&output_dir.join("hybrid-layer-output.f32le"), &hybrid.layer_output)?;
        let manifest_path = std::path::Path::new(&path).join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path)
            .map_err(|err| format!("failed to read package manifest: {err}"))?;
        let package_manifest_sha256 = z_hybrid_sha256(&manifest_bytes);
        let baseline_source_z = z_hybrid_metric(&baseline.attention_z_projection, &source_z_values, sequence_len)?;
        let hybrid_source_z = z_hybrid_metric(&hybrid.attention_z_projection, &source_z_values, sequence_len)?;
        let hybrid_baseline_z = z_hybrid_metric(&hybrid.attention_z_projection, &baseline.attention_z_projection, sequence_len)?;
        let hybrid_baseline_block = z_hybrid_metric(&hybrid.attention_block_output, &baseline.attention_block_output, sequence_len)?;
        let hybrid_baseline_layer = z_hybrid_metric(&hybrid.layer_output, &baseline.layer_output, sequence_len)?;
        let report = serde_json::json!({
            "schema_version": "ullm.aq4_layer0_z_hybrid_fidelity.cpu.v1",
            "status": "valid",
            "classification": "unclassified",
            "promotion": false,
            "holdout": "not_run",
            "policy_evaluation": "policy_not_evaluated",
            "thresholds": serde_json::Value::Null,
            "device": {"backend": "cpu", "requested_index": device_index},
            "package": {"root": path, "manifest_sha256": package_manifest_sha256, "layer_index": 0},
            "input": {"path": input, "sha256": input_sha256, "schema": "ullm.aq4_layer0_input_normed_jsonl.v1", "rows": sequence_len, "shape": [4096], "cases": cases.iter().map(|case| serde_json::json!({"case_id": case.case_id, "step": case.step, "context_length": case.context_length, "context_token_ids_sha256": case.context_token_ids_sha256, "input_sha256": case.input_sha256})).collect::<Vec<_>>()},
            "source_z": {"path": source_z, "sha256": z_hybrid_sha256(&source_z_bytes), "shape": [sequence_len, hidden], "dtype": "f32", "operation": "input_f32_matmul_source_bf16_weight_cast_f32_accumulate_f32"},
            "state_reset": {"boundary": "before_each_run", "recurrent_state": "zero_initialized_once_for_sequence", "same_between_baseline_and_hybrid": true},
            "override": {"family": "z", "boundary": "after_production_z_matvec_before_production_silu_mul", "default_off": true, "worker_reachable": false, "promotion": false},
            "baseline": {"mode": "all_aq4", "z_sha256": baseline_z_sha, "attention_block_sha256": baseline_block_sha, "layer_output_sha256": baseline_layer_sha},
            "hybrid": {"mode": "aq4_except_z_source_bf16", "z_sha256": hybrid_z_sha, "attention_block_sha256": hybrid_block_sha, "layer_output_sha256": hybrid_layer_sha},
            "metrics": {
                "baseline_vs_source_z": {"aggregate": baseline_source_z, "per_step": z_hybrid_per_step_metrics(&baseline.attention_z_projection, &source_z_values, sequence_len)?},
                "hybrid_vs_source_z": {"aggregate": hybrid_source_z, "per_step": z_hybrid_per_step_metrics(&hybrid.attention_z_projection, &source_z_values, sequence_len)?},
                "hybrid_vs_baseline_z": {"aggregate": hybrid_baseline_z, "per_step": z_hybrid_per_step_metrics(&hybrid.attention_z_projection, &baseline.attention_z_projection, sequence_len)?},
                "hybrid_vs_baseline_attention_block": {"aggregate": hybrid_baseline_block, "per_step": z_hybrid_per_step_metrics(&hybrid.attention_block_output, &baseline.attention_block_output, sequence_len)?},
                "hybrid_vs_baseline_layer_output": {"aggregate": hybrid_baseline_layer, "per_step": z_hybrid_per_step_metrics(&hybrid.layer_output, &baseline.layer_output, sequence_len)?}
            },
            "production_path": {"input_norm": "same_fixed_input_sidecar", "qkv_a_b": "same_production_aq4_matvec", "conv_gate_beta_recurrent": "same_production_runtime", "post_norm_mlp_residual": "same_production_runtime"}
        });
        let report_path = output_dir.join("report.json");
        let report_bytes = serde_json::to_vec_pretty(&report).map_err(|err| format!("failed to encode report: {err}"))?;
        fs::write(&report_path, &report_bytes).map_err(|err| format!("failed to write report: {err}"))?;
        Ok(report)
    })();
    match result {
        Ok(report) => {
            let metrics = report.get("metrics").and_then(serde_json::Value::as_object);
            let block_l2 = metrics
                .and_then(|values| values.get("hybrid_vs_baseline_attention_block"))
                .and_then(|value| value.get("aggregate"))
                .and_then(|value| value.get("relative_l2"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            let layer_l2 = metrics
                .and_then(|values| values.get("hybrid_vs_baseline_layer_output"))
                .and_then(|value| value.get("aggregate"))
                .and_then(|value| value.get("relative_l2"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            println!("package-linear-attn-z-hybrid-diagnostic report={output}/report.json rows={sequence_len} attention_block_relative_l2={block_l2:.9} layer_output_relative_l2={layer_l2:.9} promotion=false holdout=not_run");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("package-linear-attn-z-hybrid-diagnostic: {err}");
            ExitCode::from(1)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NormKind {
    Input,
    Post,
}

#[derive(Debug, Clone, Copy)]
enum LinearAttnProjection {
    A,
    B,
    Qkv,
    Z,
    Out,
    All,
}

#[derive(Debug, Clone, Copy)]
enum SelfAttnProjection {
    Q,
    K,
    V,
    O,
    All,
}

impl NormKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Post => "post",
        }
    }
}

fn parse_linear_attn_projection(value: Option<&str>) -> Result<LinearAttnProjection, ExitCode> {
    let raw = value.unwrap_or("all");
    match raw {
        "a" => Ok(LinearAttnProjection::A),
        "b" => Ok(LinearAttnProjection::B),
        "qkv" => Ok(LinearAttnProjection::Qkv),
        "z" => Ok(LinearAttnProjection::Z),
        "out" => Ok(LinearAttnProjection::Out),
        "all" => Ok(LinearAttnProjection::All),
        _raw => {
            eprintln!("invalid projection: {raw}; expected a, b, qkv, z, out, or all");
            Err(ExitCode::from(2))
        }
    }
}

fn parse_self_attn_projection(value: Option<&str>) -> Result<SelfAttnProjection, ExitCode> {
    let raw = value.unwrap_or("all");
    match raw {
        "q" => Ok(SelfAttnProjection::Q),
        "k" => Ok(SelfAttnProjection::K),
        "v" => Ok(SelfAttnProjection::V),
        "o" | "out" => Ok(SelfAttnProjection::O),
        "all" => Ok(SelfAttnProjection::All),
        _raw => {
            eprintln!("invalid self-attn projection: {raw}; expected q, k, v, o, or all");
            Err(ExitCode::from(2))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LinearAttnAux {
    ALog,
    DtBias,
    Conv1d,
    Norm,
    All,
}

fn parse_linear_attn_aux(value: Option<&str>) -> Result<LinearAttnAux, ExitCode> {
    let raw = value.unwrap_or("all");
    let normalized = raw.replace(['-', '_'], "");
    match normalized.as_str() {
        "alog" => Ok(LinearAttnAux::ALog),
        "dtbias" => Ok(LinearAttnAux::DtBias),
        "conv1d" => Ok(LinearAttnAux::Conv1d),
        "norm" => Ok(LinearAttnAux::Norm),
        "all" => Ok(LinearAttnAux::All),
        _value => {
            eprintln!(
                "invalid aux: {raw}; expected a-log, dt-bias, conv1d, norm, or all (aliases: a_log, alog, dt_bias)"
            );
            Err(ExitCode::from(2))
        }
    }
}

fn normalize_norm_kind(kind: Option<&str>) -> Result<NormKind, ExitCode> {
    match kind.unwrap_or("input") {
        "input" => Ok(NormKind::Input),
        "post" => Ok(NormKind::Post),
        value => {
            eprintln!("invalid norm kind: {value}; expected input or post");
            Err(ExitCode::from(2))
        }
    }
}

fn runtime_matvec_to_host_f32(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &ullm_runtime_sys::RuntimeBuffer,
    input: &ullm_runtime_sys::RuntimeBuffer,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut output = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    ullm_runtime_sys::matvec_f32(matrix, input, rows, cols, &mut output, Some(stream))
        .map_err(|err| format!("failed to run {label} matvec: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} matvec: {err}"))?;
    let mut output_host = vec![0_u8; output_bytes];
    output
        .copy_to_host(0, &mut output_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} host copy: {err}"))?;
    Ok(decode_f32_le_values(&output_host))
}

fn runtime_headwise_rmsnorm_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    let head_dim = weight.len();
    if head_dim == 0 {
        return Err(format!("{label} weight must not be empty"));
    }
    if !input.len().is_multiple_of(head_dim) {
        return Err(format!(
            "{label} input length {} is not a multiple of head_dim {head_dim}",
            input.len()
        ));
    }

    let output = qwen3_headwise_rmsnorm_to_host_f32(context, stream, input, weight, epsilon)
        .map_err(|err| format!("failed to run {label} RMSNorm: {err}"))?;
    if output.len() != input.len() {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            input.len(),
            output.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;

    for (head_index, head_input) in input.chunks_exact(head_dim).enumerate() {
        let actual_head_start = head_index
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} head index multiplication overflow"))?;
        let actual_head_end = actual_head_start
            .checked_add(head_dim)
            .ok_or_else(|| format!("{label} head length multiplication overflow"))?;
        let actual = &output[actual_head_start..actual_head_end];
        let expected = runtime_host_rmsnorm_f32(head_input, weight, epsilon);
        let head_max_abs_diff = verify_f32_close(
            &format!("{label} head {head_index}"),
            &actual,
            &expected,
            1e-4_f32,
            1e-4_f32,
        )?;
        max_abs_diff = max_abs_diff.max(head_max_abs_diff);
    }
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_rope_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    let output = qwen3_rope_to_host_f32(
        context,
        stream,
        input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    )
    .map_err(|err| format!("failed to run {label} RoPE: {err}"))?;
    if output.len() != input.len() {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            input.len(),
            output.len()
        ));
    }
    let expected = runtime_host_rope_f32(
        input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_causal_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    let output = qwen3_causal_attn_to_host_f32(
        context,
        stream,
        q,
        k,
        v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    )
    .map_err(|err| format!("failed to run {label} causal attention: {err}"))?;
    let output_elements = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| format!("{label} output element count overflows"))?
        .checked_mul(value_dim)
        .ok_or_else(|| format!("{label} output element count overflows"))?;
    if output.len() != output_elements {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            output_elements,
            output.len()
        ));
    }
    let expected = runtime_host_causal_attn_f32(
        q,
        k,
        v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_decode_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if q.len() != q_heads * head_dim {
        return Err(format!(
            "{label} q length {} does not match q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    if k_cache.len() != cache_len * kv_heads * head_dim {
        return Err(format!(
            "{label} k cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}",
            k_cache.len()
        ));
    }
    if v_cache.len() != cache_len * kv_heads * value_dim {
        return Err(format!(
            "{label} v cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} value_dim={value_dim}",
            v_cache.len()
        ));
    }
    let q_bytes = encode_f32_to_bytes(q);
    let k_bytes = encode_f32_to_bytes(k_cache);
    let v_bytes = encode_f32_to_bytes(v_cache);
    let output_elements = q_heads * value_dim;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut q_buffer = context
        .alloc_buffer(q_bytes.len())
        .map_err(|err| format!("failed to allocate {label} q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes.len())
        .map_err(|err| format!("failed to allocate {label} k cache buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes.len())
        .map_err(|err| format!("failed to allocate {label} v cache buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} k cache input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} v cache input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copies: {err}"))?;
    ullm_runtime_sys::decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} decode attention: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} decode attention: {err}"))?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes_host);
    let expected = runtime_host_decode_attn_f32(
        q,
        k_cache,
        v_cache,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_paged_decode_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if cache_len == 0 {
        return Err(format!("{label} cache_len must be greater than zero"));
    }
    if block_size == 0 {
        return Err(format!("{label} block_size must be greater than zero"));
    }
    if cache_blocks == 0 {
        return Err(format!("{label} cache_blocks must be greater than zero"));
    }
    if q.len() != q_heads * head_dim {
        return Err(format!(
            "{label} q length {} does not match q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    let block_table_entries = (cache_len - 1) / block_size + 1;
    if block_table.len() != block_table_entries {
        return Err(format!(
            "{label} block table length {} does not match expected entries {block_table_entries}",
            block_table.len()
        ));
    }
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| format!("{label} physical cache token count overflows"))?;
    if k_cache.len() != physical_tokens * kv_heads * head_dim {
        return Err(format!(
            "{label} k cache length {} does not match cache_blocks={cache_blocks} block_size={block_size} kv_heads={kv_heads} head_dim={head_dim}",
            k_cache.len()
        ));
    }
    if v_cache.len() != physical_tokens * kv_heads * value_dim {
        return Err(format!(
            "{label} v cache length {} does not match cache_blocks={cache_blocks} block_size={block_size} kv_heads={kv_heads} value_dim={value_dim}",
            v_cache.len()
        ));
    }
    let q_bytes = encode_f32_to_bytes(q);
    let k_bytes = encode_f32_to_bytes(k_cache);
    let v_bytes = encode_f32_to_bytes(v_cache);
    let block_table_bytes = encode_u32_to_bytes(block_table);
    let output_elements = q_heads * value_dim;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut q_buffer = context
        .alloc_buffer(q_bytes.len())
        .map_err(|err| format!("failed to allocate {label} q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes.len())
        .map_err(|err| format!("failed to allocate {label} paged k cache buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes.len())
        .map_err(|err| format!("failed to allocate {label} paged v cache buffer: {err}"))?;
    let mut block_table_buffer = context
        .alloc_buffer(block_table_bytes.len())
        .map_err(|err| format!("failed to allocate {label} block table buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} paged k cache input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} paged v cache input: {err}"))?;
    block_table_buffer
        .copy_from_host(0, &block_table_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} block table input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copies: {err}"))?;
    ullm_runtime_sys::paged_decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} paged decode attention: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after {label} paged decode attention: {err}")
    })?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes_host);
    let expected = runtime_host_paged_decode_attn_f32(
        q,
        k_cache,
        v_cache,
        block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

struct RuntimePagedKvWriteDecodeResult {
    output: Vec<f32>,
    step_outputs: Vec<f32>,
    cache_blocks: usize,
    block_table: Vec<u32>,
    allocator_stats: KvBlockAllocatorStats,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
    scheduler_decode_batches: usize,
    output_max_abs_diff: f32,
    step_output_max_abs_diff: f32,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    k_write_max_abs_diff: f32,
    v_write_max_abs_diff: f32,
}

#[allow(clippy::too_many_arguments)]
fn runtime_paged_kv_write_decode_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q_sequence: &[f32],
    logical_k_cache: &[f32],
    logical_v_cache: &[f32],
    cache_len: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<RuntimePagedKvWriteDecodeResult, String> {
    if q_sequence.len() != cache_len * q_heads * head_dim {
        return Err(format!(
            "{label} q sequence length {} does not match cache_len={cache_len} q_heads={q_heads} head_dim={head_dim}",
            q_sequence.len()
        ));
    }
    if logical_k_cache.len() != cache_len * kv_heads * head_dim {
        return Err(format!(
            "{label} logical k cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}",
            logical_k_cache.len()
        ));
    }
    if logical_v_cache.len() != cache_len * kv_heads * value_dim {
        return Err(format!(
            "{label} logical v cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} value_dim={value_dim}",
            logical_v_cache.len()
        ));
    }
    let prepared = prepare_fragmented_paged_decode_state(cache_len, block_size)?;
    let mut scheduler = prepared.scheduler;
    let prefill_prompt_tokens = prepared.prefill_tokens;
    let max_new_tokens = prepared.max_new_tokens;
    let block_table = prepared.block_table;
    let cache_blocks = prepared.cache_blocks;
    let scheduler_request_id = prepared.request_id;

    let readback_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: expected_k_cache,
        v: expected_v_cache,
    } = pack_paged_kv_cache_for_block_table(
        logical_k_cache,
        logical_v_cache,
        &block_table,
        cache_len,
        readback_shape,
    )?;
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let mut decode_runner = Qwen3SelfAttnRequestDecodeRunner::new();
    decode_runner.insert_request(
        context,
        stream,
        scheduler_request_id,
        shape,
        block_table.to_vec(),
        softmax_scale,
    )?;
    let q_token_elements = q_heads * head_dim;
    let k_token_elements = kv_heads * head_dim;
    let v_token_elements = kv_heads * value_dim;
    let output_elements = q_heads * value_dim;
    let mut step_outputs = Vec::with_capacity(cache_len * output_elements);
    let mut step_output_max_abs_diff = 0.0_f32;
    let mut scheduler_decode_batches = 0_usize;

    for timestep in 0..prefill_prompt_tokens {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;

        let step = decode_runner
            .run_prefill_step(
                stream,
                Qwen3SelfAttnDecodeBatchInput {
                    request_id: scheduler_request_id,
                    q: &q_sequence[q_start..q_end],
                    k: &logical_k_cache[k_start..k_end],
                    v: &logical_v_cache[v_start..v_end],
                },
            )
            .map_err(|err| {
                format!("{label} failed to run prefix/prefill decode timestep {timestep}: {err}")
            })?;
        if step.cache_position != timestep {
            return Err(format!(
                "{label} prefix/prefill decode request wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "{label} prefix/prefill decode request reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }
        if step.attention_output.len() != output_elements {
            return Err(format!(
                "{label} prefix/prefill timestep {timestep} produced {} outputs, expected {output_elements}",
                step.attention_output.len()
            ));
        }

        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_sequence[q_start..q_end],
            &expected_k_cache,
            &expected_v_cache,
            &block_table,
            timestep + 1,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let step_max_abs_diff = verify_f32_close(
            &format!("{label} timestep {timestep} paged decode step"),
            &step.attention_output,
            &expected_step_output,
            1e-4_f32,
            1e-4_f32,
        )?;
        step_output_max_abs_diff = step_output_max_abs_diff.max(step_max_abs_diff);
        step_outputs.extend_from_slice(&step.attention_output);
    }

    scheduler
        .complete_prefill(scheduler_request_id)
        .map_err(|err| format!("failed to complete decode prefill in {label}: {err}"))?;

    for timestep in prefill_prompt_tokens..cache_len {
        let decode_requests = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to ready decode batch in {label}: {err}"))?;
        let request = decode_requests.first().ok_or_else(|| {
            format!("{label} expected one ready decode request for timestep {timestep}, got none")
        })?;

        if request.cache_position != timestep {
            return Err(format!(
                "{label} ready decode request cache position {} does not match timestep {timestep}",
                request.cache_position
            ));
        }
        if request.next_cache_len != timestep + 1 {
            return Err(format!(
                "{label} ready decode request next cache len {} does not match {}",
                request.next_cache_len,
                timestep + 1
            ));
        }
        if request.request.id != scheduler_request_id {
            return Err(format!(
                "{label} ready decode request id {:?} does not match scheduler request {:?}",
                request.request.id, scheduler_request_id
            ));
        }

        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;

        let inputs = [Qwen3SelfAttnDecodeBatchInput {
            request_id: request.request.id,
            q: &q_sequence[q_start..q_end],
            k: &logical_k_cache[k_start..k_end],
            v: &logical_v_cache[v_start..v_end],
        }];
        let mut outputs = decode_runner
            .run_ready_batch(stream, &mut scheduler, &decode_requests, &inputs)
            .map_err(|err| format!("failed to run {label} timestep {timestep}: {err}"))?;
        let step = outputs.pop().ok_or_else(|| {
            format!("{label} ready decode batch produced no output for timestep {timestep}")
        })?;
        if step.request_id != scheduler_request_id {
            return Err(format!(
                "{label} output request id {:?} does not match scheduler request {:?}",
                step.request_id, scheduler_request_id
            ));
        }

        if step.cache_position != request.cache_position {
            return Err(format!(
                "{label} paged decode state wrote position {}, expected {}",
                step.cache_position, request.cache_position
            ));
        }
        if step.cache_len != request.next_cache_len {
            return Err(format!(
                "{label} paged decode state reported cache_len {}, expected {}",
                step.cache_len, request.next_cache_len
            ));
        }
        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_sequence[q_start..q_end],
            &expected_k_cache,
            &expected_v_cache,
            &block_table,
            timestep + 1,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let step_max_abs_diff = verify_f32_close(
            &format!("{label} timestep {timestep} paged decode step"),
            &step.attention_output,
            &expected_step_output,
            1e-4_f32,
            1e-4_f32,
        )?;
        step_output_max_abs_diff = step_output_max_abs_diff.max(step_max_abs_diff);
        step_outputs.extend_from_slice(&step.attention_output);

        scheduler_decode_batches += 1;
    }

    let scheduler_active = scheduler
        .active_request(scheduler_request_id)
        .ok_or_else(|| {
            format!(
                "{label} decode request {:?} missing after scheduler progress",
                scheduler_request_id
            )
        })?;

    let readback = decode_runner
        .read_cache_to_host(scheduler_request_id, stream)
        .map_err(|err| format!("failed to read {label} paged cache: {err}"))?;
    let k_write_max_abs_diff = verify_f32_close(
        &format!("{label} paged k cache write"),
        &readback.k,
        &expected_k_cache,
        1e-5_f32,
        1e-5_f32,
    )?;
    let v_write_max_abs_diff = verify_f32_close(
        &format!("{label} paged v cache write"),
        &readback.v,
        &expected_v_cache,
        1e-5_f32,
        1e-5_f32,
    )?;

    let output_start = (cache_len - 1) * output_elements;
    let output_end = output_start + output_elements;
    let output = step_outputs[output_start..output_end].to_vec();
    let expected_output = runtime_host_paged_decode_attn_f32(
        &q_sequence[(cache_len - 1) * q_token_elements..cache_len * q_token_elements],
        &expected_k_cache,
        &expected_v_cache,
        &block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let output_max_abs_diff =
        verify_f32_close(label, &output, &expected_output, 1e-4_f32, 1e-4_f32)?;

    Ok(RuntimePagedKvWriteDecodeResult {
        output,
        step_outputs,
        cache_blocks,
        block_table,
        allocator_stats: scheduler.allocator_stats(),
        scheduler_request_id,
        scheduler_prefill_tokens: prefill_prompt_tokens,
        scheduler_max_new_tokens: max_new_tokens,
        scheduler_cached_tokens: scheduler_active.cached_tokens,
        scheduler_generated_tokens: scheduler_active.generated_tokens,
        scheduler_active_len: scheduler.active_len(),
        scheduler_decode_batches,
        output_max_abs_diff,
        step_output_max_abs_diff,
        k_cache: readback.k,
        v_cache: readback.v,
        k_write_max_abs_diff,
        v_write_max_abs_diff,
    })
}

fn verify_f32_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    abs_floor: f32,
    rel_scale: f32,
) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} size mismatch: expected {} got {}",
            expected.len(),
            actual.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        let diff = (actual_value - expected_value).abs();
        let tolerance = abs_floor.max(expected_value.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} mismatch: max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    Ok(max_abs_diff)
}

fn verify_sigmoid_mul_f32_close(
    label: &str,
    gate: &[f32],
    input: &[f32],
    actual: &[f32],
    abs_floor: f32,
    rel_scale: f32,
) -> Result<f32, String> {
    if gate.len() != input.len() || input.len() != actual.len() {
        return Err(format!(
            "{label} size mismatch: gate={} input={} actual={}",
            gate.len(),
            input.len(),
            actual.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;
    for ((gate_value, input_value), actual_value) in
        gate.iter().zip(input.iter()).zip(actual.iter())
    {
        let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate_value).exp());
        let expected = sigmoid * *input_value;
        let diff = (*actual_value - expected).abs();
        let tolerance = abs_floor.max(expected.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} mismatch: max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        max_abs_diff = max_abs_diff.max(diff);
    }
    Ok(max_abs_diff)
}

fn verify_silu_mul_f32_close(
    label: &str,
    gate: &[f32],
    up: &[f32],
    actual: &[f32],
    abs_floor: f32,
    rel_scale: f32,
) -> Result<f32, String> {
    if gate.len() != up.len() || up.len() != actual.len() {
        return Err(format!(
            "{label} size mismatch: gate={} up={} actual={}",
            gate.len(),
            up.len(),
            actual.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;
    for ((gate_value, up_value), actual_value) in gate.iter().zip(up.iter()).zip(actual.iter()) {
        let gate_value = *gate_value;
        let expected = gate_value * (1.0_f32 / (1.0_f32 + (-gate_value).exp())) * *up_value;
        let diff = (*actual_value - expected).abs();
        let tolerance = abs_floor.max(expected.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} mismatch: max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        max_abs_diff = max_abs_diff.max(diff);
    }
    Ok(max_abs_diff)
}

fn verify_add_f32_close(
    label: &str,
    lhs: &[f32],
    rhs: &[f32],
    actual: &[f32],
    abs_floor: f32,
    rel_scale: f32,
) -> Result<f32, String> {
    if lhs.len() != rhs.len() || rhs.len() != actual.len() {
        return Err(format!(
            "{label} size mismatch: lhs={} rhs={} actual={}",
            lhs.len(),
            rhs.len(),
            actual.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;
    for ((lhs_value, rhs_value), actual_value) in lhs.iter().zip(rhs.iter()).zip(actual.iter()) {
        let expected = *lhs_value + *rhs_value;
        let diff = (*actual_value - expected).abs();
        let tolerance = abs_floor.max(expected.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} mismatch: max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        max_abs_diff = max_abs_diff.max(diff);
    }
    Ok(max_abs_diff)
}

#[allow(clippy::too_many_arguments)]
fn verify_aq4_matvec_batch_output_sampled(
    label: &str,
    matrix: &PackageAq4ResidentMatvec,
    input: &[f32],
    actual: &[f32],
    batch_count: usize,
    row_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    abs_floor: f32,
    rel_scale: f32,
) -> Result<(usize, f32), String> {
    let expected_input_len = batch_count
        .checked_mul(matrix.cols)
        .ok_or_else(|| format!("{label} sampled input length overflows"))?;
    let expected_actual_len = batch_count
        .checked_mul(matrix.rows)
        .ok_or_else(|| format!("{label} sampled output length overflows"))?;
    if input.len() != expected_input_len || actual.len() != expected_actual_len {
        return Err(format!(
            "{label} sampled size mismatch: input={} expected_input={} actual={} expected_actual={}",
            input.len(),
            expected_input_len,
            actual.len(),
            expected_actual_len
        ));
    }
    let sample_points = aq4_matvec_batch_sample_points(batch_count, matrix.rows);
    if sample_points.is_empty() {
        return Err(format!("{label} sampled verification has no sample points"));
    }
    let mut row_cache: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut max_abs_diff = 0.0_f32;
    for (token_index, row_index) in sample_points.iter().copied() {
        let row_position = if let Some(position) = row_cache
            .iter()
            .position(|(cached_row_index, _)| *cached_row_index == row_index)
        {
            position
        } else {
            matrix.row_f32(row_index, row_buffer, stream, label)?;
            let row = read_runtime_buffer_f32(row_buffer, stream, matrix.cols, label)?;
            row_cache.push((row_index, row));
            row_cache.len() - 1
        };
        let row = &row_cache[row_position].1;
        let input_start = token_index
            .checked_mul(matrix.cols)
            .ok_or_else(|| format!("{label} sampled input start overflows"))?;
        let input_end = input_start
            .checked_add(matrix.cols)
            .ok_or_else(|| format!("{label} sampled input end overflows"))?;
        let expected = row
            .iter()
            .zip(input[input_start..input_end].iter())
            .map(|(lhs, rhs)| *lhs * *rhs)
            .sum::<f32>();
        let actual_index = token_index
            .checked_mul(matrix.rows)
            .and_then(|value| value.checked_add(row_index))
            .ok_or_else(|| format!("{label} sampled output index overflows"))?;
        let diff = (actual[actual_index] - expected).abs();
        let tolerance = abs_floor.max(expected.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} sampled mismatch: token_index={token_index} row_index={row_index} max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        max_abs_diff = max_abs_diff.max(diff);
    }
    Ok((sample_points.len(), max_abs_diff))
}

fn aq4_matvec_batch_sample_points(batch_count: usize, rows: usize) -> Vec<(usize, usize)> {
    if batch_count == 0 || rows == 0 {
        return Vec::new();
    }
    let mut token_indices = vec![0, batch_count / 4, batch_count / 2, batch_count - 1];
    if batch_count > 1 {
        token_indices.push(1);
    }
    token_indices.sort_unstable();
    token_indices.dedup();

    let mut row_indices = vec![0, rows / 2, rows - 1];
    row_indices.sort_unstable();
    row_indices.dedup();

    let mut points = Vec::with_capacity(token_indices.len() * row_indices.len());
    for token_index in token_indices {
        for row_index in row_indices.iter().copied() {
            points.push((token_index, row_index));
        }
    }
    points
}

fn self_attn_batch_rope_abs_floor(sequence_len: usize, position_offset: usize) -> f32 {
    let max_position = position_offset
        .saturating_add(sequence_len.saturating_sub(1))
        .max(1) as f32;
    2e-4_f32.max((max_position * 2e-7_f32).min(4e-3_f32))
}

fn self_attn_batch_use_sampled_attention_verification(sequence_len: usize) -> bool {
    sequence_len >= 1024
}

fn causal_attention_sample_points(
    sequence_len: usize,
    q_heads: usize,
    value_dim: usize,
) -> Vec<(usize, usize, usize)> {
    if sequence_len == 0 || q_heads == 0 || value_dim == 0 {
        return Vec::new();
    }
    let mut timesteps = vec![0, sequence_len / 4, sequence_len / 2, sequence_len - 1];
    if sequence_len > 1 {
        timesteps.push(1);
    }
    timesteps.sort_unstable();
    timesteps.dedup();

    let mut head_values = vec![
        (0, 0),
        (q_heads / 2, value_dim / 2),
        (q_heads - 1, value_dim - 1),
    ];
    head_values.sort_unstable();
    head_values.dedup();

    let mut points = Vec::with_capacity(timesteps.len() * head_values.len());
    for timestep in timesteps {
        for (q_head, value_index) in head_values.iter().copied() {
            points.push((timestep, q_head, value_index));
        }
    }
    points
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_causal_attn_f32_sample(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    timestep: usize,
    q_head: usize,
    value_index: usize,
) -> Option<f32> {
    let q_len = sequence_len.checked_mul(q_heads)?.checked_mul(head_dim)?;
    let k_len = sequence_len.checked_mul(kv_heads)?.checked_mul(head_dim)?;
    let v_len = sequence_len.checked_mul(kv_heads)?.checked_mul(value_dim)?;
    if timestep >= sequence_len || q.len() != q_len || k.len() != k_len || v.len() != v_len {
        return None;
    }
    let q_timestep_start = timestep.checked_mul(q_heads)?.checked_mul(head_dim)?;
    let q_timestep_end = q_timestep_start.checked_add(q_heads.checked_mul(head_dim)?)?;
    runtime_host_decode_attn_f32_sample(
        &q[q_timestep_start..q_timestep_end],
        k,
        v,
        timestep.checked_add(1)?,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_head,
        value_index,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_causal_attention_output_sampled(
    label: &str,
    actual: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    abs_floor: f32,
    rel_scale: f32,
) -> Result<(usize, f32), String> {
    let expected_actual_len = sequence_len
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| format!("{label} sampled output element count overflows"))?;
    if actual.len() != expected_actual_len {
        return Err(format!(
            "{label} sampled output size mismatch: expected {} got {}",
            expected_actual_len,
            actual.len()
        ));
    }
    let sample_points = causal_attention_sample_points(sequence_len, q_heads, value_dim);
    if sample_points.is_empty() {
        return Err(format!("{label} sampled verification has no sample points"));
    }
    let mut max_abs_diff = 0.0_f32;
    for (timestep, q_head, value_index) in sample_points.iter().copied() {
        let expected = runtime_host_causal_attn_f32_sample(
            q,
            k,
            v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            timestep,
            q_head,
            value_index,
        )
        .ok_or_else(|| {
            format!(
                "{label} sampled reference failed at timestep={timestep} q_head={q_head} value_index={value_index}"
            )
        })?;
        let actual_index = (timestep * q_heads + q_head) * value_dim + value_index;
        let diff = (actual[actual_index] - expected).abs();
        let tolerance = abs_floor.max(expected.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} sampled mismatch: timestep={timestep} q_head={q_head} value_index={value_index} max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        max_abs_diff = max_abs_diff.max(diff);
    }
    Ok((sample_points.len(), max_abs_diff))
}

fn deterministic_f32_vector(elements: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(((index as f32).sin() + 1.0_f32) / 2.0_f32);
    }
    values
}

fn linear_attn_step_input(base_input: &[f32], timestep: usize) -> Vec<f32> {
    base_input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let phase = (index % 17) as f32 - 8.0_f32;
            *value + (timestep as f32) * phase * 0.00025_f32
        })
        .collect()
}

fn deterministic_linear_attn_core_output(
    sequence_len: usize,
    value_heads: usize,
    value_dim: usize,
) -> Vec<f32> {
    let elements = sequence_len * value_heads * value_dim;
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        let head_phase = ((index / value_dim) % value_heads) as f32 * 0.0007_f32;
        let dim_phase = (index % value_dim) as f32 * 0.00011_f32;
        values.push(((index as f32 * 0.013_f32).sin() * 0.05_f32) + head_phase - dim_phase);
    }
    values
}

struct LinearAttnQkvSplit {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn split_linear_attn_qkv_for_recurrent(
    conv_output: &[f32],
    sequence_len: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    qk_l2_norm: bool,
    q_scale: f32,
) -> Result<LinearAttnQkvSplit, String> {
    if sequence_len == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err("linear attention q/k/v layout contains a zero dimension".to_string());
    }
    let q_elements_per_step = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "q element count overflows".to_string())?;
    let k_elements_per_step = q_elements_per_step;
    let v_elements_per_step = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "v element count overflows".to_string())?;
    let step_elements = q_elements_per_step
        .checked_add(k_elements_per_step)
        .and_then(|value| value.checked_add(v_elements_per_step))
        .ok_or_else(|| "linear attention q/k/v step element count overflows".to_string())?;
    let expected_elements = step_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "linear attention q/k/v sequence element count overflows".to_string())?;
    if conv_output.len() != expected_elements {
        return Err(format!(
            "conv output element count mismatch: expected {expected_elements} got {}",
            conv_output.len()
        ));
    }

    let mut q = vec![0.0_f32; sequence_len * q_elements_per_step];
    let mut k = vec![0.0_f32; sequence_len * k_elements_per_step];
    let mut v = vec![0.0_f32; sequence_len * v_elements_per_step];
    for timestep in 0..sequence_len {
        let step_base = timestep * step_elements;
        let q_base = step_base;
        let k_base = q_base + q_elements_per_step;
        let v_base = k_base + k_elements_per_step;

        for head in 0..key_heads {
            let source_start = q_base + head * key_dim;
            let target_start = (timestep * key_heads + head) * key_dim;
            q[target_start..target_start + key_dim]
                .copy_from_slice(&conv_output[source_start..source_start + key_dim]);
            if qk_l2_norm {
                let norm = (q[target_start..target_start + key_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6_f32)
                    .sqrt();
                for value in &mut q[target_start..target_start + key_dim] {
                    *value = (*value / norm) * q_scale;
                }
            } else {
                for value in &mut q[target_start..target_start + key_dim] {
                    *value *= q_scale;
                }
            }

            let source_start = k_base + head * key_dim;
            let target_start = (timestep * key_heads + head) * key_dim;
            k[target_start..target_start + key_dim]
                .copy_from_slice(&conv_output[source_start..source_start + key_dim]);
            if qk_l2_norm {
                let norm = (k[target_start..target_start + key_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6_f32)
                    .sqrt();
                for value in &mut k[target_start..target_start + key_dim] {
                    *value /= norm;
                }
            }
        }

        let target_v_base = timestep * v_elements_per_step;
        v[target_v_base..target_v_base + v_elements_per_step]
            .copy_from_slice(&conv_output[v_base..v_base + v_elements_per_step]);
    }
    Ok(LinearAttnQkvSplit { q, k, v })
}

fn print_help() {
    eprintln!(
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|runtime-copy-smoke [DEVICE_INDEX]|runtime-rmsnorm-smoke [DEVICE_INDEX]|runtime-silu-mul-smoke [DEVICE_INDEX]|runtime-sigmoid-mul-smoke [DEVICE_INDEX]|runtime-add-smoke [DEVICE_INDEX]|runtime-rope-smoke [DEVICE_INDEX]|runtime-causal-attn-smoke [DEVICE_INDEX]|runtime-causal-attn-batch-smoke [DEVICE_INDEX] [BATCH_COUNT] [SEQUENCE_LEN] [MEASURED_REPEATS] [Q_HEADS] [KV_HEADS] [HEAD_DIM] [VALUE_DIM] [EXECUTOR=causal_attn_batch_f32|default|flash2|causal_attn_batch_f32_flash2]|runtime-decode-attn-smoke [DEVICE_INDEX]|runtime-cached-prefix-attn-smoke [DEVICE_INDEX] [CACHED_PREFIX_TOKENS] [NEW_TOKENS] [MEASURED_REPEATS] [Q_HEADS] [KV_HEADS] [HEAD_DIM] [VALUE_DIM] [EXECUTOR=cached_prefix_chunked|cached_prefix_flash2|cached_prefix_flash2_fp8q|cached_prefix_rocwmma_fp8|cached_prefix_rdna4_fp8_auto|decode_loop] [KV_CACHE_DTYPE=fp8_e4m3|f32]|runtime-paged-decode-attn-smoke [DEVICE_INDEX]|runtime-paged-kv-write-smoke [DEVICE_INDEX]|runtime-scheduler-paged-decode-smoke [DEVICE_INDEX]|runtime-scheduler-layer-decode-smoke [DEVICE_INDEX]|runtime-kv-paged-decode-smoke [DEVICE_INDEX]|runtime-depthwise-conv1d-smoke [DEVICE_INDEX]|runtime-wmma-fp8-probe-smoke [DEVICE_INDEX]|runtime-wmma-fp8-qk-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]|runtime-rocwmma-fp8-qk-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]|runtime-rocwmma-fp8-attn-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout]|runtime-linear-attn-gate-beta-smoke [DEVICE_INDEX]|runtime-linear-attn-recurrent-smoke [DEVICE_INDEX]|runtime-mlp-smoke [DEVICE_INDEX]|inspect-package PATH|package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]|package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]|package-materialize-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-materialize-matvec-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-rmsnorm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-rmsnorm-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-linear-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a|b|qkv|z|out|all]|package-self-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [q|k|v|o|all]|package-self-attn-qk-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-self-attn-rope-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-attention-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-decode-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-scheduler-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-model-loop-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX,...|FIRST_LAYER_INDEX SECOND_LAYER_INDEX[,...]] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-token-ids-logits-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all] [TOKEN_IDS_CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-layer-golden-smoke PACKAGE_DIR GOLDEN_FIXTURE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-golden-prefix-smoke PACKAGE_DIR GOLDEN_FIXTURE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_START] [LAYER_END_EXCLUSIVE] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET] [REPORT_PATH] [RUN_MODE] [ROW_SCALE_OVERRIDES_JSON] [INPUT_DUMP_DIR] [SAMPLED_TOKEN_INDICES] [CELL_DELTA_OVERRIDES_JSON]|package-linear-attn-qkv-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-linear-attn-conv1d-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-gate-beta-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-recurrent-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-post-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-workflow-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-aux-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a-log|dt-bias|conv1d|norm|all]|package-materialize-bench PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR] [REPEATS]|package-prefill-aq4-matvec-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_NAME] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-self-attn-qkv-rope-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-self-attn-attention-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-self-attn-block-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-linear-attn-recurrent-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-linear-attn-post-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-linear-attn-attention-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-linear-attn-mlp-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]|package-linear-attn-layer-batch-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]>"
    );
    eprintln!(
        "package-aq4-matvec-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR] [REPEATS]"
    );
    eprintln!("package-layer-kind-inventory-smoke: PACKAGE_DIR [LAYERS_CSV|manifest-all|all]");
    eprintln!(
        "package-token-ids-generate-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all|manifest-all] [TOKEN_IDS_CSV|len:N] [GENERATED_TOKENS] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET] [LM_HEAD_MODE=cpu_chunked|gpu_resident_f32] [STOP_TOKEN_IDS_CSV|none] [STOP_TOKEN_SEQUENCES=SEQ1;SEQ2|none]"
    );
    eprintln!(
        "sq-fp8-token-ids-generate-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all|manifest-all] [TOKEN_IDS_CSV|len:N] [GENERATED_TOKENS] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET] [LM_HEAD_MODE=cpu_chunked|gpu_resident_f32] [STOP_TOKEN_IDS_CSV|none] [STOP_TOKEN_SEQUENCES=SEQ1;SEQ2|none]"
    );
    eprintln!(
        "package-token-ids-logits-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all|manifest-all] [TOKEN_IDS_CSV|len:N] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-token-ids-model-loop-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|default|all-self-attn|manifest-self-attn] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-token-ids-mixed-request-state-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|manifest-all] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K|0=skip_final_logits] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "sq-fp8-token-ids-model-loop-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|default|all-self-attn|manifest-self-attn] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "sq-fp8-token-ids-mixed-request-state-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|manifest-all] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K|0=skip_final_logits] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "sq-fp8-token-ids-offline-serving-throughput: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|manifest-all] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K|0=skip_final_logits] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "sq-fp8-package-self-attn-stack-batch-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all-self-attn|manifest-self-attn] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K|0=skip_final_logits] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "sq-fp8-token-ids-logits-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all|manifest-all] [TOKEN_IDS_CSV|len:N] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "runtime-cached-prefix-attn-smoke: [DEVICE_INDEX] [CACHED_PREFIX_TOKENS] [NEW_TOKENS] [MEASURED_REPEATS] [Q_HEADS] [KV_HEADS] [HEAD_DIM] [VALUE_DIM] [EXECUTOR=cached_prefix_chunked|cached_prefix_flash2|cached_prefix_flash2_fp8q|cached_prefix_rocwmma_fp8|cached_prefix_rdna4_fp8_auto|decode_loop] [KV_CACHE_DTYPE=fp8_e4m3|f32]"
    );
    eprintln!(
        "runtime-wmma-fp8-qk-probe-smoke: [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]"
    );
    eprintln!(
        "runtime-rocwmma-fp8-qk-probe-smoke: [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]"
    );
    eprintln!("runtime-rocwmma-fp8-attn-probe-smoke: [DEVICE_INDEX] [PATTERN=ones|layout]");
    eprintln!(
        "package-token-ids-bench: same arguments as package-token-ids-generate-smoke; writes the same measured JSON report"
    );
    eprintln!(
        "sq-fp8-token-ids-bench: same arguments as sq-fp8-token-ids-generate-smoke; writes the same measured JSON report"
    );
    eprintln!(
        "package-batch-throughput-bench: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all|manifest-all] [TOKEN_IDS_BATCH|len:NxM|REQ1;REQ2] [GENERATED_TOKENS|CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET] [LM_HEAD_MODE=cpu_chunked|gpu_resident_f32] [STOP_TOKEN_IDS_CSV|none] [STOP_TOKEN_SEQUENCES=SEQ1;SEQ2|none]"
    );
    eprintln!(
        "sq-fp8-materialize-smoke: ARTIFACT_DIR [DEVICE_INDEX] [TENSOR_SELECTOR] [ROW_COUNT] [START_ROW]"
    );
    eprintln!(
        "sq-fp8-package-self-attn-layer-batch-smoke: PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS] [ROW_CHUNK] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-prefill-rmsnorm-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-prefill-aq4-matvec-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_NAME] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-self-attn-qkv-rope-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-self-attn-attention-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-linear-attn-proj-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-qkv-prepare-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-recurrent-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-post-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-attention-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-self-attn-block-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-self-attn-layer-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
    );
    eprintln!(
        "package-linear-attn-mlp-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-layer-batch-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [TOKEN_IDS_CSV|len:N] [MEASURED_REPEATS]"
    );
    eprintln!(
        "package-linear-attn-stateful-step-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]"
    );
    eprintln!(
        "package-linear-attn-request-state-smoke: PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [REQUEST_COUNT<=4] [SEQUENCE_LEN]"
    );
    eprintln!("linear attention projection selector: a|b|qkv|z|out|all");
    eprintln!("self attention projection selector: q|k|v|o|all (alias: out for o)");
    eprintln!(
        "model-loop layer list: use LAYER_INDEX,... or FIRST_LAYER_INDEX SECOND_LAYER_INDEX[,...]"
    );
    eprintln!(
        "linear attention aux selector: a-log|dt-bias|conv1d|norm|all (aliases: a_log|alog|dt_bias)"
    );
    eprintln!(
        "payload roles: smallest|tensor-index|tensor-scale|tensor-codebook|codebook|passthrough"
    );
    eprintln!("tensor selector: omitted or numeric index, exact tensor name, or unique substring");
}

fn format_u64_shape(shape: &[u64]) -> String {
    let rendered = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rendered}]")
}

fn parse_optional_device_index(device_index: Option<String>) -> Result<u32, ExitCode> {
    match device_index {
        Some(value) => match value.parse::<u32>() {
            Ok(value) => Ok(value),
            Err(err) => {
                eprintln!("invalid device index: {err}");
                Err(ExitCode::from(2))
            }
        },
        None => Ok(0),
    }
}

fn is_rdna4_device(info: &ullm_runtime_sys::DeviceInfo) -> bool {
    info.compute_major == 12 || info.gcn_arch_name.starts_with("gfx12")
}

fn parse_optional_usize(
    value: Option<String>,
    default: usize,
    label: &str,
) -> Result<usize, ExitCode> {
    match value {
        Some(value) => match value.parse::<usize>() {
            Ok(value) => Ok(value),
            Err(err) => {
                eprintln!("invalid {label}: {err}");
                Err(ExitCode::from(2))
            }
        },
        None => Ok(default),
    }
}

fn parse_optional_f32(value: Option<String>, default: f32, label: &str) -> Result<f32, ExitCode> {
    match value {
        Some(value) => match value.parse::<f32>() {
            Ok(value) if value.is_finite() => Ok(value),
            Ok(_) => {
                eprintln!("invalid {label}: value must be finite");
                Err(ExitCode::from(2))
            }
            Err(err) => {
                eprintln!("invalid {label}: {err}");
                Err(ExitCode::from(2))
            }
        },
        None => Ok(default),
    }
}

fn parse_optional_payload_role(value: Option<String>) -> Result<ReferencedFileRole, ExitCode> {
    match value {
        Some(value) => ReferencedFileRole::parse(&value).ok_or_else(|| {
            eprintln!(
                "invalid payload role: {value}; expected smallest, tensor-index, tensor-scale, tensor-codebook, codebook, or passthrough"
            );
            ExitCode::from(2)
        }),
        None => Ok(ReferencedFileRole::Smallest),
    }
}

fn read_bounded_file(path: &std::path::Path, max_bytes: usize) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let limit = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    let mut reader = file.take(limit);
    let mut data = Vec::with_capacity(max_bytes.min(1024 * 1024));
    reader
        .read_to_end(&mut data)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    Ok(data)
}

#[derive(Debug, Clone, Copy)]
struct FileRoundtripSummary {
    bytes: u64,
    chunks: u64,
}

fn roundtrip_file_chunks(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    referenced: &ReferencedFile,
    chunk_bytes: usize,
) -> Result<FileRoundtripSummary, String> {
    if chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let mut file = File::open(&referenced.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            referenced.absolute_path.display()
        )
    })?;
    let capacity = usize::try_from(referenced.bytes)
        .ok()
        .map_or(chunk_bytes, |bytes| bytes.min(chunk_bytes));
    if capacity == 0 {
        return Err(format!(
            "referenced file {} is empty",
            referenced.absolute_path.display()
        ));
    }
    let mut buffer = context.alloc_buffer(capacity)?;
    let mut input = vec![0_u8; capacity];
    let mut output = vec![0_u8; capacity];
    let mut total = 0_u64;
    let mut chunks = 0_u64;

    loop {
        let read = file.read(&mut input).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                referenced.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        buffer.copy_from_host(0, &input[..read], Some(stream))?;
        stream.synchronize()?;
        buffer.copy_to_host(0, &mut output[..read], Some(stream))?;
        stream.synchronize()?;
        if input[..read] != output[..read] {
            return Err(format!(
                "runtime roundtrip mismatch for {} at chunk {}",
                referenced.relative_path, chunks
            ));
        }
        total += read as u64;
        chunks += 1;
    }

    if total != referenced.bytes {
        return Err(format!(
            "roundtrip byte count mismatch for {}: expected {} got {}",
            referenced.relative_path, referenced.bytes, total
        ));
    }
    Ok(FileRoundtripSummary {
        bytes: total,
        chunks,
    })
}

fn print_file_roundtrip_summary(
    role: &str,
    referenced: &ReferencedFile,
    summary: &FileRoundtripSummary,
) {
    println!(
        "  file role={} path={} bytes={} chunks={} verified=true",
        role, referenced.relative_path, summary.bytes, summary.chunks
    );
}

fn print_loaded_payload_summary(payload: &LoadedPayload) {
    let buffer_bytes = payload
        .buffer
        .size()
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|err| format!("error:{err}"));
    println!(
        "  registered role={} path={} bytes={} chunks={} buffer_bytes={} resident=true",
        payload.role.as_str(),
        payload.relative_path,
        payload.bytes,
        payload.chunks,
        buffer_bytes
    );
}

fn runtime_host_rmsnorm_f32(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    if input.len() != weight.len() || input.is_empty() {
        return Vec::new();
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(input_value, weight_value)| input_value * inv_rms * weight_value)
        .collect()
}

fn runtime_host_matvec_f32(matrix: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if rows == 0 || cols == 0 || matrix.len() != rows * cols || input.len() != cols {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut value = 0.0_f32;
        let row_start = row * cols;
        for col in 0..cols {
            value += matrix[row_start + col] * input[col];
        }
        output.push(value);
    }
    output
}

fn runtime_host_silu_mul_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    if gate.len() != up.len() {
        return Vec::new();
    }
    gate.iter()
        .zip(up.iter())
        .map(|(gate_value, up_value)| {
            let gate_value = *gate_value;
            gate_value * (1.0_f32 / (1.0_f32 + (-gate_value).exp())) * *up_value
        })
        .collect()
}

fn runtime_host_silu_f32(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            let value = *value;
            value * (1.0_f32 / (1.0_f32 + (-value).exp()))
        })
        .collect()
}

fn runtime_host_sigmoid_mul_f32(gate: &[f32], input: &[f32]) -> Vec<f32> {
    if gate.len() != input.len() {
        return Vec::new();
    }
    gate.iter()
        .zip(input.iter())
        .map(|(gate_value, input_value)| {
            let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate_value).exp());
            sigmoid * *input_value
        })
        .collect()
}

fn runtime_host_add_f32(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    if lhs.len() != rhs.len() {
        return Vec::new();
    }
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| lhs_value + rhs_value)
        .collect()
}

fn runtime_host_rope_f32(
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
) -> Vec<f32> {
    if sequence_len == 0
        || heads == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.len() != sequence_len * heads * head_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; input.len()];
    let half = rotary_dim / 2;
    for timestep in 0..sequence_len {
        let position = (position_offset + timestep) as f32;
        for head in 0..heads {
            let base = (timestep * heads + head) * head_dim;
            for pair_dim in 0..half {
                let exponent = (2.0 * pair_dim as f32) / rotary_dim as f32;
                let theta = position / rope_base.powf(exponent);
                let c = theta.cos();
                let s = theta.sin();
                let first = input[base + pair_dim];
                let second = input[base + half + pair_dim];
                output[base + pair_dim] = first * c - second * s;
                output[base + half + pair_dim] = second * c + first * s;
            }
            output[base + rotary_dim..base + head_dim]
                .copy_from_slice(&input[base + rotary_dim..base + head_dim]);
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_causal_attn_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if sequence_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != sequence_len * q_heads * head_dim
        || k.len() != sequence_len * kv_heads * head_dim
        || v.len() != sequence_len * kv_heads * value_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; sequence_len * q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for timestep in 0..sequence_len {
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = (timestep * q_heads + q_head) * head_dim;
            let mut scores = Vec::with_capacity(timestep + 1);
            for source_timestep in 0..=timestep {
                let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                let score = (0..head_dim)
                    .map(|dim| q[q_base + dim] * k[k_base + dim])
                    .sum::<f32>()
                    * softmax_scale;
                scores.push(score);
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
            let output_base = (timestep * q_heads + q_head) * value_dim;
            for value in 0..value_dim {
                let mut weighted = 0.0_f32;
                for (source_timestep, weight) in weights.iter().enumerate() {
                    let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted += *weight * v[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_decode_attn_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != q_heads * head_dim
        || k_cache.len() != cache_len * kv_heads * head_dim
        || v_cache.len() != cache_len * kv_heads * value_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache_len);
        for source_timestep in 0..cache_len {
            let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            let score = (0..head_dim)
                .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                .sum::<f32>()
                * softmax_scale;
            scores.push(score);
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
        let output_base = q_head * value_dim;
        for value in 0..value_dim {
            let mut weighted = 0.0_f32;
            for (source_timestep, weight) in weights.iter().enumerate() {
                let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += *weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_paged_decode_attn_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if cache_len == 0
        || block_size == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != q_heads * head_dim
        || block_table.len() < (cache_len - 1) / block_size + 1
    {
        return Vec::new();
    }
    let physical_tokens = k_cache.len() / (kv_heads * head_dim);
    if physical_tokens * kv_heads * head_dim != k_cache.len()
        || physical_tokens * kv_heads * value_dim != v_cache.len()
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache_len);
        for source_timestep in 0..cache_len {
            let block_index = source_timestep / block_size;
            let block_offset = source_timestep - block_index * block_size;
            let physical_timestep = block_table[block_index] as usize * block_size + block_offset;
            if physical_timestep >= physical_tokens {
                return Vec::new();
            }
            let k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            let score = (0..head_dim)
                .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                .sum::<f32>()
                * softmax_scale;
            scores.push(score);
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
        let output_base = q_head * value_dim;
        for value in 0..value_dim {
            let mut weighted = 0.0_f32;
            for (source_timestep, weight) in weights.iter().enumerate() {
                let block_index = source_timestep / block_size;
                let block_offset = source_timestep - block_index * block_size;
                let physical_timestep =
                    block_table[block_index] as usize * block_size + block_offset;
                let v_index = (physical_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += *weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
    output
}

struct ScheduledPagedDecodeBlocks {
    block_table: Vec<u32>,
    cache_blocks: usize,
    allocator_stats: KvBlockAllocatorStats,
    request_id: RequestId,
    prefill_tokens: usize,
    max_new_tokens: usize,
    cached_tokens: usize,
    generated_tokens: usize,
    active_len: usize,
}

struct PreparedFragmentedPagedDecodeState {
    scheduler: SchedulerState,
    block_table: Vec<u32>,
    cache_blocks: usize,
    request_id: RequestId,
    prefill_tokens: usize,
    max_new_tokens: usize,
}

fn prepare_fragmented_paged_decode_state(
    cache_len: usize,
    block_size: usize,
) -> Result<PreparedFragmentedPagedDecodeState, String> {
    if cache_len == 0 {
        return Err("paged decode cache_len must be greater than zero".to_string());
    }
    if block_size == 0 {
        return Err("paged decode block_size must be greater than zero".to_string());
    }
    if block_size > u32::MAX as usize {
        return Err(format!(
            "paged decode block_size={block_size} exceeds u32 block size range"
        ));
    }

    let block_count = (cache_len - 1) / block_size + 1;
    if block_count > u32::MAX as usize - 2 {
        return Err(format!(
            "paged decode block_count={block_count} is too large for allocator smoke"
        ));
    }
    let cache_blocks = block_count
        .checked_add(2)
        .ok_or_else(|| "paged decode cache_blocks overflows".to_string())?;

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    let fragment_blocks = cache_blocks - 1;
    let fragment_tokens = fragment_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "paged decode fragment token count overflows".to_string())?;
    scheduler.enqueue(Request {
        id: RequestId(100),
        prompt_tokens: fragment_tokens,
        max_new_tokens: 0,
    });
    let fragment_batch = scheduler
        .pop_prefill_batch_with_allocation(fragment_tokens)
        .map_err(|err| format!("failed to allocate fragmenting KV blocks: {err}"))?;
    let fragment = fragment_batch
        .first()
        .ok_or_else(|| "fragmenting KV allocation returned an empty batch".to_string())?;
    let freed = scheduler.release_request(fragment.allocation.request_id);
    if freed != fragment.allocation.blocks.len() {
        return Err(format!(
            "freed KV block count {freed} does not match allocated fragment blocks {}",
            fragment.allocation.blocks.len()
        ));
    }

    let request_id = RequestId(101);
    let (prefill_prompt_tokens, max_new_tokens) = if cache_len > 1 {
        (cache_len - 1, 1)
    } else {
        (cache_len, 0)
    };
    scheduler.enqueue(Request {
        id: request_id,
        prompt_tokens: prefill_prompt_tokens,
        max_new_tokens,
    });
    let mut decode_batch = scheduler
        .pop_prefill_batch_with_allocation(prefill_prompt_tokens)
        .map_err(|err| format!("failed to allocate decode KV blocks: {err}"))?;
    if decode_batch.len() != 1 {
        return Err(format!(
            "decode KV allocation selected {} requests, expected 1",
            decode_batch.len()
        ));
    }
    let allocation = decode_batch.remove(0).allocation;
    if allocation.blocks.len() != block_count {
        return Err(format!(
            "decode KV allocation block count {} does not match expected {block_count}",
            allocation.blocks.len()
        ));
    }

    Ok(PreparedFragmentedPagedDecodeState {
        scheduler,
        block_table: allocation.blocks,
        cache_blocks,
        request_id,
        prefill_tokens: prefill_prompt_tokens,
        max_new_tokens,
    })
}

fn allocate_fragmented_paged_decode_blocks(
    cache_len: usize,
    block_size: usize,
) -> Result<ScheduledPagedDecodeBlocks, String> {
    let prepared = prepare_fragmented_paged_decode_state(cache_len, block_size)?;
    let mut scheduler = prepared.scheduler;
    let cache_blocks = prepared.cache_blocks;
    let request_id = prepared.request_id;

    scheduler
        .complete_prefill(request_id)
        .map_err(|err| format!("failed to complete decode prefill: {err}"))?;

    if prepared.max_new_tokens > 0 {
        let mut ready = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to prepare ready decode batch: {err}"))?;
        let request = ready
            .pop()
            .ok_or_else(|| "expected one ready decode request after prefill".to_string())?;
        if request.request.id != request_id {
            return Err(format!(
                "ready decode request {:?} does not match expected {:?}",
                request.request.id, request_id
            ));
        }
        if request.cache_position != prepared.prefill_tokens {
            return Err(format!(
                "ready decode cache_position {} does not match prefill tokens {}",
                request.cache_position, prepared.prefill_tokens
            ));
        }
        scheduler
            .advance_decode(request_id)
            .map_err(|err| format!("failed to advance decode by one token: {err}"))?;
    }

    let active = scheduler
        .active_request(request_id)
        .ok_or_else(|| "decode request is not active after scheduler progress".to_string())?;
    let cached_tokens = active.cached_tokens;
    let generated_tokens = active.generated_tokens;
    let active_len = scheduler.active_len();
    let stats = scheduler.allocator_stats();
    Ok(ScheduledPagedDecodeBlocks {
        block_table: prepared.block_table,
        cache_blocks,
        allocator_stats: stats,
        request_id,
        prefill_tokens: prepared.prefill_tokens,
        max_new_tokens: prepared.max_new_tokens,
        cached_tokens,
        generated_tokens,
        active_len,
    })
}

fn runtime_host_depthwise_conv1d_f32(
    input: &[f32],
    weight: &[f32],
    channels: usize,
    sequence_len: usize,
    kernel_size: usize,
) -> Vec<f32> {
    if channels == 0
        || sequence_len == 0
        || kernel_size == 0
        || input.len() != channels * sequence_len
        || weight.len() != channels * kernel_size
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; channels * sequence_len];
    for timestep in 0..sequence_len {
        for channel in 0..channels {
            let mut value = 0.0_f32;
            for kernel in 0..kernel_size {
                let left_padding = kernel_size - 1 - kernel;
                if timestep < left_padding {
                    continue;
                }
                value += input[(timestep - left_padding) * channels + channel]
                    * weight[channel * kernel_size + kernel];
            }
            output[timestep * channels + channel] = value;
        }
    }
    output
}

#[derive(Debug, Clone)]
struct LinearAttnConv1dStepState {
    channels: usize,
    kernel_size: usize,
    history: Vec<f32>,
    seen_tokens: usize,
}

impl LinearAttnConv1dStepState {
    fn new(channels: usize, kernel_size: usize) -> Result<Self, String> {
        if channels == 0 {
            return Err("linear attention conv1d step channels must be greater than zero".into());
        }
        if kernel_size == 0 {
            return Err(
                "linear attention conv1d step kernel_size must be greater than zero".into(),
            );
        }
        let history_len = channels
            .checked_mul(kernel_size)
            .ok_or_else(|| "linear attention conv1d step history size overflows".to_string())?;
        Ok(Self {
            channels,
            kernel_size,
            history: vec![0.0_f32; history_len],
            seen_tokens: 0,
        })
    }

    fn step(&mut self, current: &[f32], weight: &[f32]) -> Result<Vec<f32>, String> {
        if current.len() != self.channels {
            return Err(format!(
                "linear attention conv1d step input length mismatch: got {} expected {}",
                current.len(),
                self.channels
            ));
        }
        let expected_weight = self
            .channels
            .checked_mul(self.kernel_size)
            .ok_or_else(|| "linear attention conv1d step weight size overflows".to_string())?;
        if weight.len() != expected_weight {
            return Err(format!(
                "linear attention conv1d step weight length mismatch: got {} expected {}",
                weight.len(),
                expected_weight
            ));
        }

        if self.kernel_size > 1 {
            self.history.rotate_left(self.channels);
        }
        let latest_start = (self.kernel_size - 1) * self.channels;
        self.history[latest_start..latest_start + self.channels].copy_from_slice(current);
        self.seen_tokens = self
            .seen_tokens
            .checked_add(1)
            .ok_or_else(|| "linear attention conv1d step count overflows".to_string())?;

        let mut output = vec![0.0_f32; self.channels];
        for channel in 0..self.channels {
            let mut value = 0.0_f32;
            for kernel in 0..self.kernel_size {
                value += self.history[kernel * self.channels + channel]
                    * weight[channel * self.kernel_size + kernel];
            }
            output[channel] = value;
        }
        Ok(output)
    }
}

fn checked_f32_byte_len(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte size overflows"))
}

fn read_runtime_buffer_f32(
    buffer: &ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    elements: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; checked_f32_byte_len(elements, label)?];
    buffer
        .copy_to_host(0, &mut bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} from runtime: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize {label} runtime copy: {err}"))?;
    Ok(decode_f32_le_values(&bytes))
}

fn read_runtime_buffer_f32_scalar(
    buffer: &ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    element_index: usize,
    label: &str,
) -> Result<f32, String> {
    let offset = element_index
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} scalar byte offset overflows"))?;
    let mut bytes = [0_u8; std::mem::size_of::<f32>()];
    buffer
        .copy_to_host(offset, &mut bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} scalar from runtime: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize {label} scalar runtime copy: {err}"))?;
    Ok(f32::from_le_bytes(bytes))
}
