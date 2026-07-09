fn package_aq4_matvec_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
    repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-aq4-matvec-smoke requires a .ullm.d path");
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
    let repeats = match parse_optional_usize(repeats, 10, "repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };
    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (rows, cols) = match matrix_shape_rows_cols(&loaded.shape, materialize.elements) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let output_byte_count = match rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("matvec output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut row_scale_values = None;
    let mut row_scale_buffer = None;
    if !bundle.row_scale_overrides.is_empty() {
        let mut row_scales = vec![1.0_f32; rows];
        for entry in &bundle.row_scale_overrides {
            if entry.row_index >= rows || !entry.scale.is_finite() {
                eprintln!(
                    "invalid row scale override for {} row {} scale {}",
                    loaded.tensor_name, entry.row_index, entry.scale
                );
                return ExitCode::from(1);
            }
            row_scales[entry.row_index] *= entry.scale;
        }
        let mut buffer = match context.alloc_buffer(rows * std::mem::size_of::<f32>()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate row scale buffer: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) =
            buffer.copy_from_host(0, &encode_f32_to_bytes(&row_scales), Some(&mut stream))
        {
            eprintln!("failed to copy row scales into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        row_scale_values = Some(row_scales);
        row_scale_buffer = Some(buffer);
    }
    let row_scale_count = if row_scale_buffer.is_some() { rows } else { 0 };

    let mut scale_values_buffer =
        match context.alloc_buffer(materialize.scale_values.len() * std::mem::size_of::<f32>()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate scale values buffer: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = scale_values_buffer.copy_from_host(
        0,
        &encode_f32_to_bytes(&materialize.scale_values),
        Some(&mut stream),
    ) {
        eprintln!("failed to copy scale values into runtime buffer: {err}");
        return ExitCode::from(1);
    }

    let mut input = Vec::with_capacity(cols);
    for i in 0..cols {
        input.push(((i % 17) as f32 - 8.0) / 16.0);
    }
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input vector into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after AQ4 matvec setup: {err}");
        return ExitCode::from(1);
    }

    let mut matrix = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut matrix,
        Some(&mut stream),
    ) {
        eprintln!("failed to materialize AQ4 tensor: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after materialize: {err}");
        return ExitCode::from(1);
    }

    let mut f32_output = match context.alloc_buffer(output_byte_count) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate f32 output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut aq4_output = match context.alloc_buffer(output_byte_count) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate AQ4 output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &matrix,
        &input_buffer,
        rows,
        cols,
        &mut f32_output,
        Some(&mut stream),
    ) {
        eprintln!("failed to warm up f32 matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::aq4_matvec_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &scale_values_buffer,
        &input_buffer,
        row_scale_buffer.as_ref(),
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        row_scale_count,
        rows,
        cols,
        &mut aq4_output,
        Some(&mut stream),
    ) {
        eprintln!("failed to warm up AQ4 matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after matvec warmup: {err}");
        return ExitCode::from(1);
    }

    let mut f32_step_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &matrix,
            &input_buffer,
            rows,
            cols,
            &mut f32_output,
            Some(&mut stream),
        ) {
            eprintln!("failed to run f32 matvec: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after f32 matvec: {err}");
            return ExitCode::from(1);
        }
        f32_step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let mut aq4_step_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        if let Err(err) = ullm_runtime_sys::aq4_matvec_f32(
            loaded.index.buffer.as_ref(),
            loaded.scale.buffer.as_ref(),
            loaded.codebook.buffer.as_ref(),
            &scale_values_buffer,
            &input_buffer,
            row_scale_buffer.as_ref(),
            materialize.scale_values.len(),
            materialize.group_size,
            materialize.tensor_scale,
            row_scale_count,
            rows,
            cols,
            &mut aq4_output,
            Some(&mut stream),
        ) {
            eprintln!("failed to run AQ4 matvec: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after AQ4 matvec: {err}");
            return ExitCode::from(1);
        }
        aq4_step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let mut f32_bytes = vec![0_u8; output_byte_count];
    if let Err(err) = f32_output.copy_to_host(0, &mut f32_bytes, Some(&mut stream)) {
        eprintln!("failed to copy f32 matvec output back to host: {err}");
        return ExitCode::from(1);
    }
    let mut aq4_bytes = vec![0_u8; output_byte_count];
    if let Err(err) = aq4_output.copy_to_host(0, &mut aq4_bytes, Some(&mut stream)) {
        eprintln!("failed to copy AQ4 matvec output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copies: {err}");
        return ExitCode::from(1);
    }
    let mut f32_values = decode_f32_le_values(&f32_bytes);
    if let Some(row_scales) = row_scale_values.as_ref() {
        for (value, row_scale) in f32_values.iter_mut().zip(row_scales.iter()) {
            *value *= *row_scale;
        }
    }
    let aq4_values = decode_f32_le_values(&aq4_bytes);
    let mut max_abs_diff = 0.0_f32;
    let mut sum_sq_diff = 0.0_f64;
    let mut max_ref_abs = 0.0_f32;
    let mut nan_or_inf = false;
    for (aq4_value, f32_value) in aq4_values.iter().zip(f32_values.iter()) {
        if !aq4_value.is_finite() || !f32_value.is_finite() {
            nan_or_inf = true;
        }
        let diff = (aq4_value - f32_value).abs();
        max_abs_diff = max_abs_diff.max(diff);
        max_ref_abs = max_ref_abs.max(f32_value.abs());
        sum_sq_diff += f64::from(diff) * f64::from(diff);
    }
    let rms_diff = if rows > 0 {
        (sum_sq_diff / rows as f64).sqrt()
    } else {
        0.0
    };
    let tolerance = 1e-3_f32.max(max_ref_abs * 1e-5_f32);
    if nan_or_inf || max_abs_diff > tolerance {
        eprintln!(
            "AQ4 matvec mismatch: max_abs_diff={max_abs_diff:.9} tolerance={tolerance:.9} rms_diff={rms_diff:.9}"
        );
        return ExitCode::from(1);
    }

    let f32_mean_ms = f32_step_ms.iter().sum::<f64>() / f32_step_ms.len() as f64;
    let aq4_mean_ms = aq4_step_ms.iter().sum::<f64>() / aq4_step_ms.len() as f64;
    println!(
        "package-aq4-matvec-smoke package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} rows={} cols={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} row_scale_overrides={} repeats={} backend={} device_index={} name=\"{}\" f32_mean_ms={:.6} aq4_mean_ms={:.6} speedup_vs_f32={:.6} max_abs_diff={max_abs_diff:.9} rms_diff={rms_diff:.9} tolerance={tolerance:.9} f32_preview={} aq4_preview={} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        rows,
        cols,
        output_byte_count,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        bundle.row_scale_overrides.len(),
        repeats,
        info.backend,
        device_index,
        info.name,
        f32_mean_ms,
        aq4_mean_ms,
        f32_mean_ms / aq4_mean_ms,
        format_f32_preview(&f32_values[..rows.min(8)]),
        format_f32_preview(&aq4_values[..rows.min(8)])
    );
    ExitCode::SUCCESS
}

fn package_rmsnorm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-rmsnorm-smoke requires a .ullm.d path");
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
    let norm_kind = match normalize_norm_kind(norm_kind.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let tensor_name = match norm_kind {
        NormKind::Input => {
            format!("model.language_model.layers.{layer_index}.input_layernorm.weight")
        }
        NormKind::Post => {
            format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight")
        }
    };
    let selector = TensorSelector::Name(tensor_name.clone());
    let bundle = match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package passthrough tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let elements = match usize::try_from(bundle.elements) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("passthrough tensor has zero elements");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("passthrough tensor element count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&bundle, &tensor_name) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let payload = match read_passthrough_payload_f32_bytes(&bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {tensor_name}: {err}");
            return ExitCode::from(1);
        }
    };
    if payload.len() != elements {
        eprintln!(
            "passthrough tensor element count mismatch for {tensor_name}: expected {} got {}",
            elements,
            payload.len()
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(elements);
    let epsilon = 1e-5_f32;
    let expected = runtime_host_rmsnorm_f32(&input, &payload, epsilon);
    if expected.len() != elements {
        eprintln!("failed to build deterministic RMSNorm reference");
        return ExitCode::from(1);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&payload);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RMSNorm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        elements,
        epsilon,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; weight_bytes.len()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy runtime RMSNorm output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len() {
        eprintln!(
            "runtime RMSNorm output size mismatch: expected {} got {}",
            expected.len(),
            output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-rmsnorm-smoke mismatch for tensor={tensor_name}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!(
        "package-rmsnorm-smoke package={} tensor_index={} tensor=\"{}\" dtype={} elements={} shape_len={} payload_bytes={} payload_path={} device_index={} name=\"{}\" epsilon={} norm_kind={} chunk_bytes={} max_abs_diff={max_abs_diff:.9} preview={} verified=true",
        path,
        bundle.tensor_index,
        bundle.tensor_name,
        dtype,
        elements,
        bundle.shape.len(),
        bundle.payload_bytes,
        bundle.payload_file.relative_path,
        device_index,
        info.name,
        epsilon,
        norm_kind.as_str(),
        chunk_bytes,
        format_f32_preview(&output[..output.len().min(8)])
    );
    ExitCode::SUCCESS
}

fn package_mlp_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-mlp-smoke requires a .ullm.d path");
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

    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &gate_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize gate tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (up_rows, up_cols, up_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &up_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize up tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (down_rows, down_cols, down_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &down_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize down tensor: {err}");
            return ExitCode::from(1);
        }
    };

    if gate_rows != up_rows || gate_cols != up_cols {
        eprintln!("gate and up tensor shapes must match");
        return ExitCode::from(1);
    }
    if down_cols != up_rows {
        eprintln!(
            "down tensor shape mismatch: expected cols={} but got {}",
            up_rows, down_cols
        );
        return ExitCode::from(1);
    }
    if down_rows != up_cols {
        eprintln!(
            "down tensor shape mismatch: expected rows={} but got {}",
            up_cols, down_rows
        );
        return ExitCode::from(1);
    }

    let intermediate = gate_rows;
    let hidden = gate_cols;

    let mut input = Vec::with_capacity(hidden);
    for i in 0..hidden {
        input.push((i % 23) as f32 / 16.0 - 11.0 / 16.0);
    }
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input vector into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let intermediate_bytes = match intermediate.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("intermediate byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix,
        &input_buffer,
        gate_rows,
        gate_cols,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix,
        &input_buffer,
        up_rows,
        up_cols,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        intermediate,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let hidden_bytes = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix,
        &activated_buffer,
        down_rows,
        down_cols,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let preview_count = hidden.min(8);
    let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy output preview to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&output_preview_bytes);

    println!(
        "package-mlp-smoke package={} layer={} gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        layer_index,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        intermediate,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn package_rmsnorm_mlp_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    package_rmsnorm_mlp_smoke_impl(
        "package-rmsnorm-mlp-smoke",
        false,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        norm_kind,
    )
}

fn package_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    package_rmsnorm_mlp_smoke_impl(
        "package-mlp-block-smoke",
        true,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        norm_kind,
    )
}

fn package_rmsnorm_mlp_smoke_impl(
    command_name: &str,
    include_block: bool,
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("{command_name} requires a .ullm.d path");
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
    let norm_kind = match normalize_norm_kind(Some(norm_kind.as_deref().unwrap_or("post"))) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let norm_tensor = match norm_kind {
        NormKind::Input => {
            format!("model.language_model.layers.{layer_index}.input_layernorm.weight")
        }
        NormKind::Post => {
            format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight")
        }
    };
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    let norm_elements = match usize::try_from(norm_bundle.elements) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("RMSNorm tensor has zero elements");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("RMSNorm tensor element count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight = match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if norm_weight.len() != norm_elements {
        eprintln!(
            "passthrough tensor element count mismatch for {norm_tensor}: expected {} got {}",
            norm_elements,
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &gate_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize gate tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (up_rows, up_cols, up_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &up_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize up tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (down_rows, down_cols, down_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &down_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize down tensor: {err}");
            return ExitCode::from(1);
        }
    };

    if gate_rows != up_rows || gate_cols != up_cols {
        eprintln!(
            "gate and up tensor shapes must match: gate=({gate_rows}, {gate_cols}), up=({up_rows}, {up_cols})"
        );
        return ExitCode::from(1);
    }
    if down_rows != up_cols || down_cols != gate_rows {
        eprintln!(
            "down tensor shape mismatch: expected shape ({}, {}) from gate/up, got ({}, {})",
            gate_cols, gate_rows, down_rows, down_cols
        );
        return ExitCode::from(1);
    }

    let hidden = gate_cols;
    let intermediate = gate_rows;
    if norm_elements != hidden {
        eprintln!(
            "RMSNorm element count must match MLP hidden dimension: norm={norm_elements}, hidden={hidden}"
        );
        return ExitCode::from(1);
    }

    let epsilon = 1e-5_f32;
    let input = deterministic_f32_vector(hidden);
    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&norm_weight);

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic RMSNorm input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RMSNorm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after RMSNorm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut normed_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        hidden,
        epsilon,
        &mut normed_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let intermediate_bytes = match intermediate.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("intermediate byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix,
        &normed_buffer,
        gate_rows,
        gate_cols,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix,
        &normed_buffer,
        up_rows,
        up_cols,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        intermediate,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let hidden_bytes = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix,
        &activated_buffer,
        down_rows,
        down_cols,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let mut output_bytes = vec![0_u8; hidden_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_bytes, Some(&mut stream)) {
        eprintln!("failed to copy MLP output to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after MLP output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_bytes);

    if include_block {
        let mut block_output_buffer = match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate MLP block output buffer: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = ullm_runtime_sys::add_f32(
            &input_buffer,
            &output_buffer,
            hidden,
            &mut block_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run MLP residual add: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after MLP residual add: {err}");
            return ExitCode::from(1);
        }

        let mut block_output_bytes = vec![0_u8; hidden_bytes];
        if let Err(err) =
            block_output_buffer.copy_to_host(0, &mut block_output_bytes, Some(&mut stream))
        {
            eprintln!("failed to copy MLP block output to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after MLP block output copy: {err}");
            return ExitCode::from(1);
        }
        let block_output = decode_f32_le_values(&block_output_bytes);
        let expected_block_output = runtime_host_add_f32(&input, &output);
        if expected_block_output.len() != block_output.len() {
            eprintln!(
                "{command_name} output size mismatch: expected {} got {}",
                expected_block_output.len(),
                block_output.len()
            );
            return ExitCode::from(1);
        }

        let mut block_max_abs_diff = 0.0_f32;
        for (lhs, rhs) in block_output.iter().zip(expected_block_output.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 1e-6_f32.max(rhs.abs() * 1e-6_f32);
            if diff > tolerance {
                eprintln!(
                    "{command_name} residual output mismatch: max_abs_diff={diff} tolerance={tolerance}"
                );
                return ExitCode::from(1);
            }
            if diff > block_max_abs_diff {
                block_max_abs_diff = diff;
            }
        }

        println!(
            "{command_name} package={} layer={} norm_kind={} norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" residual_preview={} mlp_output_preview={} block_output_preview={} block_max_abs_diff={block_max_abs_diff:.9} verified=true",
            path,
            layer_index,
            norm_kind.as_str(),
            norm_tensor,
            gate_tensor,
            up_tensor,
            down_tensor,
            hidden,
            intermediate,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&input[..8.min(input.len())]),
            format_f32_preview(&output[..8.min(output.len())]),
            format_f32_preview(&block_output[..8.min(block_output.len())])
        );
    } else {
        println!(
            "{command_name} package={} layer={} norm_kind={} norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            norm_kind.as_str(),
            norm_tensor,
            gate_tensor,
            up_tensor,
            down_tensor,
            hidden,
            intermediate,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&output[..8.min(output.len())])
        );
    }
    ExitCode::SUCCESS
}

fn package_linear_attn_proj_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    projection: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-proj-smoke requires a .ullm.d path");
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
    let projection = match parse_linear_attn_projection(projection.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_projections = match projection {
        LinearAttnProjection::A => {
            vec![(
                "a",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
            )]
        }
        LinearAttnProjection::B => {
            vec![(
                "b",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
            )]
        }
        LinearAttnProjection::Qkv => {
            vec![(
                "qkv",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
            )]
        }
        LinearAttnProjection::Z => {
            vec![(
                "z",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
            )]
        }
        LinearAttnProjection::Out => {
            vec![(
                "out",
                format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
            )]
        }
        LinearAttnProjection::All => vec![
            (
                "a",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
            ),
            (
                "b",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
            ),
            (
                "qkv",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
            ),
            (
                "z",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
            ),
            (
                "out",
                format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
            ),
        ],
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let mut input_buffer: Option<ullm_runtime_sys::RuntimeBuffer> = None;
    let mut hidden = None;
    for (projection_name, tensor_name) in requested_projections {
        let (rows, cols, matrix) = match materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            &path,
            &tensor_name,
            chunk_bytes,
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to materialize projection {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };

        match hidden {
            Some(expected) if expected != cols => {
                eprintln!(
                    "projection {projection_name} tensor {tensor_name} has cols={cols}, expected hidden={expected}"
                );
                return ExitCode::from(1);
            }
            Some(_) => {}
            None => {
                hidden = Some(cols);
                let input = deterministic_f32_vector(cols);
                let input_bytes = encode_f32_to_bytes(&input);
                let mut buffer = match context.alloc_buffer(input_bytes.len()) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        eprintln!("failed to allocate shared input buffer: {err}");
                        return ExitCode::from(1);
                    }
                };
                if let Err(err) = buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
                    eprintln!(
                        "failed to copy deterministic input data into shared runtime buffer: {err}"
                    );
                    return ExitCode::from(1);
                }
                if let Err(err) = stream.synchronize() {
                    eprintln!("failed to synchronize runtime stream after input copy: {err}");
                    return ExitCode::from(1);
                }
                input_buffer = Some(buffer);
            }
        }

        let Some(shared_input) = input_buffer.as_mut() else {
            eprintln!("shared runtime input buffer was not initialized");
            return ExitCode::from(1);
        };

        let output_bytes = match rows.checked_mul(std::mem::size_of::<f32>()) {
            Some(value) => value,
            None => {
                eprintln!("output byte size overflows for projection {projection_name}");
                return ExitCode::from(1);
            }
        };
        let mut output = match context.alloc_buffer(output_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate output buffer for {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &matrix,
            shared_input,
            rows,
            cols,
            &mut output,
            Some(&mut stream),
        ) {
            eprintln!("failed to run matvec for projection {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after matvec: {err}");
            return ExitCode::from(1);
        }

        let preview_count = rows.min(8);
        let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
        if let Err(err) = output.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
            eprintln!("failed to copy matvec preview for {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after preview copy: {err}");
            return ExitCode::from(1);
        }
        let preview = decode_f32_le_values(&output_preview_bytes);
        println!(
            "package-linear-attn-proj-smoke package={} layer={} projection={} tensor=\"{}\" hidden={} rows={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            projection_name,
            tensor_name,
            cols,
            rows,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&preview)
        );
    }
    ExitCode::SUCCESS
}

fn package_self_attn_proj_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    projection: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-proj-smoke requires a .ullm.d path");
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
    let projection = match parse_self_attn_projection(projection.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_projections = match projection {
        SelfAttnProjection::Q => {
            vec![(
                "q",
                format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight"),
            )]
        }
        SelfAttnProjection::K => {
            vec![(
                "k",
                format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight"),
            )]
        }
        SelfAttnProjection::V => {
            vec![(
                "v",
                format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight"),
            )]
        }
        SelfAttnProjection::O => {
            vec![(
                "o",
                format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight"),
            )]
        }
        SelfAttnProjection::All => vec![
            (
                "q",
                format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight"),
            ),
            (
                "k",
                format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight"),
            ),
            (
                "v",
                format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight"),
            ),
            (
                "o",
                format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight"),
            ),
        ],
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let mut input_buffer: Option<ullm_runtime_sys::RuntimeBuffer> = None;
    let mut hidden = None;
    for (projection_name, tensor_name) in requested_projections {
        let (rows, cols, matrix) = match materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            &path,
            &tensor_name,
            chunk_bytes,
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to materialize self-attn projection {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };

        match hidden {
            Some(expected) if expected != cols => {
                eprintln!(
                    "self-attn projection {projection_name} tensor {tensor_name} has cols={cols}, expected hidden={expected}"
                );
                return ExitCode::from(1);
            }
            Some(_) => {}
            None => {
                hidden = Some(cols);
                let input = deterministic_f32_vector(cols);
                let input_bytes = encode_f32_to_bytes(&input);
                let mut buffer = match context.alloc_buffer(input_bytes.len()) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        eprintln!("failed to allocate shared self-attn input buffer: {err}");
                        return ExitCode::from(1);
                    }
                };
                if let Err(err) = buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
                    eprintln!(
                        "failed to copy deterministic self-attn input data into shared runtime buffer: {err}"
                    );
                    return ExitCode::from(1);
                }
                if let Err(err) = stream.synchronize() {
                    eprintln!(
                        "failed to synchronize runtime stream after self-attn input copy: {err}"
                    );
                    return ExitCode::from(1);
                }
                input_buffer = Some(buffer);
            }
        }

        let Some(shared_input) = input_buffer.as_mut() else {
            eprintln!("shared self-attn runtime input buffer was not initialized");
            return ExitCode::from(1);
        };

        let output_bytes = match rows.checked_mul(std::mem::size_of::<f32>()) {
            Some(value) => value,
            None => {
                eprintln!("output byte size overflows for self-attn projection {projection_name}");
                return ExitCode::from(1);
            }
        };
        let mut output = match context.alloc_buffer(output_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!(
                    "failed to allocate output buffer for self-attn projection {projection_name}: {err}"
                );
                return ExitCode::from(1);
            }
        };
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &matrix,
            shared_input,
            rows,
            cols,
            &mut output,
            Some(&mut stream),
        ) {
            eprintln!("failed to run matvec for self-attn projection {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after self-attn matvec: {err}");
            return ExitCode::from(1);
        }

        let preview_count = rows.min(8);
        let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
        if let Err(err) = output.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn matvec preview for {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after self-attn preview copy: {err}");
            return ExitCode::from(1);
        }
        let preview = decode_f32_le_values(&output_preview_bytes);
        println!(
            "package-self-attn-proj-smoke package={} layer={} projection={} tensor=\"{}\" hidden={} rows={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            projection_name,
            tensor_name,
            cols,
            rows,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&preview)
        );
    }
    ExitCode::SUCCESS
}

fn package_self_attn_qk_norm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-qk-norm-smoke requires a .ullm.d path");
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

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    if q_norm.values.is_empty() || k_norm.values.is_empty() {
        eprintln!("self-attn q/k norm weights must not be empty");
        return ExitCode::from(1);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols {
        eprintln!("self-attn q/k projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}");
        return ExitCode::from(1);
    }
    if q_rows % q_norm.values.len() != 0 {
        eprintln!(
            "q projection rows must be a multiple of q_norm elements: rows={q_rows}, q_norm={}",
            q_norm.values.len()
        );
        return ExitCode::from(1);
    }
    if k_rows % k_norm.values.len() != 0 {
        eprintln!(
            "k projection rows must be a multiple of k_norm elements: rows={k_rows}, k_norm={}",
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(q_cols);
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn q/k input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic self-attn q/k input data: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after self-attn q/k input copy: {err}");
        return ExitCode::from(1);
    }

    let q_projected = match runtime_matvec_to_host_f32(
        &mut context,
        &mut stream,
        &q_matrix,
        &input_buffer,
        q_rows,
        q_cols,
        "self-attn q projection",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_projected = match runtime_matvec_to_host_f32(
        &mut context,
        &mut stream,
        &k_matrix,
        &input_buffer,
        k_rows,
        k_cols,
        "self-attn k projection",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let epsilon = 1e-5_f32;
    let (q_normed, q_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-qk-norm-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-qk-norm-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-qk-norm-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} q_rows={} k_rows={} q_head_dim={} k_head_dim={} q_heads={} k_heads={} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_preview={} k_preview={} q_norm_preview={} k_norm_preview={} q_norm_max_abs_diff={q_max_abs_diff:.9} k_norm_max_abs_diff={k_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        q_rows,
        k_rows,
        q_norm.values.len(),
        k_norm.values.len(),
        q_rows / q_norm.values.len(),
        k_rows / k_norm.values.len(),
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_projected[..8.min(q_projected.len())]),
        format_f32_preview(&k_projected[..8.min(k_projected.len())]),
        format_f32_preview(&q_normed[..8.min(q_normed.len())]),
        format_f32_preview(&k_normed[..8.min(k_normed.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_rope_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-rope-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let q_head_dim = q_norm.values.len();
    let k_head_dim = k_norm.values.len();
    if q_head_dim == 0 || k_head_dim == 0 {
        eprintln!("self-attn q/k norm weights must not be empty");
        return ExitCode::from(1);
    }
    if q_head_dim != k_head_dim {
        eprintln!(
            "self-attn q/k head dims differ: q_head_dim={q_head_dim}, k_head_dim={k_head_dim}"
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if q_head_dim >= 4 {
            q_head_dim / 4
        } else {
            q_head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for q_head_dim={q_head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > q_head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={q_head_dim}"
        );
        return ExitCode::from(2);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols {
        eprintln!("self-attn q/k projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}");
        return ExitCode::from(1);
    }
    if q_rows % q_head_dim != 0 {
        eprintln!(
            "q projection rows must be a multiple of q_head_dim: rows={q_rows}, q_head_dim={q_head_dim}"
        );
        return ExitCode::from(1);
    }
    if k_rows % k_head_dim != 0 {
        eprintln!(
            "k projection rows must be a multiple of k_head_dim: rows={k_rows}, k_head_dim={k_head_dim}"
        );
        return ExitCode::from(1);
    }
    let q_heads = q_rows / q_head_dim;
    let k_heads = k_rows / k_head_dim;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn rope input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn rope timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn rope timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn rope q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn rope k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
    }

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-rope-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-rope-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        q_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-rope-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        k_heads,
        k_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-rope-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-rope-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} q_rows={} k_rows={} q_heads={} k_heads={} head_dim={} rotary_dim={} position_offset={} rope_base={} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_norm_preview={} k_norm_preview={} q_rope_preview={} k_rope_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        q_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_normed[..8.min(q_normed.len())]),
        format_f32_preview(&k_normed[..8.min(k_normed.len())]),
        format_f32_preview(&q_rope[..8.min(q_rope.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_attention_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-attention-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
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
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (v_rows, v_cols, v_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &v_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {v_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols || q_cols != v_cols {
        eprintln!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        );
        return ExitCode::from(1);
    }
    if k_rows % head_dim != 0 {
        eprintln!("k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        eprintln!("kv_heads must be greater than zero");
        return ExitCode::from(1);
    }
    if v_rows % kv_heads != 0 {
        eprintln!("v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn attention timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn attention timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn attention q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn attention k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let v_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            v_cols,
            "self-attn attention v projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }
    let q_projection_split = match split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        q_rows,
        q_cols,
        head_dim,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        eprintln!("q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate_elements = q_projection_split.gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-attention-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-attention-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-attention-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-attention-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = match runtime_causal_attn_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-attention-smoke attention",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-attention-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_rope_preview={} k_rope_preview={} v_preview={} attention_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        q_projection_layout,
        q_gate_elements,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_rope[..8.min(q_rope.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
        format_f32_preview(&v_projected[..8.min(v_projected.len())]),
        format_f32_preview(&attention_output[..8.min(attention_output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_decode_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-decode-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 3, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
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
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (v_rows, v_cols, v_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &v_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {v_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols || q_cols != v_cols {
        eprintln!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        );
        return ExitCode::from(1);
    }
    if k_rows % head_dim != 0 {
        eprintln!("k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        eprintln!("kv_heads must be greater than zero");
        return ExitCode::from(1);
    }
    if v_rows % kv_heads != 0 {
        eprintln!("v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn decode input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn decode timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn decode timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn decode q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn decode k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let v_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            v_cols,
            "self-attn decode v projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }
    let q_projection_split = match split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        q_rows,
        q_cols,
        head_dim,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        eprintln!("q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate_elements = q_projection_split.gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-decode-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-decode-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-decode-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-decode-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = match runtime_causal_attn_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke causal reference",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let decode_q_start = (sequence_len - 1) * q_heads * head_dim;
    let decode_q_end = decode_q_start + q_heads * head_dim;
    let decode_q = &q_rope[decode_q_start..decode_q_end];
    let (decode_output, decode_max_abs_diff) = match runtime_decode_attn_verify(
        &mut context,
        &mut stream,
        decode_q,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke decode",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let paged_block_size = 2_usize;
    let paged_decode = match runtime_paged_kv_write_decode_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        paged_block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke runtime_paged_kv_write_decode",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let decode_paged_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke decode-vs-paged-decode",
        &paged_decode.output,
        &decode_output,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let causal_last_start = (sequence_len - 1) * q_heads * value_dim;
    let causal_last_end = causal_last_start + q_heads * value_dim;
    let causal_last = &attention_output[causal_last_start..causal_last_end];
    let causal_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-decode",
        &decode_output,
        causal_last,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let causal_paged_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-paged-decode",
        &paged_decode.output,
        causal_last,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let paged_kv_write_k_max_abs_diff = paged_decode.k_write_max_abs_diff;
    let paged_kv_write_v_max_abs_diff = paged_decode.v_write_max_abs_diff;
    let paged_decode_max_abs_diff = paged_decode.output_max_abs_diff;
    let paged_step_decode_max_abs_diff = paged_decode.step_output_max_abs_diff;
    let causal_paged_step_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-paged-step-decode",
        &paged_decode.step_outputs,
        &attention_output,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-decode-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} cache_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_decode_batches={} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} paged_allocator_free_blocks={} paged_allocator_allocated_blocks={} paged_allocator_free_runs={} paged_allocator_largest_free_run={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" decode_q_preview={} k_cache_preview={} v_cache_preview={} paged_k_cache_preview={} paged_v_cache_preview={} causal_last_preview={} decode_preview={} paged_decode_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} decode_max_abs_diff={decode_max_abs_diff:.9} paged_kv_write_k_max_abs_diff={paged_kv_write_k_max_abs_diff:.9} paged_kv_write_v_max_abs_diff={paged_kv_write_v_max_abs_diff:.9} paged_decode_max_abs_diff={paged_decode_max_abs_diff:.9} paged_step_decode_max_abs_diff={paged_step_decode_max_abs_diff:.9} decode_paged_max_abs_diff={decode_paged_max_abs_diff:.9} causal_decode_max_abs_diff={causal_decode_max_abs_diff:.9} causal_paged_decode_max_abs_diff={causal_paged_decode_max_abs_diff:.9} causal_paged_step_decode_max_abs_diff={causal_paged_step_decode_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        paged_block_size,
        paged_decode.cache_blocks,
        paged_decode.block_table,
        paged_decode.scheduler_decode_batches,
        paged_decode.scheduler_request_id.0,
        paged_decode.scheduler_prefill_tokens,
        paged_decode.scheduler_max_new_tokens,
        paged_decode.scheduler_cached_tokens,
        paged_decode.scheduler_generated_tokens,
        paged_decode.scheduler_active_len,
        paged_decode.allocator_stats.free_blocks,
        paged_decode.allocator_stats.allocated_blocks,
        paged_decode.allocator_stats.free_runs,
        paged_decode.allocator_stats.largest_free_run,
        q_projection_layout,
        q_gate_elements,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&decode_q[..8.min(decode_q.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
        format_f32_preview(&v_projected[..8.min(v_projected.len())]),
        format_f32_preview(&paged_decode.k_cache[..8.min(paged_decode.k_cache.len())]),
        format_f32_preview(&paged_decode.v_cache[..8.min(paged_decode.v_cache.len())]),
        format_f32_preview(&causal_last[..8.min(causal_last.len())]),
        format_f32_preview(&decode_output[..8.min(decode_output.len())]),
        format_f32_preview(&paged_decode.output[..8.min(paged_decode.output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-block-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
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
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    let result = package_self_attn_block_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
    );

    match result {
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
fn package_self_attn_block_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: PassthroughF32Data,
    k_norm: PassthroughF32Data,
) -> Result<String, String> {
    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let self_attn_weights = qwen3_self_attn_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
    )?;

    let self_attn = run_self_attn_block_sequence_smoke(
        &mut context,
        &mut stream,
        &self_attn_weights,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        &q_norm,
        &k_norm,
        "package-self-attn-block-smoke",
    )?;

    Ok(format!(
        "package-self-attn-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" attention_preview={} gated_attention_preview={} projected_preview={} block_preview={} q_norm_max_abs_diff={:.9} k_norm_max_abs_diff={:.9} q_rope_max_abs_diff={:.9} k_rope_max_abs_diff={:.9} attention_max_abs_diff={:.9} paged_kv_write_k_max_abs_diff={:.9} paged_kv_write_v_max_abs_diff={:.9} paged_step_attention_max_abs_diff={:.9} causal_paged_step_attention_max_abs_diff={:.9} output_gate_max_abs_diff={:.9} o_proj_max_abs_diff={:.9} block_max_abs_diff={:.9} causal_paged_block_max_abs_diff={:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm_tensor,
        k_norm_tensor,
        self_attn.hidden,
        sequence_len,
        self_attn.paged_block_size,
        self_attn.paged_cache_blocks,
        self_attn.paged_block_table,
        self_attn.scheduler_request_id.0,
        self_attn.scheduler_prefill_tokens,
        self_attn.scheduler_max_new_tokens,
        self_attn.scheduler_cached_tokens,
        self_attn.scheduler_generated_tokens,
        self_attn.scheduler_active_len,
        self_attn.q_projection_layout,
        self_attn.q_gate_elements,
        self_attn.output_gate_layout,
        self_attn.q_heads,
        self_attn.kv_heads,
        self_attn.head_dim,
        self_attn.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        self_attn.softmax_scale,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&self_attn.attention_output[..8.min(self_attn.attention_output.len())]),
        format_f32_preview(
            &self_attn.attention_projection_input
                [..8.min(self_attn.attention_projection_input.len())],
        ),
        format_f32_preview(&self_attn.attn_projected[..8.min(self_attn.attn_projected.len())]),
        format_f32_preview(&self_attn.block_output[..8.min(self_attn.block_output.len())]),
        self_attn.q_norm_max_abs_diff,
        self_attn.k_norm_max_abs_diff,
        self_attn.q_rope_max_abs_diff,
        self_attn.k_rope_max_abs_diff,
        self_attn.attention_max_abs_diff,
        self_attn.paged_kv_write_k_max_abs_diff,
        self_attn.paged_kv_write_v_max_abs_diff,
        self_attn.paged_step_attention_max_abs_diff,
        self_attn.causal_paged_step_attention_max_abs_diff,
        self_attn.output_gate_max_abs_diff,
        self_attn.o_proj_max_abs_diff,
        self_attn.block_max_abs_diff,
        self_attn.causal_paged_block_max_abs_diff,
    ))
}

struct Qwen3SelfAttnPreparedSequence {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
    attention_output: Vec<f32>,
    expected_paged_k_cache: Vec<f32>,
    expected_paged_v_cache: Vec<f32>,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_projection_layout: &'static str,
    q_gate_elements: usize,
    output_gate_layout: &'static str,
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    attention_max_abs_diff: f32,
    paged_block_table: Vec<u32>,
    paged_block_size: usize,
    paged_cache_blocks: usize,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
}

struct Qwen3SelfAttnModelLoopPreparedSequence {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_projection_layout: &'static str,
    q_gate_elements: usize,
    output_gate_layout: &'static str,
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    attention_max_abs_diff: f32,
}

#[allow(dead_code)]
struct SelfAttnBlockSmokeRun {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
    attention_output: Vec<f32>,
    attention_projection_input: Vec<f32>,
    attn_projected: Vec<f32>,
    block_output: Vec<f32>,
    paged_block_table: Vec<u32>,
    paged_block_size: usize,
    paged_cache_blocks: usize,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_projection_layout: &'static str,
    q_gate_elements: usize,
    output_gate_layout: &'static str,
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    attention_max_abs_diff: f32,
    paged_kv_write_k_max_abs_diff: f32,
    paged_kv_write_v_max_abs_diff: f32,
    paged_step_attention_max_abs_diff: f32,
    causal_paged_step_attention_max_abs_diff: f32,
    output_gate_max_abs_diff: f32,
    o_proj_max_abs_diff: f32,
    block_max_abs_diff: f32,
    causal_paged_block_max_abs_diff: f32,
}

#[allow(clippy::too_many_arguments)]
fn qwen3_self_attn_prepare_model_loop_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    residual_sequence: Vec<f32>,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    input_norm: &PassthroughF32Data,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    paged_block_table: &[u32],
    paged_block_size: usize,
    paged_cache_blocks: usize,
    label: &str,
) -> Result<Qwen3SelfAttnModelLoopPreparedSequence, String> {
    let Qwen3SelfAttnRuntimeShape {
        hidden,
        q_heads: shape_q_heads,
        kv_heads: _,
        head_dim: _,
        value_dim: _,
        attention_width: _,
        q_projection_layout,
    } = qwen3_self_attn_runtime_shape(self_attn_weights)?;
    let expected_residual_len = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| format!("{label} residual length overflows"))?;
    if residual_sequence.len() != expected_residual_len {
        return Err(format!(
            "{label} residual length {} does not match expected {}",
            residual_sequence.len(),
            expected_residual_len
        ));
    }
    if input_norm.values.len() != hidden {
        return Err(format!(
            "{label} input RMSNorm length {} does not match hidden={hidden}",
            input_norm.values.len()
        ));
    }

    let original_residual_sequence = residual_sequence;
    let mut attention_input_normed = Vec::with_capacity(original_residual_sequence.len());
    for residual in original_residual_sequence.chunks_exact(hidden) {
        attention_input_normed.extend(runtime_host_rmsnorm_f32(
            residual,
            &input_norm.values,
            1e-6_f32,
        ));
    }

    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        context,
        stream,
        self_attn_weights,
        attention_input_normed,
        sequence_len,
        &q_norm.values,
        &k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    )?;
    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence: _,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query,
                k_projected,
                q_normed,
                k_normed,
                q_rope,
                k_rope,
                v_projected,
                q_gate,
                attention_output,
                shape,
                softmax_scale,
                q_projection_layout: prepared_q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: _,
        paged_v_cache: _,
        paged_block_table: _,
        paged_block_size: _,
        paged_cache_blocks: _,
    } = prepared;

    if q_projection_layout != prepared_q_projection_layout {
        return Err(format!(
            "{label} q projection layout changed between shape and prepare: {q_projection_layout} vs {prepared_q_projection_layout}"
        ));
    }
    if shape.q_heads != shape_q_heads {
        return Err(format!(
            "{label} q head count changed between shape and prepare: {} vs {shape_q_heads}",
            shape.q_heads
        ));
    }

    let epsilon = 1e-5_f32;
    let mut expected_q_normed = Vec::with_capacity(q_query.len());
    for head_input in q_query.chunks_exact(shape.head_dim) {
        expected_q_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &q_norm.values,
            epsilon,
        ));
    }
    let q_norm_max_abs_diff = verify_f32_close(
        &format!("{label} q_norm"),
        &q_normed,
        &expected_q_normed,
        1e-4_f32,
        1e-4_f32,
    )?;

    let mut expected_k_normed = Vec::with_capacity(k_normed.len());
    for head_input in k_projected.chunks_exact(shape.head_dim) {
        expected_k_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &k_norm.values,
            epsilon,
        ));
    }
    let k_norm_max_abs_diff = verify_f32_close(
        &format!("{label} k_norm"),
        &k_normed,
        &expected_k_normed,
        1e-4_f32,
        1e-4_f32,
    )?;

    let expected_q_rope = runtime_host_rope_f32(
        &q_normed,
        sequence_len,
        shape.q_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let q_rope_max_abs_diff = verify_f32_close(
        &format!("{label} q_rope"),
        &q_rope,
        &expected_q_rope,
        2e-4_f32,
        1e-4_f32,
    )?;
    let expected_k_rope = runtime_host_rope_f32(
        &k_normed,
        sequence_len,
        shape.kv_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let k_rope_max_abs_diff = verify_f32_close(
        &format!("{label} k_rope"),
        &k_rope,
        &expected_k_rope,
        2e-4_f32,
        1e-4_f32,
    )?;

    let expected_attention = runtime_host_causal_attn_f32(
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        softmax_scale,
    );
    let attention_max_abs_diff = verify_f32_close(
        &format!("{label} attention"),
        &attention_output,
        &expected_attention,
        1e-4_f32,
        1e-4_f32,
    )?;

    Ok(Qwen3SelfAttnModelLoopPreparedSequence {
        residual_sequence: original_residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        hidden,
        q_heads: shape.q_heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        value_dim: shape.value_dim,
        softmax_scale,
        q_projection_layout: prepared_q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen3_self_attn_prepare_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    label: &str,
) -> Result<Qwen3SelfAttnPreparedSequence, String> {
    let Qwen3SelfAttnRuntimeShape {
        hidden,
        q_heads: shape_q_heads,
        kv_heads,
        head_dim,
        value_dim,
        attention_width: _,
        q_projection_layout,
    } = qwen3_self_attn_runtime_shape(self_attn_weights)?;
    let q_cols = hidden;

    let base_input = deterministic_f32_vector(q_cols);
    let mut residual_sequence = Vec::with_capacity(sequence_len * q_cols);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        residual_sequence.extend_from_slice(&step_input);
    }
    let paged_block_size = 2_usize;
    let scheduled_paged_decode =
        allocate_fragmented_paged_decode_blocks(sequence_len, paged_block_size)?;
    let ScheduledPagedDecodeBlocks {
        block_table: paged_block_table,
        cache_blocks: paged_cache_blocks,
        allocator_stats: _,
        request_id: scheduler_request_id,
        prefill_tokens: scheduler_prefill_tokens,
        max_new_tokens: scheduler_max_new_tokens,
        cached_tokens: scheduler_cached_tokens,
        generated_tokens: scheduler_generated_tokens,
        active_len: scheduler_active_len,
    } = scheduled_paged_decode;

    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        context,
        stream,
        self_attn_weights,
        residual_sequence,
        sequence_len,
        &q_norm.values,
        &k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        &paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    )?;

    if q_projection_layout != prepared.prepared.q_projection_layout {
        return Err(
            "self-attn q projection layout changed between helper and runtime prepare".to_string(),
        );
    }

    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query,
                k_projected,
                q_normed,
                k_normed,
                q_rope,
                k_rope,
                v_projected: prepared_v_projected,
                q_gate,
                attention_output,
                shape,
                softmax_scale,
                q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: expected_paged_k_cache,
        paged_v_cache: expected_paged_v_cache,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    } = prepared;

    if shape.q_heads != shape_q_heads {
        return Err(
            "self-attn q projection head count changed between helper and runtime prepare"
                .to_string(),
        );
    }
    let epsilon = 1e-5_f32;
    let mut expected_q_normed = Vec::with_capacity(q_query.len());
    for head_input in q_query.chunks_exact(shape.head_dim) {
        expected_q_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &q_norm.values,
            epsilon,
        ));
    }
    let q_norm_max_abs_diff = verify_f32_close(
        &format!("{label} q_norm"),
        &q_normed,
        &expected_q_normed,
        1e-4_f32,
        1e-4_f32,
    )?;
    let mut expected_k_normed = Vec::with_capacity(k_normed.len());
    for head_input in k_projected.chunks_exact(shape.head_dim) {
        expected_k_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &k_norm.values,
            epsilon,
        ));
    }
    let k_norm_max_abs_diff = verify_f32_close(
        &format!("{label} k_norm"),
        &k_normed,
        &expected_k_normed,
        1e-4_f32,
        1e-4_f32,
    )?;
    let expected_q_rope = runtime_host_rope_f32(
        &q_normed,
        sequence_len,
        shape.q_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let q_rope_max_abs_diff = verify_f32_close(
        &format!("{label} q_rope"),
        &q_rope,
        &expected_q_rope,
        1e-4_f32,
        1e-4_f32,
    )?;
    let expected_k_rope = runtime_host_rope_f32(
        &k_normed,
        sequence_len,
        shape.kv_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let k_rope_max_abs_diff = verify_f32_close(
        &format!("{label} k_rope"),
        &k_rope,
        &expected_k_rope,
        1e-4_f32,
        1e-4_f32,
    )?;

    let expected_attention = runtime_host_causal_attn_f32(
        &q_rope,
        &k_rope,
        &prepared_v_projected,
        sequence_len,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        softmax_scale,
    );
    let attention_max_abs_diff = verify_f32_close(
        &format!("{label} attention"),
        &attention_output,
        &expected_attention,
        1e-4_f32,
        1e-4_f32,
    )?;

    Ok(Qwen3SelfAttnPreparedSequence {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected: prepared_v_projected,
        q_gate,
        attention_output,
        expected_paged_k_cache,
        expected_paged_v_cache,
        hidden: q_cols,
        q_heads: shape.q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
    })
}

struct Qwen3DecoderLayerRequestSequenceRun {
    output: Qwen3DecoderLayerSequenceOutput,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
}

fn push_decoder_layer_step_output(
    output: &mut Qwen3DecoderLayerSequenceOutput,
    step: ullm_engine::decode_runner::Qwen3DecoderLayerDecodeBatchOutput,
) {
    output
        .attention_output
        .extend_from_slice(&step.attention_output);
    output
        .attention_projection_input
        .extend_from_slice(&step.attention_projection_input);
    output
        .projected_output
        .extend_from_slice(&step.projected_output);
    output.block_output.extend_from_slice(&step.block_output);
    output.post_normed.extend_from_slice(&step.post_normed);
    output.mlp_output.extend_from_slice(&step.mlp_output);
    output.layer_output.extend_from_slice(&step.layer_output);
}

#[allow(clippy::too_many_arguments)]
fn qwen3_decoder_layer_request_sequence_to_host_f32(
    layer_weights: &Qwen3DecoderLayerRuntimeWeights,
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    shape: PagedDecodeShape,
    expected_block_table: &[u32],
    softmax_scale: f32,
    mlp_epsilon: f32,
    q_sequence: &[f32],
    k_sequence: &[f32],
    v_sequence: &[f32],
    output_gate_sequence: Option<&[f32]>,
    residual_sequence: &[f32],
    sequence_len: usize,
) -> Result<Qwen3DecoderLayerRequestSequenceRun, String> {
    if sequence_len == 0 {
        return Err("Qwen3 decoder layer request sequence length must be greater than zero".into());
    }
    let prepared_scheduler = prepare_fragmented_paged_decode_state(sequence_len, shape.block_size)?;
    if prepared_scheduler.block_table != expected_block_table {
        return Err(format!(
            "Qwen3 decoder layer request runner block table {:?} does not match prepared self-attn block table {:?}",
            prepared_scheduler.block_table, expected_block_table
        ));
    }
    if prepared_scheduler.cache_blocks != shape.cache_blocks {
        return Err(format!(
            "Qwen3 decoder layer request runner cache_blocks {} does not match shape cache_blocks {}",
            prepared_scheduler.cache_blocks, shape.cache_blocks
        ));
    }

    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let hidden = layer_weights.post_attention.hidden;
    let expected_q_len = sequence_len
        .checked_mul(q_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request q length overflows".to_string())?;
    let expected_k_len = sequence_len
        .checked_mul(k_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request k length overflows".to_string())?;
    let expected_v_len = sequence_len
        .checked_mul(v_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request v length overflows".to_string())?;
    let expected_residual_len = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "Qwen3 decoder layer request residual length overflows".to_string())?;
    if q_sequence.len() != expected_q_len
        || k_sequence.len() != expected_k_len
        || v_sequence.len() != expected_v_len
        || residual_sequence.len() != expected_residual_len
    {
        return Err(format!(
            "Qwen3 decoder layer request sequence length mismatch: q={} expected_q={} k={} expected_k={} v={} expected_v={} residual={} expected_residual={}",
            q_sequence.len(),
            expected_q_len,
            k_sequence.len(),
            expected_k_len,
            v_sequence.len(),
            expected_v_len,
            residual_sequence.len(),
            expected_residual_len
        ));
    }
    if let Some(gate) = output_gate_sequence {
        let expected_gate_len = sequence_len
            .checked_mul(attention_elements)
            .ok_or_else(|| {
                "Qwen3 decoder layer request output gate length overflows".to_string()
            })?;
        if gate.len() != expected_gate_len {
            return Err(format!(
                "Qwen3 decoder layer request output gate length {} does not match expected {}",
                gate.len(),
                expected_gate_len
            ));
        }
    }

    let mut scheduler = prepared_scheduler.scheduler;
    let request_id = prepared_scheduler.request_id;
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    runner.insert_request(
        context,
        stream,
        request_id,
        layer_weights,
        shape,
        prepared_scheduler.block_table.clone(),
        softmax_scale,
        mlp_epsilon,
    )?;

    let mut output = Qwen3DecoderLayerSequenceOutput {
        attention_output: Vec::with_capacity(sequence_len * attention_elements),
        attention_projection_input: Vec::with_capacity(sequence_len * attention_elements),
        projected_output: Vec::with_capacity(sequence_len * hidden),
        block_output: Vec::with_capacity(sequence_len * hidden),
        post_normed: Vec::with_capacity(sequence_len * hidden),
        mlp_output: Vec::with_capacity(sequence_len * hidden),
        layer_output: Vec::with_capacity(sequence_len * hidden),
        paged_cache: PagedKvCacheReadback {
            k: Vec::new(),
            v: Vec::new(),
        },
    };

    for timestep in 0..prepared_scheduler.prefill_tokens {
        let q_start = timestep * q_token_elements;
        let k_start = timestep * k_token_elements;
        let v_start = timestep * v_token_elements;
        let gate_start = timestep * attention_elements;
        let residual_start = timestep * hidden;
        let step = runner.run_prefill_step(
            stream,
            Qwen3DecoderLayerDecodeBatchInput {
                request_id,
                q: &q_sequence[q_start..q_start + q_token_elements],
                k: &k_sequence[k_start..k_start + k_token_elements],
                v: &v_sequence[v_start..v_start + v_token_elements],
                output_gate: output_gate_sequence
                    .map(|gate| &gate[gate_start..gate_start + attention_elements]),
                residual: &residual_sequence[residual_start..residual_start + hidden],
            },
        )?;
        if step.cache_position != timestep || step.cache_len != timestep + 1 {
            return Err(format!(
                "Qwen3 decoder layer prefill step returned cache_position={} cache_len={} for timestep {}",
                step.cache_position, step.cache_len, timestep
            ));
        }
        push_decoder_layer_step_output(&mut output, step);
    }

    scheduler
        .complete_prefill(request_id)
        .map_err(|err| format!("failed to complete Qwen3 decoder layer request prefill: {err}"))?;

    for timestep in prepared_scheduler.prefill_tokens..sequence_len {
        let ready = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to ready Qwen3 decoder layer request batch: {err}"))?;
        let request = ready.first().ok_or_else(|| {
            format!("expected one ready Qwen3 decoder layer request at timestep {timestep}")
        })?;
        if request.request.id != request_id || request.cache_position != timestep {
            return Err(format!(
                "Qwen3 decoder layer ready request {:?} cache_position {} does not match request {:?} timestep {}",
                request.request.id, request.cache_position, request_id, timestep
            ));
        }
        let q_start = timestep * q_token_elements;
        let k_start = timestep * k_token_elements;
        let v_start = timestep * v_token_elements;
        let gate_start = timestep * attention_elements;
        let residual_start = timestep * hidden;
        let mut steps = runner.run_ready_batch(
            stream,
            &mut scheduler,
            &ready,
            &[Qwen3DecoderLayerDecodeBatchInput {
                request_id,
                q: &q_sequence[q_start..q_start + q_token_elements],
                k: &k_sequence[k_start..k_start + k_token_elements],
                v: &v_sequence[v_start..v_start + v_token_elements],
                output_gate: output_gate_sequence
                    .map(|gate| &gate[gate_start..gate_start + attention_elements]),
                residual: &residual_sequence[residual_start..residual_start + hidden],
            }],
        )?;
        let step = steps.pop().ok_or_else(|| {
            format!("Qwen3 decoder layer request runner produced no output at timestep {timestep}")
        })?;
        if step.request_id != request_id {
            return Err(format!(
                "Qwen3 decoder layer request runner output request {:?} does not match {:?}",
                step.request_id, request_id
            ));
        }
        push_decoder_layer_step_output(&mut output, step);
    }

    output.paged_cache = runner.read_cache_to_host(request_id, stream)?;
    let active = scheduler
        .active_request(request_id)
        .ok_or_else(|| "Qwen3 decoder layer request is not active after run".to_string())?;
    Ok(Qwen3DecoderLayerRequestSequenceRun {
        output,
        scheduler_request_id: request_id,
        scheduler_prefill_tokens: prepared_scheduler.prefill_tokens,
        scheduler_max_new_tokens: prepared_scheduler.max_new_tokens,
        scheduler_cached_tokens: active.cached_tokens,
        scheduler_generated_tokens: active.generated_tokens,
        scheduler_active_len: scheduler.active_len(),
    })
}

#[allow(clippy::too_many_arguments, dead_code)]
fn run_self_attn_block_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    label: &str,
) -> Result<SelfAttnBlockSmokeRun, String> {
    let prepared = qwen3_self_attn_prepare_sequence_smoke(
        context,
        stream,
        self_attn_weights,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
        label,
    )?;
    let Qwen3SelfAttnPreparedSequence {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        attention_output: prepared_attention_output,
        expected_paged_k_cache,
        expected_paged_v_cache,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
    } = prepared;

    let o_rows = self_attn_weights.o_rows;
    let o_cols = self_attn_weights.o_cols;
    let o_matrix_bytes = o_rows
        .checked_mul(o_cols)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "o projection matrix byte size overflows".to_string())?;
    let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
    self_attn_weights
        .o_matrix
        .copy_to_host(0, &mut o_matrix_raw, Some(stream))
        .map_err(|err| format!("failed to copy materialized o projection to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after o projection host copy: {err}"))?;
    let o_matrix_host = decode_f32_le_values(&o_matrix_raw);

    let decode_shape = PagedDecodeShape {
        block_size: paged_block_size,
        cache_blocks: paged_cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let q_token_elements = q_heads * head_dim;
    let attention_elements = q_heads * value_dim;
    let block_sequence_output = qwen3_self_attn_block_sequence_to_host_f32(
        context,
        stream,
        decode_shape,
        &paged_block_table,
        hidden,
        softmax_scale,
        &self_attn_weights.o_matrix,
        &q_rope,
        &k_rope,
        &v_projected,
        q_gate.as_deref(),
        &residual_sequence,
        sequence_len,
    )
    .map_err(|err| format!("failed to run {label} Qwen3 self-attn block sequence: {err}"))?;
    let attention_output = block_sequence_output.attention_output;
    let attention_projection_input = block_sequence_output.attention_projection_input;
    let attn_projected = block_sequence_output.projected_output;
    let block_output = block_sequence_output.block_output;
    let paged_cache = block_sequence_output.paged_cache;

    let mut expected_paged_step_attention_output =
        Vec::with_capacity(sequence_len * attention_elements);
    for timestep in 0..sequence_len {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_rope[q_start..q_end],
            &expected_paged_k_cache,
            &expected_paged_v_cache,
            &paged_block_table,
            timestep + 1,
            paged_block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        expected_paged_step_attention_output.extend_from_slice(&expected_step_output);
    }
    let expected_paged_projection_input = if let Some(gate) = q_gate.as_ref() {
        runtime_host_sigmoid_mul_f32(gate, &attention_output)
    } else {
        attention_output.clone()
    };
    let mut expected_paged_attn_projected = Vec::with_capacity(sequence_len * o_rows);
    for timestep in 0..sequence_len {
        let input_start = timestep * attention_elements;
        let input_end = input_start + attention_elements;
        expected_paged_attn_projected.extend(runtime_host_matvec_f32(
            &o_matrix_host,
            &expected_paged_projection_input[input_start..input_end],
            o_rows,
            o_cols,
        ));
    }
    let expected_runtime_block_output = runtime_host_add_f32(&residual_sequence, &attn_projected);

    let paged_step_attention_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn step"),
        &attention_output,
        &expected_paged_step_attention_output,
        1e-4,
        1e-4,
    )?;
    let output_gate_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn output gate"),
        &attention_projection_input,
        &expected_paged_projection_input,
        1e-5,
        1e-6,
    )?;
    let o_proj_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn o projection"),
        &attn_projected,
        &expected_paged_attn_projected,
        1e-4,
        1e-5,
    )?;
    let block_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn residual add"),
        &block_output,
        &expected_runtime_block_output,
        1e-5,
        1e-6,
    )?;

    let paged_kv_write_k_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn paged k cache write"),
        &paged_cache.k,
        &expected_paged_k_cache,
        1e-5,
        1e-5,
    )?;
    let paged_kv_write_v_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn paged v cache write"),
        &paged_cache.v,
        &expected_paged_v_cache,
        1e-5,
        1e-5,
    )?;
    let causal_paged_step_attention_max_abs_diff = verify_f32_close(
        &format!("{label} causal-vs-paged-step-attention"),
        &attention_output,
        &prepared_attention_output,
        1e-4,
        1e-4,
    )?;
    let causal_attention_projection_input = if let Some(gate) = q_gate.as_ref() {
        runtime_host_sigmoid_mul_f32(gate, &prepared_attention_output)
    } else {
        prepared_attention_output.clone()
    };
    let mut causal_attn_projected = Vec::with_capacity(sequence_len * o_rows);
    for timestep in 0..sequence_len {
        let input_start = timestep * attention_elements;
        let input_end = input_start + attention_elements;
        causal_attn_projected.extend(runtime_host_matvec_f32(
            &o_matrix_host,
            &causal_attention_projection_input[input_start..input_end],
            o_rows,
            o_cols,
        ));
    }
    let expected_causal_block_output =
        runtime_host_add_f32(&residual_sequence, &causal_attn_projected);
    let causal_paged_block_max_abs_diff = verify_f32_close(
        &format!("{label} causal-vs-paged-block"),
        &block_output,
        &expected_causal_block_output,
        3e-3,
        2e-5,
    )?;

    Ok(SelfAttnBlockSmokeRun {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        attention_output,
        attention_projection_input,
        attn_projected,
        block_output,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
    })
}

fn package_self_attn_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-mlp-block-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let post_norm = match read_named_passthrough_f32(&path, &post_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
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
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    let result = package_self_attn_mlp_block_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
        post_norm,
    );

    match result {
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
fn package_self_attn_mlp_block_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: PassthroughF32Data,
    k_norm: PassthroughF32Data,
    post_norm: PassthroughF32Data,
) -> Result<String, String> {
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

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let layer_weights = qwen3_decoder_layer_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
    )?;
    let self_attn = qwen3_self_attn_prepare_sequence_smoke(
        &mut context,
        &mut stream,
        &layer_weights.self_attn,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        &q_norm,
        &k_norm,
        "package-self-attn-mlp-block-smoke",
    )?;

    let hidden = self_attn.hidden;
    let mlp_epsilon = 1e-5_f32;
    if layer_weights.post_attention.hidden != hidden {
        return Err(format!(
            "Qwen3 decoder layer runtime weight hidden mismatch: expected={hidden} got={}",
            layer_weights.post_attention.hidden
        ));
    }
    if layer_weights.post_attention.mlp.gate_rows != layer_weights.post_attention.intermediate
        || layer_weights.post_attention.mlp.gate_cols != hidden
    {
        return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
    }

    let (
        post_normed,
        mlp_output,
        layer_output,
        attention_output,
        attention_projection_input,
        attn_projected,
        layer_step_block_output,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
        post_norm_max_abs_diff,
        layer_block_max_abs_diff,
    ) = {
        let o_rows = layer_weights.self_attn.o_rows;
        let o_cols = layer_weights.self_attn.o_cols;
        let o_matrix_bytes = o_rows
            .checked_mul(o_cols)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "o projection matrix byte size overflows".to_string())?;
        let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
        layer_weights
            .self_attn
            .o_matrix
            .copy_to_host(0, &mut o_matrix_raw, Some(&mut stream))
            .map_err(|err| format!("failed to copy materialized o projection to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after o projection host copy: {err}"))?;
        let o_matrix_host = decode_f32_le_values(&o_matrix_raw);

        let decode_shape = PagedDecodeShape {
            block_size: self_attn.paged_block_size,
            cache_blocks: self_attn.paged_cache_blocks,
            q_heads: self_attn.q_heads,
            kv_heads: self_attn.kv_heads,
            head_dim: self_attn.head_dim,
            value_dim: self_attn.value_dim,
        };
        let layer_sequence_run = qwen3_decoder_layer_request_sequence_to_host_f32(
            &layer_weights,
            &mut context,
            &mut stream,
            decode_shape,
            &self_attn.paged_block_table,
            self_attn.softmax_scale,
            mlp_epsilon,
            &self_attn.q_rope,
            &self_attn.k_rope,
            &self_attn.v_projected,
            self_attn.q_gate.as_deref(),
            &self_attn.residual_sequence,
            sequence_len,
        )?;
        if layer_sequence_run.scheduler_request_id != self_attn.scheduler_request_id
            || layer_sequence_run.scheduler_prefill_tokens != self_attn.scheduler_prefill_tokens
            || layer_sequence_run.scheduler_max_new_tokens != self_attn.scheduler_max_new_tokens
            || layer_sequence_run.scheduler_cached_tokens != self_attn.scheduler_cached_tokens
            || layer_sequence_run.scheduler_generated_tokens != self_attn.scheduler_generated_tokens
            || layer_sequence_run.scheduler_active_len != self_attn.scheduler_active_len
        {
            return Err(format!(
                "package-self-attn-mlp-block-smoke layer request runner scheduler progress mismatch: runner request={} prefill={} max_new={} cached={} generated={} active={} self_attn request={} prefill={} max_new={} cached={} generated={} active={}",
                layer_sequence_run.scheduler_request_id.0,
                layer_sequence_run.scheduler_prefill_tokens,
                layer_sequence_run.scheduler_max_new_tokens,
                layer_sequence_run.scheduler_cached_tokens,
                layer_sequence_run.scheduler_generated_tokens,
                layer_sequence_run.scheduler_active_len,
                self_attn.scheduler_request_id.0,
                self_attn.scheduler_prefill_tokens,
                self_attn.scheduler_max_new_tokens,
                self_attn.scheduler_cached_tokens,
                self_attn.scheduler_generated_tokens,
                self_attn.scheduler_active_len
            ));
        }
        let layer_sequence_output = layer_sequence_run.output;
        let attention_output = layer_sequence_output.attention_output;
        let attention_projection_input = layer_sequence_output.attention_projection_input;
        let attn_projected = layer_sequence_output.projected_output;
        let layer_step_block_output = layer_sequence_output.block_output;
        let post_normed = layer_sequence_output.post_normed;
        let mlp_output = layer_sequence_output.mlp_output;
        let layer_output = layer_sequence_output.layer_output;
        let layer_cache = layer_sequence_output.paged_cache;
        let paged_kv_write_k_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged k cache write",
            &layer_cache.k,
            &self_attn.expected_paged_k_cache,
            1e-5,
            1e-5,
        )?;
        let paged_kv_write_v_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged v cache write",
            &layer_cache.v,
            &self_attn.expected_paged_v_cache,
            1e-5,
            1e-5,
        )?;
        let q_token_elements = self_attn.q_heads * self_attn.head_dim;
        let attention_elements = self_attn.q_heads * self_attn.value_dim;

        let mut expected_paged_step_attention_output =
            Vec::with_capacity(sequence_len * attention_elements);
        for timestep in 0..sequence_len {
            let q_start = timestep
                .checked_mul(q_token_elements)
                .ok_or_else(|| "self-attn q slice start overflows".to_string())?;
            let q_end = q_start
                .checked_add(q_token_elements)
                .ok_or_else(|| "self-attn q slice end overflows".to_string())?;
            let expected_step_output = runtime_host_paged_decode_attn_f32(
                &self_attn.q_rope[q_start..q_end],
                &self_attn.expected_paged_k_cache,
                &self_attn.expected_paged_v_cache,
                &self_attn.paged_block_table,
                timestep + 1,
                self_attn.paged_block_size,
                self_attn.q_heads,
                self_attn.kv_heads,
                self_attn.head_dim,
                self_attn.value_dim,
                self_attn.softmax_scale,
            );
            expected_paged_step_attention_output.extend_from_slice(&expected_step_output);
        }
        let expected_paged_projection_input = if let Some(gate) = self_attn.q_gate.as_ref() {
            runtime_host_sigmoid_mul_f32(gate, &attention_output)
        } else {
            attention_output.clone()
        };
        let mut expected_paged_attn_projected = Vec::with_capacity(sequence_len * o_rows);
        for timestep in 0..sequence_len {
            let input_start = timestep
                .checked_mul(attention_elements)
                .ok_or_else(|| "attention start overflow".to_string())?;
            let input_end = input_start
                .checked_add(attention_elements)
                .ok_or_else(|| "attention end overflow".to_string())?;
            expected_paged_attn_projected.extend(runtime_host_matvec_f32(
                &o_matrix_host,
                &expected_paged_projection_input[input_start..input_end],
                o_rows,
                o_cols,
            ));
        }

        let expected_runtime_block_output =
            runtime_host_add_f32(&self_attn.residual_sequence, &attn_projected);
        verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step attention block",
            &layer_step_block_output,
            &expected_runtime_block_output,
            1e-4,
            1e-5,
        )?;

        let paged_step_attention_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn step",
            &attention_output,
            &expected_paged_step_attention_output,
            1e-4,
            1e-4,
        )?;
        let output_gate_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn output gate",
            &attention_projection_input,
            &expected_paged_projection_input,
            1e-5,
            1e-6,
        )?;
        let o_proj_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn o projection",
            &attn_projected,
            &expected_paged_attn_projected,
            1e-4,
            1e-5,
        )?;
        let block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn residual add",
            &layer_step_block_output,
            &expected_runtime_block_output,
            1e-5,
            1e-6,
        )?;

        let causal_paged_step_attention_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke causal-vs-paged-step-attention",
            &attention_output,
            &self_attn.attention_output,
            1e-4,
            1e-4,
        )?;
        let causal_attention_projection_input = if let Some(gate) = self_attn.q_gate.as_ref() {
            runtime_host_sigmoid_mul_f32(gate, &self_attn.attention_output)
        } else {
            self_attn.attention_output.clone()
        };
        let mut causal_attn_projected = Vec::with_capacity(sequence_len * o_rows);
        for timestep in 0..sequence_len {
            let input_start = timestep
                .checked_mul(attention_elements)
                .ok_or_else(|| "causal attention start overflow".to_string())?;
            let input_end = input_start
                .checked_add(attention_elements)
                .ok_or_else(|| "causal attention end overflow".to_string())?;
            causal_attn_projected.extend(runtime_host_matvec_f32(
                &o_matrix_host,
                &causal_attention_projection_input[input_start..input_end],
                o_rows,
                o_cols,
            ));
        }
        let expected_causal_block_output =
            runtime_host_add_f32(&self_attn.residual_sequence, &causal_attn_projected);
        let causal_paged_block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke causal-vs-paged-block",
            &layer_step_block_output,
            &expected_causal_block_output,
            3e-3,
            2e-5,
        )?;

        let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let start = timestep * hidden;
            let end = start + hidden;
            let expected = runtime_host_rmsnorm_f32(
                &layer_step_block_output[start..end],
                &post_norm.values,
                mlp_epsilon,
            );
            post_normed_expected.extend_from_slice(&expected);
        }

        let post_norm_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke post RMSNorm",
            &post_normed,
            &post_normed_expected,
            1e-4,
            1e-5,
        )?;
        let expected_layer_output = runtime_host_add_f32(&layer_step_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (
            post_normed,
            mlp_output,
            layer_output,
            attention_output,
            attention_projection_input,
            attn_projected,
            layer_step_block_output,
            paged_kv_write_k_max_abs_diff,
            paged_kv_write_v_max_abs_diff,
            paged_step_attention_max_abs_diff,
            causal_paged_step_attention_max_abs_diff,
            output_gate_max_abs_diff,
            o_proj_max_abs_diff,
            block_max_abs_diff,
            causal_paged_block_max_abs_diff,
            post_norm_max_abs_diff,
            layer_block_max_abs_diff,
        )
    };

    Ok(format!(
        "package-self-attn-mlp-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} sequence_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_preview={} gated_attention_preview={} projected_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} q_norm_max_abs_diff={:.9} k_norm_max_abs_diff={:.9} q_rope_max_abs_diff={:.9} k_rope_max_abs_diff={:.9} attention_max_abs_diff={:.9} paged_kv_write_k_max_abs_diff={:.9} paged_kv_write_v_max_abs_diff={:.9} paged_step_attention_max_abs_diff={:.9} causal_paged_step_attention_max_abs_diff={:.9} output_gate_max_abs_diff={:.9} o_proj_max_abs_diff={:.9} block_max_abs_diff={:.9} causal_paged_block_max_abs_diff={:.9} post_norm_max_abs_diff={:.9} layer_block_max_abs_diff={:.9} verified=true",
        path,
        layer_index,
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
        hidden,
        sequence_len,
        self_attn.paged_block_size,
        self_attn.paged_cache_blocks,
        self_attn.paged_block_table,
        self_attn.scheduler_request_id.0,
        self_attn.scheduler_prefill_tokens,
        self_attn.scheduler_max_new_tokens,
        self_attn.scheduler_cached_tokens,
        self_attn.scheduler_generated_tokens,
        self_attn.scheduler_active_len,
        self_attn.q_projection_layout,
        self_attn.q_gate_elements,
        self_attn.output_gate_layout,
        self_attn.q_heads,
        self_attn.kv_heads,
        self_attn.head_dim,
        self_attn.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        self_attn.softmax_scale,
        q_norm.dtype,
        k_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(
            &self_attn.residual_sequence[..8.min(self_attn.residual_sequence.len())]
        ),
        format_f32_preview(&attention_output[..8.min(attention_output.len())]),
        format_f32_preview(&attention_projection_input[..8.min(attention_projection_input.len())]),
        format_f32_preview(&attn_projected[..8.min(attn_projected.len())]),
        format_f32_preview(&layer_step_block_output[..8.min(layer_step_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
        self_attn.q_norm_max_abs_diff,
        self_attn.k_norm_max_abs_diff,
        self_attn.q_rope_max_abs_diff,
        self_attn.k_rope_max_abs_diff,
        self_attn.attention_max_abs_diff,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
        post_norm_max_abs_diff,
        layer_block_max_abs_diff,
    ))
}

fn package_self_attn_mlp_block_scheduler_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-mlp-block-scheduler-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 3, "sequence length") {
        Ok(value) if value >= 3 => value,
        Ok(_) => {
            eprintln!("sequence length must be at least three for scheduler layer smoke");
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
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let post_norm = match read_named_passthrough_f32(&path, &post_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
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
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    match package_self_attn_mlp_block_scheduler_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
        post_norm,
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
fn package_self_attn_mlp_block_scheduler_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: PassthroughF32Data,
    k_norm: PassthroughF32Data,
    post_norm: PassthroughF32Data,
) -> Result<String, String> {
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

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let layer_weights = qwen3_decoder_layer_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
    )?;
    let runtime_shape = qwen3_self_attn_runtime_shape(&layer_weights.self_attn)?;
    if layer_weights.post_attention.hidden != runtime_shape.hidden {
        return Err(format!(
            "Qwen3 decoder layer runtime weight hidden mismatch: self_attn={} post_attention={}",
            runtime_shape.hidden, layer_weights.post_attention.hidden
        ));
    }
    if layer_weights.post_attention.mlp.gate_rows != layer_weights.post_attention.intermediate
        || layer_weights.post_attention.mlp.gate_cols != runtime_shape.hidden
    {
        return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
    }

    let block_size = 2_usize;
    let requests = vec![
        Request::new(201, sequence_len - 2, 2),
        Request::new(202, sequence_len - 1, 1),
        Request::new(203, 1, 0),
    ];
    let mut required_blocks = 0_usize;
    for request in &requests {
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }
        let request_blocks = (total_tokens - 1) / block_size + 1;
        required_blocks = required_blocks
            .checked_add(request_blocks)
            .ok_or_else(|| "package scheduler layer required block count overflows".to_string())?;
    }
    let cache_blocks = required_blocks
        .checked_add(2)
        .ok_or_else(|| "package scheduler layer cache block count overflows".to_string())?;
    if cache_blocks > u32::MAX as usize || block_size > u32::MAX as usize {
        return Err(format!(
            "package scheduler layer block layout exceeds u32 range: cache_blocks={cache_blocks} block_size={block_size}"
        ));
    }
    let decode_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: runtime_shape.q_heads,
        kv_heads: runtime_shape.kv_heads,
        head_dim: runtime_shape.head_dim,
        value_dim: runtime_shape.value_dim,
    };
    let q_token_elements = decode_shape.q_elements()?;
    let k_token_elements = decode_shape.k_token_elements()?;
    let v_token_elements = decode_shape.v_token_elements()?;
    let attention_elements = decode_shape.output_elements()?;
    let hidden = runtime_shape.hidden;
    let intermediate = layer_weights.post_attention.intermediate;
    let softmax_scale = 1.0_f32 / (runtime_shape.head_dim as f32).sqrt();
    let mlp_epsilon = 1e-5_f32;

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate package scheduler layer batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "package scheduler layer selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    let mut request_position_offsets = Vec::with_capacity(allocated.len());
    let mut q_gate_elements = Vec::with_capacity(allocated.len());
    let mut q_norm_max_abs_diff = 0.0_f32;
    let mut k_norm_max_abs_diff = 0.0_f32;
    let mut q_rope_max_abs_diff = 0.0_f32;
    let mut k_rope_max_abs_diff = 0.0_f32;
    let mut causal_attention_max_abs_diff = 0.0_f32;
    let mut q_projection_layout = None;
    let mut output_gate_layout = None;

    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        let request_position_offset = position_offset
            .checked_add(request_index.checked_mul(sequence_len).ok_or_else(|| {
                "package scheduler layer request position offset multiplier overflows".to_string()
            })?)
            .ok_or_else(|| {
                "package scheduler layer request position offset overflows".to_string()
            })?;
        request_position_offsets.push(request_position_offset);

        let prepared = qwen3_self_attn_prepare_sequence_smoke(
            &mut context,
            &mut stream,
            &layer_weights.self_attn,
            total_tokens,
            rotary_dim,
            rope_base,
            request_position_offset,
            &q_norm,
            &k_norm,
            &format!(
                "package-self-attn-mlp-block-scheduler-smoke request {:?}",
                request.id
            ),
        )?;
        if prepared.hidden != hidden
            || prepared.q_heads != runtime_shape.q_heads
            || prepared.kv_heads != runtime_shape.kv_heads
            || prepared.head_dim != runtime_shape.head_dim
            || prepared.value_dim != runtime_shape.value_dim
        {
            return Err(format!(
                "package scheduler layer prepared shape mismatch for {:?}: hidden={} q_heads={} kv_heads={} head_dim={} value_dim={}",
                request.id,
                prepared.hidden,
                prepared.q_heads,
                prepared.kv_heads,
                prepared.head_dim,
                prepared.value_dim
            ));
        }
        if let Some(layout) = q_projection_layout {
            if layout != prepared.q_projection_layout {
                return Err(format!(
                    "package scheduler layer q projection layout changed: {layout} vs {}",
                    prepared.q_projection_layout
                ));
            }
        } else {
            q_projection_layout = Some(prepared.q_projection_layout);
        }
        if let Some(layout) = output_gate_layout {
            if layout != prepared.output_gate_layout {
                return Err(format!(
                    "package scheduler layer output gate layout changed: {layout} vs {}",
                    prepared.output_gate_layout
                ));
            }
        } else {
            output_gate_layout = Some(prepared.output_gate_layout);
        }
        q_gate_elements.push(prepared.q_gate_elements);
        q_norm_max_abs_diff = q_norm_max_abs_diff.max(prepared.q_norm_max_abs_diff);
        k_norm_max_abs_diff = k_norm_max_abs_diff.max(prepared.k_norm_max_abs_diff);
        q_rope_max_abs_diff = q_rope_max_abs_diff.max(prepared.q_rope_max_abs_diff);
        k_rope_max_abs_diff = k_rope_max_abs_diff.max(prepared.k_rope_max_abs_diff);
        causal_attention_max_abs_diff =
            causal_attention_max_abs_diff.max(prepared.attention_max_abs_diff);

        let expected = qwen3_decoder_layer_sequence_to_host_f32(
            &layer_weights,
            &mut context,
            &mut stream,
            decode_shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &prepared.q_rope,
            &prepared.k_rope,
            &prepared.v_projected,
            prepared.q_gate.as_deref(),
            &prepared.residual_sequence,
            total_tokens,
        )?;
        runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            &layer_weights,
            decode_shape,
            block_table.clone(),
            softmax_scale,
            mlp_epsilon,
        )?;
        let mut run = SchedulerLayerDecodeRun {
            state: SchedulerLayerDecodeState {
                request_id: request.id,
                prompt_tokens: request.prompt_tokens,
                max_new_tokens: request.max_new_tokens,
                total_tokens,
                block_table,
                q_sequence: prepared.q_rope,
                k_sequence: prepared.k_rope,
                v_sequence: prepared.v_projected,
                output_gate_sequence: prepared.q_gate,
                residual_sequence: prepared.residual_sequence,
                decode_steps: 0,
            },
            checks: SchedulerLayerDecodeSmokeChecks::new(expected),
        };
        for timestep in 0..run.prompt_tokens {
            run_scheduler_layer_prefill_step(
                &mut runner,
                &mut stream,
                &mut run,
                timestep,
                q_token_elements,
                k_token_elements,
                v_token_elements,
                attention_elements,
                hidden,
                "package scheduler layer prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete package scheduler layer prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "package scheduler layer first batch",
    )?;
    let second_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "package scheduler layer second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query package scheduler layer final ready batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "package scheduler layer final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "package scheduler layer request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "package scheduler layer request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "package scheduler layer request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read package scheduler layer cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.checks.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "package scheduler layer request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.checks.expected.paged_cache.k,
            1e-5,
            1e-5,
        )?;
        run.checks.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "package scheduler layer request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.checks.expected.paged_cache.v,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let total_tokens = runs.iter().map(|run| run.total_tokens).collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projection_input_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projection_input_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projected_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projected_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let block_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.block_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let post_norm_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.post_norm_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let mlp_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.mlp_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let layer_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.layer_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "package-self-attn-mlp-block-scheduler-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" sequence_len={} request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} total_tokens={:?} request_position_offsets={:?} paged_block_size={} paged_cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} hidden={} intermediate={} q_projection_layout={} q_gate_elements={:?} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} mlp_epsilon={mlp_epsilon:.9} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} causal_attention_max_abs_diff={causal_attention_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
        path,
        layer_index,
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
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        total_tokens,
        request_position_offsets,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        hidden,
        intermediate,
        q_projection_layout.unwrap_or("unknown"),
        q_gate_elements,
        output_gate_layout.unwrap_or("unknown"),
        runtime_shape.q_heads,
        runtime_shape.kv_heads,
        runtime_shape.head_dim,
        runtime_shape.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
    ))
}

struct PackageModelLoopRequestPlan {
    input_source: &'static str,
    scheduler: SchedulerState,
    requests: Vec<Request>,
    request_ids: Vec<u64>,
    prompt_tokens: Vec<usize>,
    max_new_tokens: Vec<usize>,
    total_tokens: Vec<usize>,
    prompt_token_ids_by_request: Vec<Vec<usize>>,
    decode_token_ids_by_request: Vec<Vec<usize>>,
    block_tables: Vec<Vec<u32>>,
    initial_residuals: Vec<Vec<f32>>,
    block_size: usize,
    cache_blocks: usize,
    position_stride: usize,
}

#[derive(Default)]
struct PackageModelLoopPreparedDiffs {
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    causal_attention_max_abs_diff: f32,
}

struct PackageModelLoopRuntimeDiffs {
    attention_max_abs_diff: f32,
    projection_input_max_abs_diff: f32,
    projected_max_abs_diff: f32,
    block_max_abs_diff: f32,
    post_norm_max_abs_diff: f32,
    mlp_max_abs_diff: f32,
    layer_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

struct PackageModelLoopLayerRunPlan {
    runs_by_layer: Vec<Vec<SchedulerLayerDecodeRun>>,
    q_projection_layouts: Vec<&'static str>,
    output_gate_layouts: Vec<&'static str>,
    q_gate_elements_by_layer: Vec<Vec<usize>>,
    prepared_diffs: PackageModelLoopPreparedDiffs,
}

struct PackageModelLoopExecutionPlan {
    decode: Qwen3PackageModelDecodePlan,
    max_decode_batch_requests: usize,
}

struct PackageModelLoopExecutionSummary {
    first_batch_ready: usize,
    second_batch_ready: usize,
    prefill_batch_request_counts: Vec<usize>,
    decode_batch_ready_counts: Vec<usize>,
    final_ready: usize,
    prefill_wall_ms: f64,
    decode_wall_ms: f64,
    total_wall_ms: f64,
}

struct PackageModelLoopSqOverlayInfo {
    artifact: String,
    candidate: String,
    candidate_legacy: Option<String>,
    format_id: String,
    implementation_id: String,
    schema_version: String,
    fp8_tensor_count: u64,
    passthrough_tensor_count: u64,
    row_chunk: usize,
}

struct PackageModelLoopSmokeRun {
    command_name: &'static str,
    model: Qwen3PackageModelRuntime,
    request_plan: PackageModelLoopRequestPlan,
    layer_run_plan: PackageModelLoopLayerRunPlan,
    execution_plan: PackageModelLoopExecutionPlan,
    execution_summary: Option<PackageModelLoopExecutionSummary>,
    final_top_logits: Option<Vec<Vec<PackageTokenLogit>>>,
    lm_head_top_k: Option<usize>,
    lm_head_chunk_rows: Option<usize>,
    sq_overlay_info: Option<PackageModelLoopSqOverlayInfo>,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
}

fn parse_package_model_loop_rotary_dim(
    model: &Qwen3PackageModelRuntime,
    rotary_dim: Option<String>,
) -> Result<usize, String> {
    let rotary_dim = match rotary_dim {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid rotary dim {raw:?}: {err}"))?,
        None => model.default_rotary_dim()?,
    };
    if rotary_dim == 0 || rotary_dim > model.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={}",
            model.head_dim
        ));
    }
    Ok(rotary_dim)
}

fn package_model_loop_sq_overlay_info(
    artifact: &ullm_engine::sq::SqFp8Artifact,
    row_chunk: usize,
) -> PackageModelLoopSqOverlayInfo {
    let candidate_legacy = if artifact.manifest.candidate.id == FORMAT_SQ8_0 {
        None
    } else {
        Some(artifact.manifest.candidate.id.clone())
    };
    let implementation_id = artifact
        .manifest
        .candidate
        .implementation_id
        .clone()
        .or_else(|| candidate_legacy.clone())
        .unwrap_or_else(|| "none".to_string());
    PackageModelLoopSqOverlayInfo {
        artifact: artifact.artifact_dir.display().to_string(),
        candidate: FORMAT_SQ8_0.to_string(),
        candidate_legacy,
        format_id: FORMAT_SQ8_0.to_string(),
        implementation_id,
        schema_version: artifact.manifest.schema_version.clone(),
        fp8_tensor_count: artifact.manifest.storage.fp8_tensor_count,
        passthrough_tensor_count: artifact.manifest.storage.passthrough_tensor_count,
        row_chunk,
    }
}

impl PackageModelLoopRequestPlan {
    fn new(sequence_len: usize, hidden: usize, block_size: usize) -> Result<Self, String> {
        let requests = vec![
            Request::new(201, sequence_len - 2, 2),
            Request::new(202, sequence_len - 1, 1),
            Request::new(203, 1, 0),
        ];
        let mut initial_residuals = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.iter().enumerate() {
            let total_tokens = request
                .prompt_tokens
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
            let base_input = deterministic_f32_vector(hidden);
            let residual_elements = total_tokens.checked_mul(hidden).ok_or_else(|| {
                format!("request {:?} residual element count overflows", request.id)
            })?;
            let mut residual = Vec::with_capacity(residual_elements);
            for timestep in 0..total_tokens {
                let shifted_timestep = timestep
                    .checked_add(request_index.checked_mul(sequence_len).ok_or_else(|| {
                        "model-loop residual timestep multiplier overflows".to_string()
                    })?)
                    .ok_or_else(|| "model-loop residual timestep overflows".to_string())?;
                residual.extend(linear_attn_step_input(&base_input, shifted_timestep));
            }
            initial_residuals.push(residual);
        }
        let request_count = requests.len();
        Self::from_initial_residuals(
            requests,
            initial_residuals,
            vec![Vec::new(); request_count],
            vec![Vec::new(); request_count],
            hidden,
            block_size,
            sequence_len,
            "synthetic_residual",
        )
    }

    fn from_token_id_batches(
        path: &str,
        prompt_token_ids_batch: Vec<Vec<usize>>,
        generated_tokens_batch: Vec<usize>,
        hidden: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        if prompt_token_ids_batch.is_empty() {
            return Err("model-loop token-id batch requires at least one request".to_string());
        }
        if prompt_token_ids_batch.len() != generated_tokens_batch.len() {
            return Err(format!(
                "model-loop token-id prompt request count {} does not match generated token count {}",
                prompt_token_ids_batch.len(),
                generated_tokens_batch.len()
            ));
        }
        let (embedding_vocab, embedding_hidden) = package_embedding_shape(path)?;
        if embedding_hidden != hidden {
            return Err(format!(
                "model-loop token-id embedding hidden mismatch: embedding={embedding_hidden} model={hidden}"
            ));
        }
        if embedding_vocab <= 1 {
            return Err(format!(
                "model-loop token-id embedding vocab must be greater than one, got {embedding_vocab}"
            ));
        }

        let mut requests = Vec::with_capacity(prompt_token_ids_batch.len());
        let mut initial_residuals = Vec::with_capacity(prompt_token_ids_batch.len());
        let mut decode_token_ids_by_request = Vec::with_capacity(prompt_token_ids_batch.len());
        let mut position_stride = 0_usize;
        for (request_index, prompt_token_ids) in prompt_token_ids_batch.iter().enumerate() {
            if prompt_token_ids.is_empty() {
                return Err(format!(
                    "model-loop token-id request {request_index} has no prompt tokens"
                ));
            }
            if let Some(token_id) = prompt_token_ids
                .iter()
                .copied()
                .find(|token_id| *token_id >= embedding_vocab)
            {
                return Err(format!(
                    "model-loop token-id request {request_index} token id {token_id} is out of embedding range 0..{embedding_vocab}"
                ));
            }
            let generated_tokens = generated_tokens_batch[request_index];
            let mut decode_token_ids = Vec::with_capacity(generated_tokens);
            for generated_index in 0..generated_tokens {
                let future_token_id = 1
                    + ((request_index
                        .checked_mul(4096)
                        .and_then(|value| value.checked_add(prompt_token_ids.len()))
                        .and_then(|value| value.checked_add(generated_index))
                        .ok_or_else(|| {
                            "model-loop token-id future token id seed overflows".to_string()
                        })?)
                        % (embedding_vocab - 1));
                decode_token_ids.push(future_token_id);
            }
            let mut token_ids = prompt_token_ids.clone();
            token_ids.extend(decode_token_ids.iter().copied());
            let rows = read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &token_ids)
                .map_err(|err| {
                    format!(
                        "failed to read model-loop token-id embedding rows for request {request_index}: {err}"
                    )
                })?;
            if rows.columns != hidden || rows.values.len() != token_ids.len() * hidden {
                return Err(format!(
                    "model-loop token-id embedding shape mismatch for request {request_index}: columns={} values={} tokens={} hidden={hidden}",
                    rows.columns,
                    rows.values.len(),
                    token_ids.len()
                ));
            }
            let request_id = 301_u64
                .checked_add(
                    u64::try_from(request_index)
                        .map_err(|_| "model-loop token-id request index exceeds u64".to_string())?,
                )
                .ok_or_else(|| "model-loop token-id request id overflows".to_string())?;
            requests.push(Request::new(
                request_id,
                prompt_token_ids.len(),
                generated_tokens,
            ));
            position_stride = position_stride.max(token_ids.len());
            decode_token_ids_by_request.push(decode_token_ids);
            initial_residuals.push(rows.values);
        }

        Self::from_initial_residuals(
            requests,
            initial_residuals,
            prompt_token_ids_batch,
            decode_token_ids_by_request,
            hidden,
            block_size,
            position_stride,
            "embedding_token_ids",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_initial_residuals(
        requests: Vec<Request>,
        initial_residuals_input: Vec<Vec<f32>>,
        prompt_token_ids_by_request: Vec<Vec<usize>>,
        decode_token_ids_by_request: Vec<Vec<usize>>,
        hidden: usize,
        block_size: usize,
        position_stride: usize,
        input_source: &'static str,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("model-loop block size must be greater than zero".to_string());
        }
        if requests.is_empty() {
            return Err("model-loop request plan requires at least one request".to_string());
        }
        if position_stride == 0 {
            return Err("model-loop position stride must be greater than zero".to_string());
        }
        if requests.len() != initial_residuals_input.len()
            || requests.len() != prompt_token_ids_by_request.len()
            || requests.len() != decode_token_ids_by_request.len()
        {
            return Err(format!(
                "model-loop request count mismatch: requests={} residuals={} prompt_ids={} decode_ids={}",
                requests.len(),
                initial_residuals_input.len(),
                prompt_token_ids_by_request.len(),
                decode_token_ids_by_request.len()
            ));
        }
        let mut required_blocks = 0_usize;
        for request in &requests {
            let total_tokens = request
                .prompt_tokens
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
            if total_tokens == 0 {
                return Err(format!("request {:?} has zero total tokens", request.id));
            }
            let request_blocks = (total_tokens - 1) / block_size + 1;
            required_blocks = required_blocks
                .checked_add(request_blocks)
                .ok_or_else(|| "model-loop required block count overflows".to_string())?;
        }
        let cache_blocks = required_blocks
            .checked_add(2)
            .ok_or_else(|| "model-loop cache block count overflows".to_string())?;
        if cache_blocks > u32::MAX as usize || block_size > u32::MAX as usize {
            return Err(format!(
                "model-loop block layout exceeds u32 range: cache_blocks={cache_blocks} block_size={block_size}"
            ));
        }

        let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
        for request in &requests {
            scheduler.enqueue(request.clone());
        }
        let mut allocated = scheduler
            .pop_prefill_batch_with_allocation(usize::MAX)
            .map_err(|err| format!("failed to allocate model-loop package batch: {err}"))?;
        if allocated.len() != requests.len() {
            return Err(format!(
                "model-loop selected {} requests, expected {}",
                allocated.len(),
                requests.len()
            ));
        }

        let mut request_ids = Vec::with_capacity(allocated.len());
        let mut prompt_tokens = Vec::with_capacity(allocated.len());
        let mut max_new_tokens = Vec::with_capacity(allocated.len());
        let mut total_tokens = Vec::with_capacity(allocated.len());
        let mut block_tables = Vec::with_capacity(allocated.len());
        let mut initial_residuals = Vec::with_capacity(allocated.len());
        for (request_index, scheduled) in allocated.drain(..).enumerate() {
            let request = scheduled.request;
            let request_total_tokens = request
                .prompt_tokens
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
            let residual_elements = request_total_tokens.checked_mul(hidden).ok_or_else(|| {
                format!("request {:?} residual element count overflows", request.id)
            })?;
            let residual = initial_residuals_input
                .get(request_index)
                .ok_or_else(|| {
                    format!("model-loop residual input missing for request index {request_index}")
                })?
                .clone();
            if residual.len() != residual_elements {
                return Err(format!(
                    "model-loop residual length mismatch for request {:?}: got {} expected {residual_elements}",
                    request.id,
                    residual.len()
                ));
            }
            request_ids.push(request.id.0);
            prompt_tokens.push(request.prompt_tokens);
            max_new_tokens.push(request.max_new_tokens);
            total_tokens.push(request_total_tokens);
            block_tables.push(scheduled.allocation.blocks);
            initial_residuals.push(residual);
        }

        Ok(Self {
            input_source,
            scheduler,
            requests,
            request_ids,
            prompt_tokens,
            max_new_tokens,
            total_tokens,
            prompt_token_ids_by_request,
            decode_token_ids_by_request,
            block_tables,
            initial_residuals,
            block_size,
            cache_blocks,
            position_stride,
        })
    }

    fn request_count(&self) -> usize {
        self.requests.len()
    }

    fn complete_prefill_all(&mut self) -> Result<(), String> {
        for request in &self.requests {
            self.scheduler.complete_prefill(request.id).map_err(|err| {
                format!(
                    "failed to complete package model-loop prefill {:?}: {err}",
                    request.id
                )
            })?;
        }
        Ok(())
    }

    fn cached_tokens(&self) -> Result<Vec<usize>, String> {
        self.requests
            .iter()
            .map(|request| {
                self.scheduler
                    .active_request(request.id)
                    .map(|active| active.cached_tokens)
                    .ok_or_else(|| format!("request {:?} is not active", request.id))
            })
            .collect()
    }

    fn generated_tokens(&self) -> Result<Vec<usize>, String> {
        self.requests
            .iter()
            .map(|request| {
                self.scheduler
                    .active_request(request.id)
                    .map(|active| active.generated_tokens)
                    .ok_or_else(|| format!("request {:?} is not active", request.id))
            })
            .collect()
    }
}

impl PackageModelLoopPreparedDiffs {
    fn observe(&mut self, prepared: &Qwen3SelfAttnModelLoopPreparedSequence) {
        self.q_norm_max_abs_diff = self.q_norm_max_abs_diff.max(prepared.q_norm_max_abs_diff);
        self.k_norm_max_abs_diff = self.k_norm_max_abs_diff.max(prepared.k_norm_max_abs_diff);
        self.q_rope_max_abs_diff = self.q_rope_max_abs_diff.max(prepared.q_rope_max_abs_diff);
        self.k_rope_max_abs_diff = self.k_rope_max_abs_diff.max(prepared.k_rope_max_abs_diff);
        self.causal_attention_max_abs_diff = self
            .causal_attention_max_abs_diff
            .max(prepared.attention_max_abs_diff);
    }
}

impl PackageModelLoopLayerRunPlan {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        request_plan: &PackageModelLoopRequestPlan,
        decode_shape: PagedDecodeShape,
        rotary_dim: usize,
        rope_base: f32,
        position_offset: usize,
    ) -> Result<Self, String> {
        let mut runs_by_layer: Vec<Vec<SchedulerLayerDecodeRun>> =
            Vec::with_capacity(model.layer_count());
        let mut q_projection_layouts = Vec::with_capacity(model.layer_count());
        let mut output_gate_layouts = Vec::with_capacity(model.layer_count());
        let mut q_gate_elements_by_layer = Vec::with_capacity(model.layer_count());
        let mut prepared_diffs = PackageModelLoopPreparedDiffs::default();

        for (layer_position, layer) in model.layers.iter().enumerate() {
            let mut runs = Vec::with_capacity(request_plan.request_count());
            let mut q_gate_elements = Vec::with_capacity(request_plan.request_count());
            for (request_index, request) in request_plan.requests.iter().enumerate() {
                let residual_sequence = if layer_position == 0 {
                    request_plan.initial_residuals[request_index].clone()
                } else {
                    runs_by_layer[layer_position - 1][request_index]
                        .checks
                        .expected
                        .layer_output
                        .clone()
                };
                let request_position_stride = request_index
                    .checked_mul(request_plan.position_stride)
                    .ok_or_else(|| {
                        "model-loop request position offset multiplier overflows".to_string()
                    })?;
                let request_position_offset = position_offset
                    .checked_add(request_position_stride)
                    .ok_or_else(|| "model-loop request position offset overflows".to_string())?;
                let prepared = qwen3_self_attn_prepare_model_loop_sequence_smoke(
                    context,
                    stream,
                    &layer.weights.self_attn,
                    residual_sequence,
                    request_plan.total_tokens[request_index],
                    rotary_dim,
                    rope_base,
                    request_position_offset,
                    &layer.input_norm,
                    &layer.q_norm,
                    &layer.k_norm,
                    &request_plan.block_tables[request_index],
                    request_plan.block_size,
                    request_plan.cache_blocks,
                    &format!(
                        "package-self-attn-mlp-block-model-loop-smoke layer {} request {:?}",
                        layer.layer_index, request.id
                    ),
                )?;
                if prepared.hidden != model.hidden
                    || prepared.q_heads != model.q_heads
                    || prepared.kv_heads != model.kv_heads
                    || prepared.head_dim != model.head_dim
                    || prepared.value_dim != model.value_dim
                {
                    return Err(format!(
                        "model-loop prepared shape mismatch for layer {} request {:?}",
                        layer.layer_index, request.id
                    ));
                }
                q_gate_elements.push(prepared.q_gate_elements);
                prepared_diffs.observe(&prepared);
                if request_index == 0 {
                    q_projection_layouts.push(prepared.q_projection_layout);
                    output_gate_layouts.push(prepared.output_gate_layout);
                }

                let expected = qwen3_decoder_layer_sequence_to_host_f32(
                    &layer.weights,
                    context,
                    stream,
                    decode_shape,
                    &request_plan.block_tables[request_index],
                    prepared.softmax_scale,
                    model.mlp_epsilon,
                    &prepared.q_rope,
                    &prepared.k_rope,
                    &prepared.v_projected,
                    prepared.q_gate.as_deref(),
                    &prepared.residual_sequence,
                    request_plan.total_tokens[request_index],
                )?;
                runs.push(SchedulerLayerDecodeRun {
                    state: SchedulerLayerDecodeState {
                        request_id: request.id,
                        prompt_tokens: request.prompt_tokens,
                        max_new_tokens: request.max_new_tokens,
                        total_tokens: request_plan.total_tokens[request_index],
                        block_table: request_plan.block_tables[request_index].clone(),
                        q_sequence: prepared.q_rope,
                        k_sequence: prepared.k_rope,
                        v_sequence: prepared.v_projected,
                        output_gate_sequence: prepared.q_gate,
                        residual_sequence: prepared.residual_sequence,
                        decode_steps: 0,
                    },
                    checks: SchedulerLayerDecodeSmokeChecks::new(expected),
                });
            }
            q_gate_elements_by_layer.push(q_gate_elements);
            runs_by_layer.push(runs);
        }

        Ok(Self {
            runs_by_layer,
            q_projection_layouts,
            output_gate_layouts,
            q_gate_elements_by_layer,
            prepared_diffs,
        })
    }

    fn decode_steps_by_layer(&self) -> Vec<Vec<usize>> {
        self.runs_by_layer
            .iter()
            .map(|runs| runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }

    fn runtime_diffs(&self) -> PackageModelLoopRuntimeDiffs {
        PackageModelLoopRuntimeDiffs {
            attention_max_abs_diff: self.max_run_diff(|run| run.checks.attention_max_abs_diff),
            projection_input_max_abs_diff: self
                .max_run_diff(|run| run.checks.projection_input_max_abs_diff),
            projected_max_abs_diff: self.max_run_diff(|run| run.checks.projected_max_abs_diff),
            block_max_abs_diff: self.max_run_diff(|run| run.checks.block_max_abs_diff),
            post_norm_max_abs_diff: self.max_run_diff(|run| run.checks.post_norm_max_abs_diff),
            mlp_max_abs_diff: self.max_run_diff(|run| run.checks.mlp_max_abs_diff),
            layer_max_abs_diff: self.max_run_diff(|run| run.checks.layer_max_abs_diff),
            k_cache_max_abs_diff: self.max_run_diff(|run| run.checks.k_cache_max_abs_diff),
            v_cache_max_abs_diff: self.max_run_diff(|run| run.checks.v_cache_max_abs_diff),
        }
    }

    fn stack_requests(&self) -> Vec<Vec<Qwen3PackageModelStackRequest<'_>>> {
        self.runs_by_layer
            .iter()
            .map(|runs| {
                runs.iter()
                    .map(|run| Qwen3PackageModelStackRequest {
                        request_id: run.request_id,
                        block_table: &run.block_table,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn max_run_diff<F>(&self, select: F) -> f32
    where
        F: Fn(&SchedulerLayerDecodeRun) -> f32,
    {
        self.runs_by_layer
            .iter()
            .flatten()
            .map(select)
            .fold(0.0_f32, f32::max)
    }
}

impl PackageModelLoopExecutionPlan {
    fn new(
        model: &Qwen3PackageModelRuntime,
        request_plan: &PackageModelLoopRequestPlan,
    ) -> Result<Self, String> {
        Ok(Self {
            decode: Qwen3PackageModelDecodePlan::from_model(
                model,
                request_plan.block_size,
                request_plan.cache_blocks,
            )?,
            max_decode_batch_requests: 8,
        })
    }

    fn execute(
        &self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        request_plan: &mut PackageModelLoopRequestPlan,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<PackageModelLoopExecutionSummary, String> {
        let mut layer_runner = self.build_layer_runner(context, stream, model, layer_run_plan)?;
        let prefill_started = Instant::now();
        let prefill_batch_request_counts =
            self.run_prefill_layers(&mut layer_runner, stream, model, layer_run_plan)?;
        let prefill_wall_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        request_plan.complete_prefill_all()?;

        let decode_started = Instant::now();
        let decode_batch_ready_counts =
            self.run_decode_batches(&mut layer_runner, stream, request_plan, layer_run_plan)?;
        let decode_wall_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        let final_ready = self.final_ready(request_plan)?;
        self.verify_layer_caches(&layer_runner, stream, model, layer_run_plan)?;

        Ok(PackageModelLoopExecutionSummary {
            first_batch_ready: decode_batch_ready_counts.first().copied().unwrap_or(0),
            second_batch_ready: decode_batch_ready_counts.get(1).copied().unwrap_or(0),
            prefill_batch_request_counts,
            decode_batch_ready_counts,
            final_ready,
            prefill_wall_ms,
            decode_wall_ms,
            total_wall_ms: prefill_wall_ms + decode_wall_ms,
        })
    }

    fn build_layer_runner<'weights>(
        &self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &'weights Qwen3PackageModelRuntime,
        layer_run_plan: &PackageModelLoopLayerRunPlan,
    ) -> Result<Qwen3DecoderLayerStackRequestDecodeRunner<'weights>, String> {
        let layer_requests = layer_run_plan.stack_requests();
        qwen3_package_model_stack_runner(model, context, stream, self.decode, &layer_requests)
    }

    fn run_prefill_layers(
        &self,
        layer_runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<Vec<usize>, String> {
        let mut batch_request_counts = Vec::new();
        for layer_position in 0..layer_runner.layer_count() {
            let max_prompt_tokens = layer_run_plan.runs_by_layer[layer_position]
                .iter()
                .map(|run| run.prompt_tokens)
                .max()
                .unwrap_or(0);
            for timestep in 0..max_prompt_tokens {
                let count = run_scheduler_layer_stack_prefill_batch(
                    layer_runner,
                    layer_position,
                    stream,
                    &mut layer_run_plan.runs_by_layer[layer_position],
                    timestep,
                    self.decode,
                    &format!(
                        "package model-loop layer {} prefill batch timestep {timestep}",
                        model.layers[layer_position].layer_index
                    ),
                )?;
                if count > 0 {
                    batch_request_counts.push(count);
                }
            }
        }
        Ok(batch_request_counts)
    }

    fn run_decode_batches(
        &self,
        layer_runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_plan: &mut PackageModelLoopRequestPlan,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<Vec<usize>, String> {
        let mut decode_batch_ready_counts = Vec::new();
        loop {
            let ready = request_plan
                .scheduler
                .ready_decode_batch(self.max_decode_batch_requests)
                .map_err(|err| format!("failed to query model-loop ready batch: {err}"))?;
            if ready.is_empty() {
                break;
            }
            let batch_index = decode_batch_ready_counts.len();
            let label = format!("package model-loop decode batch {batch_index}");
            let ready_count = run_scheduler_layer_stack_ready_batch(
                layer_runner,
                &mut request_plan.scheduler,
                &mut layer_run_plan.runs_by_layer,
                stream,
                &ready,
                self.decode,
                &label,
            )?;
            decode_batch_ready_counts.push(ready_count);
        }
        Ok(decode_batch_ready_counts)
    }

    fn final_ready(&self, request_plan: &PackageModelLoopRequestPlan) -> Result<usize, String> {
        let final_ready = request_plan
            .scheduler
            .ready_decode_batch(self.max_decode_batch_requests)
            .map_err(|err| format!("failed to query model-loop final ready batch: {err}"))?
            .len();
        if final_ready != 0 {
            return Err(format!(
                "package model-loop final ready count {final_ready}, expected 0"
            ));
        }
        Ok(final_ready)
    }

    fn verify_layer_caches(
        &self,
        layer_runner: &Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<(), String> {
        for layer_position in 0..layer_runner.layer_count() {
            for run in &mut layer_run_plan.runs_by_layer[layer_position] {
                let cache = layer_runner
                    .read_layer_cache_to_host(layer_position, run.request_id, stream)
                    .map_err(|err| {
                        format!(
                            "failed to read package model-loop layer {} cache for {:?}: {err}",
                            model.layers[layer_position].layer_index, run.request_id
                        )
                    })?;
                run.checks.k_cache_max_abs_diff = verify_f32_close(
                    &format!(
                        "package model-loop layer {} request {:?} k cache",
                        model.layers[layer_position].layer_index, run.request_id
                    ),
                    &cache.k,
                    &run.checks.expected.paged_cache.k,
                    1e-5,
                    1e-5,
                )?;
                run.checks.v_cache_max_abs_diff = verify_f32_close(
                    &format!(
                        "package model-loop layer {} request {:?} v cache",
                        model.layers[layer_position].layer_index, run.request_id
                    ),
                    &cache.v,
                    &run.checks.expected.paged_cache.v,
                    1e-5,
                    1e-5,
                )?;
            }
        }
        Ok(())
    }
}

impl PackageModelLoopSmokeRun {
    #[allow(clippy::too_many_arguments)]
    fn new(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
        sequence_len: usize,
        rotary_dim: Option<String>,
        rope_base: f32,
        position_offset: usize,
    ) -> Result<Self, String> {
        let model =
            Qwen3PackageModelRuntime::load(context, stream, path, chunk_bytes, layer_indices)?;
        let rotary_dim = parse_package_model_loop_rotary_dim(&model, rotary_dim)?;
        let block_size = 2_usize;
        let request_plan =
            PackageModelLoopRequestPlan::new(sequence_len, model.hidden, block_size)?;
        let execution_plan = PackageModelLoopExecutionPlan::new(&model, &request_plan)?;

        let layer_run_plan = PackageModelLoopLayerRunPlan::prepare(
            context,
            stream,
            &model,
            &request_plan,
            execution_plan.decode.decode_shape,
            rotary_dim,
            rope_base,
            position_offset,
        )?;

        Ok(Self {
            command_name: "package-self-attn-mlp-block-model-loop-smoke",
            model,
            request_plan,
            layer_run_plan,
            execution_plan,
            execution_summary: None,
            final_top_logits: None,
            lm_head_top_k: None,
            lm_head_chunk_rows: None,
            sq_overlay_info: None,
            sequence_len,
            rotary_dim,
            rope_base,
            position_offset,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_from_token_ids(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
        prompt_token_ids_batch: Vec<Vec<usize>>,
        generated_tokens_batch: Vec<usize>,
        top_k: usize,
        lm_head_chunk_rows: usize,
        rotary_dim: Option<String>,
        rope_base: f32,
        position_offset: usize,
        command_name: &'static str,
        sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
    ) -> Result<Self, String> {
        let sq_row_chunk = 256_usize;
        let sq_overlay = sq_artifact.map(|artifact| Qwen3PackageSqOverlay {
            artifact,
            row_chunk: sq_row_chunk,
        });
        let sq_overlay_info =
            sq_artifact.map(|artifact| package_model_loop_sq_overlay_info(artifact, sq_row_chunk));
        let model = Qwen3PackageModelRuntime::load_with_sq_overlay(
            context,
            stream,
            path,
            chunk_bytes,
            layer_indices,
            sq_overlay.as_ref(),
        )?;
        let rotary_dim = parse_package_model_loop_rotary_dim(&model, rotary_dim)?;
        let block_size = 2_usize;
        let request_plan = PackageModelLoopRequestPlan::from_token_id_batches(
            path,
            prompt_token_ids_batch,
            generated_tokens_batch,
            model.hidden,
            block_size,
        )?;
        let sequence_len = request_plan.position_stride;
        let execution_plan = PackageModelLoopExecutionPlan::new(&model, &request_plan)?;

        let layer_run_plan = PackageModelLoopLayerRunPlan::prepare(
            context,
            stream,
            &model,
            &request_plan,
            execution_plan.decode.decode_shape,
            rotary_dim,
            rope_base,
            position_offset,
        )?;

        Ok(Self {
            command_name,
            model,
            request_plan,
            layer_run_plan,
            execution_plan,
            execution_summary: None,
            final_top_logits: None,
            lm_head_top_k: Some(top_k),
            lm_head_chunk_rows: Some(lm_head_chunk_rows),
            sq_overlay_info,
            sequence_len,
            rotary_dim,
            rope_base,
            position_offset,
        })
    }

    fn execute(
        &mut self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        if self.execution_summary.is_some() {
            return Err("package model-loop smoke run has already executed".to_string());
        }
        let summary = self.execution_plan.execute(
            context,
            stream,
            &self.model,
            &mut self.request_plan,
            &mut self.layer_run_plan,
        )?;
        self.execution_summary = Some(summary);
        Ok(())
    }

    fn compute_final_top_logits(
        &mut self,
        path: &str,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        chunk_bytes: usize,
        top_k: usize,
        lm_head_chunk_rows: usize,
    ) -> Result<(), String> {
        let mut final_norm = read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, chunk_bytes)
            .map_err(|err| format!("failed to read model-loop final RMSNorm tensor: {err}"))?;
        final_norm.values =
            effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
        if final_norm.values.len() != self.model.hidden {
            return Err(format!(
                "model-loop final RMSNorm length mismatch: len={} hidden={}",
                final_norm.values.len(),
                self.model.hidden
            ));
        }
        let final_layer_runs =
            self.layer_run_plan.runs_by_layer.last().ok_or_else(|| {
                "model-loop final top logits require at least one layer".to_string()
            })?;
        let mut final_top_logits = Vec::with_capacity(final_layer_runs.len());
        for run in final_layer_runs {
            let final_token_index = run
                .total_tokens
                .checked_sub(1)
                .ok_or_else(|| format!("request {:?} has no final token", run.request_id))?;
            let final_start = final_token_index
                .checked_mul(self.model.hidden)
                .ok_or_else(|| "model-loop final hidden start overflows".to_string())?;
            let final_end = final_start
                .checked_add(self.model.hidden)
                .ok_or_else(|| "model-loop final hidden end overflows".to_string())?;
            if final_end > run.checks.expected.layer_output.len() {
                return Err(format!(
                    "model-loop final hidden slice {final_start}..{final_end} exceeds request {:?} layer output len {}",
                    run.request_id,
                    run.checks.expected.layer_output.len()
                ));
            }
            let final_hidden = &run.checks.expected.layer_output[final_start..final_end];
            let final_normed = runtime_host_rmsnorm_f32(final_hidden, &final_norm.values, 1e-6_f32);
            if final_normed.len() != self.model.hidden
                || final_normed.iter().any(|value| !value.is_finite())
            {
                return Err(format!(
                    "model-loop final normalized hidden state for request {:?} contains invalid values",
                    run.request_id
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
                    let mut lm_head_runtime = PackageLmHeadRuntime::load(
                        PackageLmHeadMode::GpuResidentF32,
                        context,
                        stream,
                        path,
                        chunk_bytes,
                        self.model.hidden,
                        lm_head_chunk_rows,
                    )
                    .map_err(|resident_err| {
                        format!(
                            "failed to compute model-loop lm_head top-k: cpu_chunked_error={cpu_err}; resident_error={resident_err}"
                        )
                    })?;
                    lm_head_runtime.top_logits(path, stream, &final_normed, top_k)?
                }
            };
            final_top_logits.push(top_logits);
        }
        self.final_top_logits = Some(final_top_logits);
        self.lm_head_top_k = Some(top_k);
        self.lm_head_chunk_rows = Some(lm_head_chunk_rows);
        Ok(())
    }

    fn format_output(
        &self,
        path: &str,
        device_index: u32,
        info: &ullm_runtime_sys::DeviceInfo,
    ) -> Result<String, String> {
        let execution_summary = self
            .execution_summary
            .as_ref()
            .ok_or_else(|| "package model-loop smoke run has not executed".to_string())?;
        let stats = self.request_plan.scheduler.allocator_stats();
        let cached_tokens = self.request_plan.cached_tokens()?;
        let generated_tokens = self.request_plan.generated_tokens()?;
        let prefill_total_input_tokens = self
            .request_plan
            .prompt_tokens
            .iter()
            .try_fold(0_usize, |acc, value| acc.checked_add(*value))
            .ok_or_else(|| "package model-loop prefill total input tokens overflow".to_string())?;
        let decode_total_generated_tokens = generated_tokens
            .iter()
            .try_fold(0_usize, |acc, value| acc.checked_add(*value))
            .ok_or_else(|| {
                "package model-loop decode total generated tokens overflow".to_string()
            })?;
        let end_to_end_total_tokens = prefill_total_input_tokens
            .checked_add(decode_total_generated_tokens)
            .ok_or_else(|| "package model-loop end-to-end total tokens overflow".to_string())?;
        let decode_executor_request_parallelism = execution_summary
            .decode_batch_ready_counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let decode_real_batch = decode_executor_request_parallelism > 1;
        let prefill_executor_request_parallelism = execution_summary
            .prefill_batch_request_counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let prefill_real_batch = prefill_executor_request_parallelism > 1;
        let prefill_executor = if prefill_real_batch {
            "stack_prefill_request_batch_step"
        } else {
            "stack_prefill_step"
        };
        let batching_mode = if prefill_real_batch && decode_real_batch {
            "real"
        } else {
            "hybrid"
        };
        let layer_indices_csv = self
            .model
            .layer_indices()
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let prompt_tokens_csv = self
            .request_plan
            .prompt_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let max_new_tokens_csv = self
            .request_plan
            .max_new_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let total_tokens_csv = self
            .request_plan
            .total_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let generated_tokens_csv = generated_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let decode_batch_ready_counts_csv = execution_summary
            .decode_batch_ready_counts
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let prefill_batch_request_counts_csv = execution_summary
            .prefill_batch_request_counts
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let decode_steps_by_layer = self.layer_run_plan.decode_steps_by_layer();
        let layer_indices = self.model.layer_indices();
        let input_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.input_norm_tensor);
        let q_tensors = self.model.tensor_names_by_layer(|layer| &layer.q_tensor);
        let k_tensors = self.model.tensor_names_by_layer(|layer| &layer.k_tensor);
        let v_tensors = self.model.tensor_names_by_layer(|layer| &layer.v_tensor);
        let o_tensors = self.model.tensor_names_by_layer(|layer| &layer.o_tensor);
        let q_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.q_norm_tensor);
        let k_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.k_norm_tensor);
        let post_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.post_norm_tensor);
        let gate_tensors = self.model.tensor_names_by_layer(|layer| &layer.gate_tensor);
        let up_tensors = self.model.tensor_names_by_layer(|layer| &layer.up_tensor);
        let down_tensors = self.model.tensor_names_by_layer(|layer| &layer.down_tensor);
        let input_norm_dtypes = self.model.input_norm_dtypes();
        let q_norm_dtypes = self.model.q_norm_dtypes();
        let k_norm_dtypes = self.model.k_norm_dtypes();
        let post_norm_dtypes = self.model.post_norm_dtypes();
        let prepared_diffs = &self.layer_run_plan.prepared_diffs;
        let runtime_diffs = self.layer_run_plan.runtime_diffs();
        let final_top1_tokens = self
            .final_top_logits
            .as_ref()
            .map(|requests| {
                requests
                    .iter()
                    .filter_map(|entries| entries.first().map(|entry| entry.token_id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let final_top1_tokens_csv = final_top1_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let final_top1_logits_csv = self
            .final_top_logits
            .as_ref()
            .map(|requests| {
                requests
                    .iter()
                    .filter_map(|entries| {
                        entries.first().map(|entry| format!("{:.6}", entry.logit))
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let final_topk_tokens_csv = self
            .final_top_logits
            .as_ref()
            .map(|requests| {
                requests
                    .iter()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| entry.token_id.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();
        let final_topk_logits_csv = self
            .final_top_logits
            .as_ref()
            .map(|requests| {
                requests
                    .iter()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| format!("{:.6}", entry.logit))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();
        let final_lm_head_guard = self.final_top_logits.is_some();
        let prefill_mode = if self.request_plan.input_source == "embedding_token_ids" {
            "token_id_layer_stack"
        } else {
            "synthetic_layer_stack"
        };
        let sq_overlay = self.sq_overlay_info.is_some();
        let sq_candidate = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.candidate.as_str())
            .unwrap_or("none");
        let format_id = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.format_id.as_str())
            .unwrap_or(FORMAT_AQ4_0);
        let sq_format_id = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.format_id.as_str())
            .unwrap_or("none");
        let sq_implementation_id = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.implementation_id.as_str())
            .unwrap_or("none");
        let sq_candidate_legacy = self
            .sq_overlay_info
            .as_ref()
            .and_then(|info| info.candidate_legacy.as_deref())
            .unwrap_or("none");
        let sq_artifact = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.artifact.as_str())
            .unwrap_or("none");
        let sq_schema_version = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.schema_version.as_str())
            .unwrap_or("none");
        let sq_fp8_tensor_count = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.fp8_tensor_count)
            .unwrap_or(0);
        let sq_passthrough_tensor_count = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.passthrough_tensor_count)
            .unwrap_or(0);
        let sq_row_chunk = self
            .sq_overlay_info
            .as_ref()
            .map(|info| info.row_chunk)
            .unwrap_or(0);
        let (
            sq_fp8_projection_telemetry,
            sq_projection_boundary,
            sq_fp8_projection_implementation_ids,
        ) = if sq_overlay {
            let sq_fp8_projection_telemetry = snapshot_sq_fp8_projection_telemetry();
            let sq_projection_boundary = sq_fp8_projection_boundary(sq_fp8_projection_telemetry);
            let sq_fp8_projection_dispatches = SqFp8ProjectionDispatches::from_info(info);
            let sq_fp8_projection_implementation_ids =
                sq_fp8_projection_implementation_ids(
                    sq_fp8_projection_telemetry,
                    sq_fp8_projection_dispatches,
                );
            (
                sq_fp8_projection_telemetry,
                sq_projection_boundary,
                sq_fp8_projection_implementation_ids,
            )
        } else {
            (
                SqFp8ProjectionTelemetry::default(),
                "none".to_string(),
                "none".to_string(),
            )
        };
        let sq_execution_mode = if sq_overlay {
            if sq_fp8_projection_telemetry.single_matvec_count == 0
                && sq_fp8_projection_telemetry.batch_matvec_count == 0
                && sq_fp8_projection_telemetry.pair_matvec_count == 0
                && sq_fp8_projection_telemetry.triple_matvec_count == 0
            {
                "materialized_f32_fallback"
            } else {
                "direct_fp8_dequant_matvec"
            }
        } else {
            "none"
        };
        let q_norm_max_abs_diff = prepared_diffs.q_norm_max_abs_diff;
        let k_norm_max_abs_diff = prepared_diffs.k_norm_max_abs_diff;
        let q_rope_max_abs_diff = prepared_diffs.q_rope_max_abs_diff;
        let k_rope_max_abs_diff = prepared_diffs.k_rope_max_abs_diff;
        let causal_attention_max_abs_diff = prepared_diffs.causal_attention_max_abs_diff;
        let attention_max_abs_diff = runtime_diffs.attention_max_abs_diff;
        let projection_input_max_abs_diff = runtime_diffs.projection_input_max_abs_diff;
        let projected_max_abs_diff = runtime_diffs.projected_max_abs_diff;
        let block_max_abs_diff = runtime_diffs.block_max_abs_diff;
        let post_norm_max_abs_diff = runtime_diffs.post_norm_max_abs_diff;
        let mlp_max_abs_diff = runtime_diffs.mlp_max_abs_diff;
        let layer_max_abs_diff = runtime_diffs.layer_max_abs_diff;
        let k_cache_max_abs_diff = runtime_diffs.k_cache_max_abs_diff;
        let v_cache_max_abs_diff = runtime_diffs.v_cache_max_abs_diff;

        Ok(format!(
            "{} package={} layers={:?} layers_csv={} input_source={} prefill_mode={} format_id={} sq_overlay={} sq_candidate={} sq_candidate_legacy={} sq_format_id={} sq_implementation_id={} sq_artifact={} sq_schema_version={} sq_fp8_tensor_count={} sq_passthrough_tensor_count={} sq_row_chunk={} sq_execution_mode={} sq_projection_boundary={} sq_projection_implementation_ids={} sq_fp8_single_matvec_count={} sq_fp8_batch_matvec_count={} sq_fp8_pair_matvec_count={} sq_fp8_triple_matvec_count={} batching_mode={} prefill_executor={} decode_executor=stack_ready_batch prefill_real_batch={} decode_real_batch={} prefill_executor_request_parallelism={} decode_executor_request_parallelism={} prompt_token_ids_by_request={:?} decode_token_ids_by_request={:?} final_lm_head_guard={} lm_head_top_k={} lm_head_chunk_rows={} final_top1_tokens={:?} final_top1_tokens_csv={} final_top1_logits_csv={} final_topk_tokens_csv={} final_topk_logits_csv={} final_top_logits_source=verified_expected_layer_output input_norm_tensors={:?} q_tensors={:?} k_tensors={:?} v_tensors={:?} o_tensors={:?} q_norm_tensors={:?} k_norm_tensors={:?} post_norm_tensors={:?} gate_tensors={:?} up_tensors={:?} down_tensors={:?} sequence_len={} request_count={} concurrent_requests={} request_ids={:?} prompt_tokens={:?} prompt_tokens_csv={} max_new_tokens={:?} max_new_tokens_csv={} total_tokens={:?} total_tokens_csv={} prefill_total_input_tokens={} decode_total_generated_tokens={} end_to_end_total_tokens={} prefill_wall_ms={:.6} decode_wall_ms={:.6} total_wall_ms={:.6} prefill_total_input_tps={} decode_total_generated_tps={} end_to_end_total_tps={} paged_block_size={} paged_cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} prefill_batch_request_counts={:?} prefill_batch_request_counts_csv={} decode_batch_ready_counts={:?} decode_batch_ready_counts_csv={} final_ready={} decode_steps_by_layer={:?} cached_tokens={:?} generated_tokens={:?} generated_tokens_csv={} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} hidden={} intermediate_by_layer={:?} q_projection_layouts={:?} q_gate_elements_by_layer={:?} output_gate_layouts={:?} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} mlp_epsilon={:.9} input_norm_dtypes={:?} q_norm_dtypes={:?} k_norm_dtypes={:?} post_norm_dtypes={:?} backend={} device_index={} name=\"{}\" q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} causal_attention_max_abs_diff={causal_attention_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
            self.command_name,
            path,
            layer_indices,
            layer_indices_csv,
            self.request_plan.input_source,
            prefill_mode,
            format_id,
            sq_overlay,
            sq_candidate,
            sq_candidate_legacy,
            sq_format_id,
            sq_implementation_id,
            sq_artifact,
            sq_schema_version,
            sq_fp8_tensor_count,
            sq_passthrough_tensor_count,
            sq_row_chunk,
            sq_execution_mode,
            sq_projection_boundary,
            sq_fp8_projection_implementation_ids,
            sq_fp8_projection_telemetry.single_matvec_count,
            sq_fp8_projection_telemetry.batch_matvec_count,
            sq_fp8_projection_telemetry.pair_matvec_count,
            sq_fp8_projection_telemetry.triple_matvec_count,
            batching_mode,
            prefill_executor,
            prefill_real_batch,
            decode_real_batch,
            prefill_executor_request_parallelism,
            decode_executor_request_parallelism,
            self.request_plan.prompt_token_ids_by_request,
            self.request_plan.decode_token_ids_by_request,
            final_lm_head_guard,
            self.lm_head_top_k.unwrap_or(0),
            self.lm_head_chunk_rows.unwrap_or(0),
            final_top1_tokens,
            final_top1_tokens_csv,
            final_top1_logits_csv,
            final_topk_tokens_csv,
            final_topk_logits_csv,
            input_norm_tensors,
            q_tensors,
            k_tensors,
            v_tensors,
            o_tensors,
            q_norm_tensors,
            k_norm_tensors,
            post_norm_tensors,
            gate_tensors,
            up_tensors,
            down_tensors,
            self.sequence_len,
            self.request_plan.request_count(),
            self.request_plan.request_count(),
            self.request_plan.request_ids,
            self.request_plan.prompt_tokens,
            prompt_tokens_csv,
            self.request_plan.max_new_tokens,
            max_new_tokens_csv,
            self.request_plan.total_tokens,
            total_tokens_csv,
            prefill_total_input_tokens,
            decode_total_generated_tokens,
            end_to_end_total_tokens,
            execution_summary.prefill_wall_ms,
            execution_summary.decode_wall_ms,
            execution_summary.total_wall_ms,
            tps(
                prefill_total_input_tokens,
                execution_summary.prefill_wall_ms
            )
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
            tps(
                decode_total_generated_tokens,
                execution_summary.decode_wall_ms
            )
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
            tps(end_to_end_total_tokens, execution_summary.total_wall_ms)
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            self.request_plan.block_size,
            self.request_plan.cache_blocks,
            self.request_plan.block_tables,
            execution_summary.first_batch_ready,
            execution_summary.second_batch_ready,
            execution_summary.prefill_batch_request_counts,
            prefill_batch_request_counts_csv,
            execution_summary.decode_batch_ready_counts,
            decode_batch_ready_counts_csv,
            execution_summary.final_ready,
            decode_steps_by_layer,
            cached_tokens,
            generated_tokens,
            generated_tokens_csv,
            self.request_plan.scheduler.active_len(),
            stats.free_blocks,
            stats.allocated_blocks,
            stats.free_runs,
            stats.largest_free_run,
            self.model.hidden,
            self.model.intermediates(),
            self.layer_run_plan.q_projection_layouts,
            self.layer_run_plan.q_gate_elements_by_layer,
            self.layer_run_plan.output_gate_layouts,
            self.model.q_heads,
            self.model.kv_heads,
            self.model.head_dim,
            self.model.value_dim,
            self.rotary_dim,
            self.position_offset,
            self.rope_base,
            self.model.softmax_scale,
            self.model.mlp_epsilon,
            input_norm_dtypes,
            q_norm_dtypes,
            k_norm_dtypes,
            post_norm_dtypes,
            info.backend,
            device_index,
            info.name,
        ))
    }
}

fn package_layer_golden_smoke(
    path: Option<String>,
    fixture_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-layer-golden-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(fixture_path) = fixture_path else {
        eprintln!("package-layer-golden-smoke requires a golden fixture directory");
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
    let fixture = match GoldenTensorFixture::load(&fixture_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let layer_index = match layer_index {
        Some(raw) => match parse_usize_value(&raw, "layer index") {
            Ok(value) => value,
            Err(code) => return code,
        },
        None => {
            if fixture.layers().len() == 1 {
                fixture.layers()[0].layer_index
            } else {
                eprintln!(
                    "package-layer-golden-smoke requires LAYER_INDEX when fixture has {} layers",
                    fixture.layers().len()
                );
                return ExitCode::from(2);
            }
        }
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let default_position_offset = match fixture.metadata().position_ids.first() {
        Some(value) => match usize::try_from(*value) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("golden fixture first position id does not fit usize");
                return ExitCode::from(1);
            }
        },
        None => 0,
    };
    let position_offset =
        match parse_optional_usize(position_offset, default_position_offset, "position offset") {
            Ok(value) => value,
            Err(code) => return code,
        };

    match package_layer_golden_smoke_impl(
        &path,
        &fixture_path,
        fixture,
        device_index,
        chunk_bytes,
        layer_index,
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
fn package_golden_prefix_smoke(
    path: Option<String>,
    fixture_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_start: Option<String>,
    layer_end_exclusive: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
    report_path: Option<String>,
    run_mode: Option<String>,
    row_scale_overrides_path: Option<String>,
    input_dump_dir: Option<String>,
    sampled_token_indices: Option<String>,
    cell_delta_overrides_path: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-golden-prefix-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(fixture_path) = fixture_path else {
        eprintln!("package-golden-prefix-smoke requires a golden prefix fixture directory");
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
    let fixture = match GoldenTensorFixture::load(&fixture_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (default_start, default_end_exclusive) = match golden_fixture_default_layer_range(&fixture)
    {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let layer_start = match parse_optional_usize(layer_start, default_start, "layer start") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let layer_end_exclusive =
        match parse_optional_usize(layer_end_exclusive, default_end_exclusive, "layer end") {
            Ok(value) => value,
            Err(code) => return code,
        };
    if layer_end_exclusive <= layer_start {
        eprintln!(
            "package-golden-prefix-smoke requires layer end greater than layer start: start={layer_start} end={layer_end_exclusive}"
        );
        return ExitCode::from(2);
    }
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let default_position_offset = match fixture.metadata().position_ids.first() {
        Some(value) => match usize::try_from(*value) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("golden fixture first position id does not fit usize");
                return ExitCode::from(1);
            }
        },
        None => 0,
    };
    let position_offset =
        match parse_optional_usize(position_offset, default_position_offset, "position offset") {
            Ok(value) => value,
            Err(code) => return code,
        };
    let run_mode = match parse_package_golden_prefix_run_mode(run_mode.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let row_scale_overrides =
        match load_package_row_scale_overrides(row_scale_overrides_path.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        };
    let cell_delta_overrides =
        match load_package_cell_delta_overrides(cell_delta_overrides_path.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        };
    let sampled_token_indices = match sampled_token_indices.as_deref() {
        Some("none") | None => Vec::new(),
        Some(raw) => match parse_usize_csv(raw, "sampled token indices") {
            Ok(value) => value,
            Err(code) => return code,
        },
    };
    let input_dump_dir = input_dump_dir.as_deref().filter(|raw| *raw != "none");

    match package_golden_prefix_smoke_impl(
        &path,
        &fixture_path,
        fixture,
        device_index,
        chunk_bytes,
        layer_start,
        layer_end_exclusive,
        rotary_dim,
        rope_base,
        position_offset,
        report_path.as_deref(),
        run_mode,
        row_scale_overrides.as_ref(),
        cell_delta_overrides.as_ref(),
        input_dump_dir,
        &sampled_token_indices,
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
fn package_golden_prefix_smoke_impl(
    path: &str,
    fixture_path: &str,
    fixture: GoldenTensorFixture,
    device_index: u32,
    chunk_bytes: usize,
    layer_start: usize,
    layer_end_exclusive: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    report_path: Option<&str>,
    run_mode: PackageGoldenPrefixRunMode,
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    cell_delta_overrides: Option<&PackageCellDeltaOverrides>,
    input_dump_dir: Option<&str>,
    sampled_token_indices: &[usize],
) -> Result<String, String> {
    let manifest_row_scale_override_count = list_tensor_payload_bundles(path)?
        .iter()
        .map(|bundle| bundle.row_scale_overrides.len())
        .sum::<usize>();
    let golden_layers = fixture.select_contiguous_layers(layer_start, layer_end_exclusive)?;
    let sequence_len = fixture.metadata().sequence_len;
    let hidden = fixture.metadata().hidden_size;
    if sequence_len == 0 || hidden == 0 {
        return Err(format!(
            "golden prefix fixture has invalid sequence_len={sequence_len} hidden_size={hidden}"
        ));
    }
    validate_golden_position_ids(
        &fixture.metadata().position_ids,
        sequence_len,
        position_offset,
    )?;
    for golden_layer in &golden_layers {
        validate_golden_hidden_shape(
            &golden_layer.before_shape,
            sequence_len,
            hidden,
            "golden prefix before hidden",
        )?;
        validate_golden_hidden_shape(
            &golden_layer.after_shape,
            sequence_len,
            hidden,
            "golden prefix after hidden",
        )?;
    }

    let expected_hidden_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "golden prefix hidden element count overflows".to_string())?;
    let mut current_hidden = fixture.read_initial_before_f32(layer_start)?;
    if current_hidden.len() != expected_hidden_elements {
        return Err(format!(
            "golden prefix initial payload element mismatch: got {} expected {expected_hidden_elements}",
            current_hidden.len()
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
    let block_size = sequence_len;
    let cache_blocks = 1_usize;
    let block_table = vec![0_u32];

    let mut report_entries = Vec::with_capacity(golden_layers.len());
    let mut max_mse = 0.0_f64;
    let mut max_mean_abs_diff = 0.0_f64;
    let mut max_abs_diff = 0.0_f64;
    let mut min_cosine_similarity = 1.0_f64;
    let mut self_attn_rotary_dim = None::<usize>;

    for (layer_position, golden_layer) in golden_layers.iter().enumerate() {
        let layer_index = golden_layer.layer_index;
        if current_hidden.len() != expected_hidden_elements {
            return Err(format!(
                "package golden prefix input length mismatch before layer {}: got {} expected {expected_hidden_elements}",
                layer_index,
                current_hidden.len()
            ));
        }
        let expected_after = fixture.read_layer_after_f32(layer_index)?;
        if expected_after.len() != expected_hidden_elements {
            return Err(format!(
                "golden prefix layer {} after payload element mismatch: got {} expected {expected_hidden_elements}",
                layer_index,
                expected_after.len()
            ));
        }
        let expected_before = fixture.read_layer_before_f32(layer_index)?;
        if expected_before.len() != expected_hidden_elements {
            return Err(format!(
                "golden prefix layer {} before payload element mismatch: got {} expected {expected_hidden_elements}",
                layer_index,
                expected_before.len()
            ));
        }

        let input_metrics = compare_f32_slices(&current_hidden, &expected_before)?;
        let input_preview_len = 8.min(current_hidden.len()).min(expected_before.len());
        let input_expected_preview = expected_before[..input_preview_len].to_vec();
        let input_actual_preview = current_hidden[..input_preview_len].to_vec();
        let input_diff_preview = current_hidden
            .iter()
            .zip(expected_before.iter())
            .take(input_preview_len)
            .map(|(actual, expected)| actual - expected)
            .collect::<Vec<_>>();
        let input_failure_class = package_golden_prefix_failure_class(&input_metrics);
        let input_distribution =
            package_hidden_distribution(&current_hidden, &expected_before, sequence_len, hidden)?;

        let layer_input = match run_mode {
            PackageGoldenPrefixRunMode::ActualPrefix => current_hidden.clone(),
            PackageGoldenPrefixRunMode::GoldenBeforeEachLayer => expected_before.clone(),
        };
        let layer_input_for_delta = layer_input.clone();
        let input_dump_file = match input_dump_dir {
            Some(dump_dir) => Some(write_package_prefix_input_dump(
                dump_dir,
                layer_index,
                run_mode,
                sequence_len,
                hidden,
                &layer_input_for_delta,
            )?),
            None => None,
        };

        let layer_kind = package_decoder_layer_kind(path, layer_index)?;
        let (actual, details) = match layer_kind {
            PackageDecoderLayerKind::SelfAttention => {
                let mut layer = qwen3_package_decoder_layer_runtime_from_package(
                    &mut context,
                    &mut stream,
                    path,
                    chunk_bytes,
                    layer_index,
                )?;
                if layer.runtime_shape.hidden != hidden {
                    return Err(format!(
                        "golden hidden_size {hidden} does not match package self-attn layer {} hidden {}",
                        layer_index, layer.runtime_shape.hidden
                    ));
                }
                let mut applied_row_scale_overrides = Vec::new();
                let mut applied_cell_delta_overrides = Vec::new();
                let self_attn_o_row_scale_overrides = matching_package_row_scale_overrides(
                    row_scale_overrides,
                    layer_index,
                    "self_attn.o_proj.weight",
                );
                applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.self_attn.o_matrix,
                    layer.weights.self_attn.o_rows,
                    layer.weights.self_attn.o_cols,
                    &layer.o_tensor,
                    &self_attn_o_row_scale_overrides,
                )?);
                let self_attn_o_cell_delta_overrides = matching_package_cell_delta_overrides(
                    cell_delta_overrides,
                    layer_index,
                    "self_attn.o_proj.weight",
                );
                applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.self_attn.o_matrix,
                    layer.weights.self_attn.o_rows,
                    layer.weights.self_attn.o_cols,
                    &layer.o_tensor,
                    &self_attn_o_cell_delta_overrides,
                )?);
                let gate_cell_delta_overrides = matching_package_cell_delta_overrides(
                    cell_delta_overrides,
                    layer_index,
                    "mlp.gate_proj.weight",
                );
                applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.post_attention.mlp.gate_matrix,
                    layer.weights.post_attention.mlp.gate_rows,
                    layer.weights.post_attention.mlp.gate_cols,
                    &layer.gate_tensor,
                    &gate_cell_delta_overrides,
                )?);
                let up_cell_delta_overrides = matching_package_cell_delta_overrides(
                    cell_delta_overrides,
                    layer_index,
                    "mlp.up_proj.weight",
                );
                applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.post_attention.mlp.up_matrix,
                    layer.weights.post_attention.mlp.gate_rows,
                    layer.weights.post_attention.mlp.gate_cols,
                    &layer.up_tensor,
                    &up_cell_delta_overrides,
                )?);
                let down_row_scale_overrides = matching_package_row_scale_overrides(
                    row_scale_overrides,
                    layer_index,
                    "mlp.down_proj.weight",
                );
                applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.post_attention.mlp.down_matrix,
                    layer.weights.post_attention.hidden,
                    layer.weights.post_attention.intermediate,
                    &layer.down_tensor,
                    &down_row_scale_overrides,
                )?);
                let down_cell_delta_overrides = matching_package_cell_delta_overrides(
                    cell_delta_overrides,
                    layer_index,
                    "mlp.down_proj.weight",
                );
                applied_cell_delta_overrides.extend(apply_package_cell_delta_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.post_attention.mlp.down_matrix,
                    layer.weights.post_attention.hidden,
                    layer.weights.post_attention.intermediate,
                    &layer.down_tensor,
                    &down_cell_delta_overrides,
                )?);
                let input_norm_tensor =
                    format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
                let mut input_norm =
                    read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
                input_norm.values =
                    effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
                if input_norm.values.len() != hidden {
                    return Err(format!(
                        "self-attn input RMSNorm length {} does not match hidden={hidden}",
                        input_norm.values.len()
                    ));
                }
                let mut attention_input_normed = Vec::with_capacity(layer_input_for_delta.len());
                for residual in layer_input_for_delta.chunks_exact(hidden) {
                    attention_input_normed.extend(runtime_host_rmsnorm_f32(
                        residual,
                        &input_norm.values,
                        1e-6_f32,
                    ));
                }
                let rotary_dim = match rotary_dim.as_ref() {
                    Some(raw) => parse_package_layer_golden_rotary_dim(
                        layer.runtime_shape.head_dim,
                        Some(raw.clone()),
                    )?,
                    None => {
                        parse_package_layer_golden_rotary_dim(layer.runtime_shape.head_dim, None)?
                    }
                };
                self_attn_rotary_dim = Some(rotary_dim);
                let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
                    &mut context,
                    &mut stream,
                    &layer.weights.self_attn,
                    attention_input_normed.clone(),
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
                    residual_sequence: _,
                    prepared:
                        Qwen3SelfAttnRuntimePreparedSequence {
                            q_query,
                            k_projected,
                            q_normed,
                            k_normed,
                            q_rope,
                            k_rope,
                            v_projected,
                            q_gate,
                            attention_output,
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
                let layer_output = qwen3_decoder_layer_sequence_to_host_f32(
                    &layer.weights,
                    &mut context,
                    &mut stream,
                    decode_shape,
                    &paged_block_table,
                    softmax_scale,
                    1e-5_f32,
                    &q_rope,
                    &k_rope,
                    &v_projected,
                    q_gate.as_deref(),
                    &layer_input_for_delta,
                    sequence_len,
                )?;
                let causal_attention_runtime_diagnostic =
                    package_self_attn_causal_attention_runtime_diagnostic(
                        &attention_output,
                        &layer_output.attention_output,
                        &layer_output.attention_projection_input,
                        &q_rope,
                        &k_rope,
                        &v_projected,
                        q_gate.as_deref(),
                        sequence_len,
                        shape.q_heads,
                        shape.kv_heads,
                        shape.head_dim,
                        shape.value_dim,
                        softmax_scale,
                    )?;
                let candidate_ids = package_layer_candidate_ids(path, &layer);
                let mut details = serde_json::Map::new();
                insert_json_detail(&mut details, "candidate_ids", candidate_ids);
                insert_json_detail(&mut details, "input_norm_tensor", &input_norm_tensor);
                insert_json_detail(&mut details, "q_tensor", &layer.q_tensor);
                insert_json_detail(&mut details, "k_tensor", &layer.k_tensor);
                insert_json_detail(&mut details, "v_tensor", &layer.v_tensor);
                insert_json_detail(&mut details, "o_tensor", &layer.o_tensor);
                insert_json_detail(&mut details, "gate_tensor", &layer.gate_tensor);
                insert_json_detail(&mut details, "up_tensor", &layer.up_tensor);
                insert_json_detail(&mut details, "down_tensor", &layer.down_tensor);
                insert_json_detail(&mut details, "q_heads", shape.q_heads);
                insert_json_detail(&mut details, "kv_heads", shape.kv_heads);
                insert_json_detail(&mut details, "head_dim", shape.head_dim);
                insert_json_detail(&mut details, "value_dim", shape.value_dim);
                insert_json_detail(&mut details, "rotary_dim", rotary_dim);
                insert_json_detail(&mut details, "position_offset", position_offset);
                insert_json_detail(&mut details, "rope_base", rope_base);
                insert_json_detail(&mut details, "block_size", paged_block_size);
                insert_json_detail(&mut details, "cache_blocks", paged_cache_blocks);
                insert_json_detail(&mut details, "block_table", paged_block_table);
                insert_json_detail(&mut details, "softmax_scale", softmax_scale);
                insert_json_detail(&mut details, "mlp_epsilon", 1e-5_f32);
                insert_json_detail(
                    &mut details,
                    "q_projection_layout",
                    q_projection_layout.to_string(),
                );
                insert_json_detail(&mut details, "q_gate_elements", q_gate_elements);
                insert_json_detail(
                    &mut details,
                    "output_gate_layout",
                    output_gate_layout.to_string(),
                );
                insert_json_detail(&mut details, "input_norm_dtype", &input_norm.dtype);
                insert_json_detail(&mut details, "q_norm_dtype", &layer.q_norm.dtype);
                insert_json_detail(&mut details, "k_norm_dtype", &layer.k_norm.dtype);
                insert_json_detail(&mut details, "post_norm_dtype", &layer.post_norm.dtype);
                if let Some(overrides) = row_scale_overrides {
                    insert_json_detail(
                        &mut details,
                        "row_scale_override_source",
                        &overrides.source_path,
                    );
                }
                if let Some(overrides) = cell_delta_overrides {
                    insert_json_detail(
                        &mut details,
                        "cell_delta_override_source",
                        &overrides.source_path,
                    );
                }
                if !applied_row_scale_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_row_scale_overrides",
                        &applied_row_scale_overrides,
                    );
                }
                if !applied_cell_delta_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_cell_delta_overrides",
                        &applied_cell_delta_overrides,
                    );
                }
                insert_json_detail(
                    &mut details,
                    "causal_attention_runtime_diagnostic",
                    causal_attention_runtime_diagnostic,
                );
                let extra_hot_input_vectors = [
                    (
                        "attention_input_normed",
                        attention_input_normed.as_slice(),
                        hidden,
                    ),
                    ("attention_q_query", q_query.as_slice(), hidden),
                    (
                        "attention_k_projected",
                        k_projected.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    (
                        "attention_v_projected",
                        v_projected.as_slice(),
                        shape.kv_heads * shape.value_dim,
                    ),
                    ("attention_q_normed", q_normed.as_slice(), hidden),
                    (
                        "attention_k_normed",
                        k_normed.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    ("attention_q_rope", q_rope.as_slice(), hidden),
                    (
                        "attention_k_rope",
                        k_rope.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    ("attention_output", attention_output.as_slice(), hidden),
                ];
                let mut extra_hot_input_vectors = extra_hot_input_vectors.to_vec();
                if let Some(q_gate) = q_gate.as_ref() {
                    extra_hot_input_vectors.push(("attention_q_gate", q_gate.as_slice(), hidden));
                }
                insert_json_detail(
                    &mut details,
                    "module_contribution",
                    package_module_contribution_summary(
                        &layer_input_for_delta,
                        &expected_before,
                        &expected_after,
                        Some(&layer_output.attention_projection_input),
                        &layer_output.projected_output,
                        &layer_output.block_output,
                        &layer_output.post_normed,
                        None,
                        &extra_hot_input_vectors,
                        &layer_output.mlp_output,
                        &layer_output.layer_output,
                        sequence_len,
                        hidden,
                        sampled_token_indices,
                    )?,
                );
                (layer_output.layer_output, details)
            }
            PackageDecoderLayerKind::LinearAttention => {
                let run = package_linear_attn_mlp_block_sequence_run(
                    path,
                    device_index,
                    chunk_bytes,
                    layer_index,
                    sequence_len,
                    layer_input,
                    row_scale_overrides,
                    cell_delta_overrides,
                )?;
                let mut details = serde_json::Map::new();
                insert_json_detail(&mut details, "runtime_line", &run.line);
                let runtime_metrics = package_runtime_line_metrics(&run.line);
                if !runtime_metrics.is_empty() {
                    insert_json_detail(&mut details, "runtime_metrics", runtime_metrics);
                }
                if let Some(overrides) = row_scale_overrides {
                    insert_json_detail(
                        &mut details,
                        "row_scale_override_source",
                        &overrides.source_path,
                    );
                }
                if let Some(overrides) = cell_delta_overrides {
                    insert_json_detail(
                        &mut details,
                        "cell_delta_override_source",
                        &overrides.source_path,
                    );
                }
                if !run.applied_row_scale_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_row_scale_overrides",
                        &run.applied_row_scale_overrides,
                    );
                }
                if !run.applied_cell_delta_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_cell_delta_overrides",
                        &run.applied_cell_delta_overrides,
                    );
                }
                let extra_hot_input_vectors = [
                    (
                        "attention_input_normed",
                        run.attention_input_normed.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_qkv_projection",
                        run.attention_qkv_projection.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_z_projection",
                        run.attention_z_projection.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_gate_silu",
                        run.attention_gate_silu.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_a_projection",
                        run.attention_a_projection.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_b_projection",
                        run.attention_b_projection.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_conv_pre_silu",
                        run.attention_conv_pre_silu.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_conv",
                        run.attention_conv.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_recurrent_q",
                        run.attention_recurrent_q.as_slice(),
                        run.attention_recurrent_qk_dim,
                    ),
                    (
                        "attention_recurrent_k",
                        run.attention_recurrent_k.as_slice(),
                        run.attention_recurrent_qk_dim,
                    ),
                    (
                        "attention_recurrent_v",
                        run.attention_recurrent_v.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_gate",
                        run.attention_gate.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_beta",
                        run.attention_beta.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_recurrent",
                        run.attention_recurrent.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_pre_gate_normed",
                        run.attention_normed.as_slice(),
                        hidden,
                    ),
                    ("attention_normed", run.attention_normed.as_slice(), hidden),
                    (
                        "mlp_gate_projection",
                        run.mlp_gate_projection.as_slice(),
                        run.mlp_intermediate,
                    ),
                    (
                        "mlp_gate_silu",
                        run.mlp_gate_silu.as_slice(),
                        run.mlp_intermediate,
                    ),
                    (
                        "mlp_up_projection",
                        run.mlp_up_projection.as_slice(),
                        run.mlp_intermediate,
                    ),
                ];
                insert_json_detail(
                    &mut details,
                    "candidate_ids",
                    package_linear_attn_candidate_ids(path, layer_index),
                );
                insert_json_detail(
                    &mut details,
                    "qkv_tensor",
                    format!(
                        "model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"
                    ),
                );
                insert_json_detail(
                    &mut details,
                    "out_tensor",
                    format!(
                        "model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"
                    ),
                );
                insert_json_detail(
                    &mut details,
                    "gate_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "up_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "down_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "module_contribution",
                    package_module_contribution_summary(
                        &layer_input_for_delta,
                        &expected_before,
                        &expected_after,
                        Some(&run.attention_projection_input),
                        &run.attention_output,
                        &run.attention_block_output,
                        &run.post_normed,
                        Some((&run.mlp_activation, run.mlp_intermediate)),
                        &extra_hot_input_vectors,
                        &run.mlp_output,
                        &run.layer_output,
                        sequence_len,
                        hidden,
                        sampled_token_indices,
                    )?,
                );
                (run.layer_output, details)
            }
        };
        let metrics = compare_f32_slices(&actual, &expected_after)?;
        max_mse = max_mse.max(metrics.mse);
        max_mean_abs_diff = max_mean_abs_diff.max(metrics.mean_abs_diff);
        max_abs_diff = max_abs_diff.max(metrics.max_abs_diff);
        min_cosine_similarity = min_cosine_similarity.min(metrics.cosine_similarity);

        let preview_len = 8.min(expected_after.len()).min(actual.len());
        let expected_preview = expected_after[..preview_len].to_vec();
        let actual_preview = actual[..preview_len].to_vec();
        let diff_preview = actual
            .iter()
            .zip(expected_after.iter())
            .take(preview_len)
            .map(|(actual, expected)| actual - expected)
            .collect::<Vec<_>>();
        let failure_class = package_golden_prefix_failure_class(&metrics);
        let mut details = details;
        if let Some(input_dump_file) = input_dump_file.as_ref() {
            insert_json_detail(&mut details, "input_dump_file", input_dump_file);
        }
        insert_json_detail(
            &mut details,
            "manifest_row_scale_override_count",
            manifest_row_scale_override_count,
        );
        let output_distribution =
            package_hidden_distribution(&actual, &expected_after, sequence_len, hidden)?;
        insert_json_detail(&mut details, "input_distribution", input_distribution);
        insert_json_detail(&mut details, "output_distribution", output_distribution);
        append_package_golden_prefix_report_entry(
            &mut report_entries,
            path,
            fixture_path,
            fixture.metadata().fixture_kind.as_deref(),
            device_index,
            &info.backend.to_string(),
            &info.name,
            layer_position,
            layer_index,
            layer_kind.as_str(),
            layer_start,
            layer_end_exclusive,
            sequence_len,
            hidden,
            run_mode,
            &input_metrics,
            input_failure_class,
            input_expected_preview,
            input_actual_preview,
            input_diff_preview,
            &metrics,
            failure_class,
            expected_preview,
            actual_preview,
            diff_preview,
            details,
        );

        current_hidden = actual;
    }

    if let Some(report_path) = report_path {
        write_jsonl_report(report_path, &report_entries)?;
    }

    Ok(format!(
        "package-golden-prefix-smoke package={} fixture={} layers={}..{} layer_count={} sequence_len={} hidden={} run_mode={} block_size={} cache_blocks={} block_table={:?} rotary_dim={} position_offset={} rope_base={} row_scale_overrides={} cell_delta_overrides={} manifest_row_scale_overrides={} input_dump_dir={} sampled_tokens={} backend={} device_index={} name=\"{}\" max_mse={:.12} max_mean_abs_diff={:.9} max_abs_diff={:.9} min_cosine_similarity={:.9} report={} verified=true",
        path,
        fixture_path,
        layer_start,
        layer_end_exclusive,
        golden_layers.len(),
        sequence_len,
        hidden,
        run_mode.as_str(),
        block_size,
        cache_blocks,
        block_table,
        self_attn_rotary_dim
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        position_offset,
        rope_base,
        row_scale_overrides
            .map(|overrides| overrides.source_path.as_str())
            .unwrap_or("none"),
        cell_delta_overrides
            .map(|overrides| overrides.source_path.as_str())
            .unwrap_or("none"),
        manifest_row_scale_override_count,
        input_dump_dir.unwrap_or("none"),
        if sampled_token_indices.is_empty() {
            "none".to_string()
        } else {
            sampled_token_indices
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        },
        info.backend,
        device_index,
        info.name,
        max_mse,
        max_mean_abs_diff,
        max_abs_diff,
        min_cosine_similarity,
        report_path.unwrap_or("none"),
    ))
}
