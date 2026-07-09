fn package_linear_attn_qkv_prepare_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-qkv-prepare-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_qkv_prepare_batch_smoke_impl(
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

fn package_linear_attn_qkv_prepare_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn qkv prepare batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn qkv prepare batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }

    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let q_elements = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear-attn qkv prepare batch q element count overflows".to_string())?;
    let v_elements = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear-attn qkv prepare batch v element count overflows".to_string())?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| "linear-attn qkv prepare batch channel count overflows".to_string())?;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;

    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn qkv prepare batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn qkv prepare batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }

    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read conv1d tensor: {err}"))?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "linear-attn qkv prepare batch conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "linear-attn qkv prepare batch conv channel count is too large".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "linear-attn qkv prepare batch kernel size is too large".to_string())?;
    if conv_channels != channels {
        return Err(format!(
            "linear-attn qkv prepare batch conv channels mismatch: got {conv_channels} expected {channels}"
        ));
    }
    if kernel_size == 0 {
        return Err(
            "linear-attn qkv prepare batch kernel size must be greater than zero".to_string(),
        );
    }
    if conv.values.len() != channels * kernel_size {
        return Err(format!(
            "linear-attn qkv prepare batch conv weight element count mismatch: got {} expected {}",
            conv.values.len(),
            channels * kernel_size
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden
        || embedding_rows.values.len() != prompt_token_ids.len() * hidden
    {
        return Err(format!(
            "linear-attn qkv prepare batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            prompt_token_ids.len()
        ));
    }
    let mut normed_expected = Vec::with_capacity(embedding_rows.values.len());
    for token_index in 0..prompt_token_ids.len() {
        let start = token_index.checked_mul(hidden).ok_or_else(|| {
            "linear-attn qkv prepare batch norm input start overflows".to_string()
        })?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "linear-attn qkv prepare batch norm input end overflows".to_string())?;
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
    if qkv_matrix.rows != channels || qkv_matrix.cols != hidden {
        return Err(format!(
            "linear-attn qkv prepare batch qkv matrix shape mismatch: rows={} cols={} expected rows={channels} cols={hidden}",
            qkv_matrix.rows, qkv_matrix.cols
        ));
    }

    let history_elements = channels.checked_mul(kernel_size).ok_or_else(|| {
        "linear-attn qkv prepare batch history element count overflows".to_string()
    })?;

    let input_elements = prompt_token_ids
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn qkv prepare batch input element count overflows".to_string())?;
    let qkv_output_elements = prompt_token_ids
        .len()
        .checked_mul(channels)
        .ok_or_else(|| {
            "linear-attn qkv prepare batch qkv output element count overflows".to_string()
        })?;
    let q_output_elements = prompt_token_ids
        .len()
        .checked_mul(q_elements)
        .ok_or_else(|| {
            "linear-attn qkv prepare batch q output element count overflows".to_string()
        })?;
    let v_output_elements = prompt_token_ids
        .len()
        .checked_mul(v_elements)
        .ok_or_else(|| {
            "linear-attn qkv prepare batch v output element count overflows".to_string()
        })?;
    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn qkv prepare batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn qkv prepare batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn qkv prepare batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn qkv prepare batch input normed",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch normed input: {err}")
        })?;
    let mut qkv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn qkv prepare batch qkv output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch qkv output: {err}")
        })?;
    let mut conv_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            conv.values.len(),
            "linear-attn qkv prepare batch conv weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch conv weight: {err}")
        })?;
    let mut conv_history_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            history_elements,
            "linear-attn qkv prepare batch conv history",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch conv history: {err}")
        })?;
    let mut conv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn qkv prepare batch conv output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch conv output: {err}")
        })?;
    let mut q_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn qkv prepare batch q output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch q output: {err}")
        })?;
    let mut k_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn qkv prepare batch k output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch k output: {err}")
        })?;
    let mut v_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            v_output_elements,
            "linear-attn qkv prepare batch v output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn qkv prepare batch v output: {err}")
        })?;
    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn qkv prepare batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to copy linear-attn qkv prepare batch norm weight: {err}")
        })?;
    conv_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&conv.values), Some(&mut stream))
        .map_err(|err| {
            format!("failed to copy linear-attn qkv prepare batch conv weight: {err}")
        })?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn qkv prepare batch setup: {err}")
    })?;

    let zero_history_bytes = vec![0_u8; checked_f32_byte_len(history_elements, "zero history")?];
    macro_rules! run_qkv_prepare_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                prompt_token_ids.len(),
                hidden,
                1e-6_f32,
                &mut input_normed_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn qkv prepare batch input RMSNorm: {err}")
            })?;
            qkv_matrix.matvec_batch(
                &input_normed_buffer,
                prompt_token_ids.len(),
                &mut qkv_output_buffer,
                $stream,
                "linear-attn qkv prepare batch projection",
            )?;
            ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
                &qkv_output_buffer,
                &conv_weight_buffer,
                &mut conv_history_buffer,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                prompt_token_ids.len(),
                q_scale,
                qk_l2_norm,
                &mut conv_output_buffer,
                &mut q_output_buffer,
                &mut k_output_buffer,
                &mut v_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn qkv prepare batch: {err}"))
        }};
    }

    conv_history_buffer
        .copy_from_host(0, &zero_history_bytes, Some(&mut stream))
        .map_err(|err| {
            format!("failed to reset linear-attn qkv prepare batch warmup history: {err}")
        })?;
    run_qkv_prepare_batch!(&mut stream)?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn qkv prepare batch warmup: {err}")
    })?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        conv_history_buffer
            .copy_from_host(0, &zero_history_bytes, Some(&mut stream))
            .map_err(|err| {
                format!("failed to reset linear-attn qkv prepare batch history: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn qkv prepare batch history reset: {err}")
        })?;
        let started = Instant::now();
        run_qkv_prepare_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn qkv prepare batch measured run: {err}")
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
        qkv_output_elements,
        "linear-attn qkv prepare batch qkv output",
    )?;
    let conv_output = read_runtime_buffer_f32(
        &conv_output_buffer,
        &mut stream,
        qkv_output_elements,
        "linear-attn qkv prepare batch conv output",
    )?;
    let q_output = read_runtime_buffer_f32(
        &q_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn qkv prepare batch q output",
    )?;
    let k_output = read_runtime_buffer_f32(
        &k_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn qkv prepare batch k output",
    )?;
    let v_output = read_runtime_buffer_f32(
        &v_output_buffer,
        &mut stream,
        v_output_elements,
        "linear-attn qkv prepare batch v output",
    )?;
    let history_output = read_runtime_buffer_f32(
        &conv_history_buffer,
        &mut stream,
        history_elements,
        "linear-attn qkv prepare batch conv history",
    )?;
    let conv_sum_expected = runtime_host_depthwise_conv1d_f32(
        &qkv_output,
        &conv.values,
        channels,
        prompt_token_ids.len(),
        kernel_size,
    );
    if conv_sum_expected.len() != qkv_output.len() {
        return Err("failed to build linear-attn qkv prepare batch conv reference".to_string());
    }
    let conv_expected = runtime_host_silu_f32(&conv_sum_expected);
    let qkv_split = split_linear_attn_qkv_for_recurrent(
        &conv_expected,
        prompt_token_ids.len(),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    )?;
    let mut expected_history = vec![0.0_f32; history_elements];
    for token_qkv in qkv_output.chunks_exact(channels) {
        for channel in 0..channels {
            for kernel in 0..kernel_size - 1 {
                expected_history[kernel * channels + channel] =
                    expected_history[(kernel + 1) * channels + channel];
            }
            expected_history[(kernel_size - 1) * channels + channel] = token_qkv[channel];
        }
    }
    let conv_max_abs_diff = verify_f32_close(
        "linear-attn qkv prepare batch conv output",
        &conv_output,
        &conv_expected,
        1e-3_f32,
        1e-5_f32,
    )?;
    let q_max_abs_diff = verify_f32_close(
        "linear-attn qkv prepare batch q output",
        &q_output,
        &qkv_split.q,
        1e-3_f32,
        1e-5_f32,
    )?;
    let k_max_abs_diff = verify_f32_close(
        "linear-attn qkv prepare batch k output",
        &k_output,
        &qkv_split.k,
        1e-3_f32,
        1e-5_f32,
    )?;
    let v_max_abs_diff = verify_f32_close(
        "linear-attn qkv prepare batch v output",
        &v_output,
        &qkv_split.v,
        1e-3_f32,
        1e-5_f32,
    )?;
    let history_max_abs_diff = verify_f32_close(
        "linear-attn qkv prepare batch history output",
        &history_output,
        &expected_history,
        1e-3_f32,
        1e-5_f32,
    )?;
    let output_elements = conv_output
        .len()
        .checked_add(q_output.len())
        .and_then(|value| value.checked_add(k_output.len()))
        .and_then(|value| value.checked_add(v_output.len()))
        .ok_or_else(|| {
            "linear-attn qkv prepare batch output element count overflows".to_string()
        })?;
    let preview_len = q_output.len().min(8);
    Ok(format!(
        "package-linear-attn-qkv-prepare-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} key_heads={} value_heads={} key_dim={} value_dim={} channels={} kernel_size={} q_scale={q_scale:.9} qk_l2_norm={} input_elements={} output_elements={} qkv_source=runtime_aq4_batch_projection executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} conv_max_abs_diff={conv_max_abs_diff:.9} q_max_abs_diff={q_max_abs_diff:.9} k_max_abs_diff={k_max_abs_diff:.9} v_max_abs_diff={v_max_abs_diff:.9} history_max_abs_diff={history_max_abs_diff:.9} q_preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        conv_tensor,
        prompt_token_ids.len(),
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        channels,
        kernel_size,
        qk_l2_norm,
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
        format_f32_preview(&q_output[..preview_len]),
    ))
}

fn package_linear_attn_recurrent_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-recurrent-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_recurrent_batch_smoke_impl(
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

fn package_linear_attn_recurrent_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn recurrent batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn recurrent batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }

    let sequence_len = prompt_token_ids.len();
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let q_elements = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear-attn recurrent batch q element count overflows".to_string())?;
    let v_elements = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear-attn recurrent batch v element count overflows".to_string())?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| "linear-attn recurrent batch channel count overflows".to_string())?;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;

    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn recurrent batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn recurrent batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read conv1d tensor: {err}"))?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "linear-attn recurrent batch conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "linear-attn recurrent batch conv channel count is too large".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "linear-attn recurrent batch kernel size is too large".to_string())?;
    if conv_channels != channels {
        return Err(format!(
            "linear-attn recurrent batch conv channels mismatch: got {conv_channels} expected {channels}"
        ));
    }
    if kernel_size == 0 {
        return Err(
            "linear-attn recurrent batch kernel size must be greater than zero".to_string(),
        );
    }
    if conv.values.len() != channels * kernel_size {
        return Err(format!(
            "linear-attn recurrent batch conv weight element count mismatch: got {} expected {}",
            conv.values.len(),
            channels * kernel_size
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read A_log tensor: {err}"))?;
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read dt_bias tensor: {err}"))?;
    if a_log.values.len() != value_heads || dt_bias.values.len() != value_heads {
        return Err(format!(
            "linear-attn recurrent batch a_log/dt_bias length mismatch: a_log={} dt_bias={} expected {value_heads}",
            a_log.values.len(),
            dt_bias.values.len()
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "linear-attn recurrent batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
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
    if qkv_matrix.rows != channels || qkv_matrix.cols != hidden {
        return Err(format!(
            "linear-attn recurrent batch qkv matrix shape mismatch: rows={} cols={} expected rows={channels} cols={hidden}",
            qkv_matrix.rows, qkv_matrix.cols
        ));
    }
    if a_matrix.rows != value_heads
        || b_matrix.rows != value_heads
        || a_matrix.cols != hidden
        || b_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn recurrent batch a/b matrix shape mismatch: a=[{},{}] b=[{},{}] expected [{value_heads},{hidden}]",
            a_matrix.rows, a_matrix.cols, b_matrix.rows, b_matrix.cols
        ));
    }

    let input_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn recurrent batch input element count overflows".to_string())?;
    let qkv_output_elements = sequence_len.checked_mul(channels).ok_or_else(|| {
        "linear-attn recurrent batch qkv output element count overflows".to_string()
    })?;
    let q_output_elements = sequence_len.checked_mul(q_elements).ok_or_else(|| {
        "linear-attn recurrent batch q output element count overflows".to_string()
    })?;
    let v_output_elements = sequence_len.checked_mul(v_elements).ok_or_else(|| {
        "linear-attn recurrent batch v output element count overflows".to_string()
    })?;
    let gate_beta_elements = sequence_len.checked_mul(value_heads).ok_or_else(|| {
        "linear-attn recurrent batch gate/beta element count overflows".to_string()
    })?;
    let history_elements = channels
        .checked_mul(kernel_size)
        .ok_or_else(|| "linear-attn recurrent batch history element count overflows".to_string())?;
    let state_elements = value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "linear-attn recurrent batch state element count overflows".to_string())?;

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn recurrent batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn recurrent batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn recurrent batch norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_elements,
            "linear-attn recurrent batch input normed",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn recurrent batch normed input: {err}")
        })?;
    let mut qkv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn recurrent batch qkv output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch qkv: {err}"))?;
    let mut a_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn recurrent batch a output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch a: {err}"))?;
    let mut b_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn recurrent batch b output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch b: {err}"))?;
    let mut conv_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            conv.values.len(),
            "linear-attn recurrent batch conv weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn recurrent batch conv weight: {err}")
        })?;
    let mut conv_history_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            history_elements,
            "linear-attn recurrent batch conv history",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn recurrent batch conv history: {err}")
        })?;
    let mut conv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn recurrent batch conv output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn recurrent batch conv output: {err}")
        })?;
    let mut q_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn recurrent batch q output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch q: {err}"))?;
    let mut k_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn recurrent batch k output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch k: {err}"))?;
    let mut v_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            v_output_elements,
            "linear-attn recurrent batch v output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch v: {err}"))?;
    let mut a_log_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            a_log.values.len(),
            "linear-attn recurrent batch A_log",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch A_log: {err}"))?;
    let mut dt_bias_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            dt_bias.values.len(),
            "linear-attn recurrent batch dt_bias",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch dt_bias: {err}"))?;
    let mut gate_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn recurrent batch gate output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch gate: {err}"))?;
    let mut beta_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn recurrent batch beta output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch beta: {err}"))?;
    let mut recurrent_state_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            state_elements,
            "linear-attn recurrent batch state",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch state: {err}"))?;
    let mut recurrent_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            v_output_elements,
            "linear-attn recurrent batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn recurrent batch output: {err}"))?;

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn recurrent batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn recurrent batch norm weight: {err}"))?;
    conv_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&conv.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn recurrent batch conv weight: {err}"))?;
    a_log_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&a_log.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn recurrent batch A_log: {err}"))?;
    dt_bias_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&dt_bias.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn recurrent batch dt_bias: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn recurrent batch setup: {err}"))?;

    let zero_history_bytes = vec![0_u8; checked_f32_byte_len(history_elements, "zero history")?];
    let zero_state_bytes =
        vec![0_u8; checked_f32_byte_len(state_elements, "zero recurrent state")?];
    macro_rules! reset_recurrent_batch_state {
        ($stream:expr) => {{
            conv_history_buffer
                .copy_from_host(0, &zero_history_bytes, Some($stream))
                .map_err(|err| {
                    format!("failed to reset linear-attn recurrent batch conv history: {err}")
                })?;
            recurrent_state_buffer
                .copy_from_host(0, &zero_state_bytes, Some($stream))
                .map_err(|err| {
                    format!("failed to reset linear-attn recurrent batch state: {err}")
                })?;
            Ok::<(), String>(())
        }};
    }
    macro_rules! run_recurrent_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                sequence_len,
                hidden,
                1e-6_f32,
                &mut input_normed_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn recurrent batch input RMSNorm: {err}")
            })?;
            qkv_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut qkv_output_buffer,
                $stream,
                "linear-attn recurrent batch qkv projection",
            )?;
            a_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut a_output_buffer,
                $stream,
                "linear-attn recurrent batch a projection",
            )?;
            b_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut b_output_buffer,
                $stream,
                "linear-attn recurrent batch b projection",
            )?;
            ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
                &qkv_output_buffer,
                &conv_weight_buffer,
                &mut conv_history_buffer,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                sequence_len,
                q_scale,
                qk_l2_norm,
                &mut conv_output_buffer,
                &mut q_output_buffer,
                &mut k_output_buffer,
                &mut v_output_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn recurrent batch qkv prepare: {err}")
            })?;
            ullm_runtime_sys::linear_attn_gate_beta_f32(
                &a_output_buffer,
                &b_output_buffer,
                &a_log_buffer,
                &dt_bias_buffer,
                value_heads,
                sequence_len,
                &mut gate_output_buffer,
                &mut beta_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn recurrent batch gate/beta: {err}"))?;
            ullm_runtime_sys::linear_attn_recurrent_f32(
                &q_output_buffer,
                &k_output_buffer,
                &v_output_buffer,
                &gate_output_buffer,
                &beta_output_buffer,
                key_heads,
                value_heads,
                sequence_len,
                key_dim,
                value_dim,
                &mut recurrent_state_buffer,
                &mut recurrent_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn recurrent batch scan: {err}"))
        }};
    }

    reset_recurrent_batch_state!(&mut stream)?;
    run_recurrent_batch!(&mut stream)?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn recurrent batch warmup: {err}")
    })?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        reset_recurrent_batch_state!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn recurrent batch reset: {err}")
        })?;
        let started = Instant::now();
        run_recurrent_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn recurrent batch measured run: {err}")
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

    let a_output = read_runtime_buffer_f32(
        &a_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn recurrent batch a output",
    )?;
    let b_output = read_runtime_buffer_f32(
        &b_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn recurrent batch b output",
    )?;
    let gate_output = read_runtime_buffer_f32(
        &gate_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn recurrent batch gate output",
    )?;
    let beta_output = read_runtime_buffer_f32(
        &beta_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn recurrent batch beta output",
    )?;
    let q_output = read_runtime_buffer_f32(
        &q_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn recurrent batch q output",
    )?;
    let k_output = read_runtime_buffer_f32(
        &k_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn recurrent batch k output",
    )?;
    let v_output = read_runtime_buffer_f32(
        &v_output_buffer,
        &mut stream,
        v_output_elements,
        "linear-attn recurrent batch v output",
    )?;
    let recurrent_output = read_runtime_buffer_f32(
        &recurrent_output_buffer,
        &mut stream,
        v_output_elements,
        "linear-attn recurrent batch output",
    )?;
    let final_state = read_runtime_buffer_f32(
        &recurrent_state_buffer,
        &mut stream,
        state_elements,
        "linear-attn recurrent batch state",
    )?;

    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_output,
        &b_output,
        &a_log.values,
        &dt_bias.values,
        value_heads,
        sequence_len,
    );
    if expected_gate.len() != gate_beta_elements || expected_beta.len() != gate_beta_elements {
        return Err("failed to build linear-attn recurrent batch gate/beta reference".to_string());
    }
    let gate_max_abs_diff = verify_f32_close(
        "linear-attn recurrent batch gate output",
        &gate_output,
        &expected_gate,
        1e-4_f32,
        1e-5_f32,
    )?;
    let beta_max_abs_diff = verify_f32_close(
        "linear-attn recurrent batch beta output",
        &beta_output,
        &expected_beta,
        1e-4_f32,
        1e-5_f32,
    )?;

    let mut expected_state = vec![0.0_f32; state_elements];
    let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
        &q_output,
        &k_output,
        &v_output,
        &gate_output,
        &beta_output,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_recurrent.len() != v_output_elements {
        return Err("failed to build linear-attn recurrent batch reference".to_string());
    }
    let recurrent_output_max_abs_diff = verify_f32_close(
        "linear-attn recurrent batch output",
        &recurrent_output,
        &expected_recurrent,
        2e-3_f32,
        2e-5_f32,
    )?;
    let recurrent_state_max_abs_diff = verify_f32_close(
        "linear-attn recurrent batch state",
        &final_state,
        &expected_state,
        2e-3_f32,
        2e-5_f32,
    )?;

    let output_elements = qkv_output_elements
        .checked_add(gate_beta_elements)
        .and_then(|value| value.checked_add(gate_beta_elements))
        .and_then(|value| value.checked_add(v_output_elements))
        .ok_or_else(|| "linear-attn recurrent batch output element count overflows".to_string())?;
    let preview_len = recurrent_output.len().min(8);
    Ok(format!(
        "package-linear-attn-recurrent-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} key_heads={} value_heads={} key_dim={} value_dim={} channels={} kernel_size={} q_scale={q_scale:.9} qk_l2_norm={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} gate_max_abs_diff={gate_max_abs_diff:.9} beta_max_abs_diff={beta_max_abs_diff:.9} recurrent_output_max_abs_diff={recurrent_output_max_abs_diff:.9} recurrent_state_max_abs_diff={recurrent_state_max_abs_diff:.9} recurrent_preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        a_tensor,
        b_tensor,
        conv_tensor,
        a_log_tensor,
        dt_bias_tensor,
        sequence_len,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        channels,
        kernel_size,
        qk_l2_norm,
        input_elements,
        output_elements,
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
        format_f32_preview(&recurrent_output[..preview_len]),
    ))
}

fn package_linear_attn_post_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-post-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_post_batch_smoke_impl(
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

fn package_linear_attn_post_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn post batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn post batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }

    let sequence_len = prompt_token_ids.len();
    let value_heads = 32_usize;
    let value_dim = 128_usize;
    let hidden = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear-attn post batch hidden size overflows".to_string())?;
    let epsilon = 1e-6_f32;

    let (embedding_vocab, package_hidden) = package_embedding_shape(path)?;
    if package_hidden != hidden {
        return Err(format!(
            "linear-attn post batch hidden mismatch: package hidden={package_hidden} expected {hidden}"
        ));
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn post batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn post batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read linear attention norm tensor: {err}"))?;
    if norm.values.len() != value_dim {
        return Err(format!(
            "linear-attn post batch norm length {} does not match value_dim {value_dim}",
            norm.values.len()
        ));
    }
    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "linear-attn post batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
        ));
    }
    let recurrent_input =
        deterministic_linear_attn_core_output(sequence_len, value_heads, value_dim);
    if recurrent_input.len() != sequence_len * hidden {
        return Err(
            "failed to build deterministic recurrent output for post batch smoke".to_string(),
        );
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
    let z_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &z_tensor,
        chunk_bytes,
    )?;
    let out_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &out_tensor,
        chunk_bytes,
    )?;
    if z_matrix.rows != hidden || z_matrix.cols != hidden {
        return Err(format!(
            "linear-attn post batch z matrix shape mismatch: got [{},{}] expected [{hidden},{hidden}]",
            z_matrix.rows, z_matrix.cols
        ));
    }
    if out_matrix.rows != hidden || out_matrix.cols != hidden {
        return Err(format!(
            "linear-attn post batch out matrix shape mismatch: got [{},{}] expected [{hidden},{hidden}]",
            out_matrix.rows, out_matrix.cols
        ));
    }

    let sequence_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn post batch sequence element count overflows".to_string())?;
    let segments = sequence_len
        .checked_mul(value_heads)
        .ok_or_else(|| "linear-attn post batch segment count overflows".to_string())?;

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn post batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn post batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn post batch input norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch input normed",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn post batch input normed: {err}"))?;
    let mut z_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch z output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn post batch z output: {err}"))?;
    let mut recurrent_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch recurrent output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn post batch recurrent output: {err}")
        })?;
    let mut norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            norm.values.len(),
            "linear-attn post batch norm weight",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn post batch norm weight: {err}"))?;
    let mut attn_projection_input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch projection input",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn post batch projection input: {err}")
        })?;
    let mut attn_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch projected output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn post batch projected output: {err}")
        })?;
    let mut attention_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn post batch attention output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn post batch attention output: {err}")
        })?;

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn post batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn post batch input norm: {err}"))?;
    recurrent_output_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&recurrent_input), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn post batch recurrent output: {err}"))?;
    norm_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&norm.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn post batch norm weight: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn post batch setup: {err}"))?;

    macro_rules! run_post_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                sequence_len,
                hidden,
                1e-6_f32,
                &mut input_normed_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn post batch input RMSNorm: {err}"))?;
            z_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut z_output_buffer,
                $stream,
                "linear-attn post batch z projection",
            )?;
            ullm_runtime_sys::segmented_rmsnorm_silu_mul_f32(
                &recurrent_output_buffer,
                &norm_weight_buffer,
                &z_output_buffer,
                segments,
                value_dim,
                epsilon,
                &mut attn_projection_input_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn post batch RMSNorm SiLU-mul: {err}")
            })?;
            out_matrix.matvec_batch(
                &attn_projection_input_buffer,
                sequence_len,
                &mut attn_projected_buffer,
                $stream,
                "linear-attn post batch out projection",
            )?;
            ullm_runtime_sys::add_f32(
                &attn_projected_buffer,
                &input_buffer,
                sequence_elements,
                &mut attention_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn post batch residual add: {err}"))
        }};
    }

    run_post_batch!(&mut stream)?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn post batch warmup: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        run_post_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn post batch measured run: {err}")
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

    let z_output = read_runtime_buffer_f32(
        &z_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn post batch z output",
    )?;
    let attn_projection_input = read_runtime_buffer_f32(
        &attn_projection_input_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn post batch projection input",
    )?;
    let attn_projected = read_runtime_buffer_f32(
        &attn_projected_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn post batch projected output",
    )?;
    let attention_output = read_runtime_buffer_f32(
        &attention_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn post batch attention output",
    )?;

    let mut expected_normed = vec![0.0_f32; sequence_elements];
    for row in 0..segments {
        let start = row
            .checked_mul(value_dim)
            .ok_or_else(|| "linear-attn post batch expected row start overflows".to_string())?;
        let end = start
            .checked_add(value_dim)
            .ok_or_else(|| "linear-attn post batch expected row end overflows".to_string())?;
        let normed = runtime_host_rmsnorm_f32(&recurrent_input[start..end], &norm.values, epsilon);
        if normed.len() != value_dim {
            return Err("failed to build linear-attn post batch norm reference".to_string());
        }
        expected_normed[start..end].copy_from_slice(&normed);
    }
    let expected_post = runtime_host_silu_mul_f32(&z_output, &expected_normed);
    if expected_post.len() != sequence_elements {
        return Err("failed to build linear-attn post batch activation reference".to_string());
    }
    let post_max_abs_diff = verify_f32_close(
        "linear-attn post batch projection input",
        &attn_projection_input,
        &expected_post,
        1e-4_f32,
        1e-5_f32,
    )?;

    let expected_attention_output = attn_projected
        .iter()
        .zip(embedding_rows.values.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let residual_max_abs_diff = verify_f32_close(
        "linear-attn post batch residual output",
        &attention_output,
        &expected_attention_output,
        1e-4_f32,
        1e-5_f32,
    )?;
    let preview_len = attention_output.len().min(8);
    Ok(format!(
        "package-linear-attn-post-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} value_heads={} value_dim={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} post_max_abs_diff={post_max_abs_diff:.9} residual_max_abs_diff={residual_max_abs_diff:.9} attention_preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        sequence_len,
        hidden,
        value_heads,
        value_dim,
        sequence_elements,
        sequence_elements
            .checked_mul(3)
            .ok_or_else(|| "linear-attn post batch output element count overflows".to_string())?,
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
        tps(sequence_elements * 3, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        format_f32_preview(&attention_output[..preview_len]),
    ))
}

fn package_linear_attn_attention_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-attention-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_attention_batch_smoke_impl(
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

fn package_linear_attn_attention_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn attention batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn attention batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }

    let sequence_len = prompt_token_ids.len();
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear-attn attention batch hidden size overflows".to_string())?;
    let q_elements = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear-attn attention batch q element count overflows".to_string())?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(hidden))
        .ok_or_else(|| "linear-attn attention batch qkv channel count overflows".to_string())?;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let input_epsilon = 1e-6_f32;
    let post_epsilon = 1e-6_f32;

    let (embedding_vocab, package_hidden) = package_embedding_shape(path)?;
    if package_hidden != hidden {
        return Err(format!(
            "linear-attn attention batch hidden mismatch: package hidden={package_hidden} expected {hidden}"
        ));
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn attention batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
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
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read input RMSNorm tensor: {err}"))?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    if input_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn attention batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read conv1d tensor: {err}"))?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "linear-attn attention batch conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "linear-attn attention batch conv channel count is too large".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "linear-attn attention batch kernel size is too large".to_string())?;
    if conv_channels != channels {
        return Err(format!(
            "linear-attn attention batch conv channels mismatch: got {conv_channels} expected {channels}"
        ));
    }
    if kernel_size == 0 {
        return Err(
            "linear-attn attention batch kernel size must be greater than zero".to_string(),
        );
    }
    if conv.values.len() != channels * kernel_size {
        return Err(format!(
            "linear-attn attention batch conv weight element count mismatch: got {} expected {}",
            conv.values.len(),
            channels * kernel_size
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read A_log tensor: {err}"))?;
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read dt_bias tensor: {err}"))?;
    if a_log.values.len() != value_heads || dt_bias.values.len() != value_heads {
        return Err(format!(
            "linear-attn attention batch a_log/dt_bias length mismatch: a_log={} dt_bias={} expected {value_heads}",
            a_log.values.len(),
            dt_bias.values.len()
        ));
    }
    let norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read linear attention norm tensor: {err}"))?;
    if norm.values.len() != value_dim {
        return Err(format!(
            "linear-attn attention batch norm length {} does not match value_dim {value_dim}",
            norm.values.len()
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "linear-attn attention batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
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
    let out_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &out_tensor,
        chunk_bytes,
    )?;
    if qkv_matrix.rows != channels || qkv_matrix.cols != hidden {
        return Err(format!(
            "linear-attn attention batch qkv matrix shape mismatch: rows={} cols={} expected rows={channels} cols={hidden}",
            qkv_matrix.rows, qkv_matrix.cols
        ));
    }
    if z_matrix.rows != hidden
        || z_matrix.cols != hidden
        || out_matrix.rows != hidden
        || out_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn attention batch z/out matrix shape mismatch: z=[{},{}] out=[{},{}] expected [{hidden},{hidden}]",
            z_matrix.rows, z_matrix.cols, out_matrix.rows, out_matrix.cols
        ));
    }
    if a_matrix.rows != value_heads
        || b_matrix.rows != value_heads
        || a_matrix.cols != hidden
        || b_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn attention batch a/b matrix shape mismatch: a=[{},{}] b=[{},{}] expected [{value_heads},{hidden}]",
            a_matrix.rows, a_matrix.cols, b_matrix.rows, b_matrix.cols
        ));
    }

    let sequence_elements = sequence_len.checked_mul(hidden).ok_or_else(|| {
        "linear-attn attention batch sequence element count overflows".to_string()
    })?;
    let qkv_output_elements = sequence_len.checked_mul(channels).ok_or_else(|| {
        "linear-attn attention batch qkv output element count overflows".to_string()
    })?;
    let q_output_elements = sequence_len.checked_mul(q_elements).ok_or_else(|| {
        "linear-attn attention batch q output element count overflows".to_string()
    })?;
    let gate_beta_elements = sequence_len.checked_mul(value_heads).ok_or_else(|| {
        "linear-attn attention batch gate/beta element count overflows".to_string()
    })?;
    let history_elements = channels.checked_mul(kernel_size).ok_or_else(|| {
        "linear-attn attention batch conv history element count overflows".to_string()
    })?;
    let state_elements = value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "linear-attn attention batch state element count overflows".to_string())?;
    let post_segments = sequence_len
        .checked_mul(value_heads)
        .ok_or_else(|| "linear-attn attention batch post segment count overflows".to_string())?;

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn attention batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch input norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch input normed",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch input normed: {err}")
        })?;
    let mut qkv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn attention batch qkv output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch qkv: {err}"))?;
    let mut z_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch z output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch z: {err}"))?;
    let mut a_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn attention batch a output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch a: {err}"))?;
    let mut b_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn attention batch b output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch b: {err}"))?;
    let mut conv_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            conv.values.len(),
            "linear-attn attention batch conv weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch conv weight: {err}")
        })?;
    let mut conv_history_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            history_elements,
            "linear-attn attention batch conv history",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch conv history: {err}")
        })?;
    let mut conv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn attention batch conv output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch conv output: {err}")
        })?;
    let mut q_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn attention batch q output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch q: {err}"))?;
    let mut k_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn attention batch k output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch k: {err}"))?;
    let mut v_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch v output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch v: {err}"))?;
    let mut a_log_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            a_log.values.len(),
            "linear-attn attention batch A_log",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch A_log: {err}"))?;
    let mut dt_bias_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            dt_bias.values.len(),
            "linear-attn attention batch dt_bias",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch dt_bias: {err}"))?;
    let mut gate_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn attention batch gate output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch gate: {err}"))?;
    let mut beta_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn attention batch beta output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch beta: {err}"))?;
    let mut recurrent_state_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            state_elements,
            "linear-attn attention batch state",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch state: {err}"))?;
    let mut recurrent_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch recurrent output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch recurrent output: {err}")
        })?;
    let mut attn_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            norm.values.len(),
            "linear-attn attention batch norm weight",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch norm: {err}"))?;
    let mut attn_projection_input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch projection input",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch projection input: {err}")
        })?;
    let mut attn_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch projected output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn attention batch projected output: {err}")
        })?;
    let mut attention_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn attention batch output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn attention batch output: {err}"))?;

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn attention batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn attention batch input norm: {err}"))?;
    conv_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&conv.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn attention batch conv weight: {err}"))?;
    a_log_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&a_log.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn attention batch A_log: {err}"))?;
    dt_bias_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&dt_bias.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn attention batch dt_bias: {err}"))?;
    attn_norm_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&norm.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn attention batch norm weight: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn attention batch setup: {err}"))?;

    let zero_history_bytes =
        vec![0_u8; checked_f32_byte_len(history_elements, "zero conv history")?];
    let zero_state_bytes =
        vec![0_u8; checked_f32_byte_len(state_elements, "zero recurrent state")?];
    macro_rules! reset_attention_batch_state {
        ($stream:expr) => {{
            conv_history_buffer
                .copy_from_host(0, &zero_history_bytes, Some($stream))
                .map_err(|err| {
                    format!("failed to reset linear-attn attention batch conv history: {err}")
                })?;
            recurrent_state_buffer
                .copy_from_host(0, &zero_state_bytes, Some($stream))
                .map_err(|err| {
                    format!("failed to reset linear-attn attention batch state: {err}")
                })?;
            Ok::<(), String>(())
        }};
    }
    macro_rules! run_attention_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                sequence_len,
                hidden,
                input_epsilon,
                &mut input_normed_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn attention batch input RMSNorm: {err}")
            })?;
            qkv_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut qkv_output_buffer,
                $stream,
                "linear-attn attention batch qkv projection",
            )?;
            z_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut z_output_buffer,
                $stream,
                "linear-attn attention batch z projection",
            )?;
            a_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut a_output_buffer,
                $stream,
                "linear-attn attention batch a projection",
            )?;
            b_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut b_output_buffer,
                $stream,
                "linear-attn attention batch b projection",
            )?;
            ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
                &qkv_output_buffer,
                &conv_weight_buffer,
                &mut conv_history_buffer,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                sequence_len,
                q_scale,
                qk_l2_norm,
                &mut conv_output_buffer,
                &mut q_output_buffer,
                &mut k_output_buffer,
                &mut v_output_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn attention batch qkv prepare: {err}")
            })?;
            ullm_runtime_sys::linear_attn_gate_beta_f32(
                &a_output_buffer,
                &b_output_buffer,
                &a_log_buffer,
                &dt_bias_buffer,
                value_heads,
                sequence_len,
                &mut gate_output_buffer,
                &mut beta_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn attention batch gate/beta: {err}"))?;
            ullm_runtime_sys::linear_attn_recurrent_f32(
                &q_output_buffer,
                &k_output_buffer,
                &v_output_buffer,
                &gate_output_buffer,
                &beta_output_buffer,
                key_heads,
                value_heads,
                sequence_len,
                key_dim,
                value_dim,
                &mut recurrent_state_buffer,
                &mut recurrent_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn attention batch recurrent: {err}"))?;
            ullm_runtime_sys::segmented_rmsnorm_silu_mul_f32(
                &recurrent_output_buffer,
                &attn_norm_weight_buffer,
                &z_output_buffer,
                post_segments,
                value_dim,
                post_epsilon,
                &mut attn_projection_input_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn attention batch post RMSNorm SiLU-mul: {err}")
            })?;
            out_matrix.matvec_batch(
                &attn_projection_input_buffer,
                sequence_len,
                &mut attn_projected_buffer,
                $stream,
                "linear-attn attention batch out projection",
            )?;
            ullm_runtime_sys::add_f32(
                &attn_projected_buffer,
                &input_buffer,
                sequence_elements,
                &mut attention_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn attention batch residual add: {err}"))
        }};
    }

    reset_attention_batch_state!(&mut stream)?;
    run_attention_batch!(&mut stream)?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize linear-attn attention batch warmup: {err}")
    })?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        reset_attention_batch_state!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn attention batch reset: {err}")
        })?;
        let started = Instant::now();
        run_attention_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn attention batch measured run: {err}")
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

    let a_output = read_runtime_buffer_f32(
        &a_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn attention batch a output",
    )?;
    let b_output = read_runtime_buffer_f32(
        &b_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn attention batch b output",
    )?;
    let gate_output = read_runtime_buffer_f32(
        &gate_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn attention batch gate output",
    )?;
    let beta_output = read_runtime_buffer_f32(
        &beta_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn attention batch beta output",
    )?;
    let q_output = read_runtime_buffer_f32(
        &q_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn attention batch q output",
    )?;
    let k_output = read_runtime_buffer_f32(
        &k_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn attention batch k output",
    )?;
    let v_output = read_runtime_buffer_f32(
        &v_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch v output",
    )?;
    let z_output = read_runtime_buffer_f32(
        &z_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch z output",
    )?;
    let recurrent_output = read_runtime_buffer_f32(
        &recurrent_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch recurrent output",
    )?;
    let final_state = read_runtime_buffer_f32(
        &recurrent_state_buffer,
        &mut stream,
        state_elements,
        "linear-attn attention batch state",
    )?;
    let attn_projection_input = read_runtime_buffer_f32(
        &attn_projection_input_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch projection input",
    )?;
    let attn_projected = read_runtime_buffer_f32(
        &attn_projected_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch projected output",
    )?;
    let attention_output = read_runtime_buffer_f32(
        &attention_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn attention batch output",
    )?;

    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_output,
        &b_output,
        &a_log.values,
        &dt_bias.values,
        value_heads,
        sequence_len,
    );
    if expected_gate.len() != gate_beta_elements || expected_beta.len() != gate_beta_elements {
        return Err("failed to build linear-attn attention batch gate/beta reference".to_string());
    }
    let gate_max_abs_diff = verify_f32_close(
        "linear-attn attention batch gate output",
        &gate_output,
        &expected_gate,
        1e-4_f32,
        1e-5_f32,
    )?;
    let beta_max_abs_diff = verify_f32_close(
        "linear-attn attention batch beta output",
        &beta_output,
        &expected_beta,
        1e-4_f32,
        1e-5_f32,
    )?;

    let mut expected_state = vec![0.0_f32; state_elements];
    let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
        &q_output,
        &k_output,
        &v_output,
        &gate_output,
        &beta_output,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_recurrent.len() != sequence_elements {
        return Err("failed to build linear-attn attention batch recurrent reference".to_string());
    }
    let recurrent_output_max_abs_diff = verify_f32_close(
        "linear-attn attention batch recurrent output",
        &recurrent_output,
        &expected_recurrent,
        2e-3_f32,
        2e-5_f32,
    )?;
    let recurrent_state_max_abs_diff = verify_f32_close(
        "linear-attn attention batch recurrent state",
        &final_state,
        &expected_state,
        2e-3_f32,
        2e-5_f32,
    )?;

    let mut expected_normed = vec![0.0_f32; sequence_elements];
    for row in 0..post_segments {
        let start = row.checked_mul(value_dim).ok_or_else(|| {
            "linear-attn attention batch expected norm start overflows".to_string()
        })?;
        let end = start
            .checked_add(value_dim)
            .ok_or_else(|| "linear-attn attention batch expected norm end overflows".to_string())?;
        let normed =
            runtime_host_rmsnorm_f32(&recurrent_output[start..end], &norm.values, post_epsilon);
        if normed.len() != value_dim {
            return Err("failed to build linear-attn attention batch post reference".to_string());
        }
        expected_normed[start..end].copy_from_slice(&normed);
    }
    let expected_post = runtime_host_silu_mul_f32(&z_output, &expected_normed);
    if expected_post.len() != sequence_elements {
        return Err("failed to build linear-attn attention batch activation reference".to_string());
    }
    let post_max_abs_diff = verify_f32_close(
        "linear-attn attention batch post output",
        &attn_projection_input,
        &expected_post,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_attention_output = attn_projected
        .iter()
        .zip(embedding_rows.values.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let residual_max_abs_diff = verify_f32_close(
        "linear-attn attention batch residual output",
        &attention_output,
        &expected_attention_output,
        1e-4_f32,
        1e-5_f32,
    )?;
    let output_elements = qkv_output_elements
        .checked_add(sequence_elements)
        .and_then(|value| value.checked_add(gate_beta_elements))
        .and_then(|value| value.checked_add(gate_beta_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .ok_or_else(|| "linear-attn attention batch output element count overflows".to_string())?;
    let preview_len = attention_output.len().min(8);
    Ok(format!(
        "package-linear-attn-attention-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} key_heads={} value_heads={} key_dim={} value_dim={} channels={} kernel_size={} q_scale={q_scale:.9} qk_l2_norm={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} gate_max_abs_diff={gate_max_abs_diff:.9} beta_max_abs_diff={beta_max_abs_diff:.9} recurrent_output_max_abs_diff={recurrent_output_max_abs_diff:.9} recurrent_state_max_abs_diff={recurrent_state_max_abs_diff:.9} post_max_abs_diff={post_max_abs_diff:.9} residual_max_abs_diff={residual_max_abs_diff:.9} attention_preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        z_tensor,
        a_tensor,
        b_tensor,
        conv_tensor,
        a_log_tensor,
        dt_bias_tensor,
        norm_tensor,
        out_tensor,
        sequence_len,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        channels,
        kernel_size,
        qk_l2_norm,
        sequence_elements,
        output_elements,
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
        format_f32_preview(&attention_output[..preview_len]),
    ))
}

fn package_linear_attn_mlp_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-mlp-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_mlp_batch_smoke_impl(
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

fn package_linear_attn_mlp_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn MLP batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn MLP batch smoke requires a linear attention layer, got layer {layer_index}"
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
            "linear-attn MLP batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let mut post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read post RMSNorm tensor: {err}"))?;
    post_norm.values = effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    if post_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn MLP batch post norm length {} does not match hidden {hidden}",
            post_norm.values.len()
        ));
    }
    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "linear-attn MLP batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
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
    let gate_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &gate_tensor,
        chunk_bytes,
    )?;
    let up_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &up_tensor,
        chunk_bytes,
    )?;
    let down_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &down_tensor,
        chunk_bytes,
    )?;
    if gate_matrix.rows != up_matrix.rows
        || gate_matrix.cols != up_matrix.cols
        || gate_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn MLP batch gate/up shape mismatch: gate=[{},{}] up=[{},{}] hidden={hidden}",
            gate_matrix.rows, gate_matrix.cols, up_matrix.rows, up_matrix.cols
        ));
    }
    let intermediate = gate_matrix.rows;
    if down_matrix.rows != hidden || down_matrix.cols != intermediate {
        return Err(format!(
            "linear-attn MLP batch down shape mismatch: down=[{},{}] expected [{hidden},{intermediate}]",
            down_matrix.rows, down_matrix.cols
        ));
    }

    let sequence_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn MLP batch sequence element count overflows".to_string())?;
    let intermediate_elements = sequence_len
        .checked_mul(intermediate)
        .ok_or_else(|| "linear-attn MLP batch intermediate element count overflows".to_string())?;

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn MLP batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch input: {err}"))?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            post_norm.values.len(),
            "linear-attn MLP batch post norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn MLP batch post norm weight: {err}")
        })?;
    let mut post_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn MLP batch post normed",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch normed: {err}"))?;
    let mut gate_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn MLP batch gate output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch gate: {err}"))?;
    let mut up_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn MLP batch up output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch up: {err}"))?;
    let mut activation_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn MLP batch activation",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch activation: {err}"))?;
    let mut down_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn MLP batch down output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch down output: {err}"))?;
    let mut layer_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn MLP batch layer output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn MLP batch layer output: {err}"))?;

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn MLP batch input: {err}"))?;
    post_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&post_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn MLP batch post norm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn MLP batch setup: {err}"))?;

    macro_rules! run_mlp_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &post_norm_weight_buffer,
                sequence_len,
                hidden,
                1e-5_f32,
                &mut post_normed_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn MLP batch post RMSNorm: {err}"))?;
            gate_matrix.matvec_batch(
                &post_normed_buffer,
                sequence_len,
                &mut gate_output_buffer,
                $stream,
                "linear-attn MLP batch gate projection",
            )?;
            up_matrix.matvec_batch(
                &post_normed_buffer,
                sequence_len,
                &mut up_output_buffer,
                $stream,
                "linear-attn MLP batch up projection",
            )?;
            ullm_runtime_sys::silu_mul_f32(
                &gate_output_buffer,
                &up_output_buffer,
                intermediate_elements,
                &mut activation_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn MLP batch SiLU-mul: {err}"))?;
            down_matrix.matvec_batch(
                &activation_buffer,
                sequence_len,
                &mut down_output_buffer,
                $stream,
                "linear-attn MLP batch down projection",
            )?;
            ullm_runtime_sys::add_f32(
                &down_output_buffer,
                &input_buffer,
                sequence_elements,
                &mut layer_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn MLP batch residual add: {err}"))
        }};
    }

    run_mlp_batch!(&mut stream)?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn MLP batch warmup: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        run_mlp_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn MLP batch measured run: {err}")
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

    let post_normed = read_runtime_buffer_f32(
        &post_normed_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn MLP batch post normed",
    )?;
    let gate_output = read_runtime_buffer_f32(
        &gate_output_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn MLP batch gate output",
    )?;
    let up_output = read_runtime_buffer_f32(
        &up_output_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn MLP batch up output",
    )?;
    let activation = read_runtime_buffer_f32(
        &activation_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn MLP batch activation",
    )?;
    let down_output = read_runtime_buffer_f32(
        &down_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn MLP batch down output",
    )?;
    let layer_output = read_runtime_buffer_f32(
        &layer_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn MLP batch layer output",
    )?;

    let mut expected_normed = Vec::with_capacity(sequence_elements);
    for token_index in 0..sequence_len {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "linear-attn MLP batch expected norm start overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "linear-attn MLP batch expected norm end overflows".to_string())?;
        expected_normed.extend(runtime_host_rmsnorm_f32(
            &embedding_rows.values[start..end],
            &post_norm.values,
            1e-5_f32,
        ));
    }
    if expected_normed.len() != sequence_elements {
        return Err("failed to build linear-attn MLP batch norm reference".to_string());
    }
    let norm_max_abs_diff = verify_f32_close(
        "linear-attn MLP batch post norm",
        &post_normed,
        &expected_normed,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_activation = runtime_host_silu_mul_f32(&gate_output, &up_output);
    if expected_activation.len() != intermediate_elements {
        return Err("failed to build linear-attn MLP batch activation reference".to_string());
    }
    let activation_max_abs_diff = verify_f32_close(
        "linear-attn MLP batch activation",
        &activation,
        &expected_activation,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_layer_output = down_output
        .iter()
        .zip(embedding_rows.values.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let residual_max_abs_diff = verify_f32_close(
        "linear-attn MLP batch residual output",
        &layer_output,
        &expected_layer_output,
        1e-4_f32,
        1e-5_f32,
    )?;
    let output_elements = sequence_elements
        .checked_add(intermediate_elements)
        .and_then(|value| value.checked_add(intermediate_elements))
        .and_then(|value| value.checked_add(intermediate_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .ok_or_else(|| "linear-attn MLP batch output element count overflows".to_string())?;
    let preview_len = layer_output.len().min(8);
    Ok(format!(
        "package-linear-attn-mlp-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} intermediate={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} residual_max_abs_diff={residual_max_abs_diff:.9} layer_preview={} verified=true",
        path,
        layer_index,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        sequence_len,
        hidden,
        intermediate,
        sequence_elements,
        output_elements,
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
        format_f32_preview(&layer_output[..preview_len]),
    ))
}

fn package_linear_attn_layer_batch_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    prompt_token_ids: Option<String>,
    measured_repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-layer-batch-smoke requires a .ullm.d path");
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

    match package_linear_attn_layer_batch_smoke_impl(
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

fn package_linear_attn_layer_batch_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    prompt_token_ids: Vec<usize>,
    measured_repeats: usize,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err("linear-attn layer batch smoke requires at least one token".to_string());
    }
    if package_decoder_layer_kind(path, layer_index)? != PackageDecoderLayerKind::LinearAttention {
        return Err(format!(
            "linear-attn layer batch smoke requires a linear attention layer, got layer {layer_index}"
        ));
    }

    let sequence_len = prompt_token_ids.len();
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear-attn layer batch hidden size overflows".to_string())?;
    let q_elements = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear-attn layer batch q element count overflows".to_string())?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(hidden))
        .ok_or_else(|| "linear-attn layer batch qkv channel count overflows".to_string())?;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let input_epsilon = 1e-6_f32;
    let attention_post_epsilon = 1e-6_f32;
    let mlp_epsilon = 1e-5_f32;

    let (embedding_vocab, package_hidden) = package_embedding_shape(path)?;
    if package_hidden != hidden {
        return Err(format!(
            "linear-attn layer batch hidden mismatch: package hidden={package_hidden} expected {hidden}"
        ));
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "linear-attn layer batch token id {token_id} is out of embedding range 0..{embedding_vocab}"
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
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");
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
            "linear-attn layer batch input norm length {} does not match hidden {hidden}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read conv1d tensor: {err}"))?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "linear-attn layer batch conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "linear-attn layer batch conv channel count is too large".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "linear-attn layer batch kernel size is too large".to_string())?;
    if conv_channels != channels {
        return Err(format!(
            "linear-attn layer batch conv channels mismatch: got {conv_channels} expected {channels}"
        ));
    }
    if kernel_size == 0 {
        return Err("linear-attn layer batch kernel size must be greater than zero".to_string());
    }
    if conv.values.len() != channels * kernel_size {
        return Err(format!(
            "linear-attn layer batch conv weight element count mismatch: got {} expected {}",
            conv.values.len(),
            channels * kernel_size
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read A_log tensor: {err}"))?;
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read dt_bias tensor: {err}"))?;
    if a_log.values.len() != value_heads || dt_bias.values.len() != value_heads {
        return Err(format!(
            "linear-attn layer batch a_log/dt_bias length mismatch: a_log={} dt_bias={} expected {value_heads}",
            a_log.values.len(),
            dt_bias.values.len()
        ));
    }
    let norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read linear attention norm tensor: {err}"))?;
    if norm.values.len() != value_dim {
        return Err(format!(
            "linear-attn layer batch norm length {} does not match value_dim {value_dim}",
            norm.values.len()
        ));
    }
    let mut post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)
        .map_err(|err| format!("failed to read post RMSNorm tensor: {err}"))?;
    post_norm.values = effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    if post_norm.values.len() != hidden {
        return Err(format!(
            "linear-attn layer batch post norm length {} does not match hidden {hidden}",
            post_norm.values.len()
        ));
    }

    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
            .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
    if embedding_rows.columns != hidden || embedding_rows.values.len() != sequence_len * hidden {
        return Err(format!(
            "linear-attn layer batch embedding shape mismatch: columns={} values={} prompt_tokens={} hidden={hidden}",
            embedding_rows.columns,
            embedding_rows.values.len(),
            sequence_len
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
    let out_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &out_tensor,
        chunk_bytes,
    )?;
    let gate_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &gate_tensor,
        chunk_bytes,
    )?;
    let up_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &up_tensor,
        chunk_bytes,
    )?;
    let down_matrix = PackageAq4ResidentMatvec::load(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &down_tensor,
        chunk_bytes,
    )?;
    if qkv_matrix.rows != channels || qkv_matrix.cols != hidden {
        return Err(format!(
            "linear-attn layer batch qkv matrix shape mismatch: rows={} cols={} expected rows={channels} cols={hidden}",
            qkv_matrix.rows, qkv_matrix.cols
        ));
    }
    if z_matrix.rows != hidden
        || z_matrix.cols != hidden
        || out_matrix.rows != hidden
        || out_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn layer batch z/out matrix shape mismatch: z=[{},{}] out=[{},{}] expected [{hidden},{hidden}]",
            z_matrix.rows, z_matrix.cols, out_matrix.rows, out_matrix.cols
        ));
    }
    if a_matrix.rows != value_heads
        || b_matrix.rows != value_heads
        || a_matrix.cols != hidden
        || b_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn layer batch a/b matrix shape mismatch: a=[{},{}] b=[{},{}] expected [{value_heads},{hidden}]",
            a_matrix.rows, a_matrix.cols, b_matrix.rows, b_matrix.cols
        ));
    }
    if gate_matrix.rows != up_matrix.rows
        || gate_matrix.cols != up_matrix.cols
        || gate_matrix.cols != hidden
    {
        return Err(format!(
            "linear-attn layer batch gate/up shape mismatch: gate=[{},{}] up=[{},{}] hidden={hidden}",
            gate_matrix.rows, gate_matrix.cols, up_matrix.rows, up_matrix.cols
        ));
    }
    let intermediate = gate_matrix.rows;
    if down_matrix.rows != hidden || down_matrix.cols != intermediate {
        return Err(format!(
            "linear-attn layer batch down shape mismatch: down=[{},{}] expected [{hidden},{intermediate}]",
            down_matrix.rows, down_matrix.cols
        ));
    }

    let sequence_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "linear-attn layer batch sequence element count overflows".to_string())?;
    let qkv_output_elements = sequence_len
        .checked_mul(channels)
        .ok_or_else(|| "linear-attn layer batch qkv output element count overflows".to_string())?;
    let q_output_elements = sequence_len
        .checked_mul(q_elements)
        .ok_or_else(|| "linear-attn layer batch q output element count overflows".to_string())?;
    let gate_beta_elements = sequence_len
        .checked_mul(value_heads)
        .ok_or_else(|| "linear-attn layer batch gate/beta element count overflows".to_string())?;
    let intermediate_elements = sequence_len.checked_mul(intermediate).ok_or_else(|| {
        "linear-attn layer batch intermediate element count overflows".to_string()
    })?;
    let history_elements = channels.checked_mul(kernel_size).ok_or_else(|| {
        "linear-attn layer batch conv history element count overflows".to_string()
    })?;
    let state_elements = value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "linear-attn layer batch state element count overflows".to_string())?;
    let attention_post_segments = sequence_len
        .checked_mul(value_heads)
        .ok_or_else(|| "linear-attn layer batch post segment count overflows".to_string())?;

    let mut input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch input",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch input: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            input_norm.values.len(),
            "linear-attn layer batch input norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch input norm weight: {err}")
        })?;
    let mut input_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch input normed",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch input normed: {err}"))?;
    let mut qkv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn layer batch qkv output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch qkv: {err}"))?;
    let mut z_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch z output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch z: {err}"))?;
    let mut a_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn layer batch a output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch a: {err}"))?;
    let mut b_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn layer batch b output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch b: {err}"))?;
    let mut conv_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            conv.values.len(),
            "linear-attn layer batch conv weight",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch conv weight: {err}"))?;
    let mut conv_history_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            history_elements,
            "linear-attn layer batch conv history",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch conv history: {err}"))?;
    let mut conv_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            qkv_output_elements,
            "linear-attn layer batch conv output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch conv output: {err}"))?;
    let mut q_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn layer batch q output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch q: {err}"))?;
    let mut k_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            q_output_elements,
            "linear-attn layer batch k output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch k: {err}"))?;
    let mut v_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch v output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch v: {err}"))?;
    let mut a_log_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            a_log.values.len(),
            "linear-attn layer batch A_log",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch A_log: {err}"))?;
    let mut dt_bias_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            dt_bias.values.len(),
            "linear-attn layer batch dt_bias",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch dt_bias: {err}"))?;
    let mut gate_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn layer batch gate output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch gate: {err}"))?;
    let mut beta_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            gate_beta_elements,
            "linear-attn layer batch beta output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch beta: {err}"))?;
    let mut recurrent_state_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            state_elements,
            "linear-attn layer batch state",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch state: {err}"))?;
    let mut recurrent_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch recurrent output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch recurrent output: {err}")
        })?;
    let mut attn_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            norm.values.len(),
            "linear-attn layer batch norm weight",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch norm: {err}"))?;
    let mut attn_projection_input_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch projection input",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch projection input: {err}")
        })?;
    let mut attn_projected_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch projected output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch projected output: {err}")
        })?;
    let mut attention_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch attention output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch attention output: {err}")
        })?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            post_norm.values.len(),
            "linear-attn layer batch post norm weight",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch post norm weight: {err}")
        })?;
    let mut post_normed_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch post normed",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch post normed: {err}"))?;
    let mut mlp_gate_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn layer batch MLP gate output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch MLP gate: {err}"))?;
    let mut mlp_up_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn layer batch MLP up output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch MLP up: {err}"))?;
    let mut mlp_activation_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            intermediate_elements,
            "linear-attn layer batch MLP activation",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch MLP activation: {err}")
        })?;
    let mut mlp_down_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch MLP down output",
        )?)
        .map_err(|err| {
            format!("failed to allocate linear-attn layer batch MLP down output: {err}")
        })?;
    let mut layer_output_buffer = context
        .alloc_buffer(checked_f32_byte_len(
            sequence_elements,
            "linear-attn layer batch layer output",
        )?)
        .map_err(|err| format!("failed to allocate linear-attn layer batch layer output: {err}"))?;

    input_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&embedding_rows.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn layer batch input: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&input_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn layer batch input norm: {err}"))?;
    conv_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&conv.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn layer batch conv weight: {err}"))?;
    a_log_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&a_log.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn layer batch A_log: {err}"))?;
    dt_bias_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&dt_bias.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn layer batch dt_bias: {err}"))?;
    attn_norm_weight_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&norm.values), Some(&mut stream))
        .map_err(|err| format!("failed to copy linear-attn layer batch norm weight: {err}"))?;
    post_norm_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&post_norm.values),
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to copy linear-attn layer batch post norm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn layer batch setup: {err}"))?;

    let zero_history_bytes =
        vec![0_u8; checked_f32_byte_len(history_elements, "zero conv history")?];
    let zero_state_bytes =
        vec![0_u8; checked_f32_byte_len(state_elements, "zero recurrent state")?];
    macro_rules! reset_layer_batch_state {
        ($stream:expr) => {{
            conv_history_buffer
                .copy_from_host(0, &zero_history_bytes, Some($stream))
                .map_err(|err| {
                    format!("failed to reset linear-attn layer batch conv history: {err}")
                })?;
            recurrent_state_buffer
                .copy_from_host(0, &zero_state_bytes, Some($stream))
                .map_err(|err| format!("failed to reset linear-attn layer batch state: {err}"))?;
            Ok::<(), String>(())
        }};
    }
    macro_rules! run_layer_batch {
        ($stream:expr) => {{
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &input_buffer,
                &input_norm_weight_buffer,
                sequence_len,
                hidden,
                input_epsilon,
                &mut input_normed_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch input RMSNorm: {err}"))?;
            qkv_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut qkv_output_buffer,
                $stream,
                "linear-attn layer batch qkv projection",
            )?;
            z_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut z_output_buffer,
                $stream,
                "linear-attn layer batch z projection",
            )?;
            a_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut a_output_buffer,
                $stream,
                "linear-attn layer batch a projection",
            )?;
            b_matrix.matvec_batch(
                &input_normed_buffer,
                sequence_len,
                &mut b_output_buffer,
                $stream,
                "linear-attn layer batch b projection",
            )?;
            ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
                &qkv_output_buffer,
                &conv_weight_buffer,
                &mut conv_history_buffer,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                sequence_len,
                q_scale,
                qk_l2_norm,
                &mut conv_output_buffer,
                &mut q_output_buffer,
                &mut k_output_buffer,
                &mut v_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch qkv prepare: {err}"))?;
            ullm_runtime_sys::linear_attn_gate_beta_f32(
                &a_output_buffer,
                &b_output_buffer,
                &a_log_buffer,
                &dt_bias_buffer,
                value_heads,
                sequence_len,
                &mut gate_output_buffer,
                &mut beta_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch gate/beta: {err}"))?;
            ullm_runtime_sys::linear_attn_recurrent_f32(
                &q_output_buffer,
                &k_output_buffer,
                &v_output_buffer,
                &gate_output_buffer,
                &beta_output_buffer,
                key_heads,
                value_heads,
                sequence_len,
                key_dim,
                value_dim,
                &mut recurrent_state_buffer,
                &mut recurrent_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch recurrent: {err}"))?;
            ullm_runtime_sys::segmented_rmsnorm_silu_mul_f32(
                &recurrent_output_buffer,
                &attn_norm_weight_buffer,
                &z_output_buffer,
                attention_post_segments,
                value_dim,
                attention_post_epsilon,
                &mut attn_projection_input_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn layer batch attention RMSNorm SiLU-mul: {err}")
            })?;
            out_matrix.matvec_batch(
                &attn_projection_input_buffer,
                sequence_len,
                &mut attn_projected_buffer,
                $stream,
                "linear-attn layer batch out projection",
            )?;
            ullm_runtime_sys::add_f32(
                &attn_projected_buffer,
                &input_buffer,
                sequence_elements,
                &mut attention_output_buffer,
                Some($stream),
            )
            .map_err(|err| {
                format!("failed to run linear-attn layer batch attention residual: {err}")
            })?;
            ullm_runtime_sys::segmented_rmsnorm_f32(
                &attention_output_buffer,
                &post_norm_weight_buffer,
                sequence_len,
                hidden,
                mlp_epsilon,
                &mut post_normed_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch post RMSNorm: {err}"))?;
            gate_matrix.matvec_batch(
                &post_normed_buffer,
                sequence_len,
                &mut mlp_gate_output_buffer,
                $stream,
                "linear-attn layer batch MLP gate projection",
            )?;
            up_matrix.matvec_batch(
                &post_normed_buffer,
                sequence_len,
                &mut mlp_up_output_buffer,
                $stream,
                "linear-attn layer batch MLP up projection",
            )?;
            ullm_runtime_sys::silu_mul_f32(
                &mlp_gate_output_buffer,
                &mlp_up_output_buffer,
                intermediate_elements,
                &mut mlp_activation_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch MLP SiLU-mul: {err}"))?;
            down_matrix.matvec_batch(
                &mlp_activation_buffer,
                sequence_len,
                &mut mlp_down_output_buffer,
                $stream,
                "linear-attn layer batch MLP down projection",
            )?;
            ullm_runtime_sys::add_f32(
                &mlp_down_output_buffer,
                &attention_output_buffer,
                sequence_elements,
                &mut layer_output_buffer,
                Some($stream),
            )
            .map_err(|err| format!("failed to run linear-attn layer batch MLP residual: {err}"))
        }};
    }

    reset_layer_batch_state!(&mut stream)?;
    run_layer_batch!(&mut stream)?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize linear-attn layer batch warmup: {err}"))?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        reset_layer_batch_state!(&mut stream)?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize linear-attn layer batch reset: {err}"))?;
        let started = Instant::now();
        run_layer_batch!(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn layer batch measured run: {err}")
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

    let a_output = read_runtime_buffer_f32(
        &a_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn layer batch a output",
    )?;
    let b_output = read_runtime_buffer_f32(
        &b_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn layer batch b output",
    )?;
    let gate_output = read_runtime_buffer_f32(
        &gate_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn layer batch gate output",
    )?;
    let beta_output = read_runtime_buffer_f32(
        &beta_output_buffer,
        &mut stream,
        gate_beta_elements,
        "linear-attn layer batch beta output",
    )?;
    let q_output = read_runtime_buffer_f32(
        &q_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn layer batch q output",
    )?;
    let k_output = read_runtime_buffer_f32(
        &k_output_buffer,
        &mut stream,
        q_output_elements,
        "linear-attn layer batch k output",
    )?;
    let v_output = read_runtime_buffer_f32(
        &v_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch v output",
    )?;
    let z_output = read_runtime_buffer_f32(
        &z_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch z output",
    )?;
    let recurrent_output = read_runtime_buffer_f32(
        &recurrent_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch recurrent output",
    )?;
    let final_state = read_runtime_buffer_f32(
        &recurrent_state_buffer,
        &mut stream,
        state_elements,
        "linear-attn layer batch state",
    )?;
    let attn_projection_input = read_runtime_buffer_f32(
        &attn_projection_input_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch attention projection input",
    )?;
    let attn_projected = read_runtime_buffer_f32(
        &attn_projected_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch attention projected output",
    )?;
    let attention_output = read_runtime_buffer_f32(
        &attention_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch attention output",
    )?;
    let post_normed = read_runtime_buffer_f32(
        &post_normed_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch post normed",
    )?;
    let mlp_gate_output = read_runtime_buffer_f32(
        &mlp_gate_output_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn layer batch MLP gate output",
    )?;
    let mlp_up_output = read_runtime_buffer_f32(
        &mlp_up_output_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn layer batch MLP up output",
    )?;
    let mlp_activation = read_runtime_buffer_f32(
        &mlp_activation_buffer,
        &mut stream,
        intermediate_elements,
        "linear-attn layer batch MLP activation",
    )?;
    let mlp_down_output = read_runtime_buffer_f32(
        &mlp_down_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch MLP down output",
    )?;
    let layer_output = read_runtime_buffer_f32(
        &layer_output_buffer,
        &mut stream,
        sequence_elements,
        "linear-attn layer batch layer output",
    )?;

    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_output,
        &b_output,
        &a_log.values,
        &dt_bias.values,
        value_heads,
        sequence_len,
    );
    if expected_gate.len() != gate_beta_elements || expected_beta.len() != gate_beta_elements {
        return Err("failed to build linear-attn layer batch gate/beta reference".to_string());
    }
    let gate_max_abs_diff = verify_f32_close(
        "linear-attn layer batch gate output",
        &gate_output,
        &expected_gate,
        1e-4_f32,
        1e-5_f32,
    )?;
    let beta_max_abs_diff = verify_f32_close(
        "linear-attn layer batch beta output",
        &beta_output,
        &expected_beta,
        1e-4_f32,
        1e-5_f32,
    )?;

    let mut expected_state = vec![0.0_f32; state_elements];
    let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
        &q_output,
        &k_output,
        &v_output,
        &gate_output,
        &beta_output,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_recurrent.len() != sequence_elements {
        return Err("failed to build linear-attn layer batch recurrent reference".to_string());
    }
    let recurrent_output_max_abs_diff = verify_f32_close(
        "linear-attn layer batch recurrent output",
        &recurrent_output,
        &expected_recurrent,
        2e-3_f32,
        2e-5_f32,
    )?;
    let recurrent_state_max_abs_diff = verify_f32_close(
        "linear-attn layer batch recurrent state",
        &final_state,
        &expected_state,
        2e-3_f32,
        2e-5_f32,
    )?;

    let mut expected_attention_normed = vec![0.0_f32; sequence_elements];
    for row in 0..attention_post_segments {
        let start = row.checked_mul(value_dim).ok_or_else(|| {
            "linear-attn layer batch expected attention norm start overflows".to_string()
        })?;
        let end = start.checked_add(value_dim).ok_or_else(|| {
            "linear-attn layer batch expected attention norm end overflows".to_string()
        })?;
        let normed = runtime_host_rmsnorm_f32(
            &recurrent_output[start..end],
            &norm.values,
            attention_post_epsilon,
        );
        if normed.len() != value_dim {
            return Err(
                "failed to build linear-attn layer batch attention post reference".to_string(),
            );
        }
        expected_attention_normed[start..end].copy_from_slice(&normed);
    }
    let expected_attention_post = runtime_host_silu_mul_f32(&z_output, &expected_attention_normed);
    if expected_attention_post.len() != sequence_elements {
        return Err(
            "failed to build linear-attn layer batch attention activation reference".to_string(),
        );
    }
    let attention_post_max_abs_diff = verify_f32_close(
        "linear-attn layer batch attention post output",
        &attn_projection_input,
        &expected_attention_post,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_attention_output = attn_projected
        .iter()
        .zip(embedding_rows.values.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let attention_residual_max_abs_diff = verify_f32_close(
        "linear-attn layer batch attention residual output",
        &attention_output,
        &expected_attention_output,
        1e-4_f32,
        1e-5_f32,
    )?;

    let mut expected_post_normed = Vec::with_capacity(sequence_elements);
    for token_index in 0..sequence_len {
        let start = token_index.checked_mul(hidden).ok_or_else(|| {
            "linear-attn layer batch expected post norm start overflows".to_string()
        })?;
        let end = start.checked_add(hidden).ok_or_else(|| {
            "linear-attn layer batch expected post norm end overflows".to_string()
        })?;
        expected_post_normed.extend(runtime_host_rmsnorm_f32(
            &attention_output[start..end],
            &post_norm.values,
            mlp_epsilon,
        ));
    }
    if expected_post_normed.len() != sequence_elements {
        return Err("failed to build linear-attn layer batch MLP norm reference".to_string());
    }
    let mlp_norm_max_abs_diff = verify_f32_close(
        "linear-attn layer batch MLP post norm",
        &post_normed,
        &expected_post_normed,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_mlp_activation = runtime_host_silu_mul_f32(&mlp_gate_output, &mlp_up_output);
    if expected_mlp_activation.len() != intermediate_elements {
        return Err("failed to build linear-attn layer batch MLP activation reference".to_string());
    }
    let mlp_activation_max_abs_diff = verify_f32_close(
        "linear-attn layer batch MLP activation",
        &mlp_activation,
        &expected_mlp_activation,
        1e-4_f32,
        1e-5_f32,
    )?;
    let expected_layer_output = mlp_down_output
        .iter()
        .zip(attention_output.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let layer_residual_max_abs_diff = verify_f32_close(
        "linear-attn layer batch layer residual output",
        &layer_output,
        &expected_layer_output,
        1e-4_f32,
        1e-5_f32,
    )?;
    let output_elements = qkv_output_elements
        .checked_add(sequence_elements)
        .and_then(|value| value.checked_add(gate_beta_elements))
        .and_then(|value| value.checked_add(gate_beta_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(intermediate_elements))
        .and_then(|value| value.checked_add(intermediate_elements))
        .and_then(|value| value.checked_add(intermediate_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .and_then(|value| value.checked_add(sequence_elements))
        .ok_or_else(|| "linear-attn layer batch output element count overflows".to_string())?;
    let preview_len = layer_output.len().min(8);
    Ok(format!(
        "package-linear-attn-layer-batch-smoke package={} layer={} tensors=[\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"] prompt_tokens={} hidden={} intermediate={} key_heads={} value_heads={} key_dim={} value_dim={} channels={} kernel_size={} q_scale={q_scale:.9} qk_l2_norm={} input_elements={} output_elements={} executor=segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32+segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32 real_batch=true token_parallelism={} request_parallelism=1 backend={} device_index={} name=\"{}\" warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} token_tps_mean={} output_element_tps_mean={} gate_max_abs_diff={gate_max_abs_diff:.9} beta_max_abs_diff={beta_max_abs_diff:.9} recurrent_output_max_abs_diff={recurrent_output_max_abs_diff:.9} recurrent_state_max_abs_diff={recurrent_state_max_abs_diff:.9} attention_post_max_abs_diff={attention_post_max_abs_diff:.9} attention_residual_max_abs_diff={attention_residual_max_abs_diff:.9} mlp_norm_max_abs_diff={mlp_norm_max_abs_diff:.9} mlp_activation_max_abs_diff={mlp_activation_max_abs_diff:.9} layer_residual_max_abs_diff={layer_residual_max_abs_diff:.9} layer_preview={} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        z_tensor,
        a_tensor,
        b_tensor,
        conv_tensor,
        a_log_tensor,
        dt_bias_tensor,
        norm_tensor,
        out_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        sequence_len,
        hidden,
        intermediate,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        channels,
        kernel_size,
        qk_l2_norm,
        sequence_elements,
        output_elements,
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
        format_f32_preview(&layer_output[..preview_len]),
    ))
}

fn current_git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        None
    } else {
        Some(commit.to_string())
    }
}

fn package_report_total_ms(report: &serde_json::Value, label: &str) -> Result<f64, String> {
    report
        .get("timing_ms")
        .and_then(|timing| timing.get("total"))
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{label} report has no timing_ms.total"))
}

fn package_report_top_token_id(report: &serde_json::Value, label: &str) -> Result<usize, String> {
    let token_id = report
        .get("top_logits")
        .and_then(serde_json::Value::as_array)
        .and_then(|entries| entries.first())
        .and_then(|entry| entry.get("token_id"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{label} report has no top_logits[0].token_id"))?;
    usize::try_from(token_id)
        .map_err(|_| format!("{label} top token id {token_id} is too large for this host"))
}

fn package_report_top_logits_json(
    report: &serde_json::Value,
    label: &str,
) -> Result<serde_json::Value, String> {
    report
        .get("top_logits")
        .cloned()
        .ok_or_else(|| format!("{label} report has no top_logits"))
}

fn tps(tokens: usize, wall_ms: f64) -> Option<f64> {
    if wall_ms > 0.0 {
        Some((tokens as f64) / (wall_ms / 1000.0))
    } else {
        None
    }
}

fn timed_step_tps(step_wall_ms: &[f64]) -> Option<f64> {
    tps(step_wall_ms.len(), step_wall_ms.iter().sum::<f64>())
}

fn timed_step_summary_json(step_wall_ms: &[f64]) -> serde_json::Value {
    let count = step_wall_ms.len();
    let wall_ms = step_wall_ms.iter().sum::<f64>();
    let mean_ms = if count > 0 {
        Some(wall_ms / count as f64)
    } else {
        None
    };
    let min_ms = step_wall_ms.iter().copied().reduce(f64::min);
    let max_ms = step_wall_ms.iter().copied().reduce(f64::max);
    let p50_ms = if count > 0 {
        let mut sorted = step_wall_ms.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        Some(sorted[count / 2])
    } else {
        None
    };
    let skip1 = if count > 1 { &step_wall_ms[1..] } else { &[] };
    let skip2 = if count > 2 { &step_wall_ms[2..] } else { &[] };
    let last4_start = count.saturating_sub(4);
    let last8_start = count.saturating_sub(8);
    let last4 = &step_wall_ms[last4_start..];
    let last8 = &step_wall_ms[last8_start..];

    serde_json::json!({
        "count": count,
        "wall_ms": wall_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "p50_ms": p50_ms,
        "max_ms": max_ms,
        "all_step_tps": timed_step_tps(step_wall_ms),
        "warmup_skip_1_step_count": skip1.len(),
        "warmup_skip_1_step_tps": timed_step_tps(skip1),
        "warmup_skip_2_step_count": skip2.len(),
        "warmup_skip_2_step_tps": timed_step_tps(skip2),
        "last_4_step_count": last4.len(),
        "last_4_step_tps": timed_step_tps(last4),
        "last_8_step_count": last8.len(),
        "last_8_step_tps": timed_step_tps(last8),
    })
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_generate_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids: Vec<usize>,
    generated_tokens: usize,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    lm_head_mode: PackageLmHeadMode,
    stop_token_ids: Vec<usize>,
    stop_token_sequences: Vec<Vec<usize>>,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    if env::var("ULLM_GENERATE_DECODE_MODE")
        .map(|value| value == "full_sequence_recompute_greedy")
        .unwrap_or(false)
    {
        if lm_head_mode != PackageLmHeadMode::CpuChunked {
            return Err(
                "gpu_resident_f32 lm head mode is only supported by incremental decode".to_string(),
            );
        }
        return package_token_ids_generate_recompute_smoke_impl(
            path,
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
            stop_token_ids,
            stop_token_sequences,
            sq_artifact,
        );
    }
    let sync_decode_layers_for_timing = env_flag_enabled("ULLM_SYNC_DECODE_LAYERS_FOR_TIMING");
    let sync_decode_each_layer_for_timing =
        env_flag_enabled("ULLM_SYNC_DECODE_EACH_LAYER_FOR_TIMING");
    package_token_ids_generate_incremental_smoke_impl(
        path,
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
        sync_decode_layers_for_timing,
        sync_decode_each_layer_for_timing,
        sq_artifact,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_generate_recompute_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids: Vec<usize>,
    generated_tokens: usize,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    stop_token_ids: Vec<usize>,
    stop_token_sequences: Vec<Vec<usize>>,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err(
            "package token-id generate smoke requires at least one prompt token".to_string(),
        );
    }
    if layer_indices.is_empty() {
        return Err("package token-id generate smoke requires at least one layer".to_string());
    }

    let run_started = Instant::now();
    let context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    drop(context);

    let mut sequence_ids = prompt_token_ids.clone();
    let prefill_report_text = package_token_ids_logits_smoke_impl_with_sq_overlay(
        path,
        device_index,
        chunk_bytes,
        layer_indices.clone(),
        sequence_ids.clone(),
        top_k,
        lm_head_chunk_rows,
        rotary_dim.clone(),
        rope_base,
        position_offset,
        sq_artifact,
    )?;
    let prefill_report = serde_json::from_str::<serde_json::Value>(&prefill_report_text)
        .map_err(|err| format!("failed to decode prefill logits report: {err}"))?;
    let prefill_ms = package_report_total_ms(&prefill_report, "prefill")?;
    let prefill_top_logits = package_report_top_logits_json(&prefill_report, "prefill")?;
    let resolved_rotary_dim = prefill_report
        .get("rotary_dim")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let layer_kinds = prefill_report
        .get("layer_kinds")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let hidden = prefill_report
        .get("hidden")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let mut generated_token_ids = Vec::with_capacity(generated_tokens);
    let mut decode_step_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_sequence_lengths = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut last_top_logits = prefill_top_logits.clone();
    let mut stopped_on_token_id = None;
    let mut stopped_on_token_sequence = None;

    if generated_tokens > 0 {
        let next = package_report_top_token_id(&prefill_report, "prefill")?;
        generated_token_ids.push(next);
        sequence_ids.push(next);
        if stop_token_ids.contains(&next) {
            stopped_on_token_id = Some(next);
        } else if let Some(sequence) =
            matched_stop_token_sequence(&generated_token_ids, &stop_token_sequences)
        {
            stopped_on_token_sequence = Some(sequence);
        }
    }

    while generated_token_ids.len() < generated_tokens
        && stopped_on_token_id.is_none()
        && stopped_on_token_sequence.is_none()
    {
        let step_started = Instant::now();
        let report_text = package_token_ids_logits_smoke_impl_with_sq_overlay(
            path,
            device_index,
            chunk_bytes,
            layer_indices.clone(),
            sequence_ids.clone(),
            top_k,
            lm_head_chunk_rows,
            rotary_dim.clone(),
            rope_base,
            position_offset,
            sq_artifact,
        )?;
        let external_step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
        let report = serde_json::from_str::<serde_json::Value>(&report_text)
            .map_err(|err| format!("failed to decode decode-step logits report: {err}"))?;
        let report_ms = package_report_total_ms(&report, "decode step")?;
        let next = package_report_top_token_id(&report, "decode step")?;
        last_top_logits = package_report_top_logits_json(&report, "decode step")?;
        decode_step_ms.push(report_ms.max(external_step_ms));
        decode_sequence_lengths.push(sequence_ids.len());
        generated_token_ids.push(next);
        sequence_ids.push(next);
        if stop_token_ids.contains(&next) {
            stopped_on_token_id = Some(next);
        } else if let Some(sequence) =
            matched_stop_token_sequence(&generated_token_ids, &stop_token_sequences)
        {
            stopped_on_token_sequence = Some(sequence);
        }
    }

    let decode_ms = if decode_step_ms.is_empty() {
        0.0
    } else {
        decode_step_ms.iter().sum::<f64>()
    };
    let total_ms = run_started.elapsed().as_secs_f64() * 1000.0;
    let timed_decode_tokens = decode_step_ms.len();
    let decode_step_summary = timed_step_summary_json(&decode_step_ms);
    let prompt_token_count = prompt_token_ids.len();
    let full_forward_tokens = prompt_token_ids
        .len()
        .checked_add(decode_sequence_lengths.iter().sum::<usize>())
        .ok_or_else(|| "full-sequence token count overflows".to_string())?;
    let kv_cache_bytes = 0_u64;
    let stop_reason = if stopped_on_token_id.is_some() {
        "stop_token"
    } else if stopped_on_token_sequence.is_some() {
        "stop_sequence"
    } else {
        "max_generated_tokens"
    };
    let stopped = stopped_on_token_id.is_some() || stopped_on_token_sequence.is_some();
    let report = serde_json::json!({
        "schema_version": "package-token-ids-generate-smoke-v0.1",
        "package": path,
        "git_commit": current_git_commit(),
        "backend": info.backend.to_string(),
        "device_index": device_index,
        "device_name": info.name,
        "device_total_global_mem": info.total_global_mem,
        "layers": layer_indices,
        "layer_kinds": layer_kinds,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "final_sequence_len": sequence_ids.len(),
        "hidden": hidden,
        "top_k": top_k,
        "lm_head_chunk_rows": lm_head_chunk_rows,
        "rotary_dim": resolved_rotary_dim,
        "rope_base": rope_base,
        "position_offset": position_offset,
        "stop": {
            "token_ids": stop_token_ids,
            "token_sequences": stop_token_sequences,
            "stopped": stopped,
            "stopped_on_token_id": stopped_on_token_id,
            "stopped_on_token_sequence": stopped_on_token_sequence,
            "reason": stop_reason,
        },
        "decode_mode": "full_sequence_recompute_greedy",
        "incremental_decode": false,
        "prefill": {
            "prompt_tokens": prompt_token_count,
            "wall_ms": prefill_ms,
            "tps": tps(prompt_token_count, prefill_ms),
            "top_logits": prefill_top_logits,
        },
        "decode": {
            "requested_generated_tokens": generated_tokens,
            "timed_recompute_steps": timed_decode_tokens,
            "sequence_lengths": decode_sequence_lengths,
            "step_wall_ms": decode_step_ms,
            "step_wall_summary": decode_step_summary,
            "wall_ms": decode_ms,
            "timed_step_tps": tps(timed_decode_tokens, decode_ms),
            "end_to_end_generated_tps": tps(generated_tokens, total_ms),
            "last_top_logits": last_top_logits,
        },
        "throughput": {
            "full_forward_tokens": full_forward_tokens,
            "full_forward_tps": tps(full_forward_tokens, prefill_ms + decode_ms),
            "total_wall_ms": total_ms,
        },
        "memory": {
            "vram_baseline_bytes": serde_json::Value::Null,
            "vram_peak_bytes": serde_json::Value::Null,
            "vram_consumed_bytes": serde_json::Value::Null,
            "kv_cache_bytes": kv_cache_bytes,
        },
        "correctness": {
            "verified": true,
            "nan_or_inf_detected": false,
        },
        "notes": [
            "This smoke uses full-sequence recompute for generated tokens; it is a T2 entry point, not the final incremental decode TPS path."
        ],
        "verified": true,
    });
    serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to encode token-id generate smoke report: {err}"))
}

enum PackageLmHeadRuntime {
    CpuChunked {
        chunk_rows: usize,
    },
    GpuResidentAq4 {
        shape: Vec<u64>,
        vocab: usize,
        hidden: usize,
        matrix: PackageAq4ResidentMatvec,
        input_buffer: ullm_runtime_sys::RuntimeBuffer,
        logits_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_count: usize,
        top1_partial_values_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_indices_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_host: Vec<u8>,
        top1_partial_indices_host: Vec<u8>,
        logits_host: Vec<u8>,
    },
    GpuResidentF32 {
        dtype: String,
        shape: Vec<u64>,
        vocab: usize,
        hidden: usize,
        matrix_storage: PackageLmHeadMatrixStorage,
        top1_partial_count: usize,
        matrix_buffer: ullm_runtime_sys::RuntimeBuffer,
        input_buffer: ullm_runtime_sys::RuntimeBuffer,
        logits_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_indices_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_host: Vec<u8>,
        top1_partial_indices_host: Vec<u8>,
        logits_host: Vec<u8>,
        matrix_bytes: usize,
    },
}

impl PackageLmHeadRuntime {
    fn load(
        mode: PackageLmHeadMode,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        hidden: usize,
        chunk_rows: usize,
    ) -> Result<Self, String> {
        match mode {
            PackageLmHeadMode::CpuChunked => Ok(Self::CpuChunked { chunk_rows }),
            PackageLmHeadMode::GpuResidentF32 => {
                let selector = TensorSelector::Name(QWEN3_LM_HEAD_TENSOR.to_string());
                if select_tensor_payload_bundle(path, &selector).is_ok() {
                    let mut registry = WeightRegistry::new();
                    let matrix = PackageAq4ResidentMatvec::load(
                        context,
                        stream,
                        &mut registry,
                        path,
                        QWEN3_LM_HEAD_TENSOR,
                        chunk_bytes,
                    )
                    .map_err(|err| format!("failed to load resident AQ4 lm_head tensor: {err}"))?;
                    let vocab = matrix.rows;
                    let cols = matrix.cols;
                    if cols != hidden {
                        return Err(format!(
                            "resident AQ4 lm_head hidden mismatch: lm_head={cols} hidden={hidden}"
                        ));
                    }
                    let shape = vec![
                        u64::try_from(vocab)
                            .map_err(|_| "resident AQ4 lm_head vocab exceeds u64".to_string())?,
                        u64::try_from(hidden)
                            .map_err(|_| "resident AQ4 lm_head hidden exceeds u64".to_string())?,
                    ];
                    let hidden_bytes = checked_f32_byte_len(hidden, "resident AQ4 lm_head input")?;
                    let logits_bytes = checked_f32_byte_len(vocab, "resident AQ4 lm_head logits")?;
                    let logits_top1_partial_count = ullm_runtime_sys::top1_partial_count(vocab)
                        .map_err(|err| {
                            format!("failed to size resident AQ4 lm_head top1: {err}")
                        })?;
                    let aq4_direct_top1_partial_count =
                        ullm_runtime_sys::aq4_matvec_top1_partial_count(vocab).map_err(|err| {
                            format!("failed to size resident AQ4 lm_head direct top1: {err}")
                        })?;
                    let top1_partial_count =
                        logits_top1_partial_count.max(aq4_direct_top1_partial_count);
                    let top1_partial_values_bytes = checked_f32_byte_len(
                        top1_partial_count,
                        "resident AQ4 lm_head top1 values",
                    )?;
                    let top1_partial_indices_bytes = top1_partial_count
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| {
                            "resident AQ4 lm_head top1 index byte size overflows".to_string()
                        })?;
                    let mut input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
                        format!("failed to allocate resident AQ4 lm_head input: {err}")
                    })?;
                    let mut logits_buffer = context.alloc_buffer(logits_bytes).map_err(|err| {
                        format!("failed to allocate resident AQ4 lm_head logits: {err}")
                    })?;
                    let mut top1_partial_values_buffer = context
                        .alloc_buffer(top1_partial_values_bytes)
                        .map_err(|err| {
                            format!("failed to allocate resident AQ4 lm_head top1 values: {err}")
                        })?;
                    let mut top1_partial_indices_buffer = context
                        .alloc_buffer(top1_partial_indices_bytes)
                        .map_err(|err| {
                            format!("failed to allocate resident AQ4 lm_head top1 indices: {err}")
                        })?;
                    stream.synchronize().map_err(|err| {
                        format!("failed to synchronize resident AQ4 lm_head load: {err}")
                    })?;
                    let mut top1_partial_values_host = vec![0_u8; top1_partial_values_bytes];
                    let mut top1_partial_indices_host = vec![0_u8; top1_partial_indices_bytes];
                    let mut logits_host = vec![0_u8; logits_bytes];
                    let zero_hidden_values = vec![0.0_f32; hidden];
                    input_buffer
                        .copy_from_host(0, &encode_f32_to_bytes(&zero_hidden_values), Some(stream))
                        .map_err(|err| {
                            format!("failed to copy resident AQ4 lm_head prewarm input: {err}")
                        })?;
                    package_gpu_resident_aq4_lm_head_top_logits(
                        stream,
                        &matrix,
                        &input_buffer,
                        vocab,
                        hidden,
                        &mut logits_buffer,
                        &mut top1_partial_values_buffer,
                        &mut top1_partial_indices_buffer,
                        &mut top1_partial_values_host,
                        &mut top1_partial_indices_host,
                        &mut logits_host,
                        1,
                    )
                    .map_err(|err| format!("failed to prewarm resident AQ4 lm_head: {err}"))?;
                    return Ok(Self::GpuResidentAq4 {
                        shape,
                        vocab,
                        hidden,
                        matrix,
                        input_buffer,
                        logits_buffer,
                        top1_partial_count,
                        top1_partial_values_buffer,
                        top1_partial_indices_buffer,
                        top1_partial_values_host,
                        top1_partial_indices_host,
                        logits_host,
                    });
                }
                let bundle = select_passthrough_payload_bundle(path, &selector)
                    .map_err(|err| format!("failed to select resident lm_head tensor: {err}"))?;
                validate_passthrough_shape_elements(&bundle)
                    .map_err(|err| format!("invalid resident lm_head shape: {err}"))?;
                let dtype = resolve_passthrough_dtype(&bundle, QWEN3_LM_HEAD_TENSOR)?.to_string();
                if bundle.shape.len() != 2 {
                    return Err(format!(
                        "resident lm_head must be 2D, got shape {:?}",
                        bundle.shape
                    ));
                }
                let vocab = usize::try_from(bundle.shape[0])
                    .map_err(|_| "resident lm_head vocab size is too large".to_string())?;
                let cols = usize::try_from(bundle.shape[1])
                    .map_err(|_| "resident lm_head hidden size is too large".to_string())?;
                if cols != hidden {
                    return Err(format!(
                        "resident lm_head hidden mismatch: lm_head={cols} hidden={hidden}"
                    ));
                }
                let expected_values = vocab
                    .checked_mul(hidden)
                    .ok_or_else(|| "resident lm_head element count overflows".to_string())?;
                if u64::try_from(expected_values).ok() != Some(bundle.elements) {
                    return Err(format!(
                        "resident lm_head element count mismatch: got {} expected {expected_values}",
                        bundle.elements
                    ));
                }
                let matrix_storage = match dtype.as_str() {
                    "BF16" => PackageLmHeadMatrixStorage::Bf16,
                    _ => PackageLmHeadMatrixStorage::F32,
                };
                let matrix_bytes = expected_values
                    .checked_mul(matrix_storage.element_size())
                    .ok_or_else(|| "resident lm_head matrix byte size overflows".to_string())?;
                let hidden_bytes = checked_f32_byte_len(hidden, "resident lm_head input")?;
                let logits_bytes = checked_f32_byte_len(vocab, "resident lm_head logits")?;
                let top1_partial_count = ullm_runtime_sys::top1_partial_count(vocab)
                    .map_err(|err| format!("failed to size resident lm_head top1: {err}"))?;
                let top1_partial_values_bytes =
                    checked_f32_byte_len(top1_partial_count, "resident lm_head top1 values")?;
                let top1_partial_indices_bytes = top1_partial_count
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| "resident lm_head top1 index byte size overflows".to_string())?;
                let mut matrix_buffer = context
                    .alloc_buffer(matrix_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head matrix: {err}"))?;
                match matrix_storage {
                    PackageLmHeadMatrixStorage::Bf16 => {
                        let payload_bytes = if bundle.payload_bytes == 0 {
                            bundle.payload_file.bytes
                        } else {
                            bundle.payload_bytes
                        };
                        if payload_bytes != bundle.payload_file.bytes {
                            return Err(format!(
                                "resident lm_head payload bytes mismatch: declared {} actual {}",
                                payload_bytes, bundle.payload_file.bytes
                            ));
                        }
                        if usize::try_from(payload_bytes).ok() != Some(matrix_bytes) {
                            return Err(format!(
                                "resident lm_head BF16 payload bytes mismatch: got {payload_bytes} expected {matrix_bytes}"
                            ));
                        }
                        match bundle.payload_encoding.as_deref() {
                            None | Some("raw_safetensors_payload") => {}
                            Some(encoding) => {
                                return Err(format!(
                                    "resident lm_head has unsupported payload encoding {encoding}"
                                ));
                            }
                        }
                        copy_file_to_runtime_buffer_chunked(
                            &mut matrix_buffer,
                            &bundle.payload_file.absolute_path,
                            matrix_bytes,
                            chunk_bytes,
                            stream,
                            "resident lm_head BF16 matrix",
                        )?;
                    }
                    PackageLmHeadMatrixStorage::F32 => {
                        let data =
                            read_named_passthrough_f32(path, QWEN3_LM_HEAD_TENSOR, chunk_bytes)
                                .map_err(|err| {
                                    format!("failed to read resident lm_head tensor: {err}")
                                })?;
                        if data.values.len() != expected_values {
                            return Err(format!(
                                "resident lm_head value count mismatch: got {} expected {expected_values}",
                                data.values.len()
                            ));
                        }
                        copy_f32_values_to_runtime_buffer_chunked(
                            &mut matrix_buffer,
                            &data.values,
                            stream,
                            "resident lm_head matrix",
                        )?;
                    }
                }
                let mut input_buffer = context
                    .alloc_buffer(hidden_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head input: {err}"))?;
                let mut logits_buffer = context
                    .alloc_buffer(logits_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head logits: {err}"))?;
                let mut top1_partial_values_buffer = context
                    .alloc_buffer(top1_partial_values_bytes)
                    .map_err(|err| {
                        format!("failed to allocate resident lm_head top1 values: {err}")
                    })?;
                let mut top1_partial_indices_buffer = context
                    .alloc_buffer(top1_partial_indices_bytes)
                    .map_err(|err| {
                        format!("failed to allocate resident lm_head top1 indices: {err}")
                    })?;
                stream
                    .synchronize()
                    .map_err(|err| format!("failed to synchronize resident lm_head load: {err}"))?;
                let mut top1_partial_values_host = vec![0_u8; top1_partial_values_bytes];
                let mut top1_partial_indices_host = vec![0_u8; top1_partial_indices_bytes];
                let mut logits_host = vec![0_u8; logits_bytes];
                let zero_hidden_values = vec![0.0_f32; hidden];
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(&zero_hidden_values), Some(stream))
                    .map_err(|err| {
                        format!("failed to copy resident lm_head prewarm input: {err}")
                    })?;
                package_gpu_resident_lm_head_top_logits(
                    stream,
                    &matrix_buffer,
                    &input_buffer,
                    matrix_storage,
                    vocab,
                    hidden,
                    &mut logits_buffer,
                    &mut top1_partial_values_buffer,
                    &mut top1_partial_indices_buffer,
                    &mut top1_partial_values_host,
                    &mut top1_partial_indices_host,
                    &mut logits_host,
                    1,
                )
                .map_err(|err| format!("failed to prewarm resident lm_head: {err}"))?;
                Ok(Self::GpuResidentF32 {
                    dtype,
                    shape: bundle.shape,
                    vocab,
                    hidden,
                    matrix_storage,
                    top1_partial_count,
                    matrix_buffer,
                    input_buffer,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    matrix_bytes,
                })
            }
        }
    }

    fn top_logits(
        &mut self,
        path: &str,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        hidden_values: &[f32],
        top_k: usize,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        match self {
            Self::CpuChunked { chunk_rows } => {
                let (_vocab, _dtype, _shape, top_logits) =
                    package_lm_head_top_k_from_rows(path, hidden_values, top_k, *chunk_rows)?;
                Ok(top_logits)
            }
            Self::GpuResidentAq4 {
                vocab,
                hidden,
                matrix,
                input_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => {
                if hidden_values.len() != *hidden {
                    return Err(format!(
                        "resident AQ4 lm_head input length mismatch: got {} expected {}",
                        hidden_values.len(),
                        hidden
                    ));
                }
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(hidden_values), Some(stream))
                    .map_err(|err| format!("failed to copy resident AQ4 lm_head input: {err}"))?;
                package_gpu_resident_aq4_lm_head_top_logits(
                    stream,
                    matrix,
                    input_buffer,
                    *vocab,
                    *hidden,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    top_k,
                )
            }
            Self::GpuResidentF32 {
                vocab,
                hidden,
                matrix_storage,
                matrix_buffer,
                input_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => {
                if hidden_values.len() != *hidden {
                    return Err(format!(
                        "resident lm_head input length mismatch: got {} expected {}",
                        hidden_values.len(),
                        hidden
                    ));
                }
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(hidden_values), Some(stream))
                    .map_err(|err| format!("failed to copy resident lm_head input: {err}"))?;
                package_gpu_resident_lm_head_top_logits(
                    stream,
                    matrix_buffer,
                    input_buffer,
                    *matrix_storage,
                    *vocab,
                    *hidden,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    top_k,
                )
            }
        }
    }

    fn supports_device_input(&self) -> bool {
        matches!(
            self,
            Self::GpuResidentAq4 { .. } | Self::GpuResidentF32 { .. }
        )
    }

    fn top_logits_from_device_buffer(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        hidden_buffer: &ullm_runtime_sys::RuntimeBuffer,
        top_k: usize,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        match self {
            Self::CpuChunked { .. } => {
                Err("device lm_head input requires gpu_resident_f32 lm_head mode".to_string())
            }
            Self::GpuResidentAq4 {
                vocab,
                hidden,
                matrix,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => package_gpu_resident_aq4_lm_head_top_logits(
                stream,
                matrix,
                hidden_buffer,
                *vocab,
                *hidden,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                top_k,
            ),
            Self::GpuResidentF32 {
                vocab,
                hidden,
                matrix_storage,
                matrix_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => package_gpu_resident_lm_head_top_logits(
                stream,
                matrix_buffer,
                hidden_buffer,
                *matrix_storage,
                *vocab,
                *hidden,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                top_k,
            ),
        }
    }

    fn report_json(&self, load_ms: f64) -> serde_json::Value {
        match self {
            Self::CpuChunked { chunk_rows } => serde_json::json!({
                "mode": PackageLmHeadMode::CpuChunked.as_str(),
                "chunk_rows": chunk_rows,
                "load_ms": load_ms,
            }),
            Self::GpuResidentAq4 {
                shape,
                vocab,
                hidden,
                matrix,
                top1_partial_count,
                ..
            } => serde_json::json!({
                "mode": PackageLmHeadMode::GpuResidentF32.as_str(),
                "tensor": QWEN3_LM_HEAD_TENSOR,
                "dtype": "AQ4",
                "shape": shape,
                "vocab": vocab,
                "hidden": hidden,
                "matrix_storage_dtype": "AQ4",
                "group_size": matrix.group_size,
                "scale_count": matrix.scale_count,
                "tensor_scale": matrix.tensor_scale,
                "top1_partial_count": top1_partial_count,
                "prewarmed_top1": true,
                "load_ms": load_ms,
            }),
            Self::GpuResidentF32 {
                dtype,
                shape,
                vocab,
                hidden,
                matrix_storage,
                top1_partial_count,
                matrix_bytes,
                ..
            } => serde_json::json!({
                "mode": PackageLmHeadMode::GpuResidentF32.as_str(),
                "tensor": QWEN3_LM_HEAD_TENSOR,
                "dtype": dtype,
                "shape": shape,
                "vocab": vocab,
                "hidden": hidden,
                "matrix_storage_dtype": matrix_storage.as_str(),
                "top1_partial_count": top1_partial_count,
                "matrix_bytes": matrix_bytes,
                "prewarmed_top1": true,
                "load_ms": load_ms,
            }),
        }
    }
}

struct PackageFinalNormRuntime {
    hidden: usize,
    weight_buffer: ullm_runtime_sys::RuntimeBuffer,
    output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

enum PackageEmbeddingStorage {
    Bf16 {
        matrix_buffer: ullm_runtime_sys::RuntimeBuffer,
        matrix_bytes: usize,
    },
    Aq4 {
        matrix: PackageAq4ResidentMatvec,
    },
}

struct PackageEmbeddingRuntime {
    dtype: String,
    shape: Vec<u64>,
    vocab: usize,
    hidden: usize,
    storage: PackageEmbeddingStorage,
    output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

impl PackageEmbeddingRuntime {
    fn load_if_available(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        hidden: usize,
    ) -> Result<Option<Self>, String> {
        let selector = TensorSelector::Name(QWEN3_EMBED_TOKENS_TENSOR.to_string());
        if select_tensor_payload_bundle(path, &selector).is_ok() {
            let mut registry = WeightRegistry::new();
            let matrix = PackageAq4ResidentMatvec::load(
                context,
                stream,
                &mut registry,
                path,
                QWEN3_EMBED_TOKENS_TENSOR,
                chunk_bytes,
            )
            .map_err(|err| format!("failed to load resident AQ4 embedding tensor: {err}"))?;
            let vocab = matrix.rows;
            let cols = matrix.cols;
            if cols != hidden {
                return Err(format!(
                    "resident AQ4 embedding hidden mismatch: embedding={cols} hidden={hidden}"
                ));
            }
            let shape = vec![
                u64::try_from(vocab)
                    .map_err(|_| "resident AQ4 embedding vocab exceeds u64".to_string())?,
                u64::try_from(hidden)
                    .map_err(|_| "resident AQ4 embedding hidden exceeds u64".to_string())?,
            ];
            let mut output_buffer = context
                .alloc_buffer(checked_f32_byte_len(
                    hidden,
                    "resident AQ4 embedding output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate resident AQ4 embedding output: {err}")
                })?;
            matrix.row_f32(
                0,
                &mut output_buffer,
                stream,
                "resident AQ4 embedding prewarm",
            )?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize resident AQ4 embedding load: {err}")
            })?;
            return Ok(Some(Self {
                dtype: "AQ4".to_string(),
                shape,
                vocab,
                hidden,
                storage: PackageEmbeddingStorage::Aq4 { matrix },
                output_buffer,
            }));
        }

        let bundle = select_passthrough_payload_bundle(path, &selector)
            .map_err(|err| format!("failed to select resident embedding tensor: {err}"))?;
        validate_passthrough_shape_elements(&bundle)
            .map_err(|err| format!("invalid resident embedding shape: {err}"))?;
        let dtype = resolve_passthrough_dtype(&bundle, QWEN3_EMBED_TOKENS_TENSOR)?.to_string();
        if dtype != "BF16" {
            return Ok(None);
        }
        if bundle.shape.len() != 2 {
            return Err(format!(
                "resident embedding must be 2D, got shape {:?}",
                bundle.shape
            ));
        }
        let vocab = usize::try_from(bundle.shape[0])
            .map_err(|_| "resident embedding vocab size is too large".to_string())?;
        let cols = usize::try_from(bundle.shape[1])
            .map_err(|_| "resident embedding hidden size is too large".to_string())?;
        if cols != hidden {
            return Err(format!(
                "resident embedding hidden mismatch: embedding={cols} hidden={hidden}"
            ));
        }
        let expected_values = vocab
            .checked_mul(hidden)
            .ok_or_else(|| "resident embedding element count overflows".to_string())?;
        if u64::try_from(expected_values).ok() != Some(bundle.elements) {
            return Err(format!(
                "resident embedding element count mismatch: got {} expected {expected_values}",
                bundle.elements
            ));
        }
        let matrix_bytes = expected_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "resident embedding matrix byte size overflows".to_string())?;
        let payload_bytes = if bundle.payload_bytes == 0 {
            bundle.payload_file.bytes
        } else {
            bundle.payload_bytes
        };
        if payload_bytes != bundle.payload_file.bytes {
            return Err(format!(
                "resident embedding payload bytes mismatch: declared {} actual {}",
                payload_bytes, bundle.payload_file.bytes
            ));
        }
        if usize::try_from(payload_bytes).ok() != Some(matrix_bytes) {
            return Err(format!(
                "resident embedding BF16 payload bytes mismatch: got {payload_bytes} expected {matrix_bytes}"
            ));
        }
        match bundle.payload_encoding.as_deref() {
            None | Some("raw_safetensors_payload") => {}
            Some(encoding) => {
                return Err(format!(
                    "resident embedding has unsupported payload encoding {encoding}"
                ));
            }
        }

        let mut matrix_buffer = context
            .alloc_buffer(matrix_bytes)
            .map_err(|err| format!("failed to allocate resident embedding matrix: {err}"))?;
        copy_file_to_runtime_buffer_chunked(
            &mut matrix_buffer,
            &bundle.payload_file.absolute_path,
            matrix_bytes,
            chunk_bytes,
            stream,
            "resident embedding BF16 matrix",
        )?;
        let mut output_buffer = context
            .alloc_buffer(checked_f32_byte_len(hidden, "resident embedding output")?)
            .map_err(|err| format!("failed to allocate resident embedding output: {err}"))?;
        ullm_runtime_sys::bf16_row_f32(
            &matrix_buffer,
            vocab,
            hidden,
            0,
            &mut output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to prewarm resident embedding row gather: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize resident embedding load: {err}"))?;
        Ok(Some(Self {
            dtype,
            shape: bundle.shape,
            vocab,
            hidden,
            storage: PackageEmbeddingStorage::Bf16 {
                matrix_buffer,
                matrix_bytes,
            },
            output_buffer,
        }))
    }

    fn gather_token(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        label: &str,
    ) -> Result<(), String> {
        if token_id >= self.vocab {
            return Err(format!(
                "{label} token id {token_id} is out of resident embedding range 0..{}",
                self.vocab
            ));
        }
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_buffer, .. } => ullm_runtime_sys::bf16_row_f32(
                matrix_buffer,
                self.vocab,
                self.hidden,
                token_id,
                &mut self.output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to gather {label} resident BF16 embedding row: {err}")),
            PackageEmbeddingStorage::Aq4 { matrix } => matrix.row_f32(
                token_id,
                &mut self.output_buffer,
                stream,
                &format!("{label} resident AQ4 embedding"),
            ),
        }
    }

    fn gather_token_to_buffer(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        let required_bytes = checked_f32_byte_len(self.hidden, "mixed request-state embedding output")?;
        let actual_bytes = output_buffer.size().map_err(|err| {
            format!("failed to query {label} mixed request-state embedding output buffer size: {err}")
        })?;
        if actual_bytes < required_bytes {
            return Err(format!(
                "{label} mixed request-state embedding output buffer is too small: got {actual_bytes} bytes expected at least {required_bytes}"
            ));
        }
        if token_id >= self.vocab {
            return Err(format!(
                "{label} token id {token_id} is out of resident embedding range 0..{}",
                self.vocab
            ));
        }
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_buffer, .. } => ullm_runtime_sys::bf16_row_f32(
                matrix_buffer,
                self.vocab,
                self.hidden,
                token_id,
                output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to gather {label} resident BF16 embedding row: {err}")),
            PackageEmbeddingStorage::Aq4 { matrix } => matrix.row_f32(
                token_id,
                output_buffer,
                stream,
                &format!("{label} resident AQ4 embedding"),
            ),
        }
    }

    fn gather_token_values(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        self.gather_token(stream, token_id, label)?;
        read_runtime_buffer_f32(&self.output_buffer, stream, self.hidden, label)
    }

    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.output_buffer
    }

    fn report_json(&self) -> serde_json::Value {
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_bytes, .. } => serde_json::json!({
                "mode": "gpu_resident_bf16",
                "tensor": QWEN3_EMBED_TOKENS_TENSOR,
                "dtype": self.dtype,
                "shape": self.shape,
                "vocab": self.vocab,
                "hidden": self.hidden,
                "matrix_bytes": matrix_bytes,
            }),
            PackageEmbeddingStorage::Aq4 { matrix } => serde_json::json!({
                "mode": "gpu_resident_aq4",
                "tensor": QWEN3_EMBED_TOKENS_TENSOR,
                "dtype": self.dtype,
                "shape": self.shape,
                "vocab": self.vocab,
                "hidden": self.hidden,
                "group_size": matrix.group_size,
                "scale_count": matrix.scale_count,
                "tensor_scale": matrix.tensor_scale,
            }),
        }
    }
}

fn package_embedding_shape(path: &str) -> Result<(usize, usize), String> {
    let selector = TensorSelector::Name(QWEN3_EMBED_TOKENS_TENSOR.to_string());
    if let Ok(bundle) = select_tensor_payload_bundle(path, &selector) {
        let elements = usize::try_from(bundle.elements)
            .map_err(|_| "resident AQ4 embedding element count exceeds usize".to_string())?;
        return matrix_shape_rows_cols(&bundle.shape, elements)
            .map_err(|err| format!("invalid resident AQ4 embedding shape: {err}"));
    }

    let bundle = select_passthrough_payload_bundle(path, &selector)
        .map_err(|err| format!("failed to select resident embedding tensor: {err}"))?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid resident embedding shape: {err}"))?;
    if bundle.shape.len() != 2 {
        return Err(format!(
            "resident embedding must be 2D, got shape {:?}",
            bundle.shape
        ));
    }
    let rows = usize::try_from(bundle.shape[0])
        .map_err(|_| "resident embedding vocab size is too large".to_string())?;
    let cols = usize::try_from(bundle.shape[1])
        .map_err(|_| "resident embedding hidden size is too large".to_string())?;
    Ok((rows, cols))
}

impl PackageFinalNormRuntime {
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        final_norm: &PassthroughF32Data,
        hidden: usize,
    ) -> Result<Self, String> {
        if final_norm.values.len() != hidden {
            return Err(format!(
                "incremental final RMSNorm length mismatch: len={} hidden={hidden}",
                final_norm.values.len()
            ));
        }
        let norm_bytes = checked_f32_byte_len(hidden, "incremental final RMSNorm")?;
        let mut weight_buffer = context
            .alloc_buffer(norm_bytes)
            .map_err(|err| format!("failed to allocate incremental final RMSNorm weight: {err}"))?;
        let output_buffer = context
            .alloc_buffer(norm_bytes)
            .map_err(|err| format!("failed to allocate incremental final RMSNorm output: {err}"))?;
        weight_buffer
            .copy_from_host(0, &encode_f32_to_bytes(&final_norm.values), Some(stream))
            .map_err(|err| format!("failed to copy incremental final RMSNorm weight: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize incremental final RMSNorm runtime setup: {err}")
        })?;
        Ok(Self {
            hidden,
            weight_buffer,
            output_buffer,
        })
    }

    fn normalize_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        let input_bytes = input_buffer
            .size()
            .map_err(|err| format!("failed to query {label} final hidden buffer size: {err}"))?;
        let required_bytes = checked_f32_byte_len(self.hidden, "incremental final RMSNorm input")?;
        if input_bytes < required_bytes {
            return Err(format!(
                "{label} final hidden buffer is too small: got {input_bytes} bytes expected at least {required_bytes}"
            ));
        }
        ullm_runtime_sys::rmsnorm_f32(
            input_buffer,
            &self.weight_buffer,
            self.hidden,
            1e-6_f32,
            &mut self.output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} final RMSNorm: {err}"))
    }

    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.output_buffer
    }

    fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": "gpu_resident_f32",
            "hidden": self.hidden,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn package_gpu_resident_lm_head_top_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix_buffer: &ullm_runtime_sys::RuntimeBuffer,
    input_buffer: &ullm_runtime_sys::RuntimeBuffer,
    matrix_storage: PackageLmHeadMatrixStorage,
    vocab: usize,
    hidden: usize,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    let required_input_bytes = checked_f32_byte_len(hidden, "resident lm_head input")?;
    let input_bytes = input_buffer
        .size()
        .map_err(|err| format!("failed to query resident lm_head input buffer size: {err}"))?;
    if input_bytes < required_input_bytes {
        return Err(format!(
            "resident lm_head input buffer is too small: got {input_bytes} bytes expected at least {required_input_bytes}"
        ));
    }
    match matrix_storage {
        PackageLmHeadMatrixStorage::F32 => ullm_runtime_sys::matvec_f32(
            matrix_buffer,
            input_buffer,
            vocab,
            hidden,
            logits_buffer,
            Some(stream),
        ),
        PackageLmHeadMatrixStorage::Bf16 => ullm_runtime_sys::matvec_bf16_f32(
            matrix_buffer,
            input_buffer,
            vocab,
            hidden,
            logits_buffer,
            Some(stream),
        ),
    }
    .map_err(|err| format!("resident lm_head matvec failed: {err}"))?;
    package_resident_lm_head_top_logits_from_logits(
        stream,
        logits_buffer,
        vocab,
        top1_partial_values_buffer,
        top1_partial_indices_buffer,
        top1_partial_values_host,
        top1_partial_indices_host,
        logits_host,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_gpu_resident_aq4_lm_head_top_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &PackageAq4ResidentMatvec,
    input_buffer: &ullm_runtime_sys::RuntimeBuffer,
    vocab: usize,
    hidden: usize,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    if matrix.rows != vocab || matrix.cols != hidden {
        return Err(format!(
            "resident AQ4 lm_head shape mismatch: matrix=[{},{}] expected=[{vocab},{hidden}]",
            matrix.rows, matrix.cols
        ));
    }
    let required_input_bytes = checked_f32_byte_len(hidden, "resident AQ4 lm_head input")?;
    let input_bytes = input_buffer
        .size()
        .map_err(|err| format!("failed to query resident AQ4 lm_head input buffer size: {err}"))?;
    if input_bytes < required_input_bytes {
        return Err(format!(
            "resident AQ4 lm_head input buffer is too small: got {input_bytes} bytes expected at least {required_input_bytes}"
        ));
    }
    if top_k == 1 && env_flag_enabled("ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1") {
        let first_stage_partial_count = matrix.matvec_top1(
            input_buffer,
            top1_partial_values_buffer,
            top1_partial_indices_buffer,
            stream,
            "resident AQ4 lm_head",
        )?;
        let partial_count = if first_stage_partial_count > 1 {
            ullm_runtime_sys::top1_pairs_f32_in_place(
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                first_stage_partial_count,
                Some(stream),
            )
            .map_err(|err| format!("resident AQ4 lm_head direct top1 pair reduce failed: {err}"))?
        } else {
            first_stage_partial_count
        };
        let partial_values_bytes = checked_f32_byte_len(
            partial_count,
            "resident AQ4 lm_head direct top1 partial values",
        )?;
        let partial_indices_bytes = partial_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                "resident AQ4 lm_head direct top1 partial index byte size overflows".to_string()
            })?;
        if top1_partial_values_host.len() < partial_values_bytes {
            return Err(format!(
                "resident AQ4 lm_head direct top1 value host buffer is too small: got {} bytes expected at least {partial_values_bytes}",
                top1_partial_values_host.len()
            ));
        }
        if top1_partial_indices_host.len() < partial_indices_bytes {
            return Err(format!(
                "resident AQ4 lm_head direct top1 index host buffer is too small: got {} bytes expected at least {partial_indices_bytes}",
                top1_partial_indices_host.len()
            ));
        }
        top1_partial_values_buffer
            .copy_to_host(
                0,
                &mut top1_partial_values_host[..partial_values_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident AQ4 lm_head direct top1 partial values: {err}")
            })?;
        top1_partial_indices_buffer
            .copy_to_host(
                0,
                &mut top1_partial_indices_host[..partial_indices_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident AQ4 lm_head direct top1 partial indices: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize resident AQ4 lm_head direct top1 partials: {err}")
        })?;
        return package_top1_from_partial_bytes(
            &top1_partial_values_host[..partial_values_bytes],
            &top1_partial_indices_host[..partial_indices_bytes],
        );
    }
    matrix.matvec(input_buffer, logits_buffer, stream, "resident AQ4 lm_head")?;
    package_resident_lm_head_top_logits_from_logits(
        stream,
        logits_buffer,
        vocab,
        top1_partial_values_buffer,
        top1_partial_indices_buffer,
        top1_partial_values_host,
        top1_partial_indices_host,
        logits_host,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_resident_lm_head_top_logits_from_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    vocab: usize,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    if top_k == 1 {
        let partial_count = ullm_runtime_sys::top1_f32(
            logits_buffer,
            vocab,
            top1_partial_values_buffer,
            top1_partial_indices_buffer,
            Some(stream),
        )
        .map_err(|err| format!("resident lm_head top1 failed: {err}"))?;
        let partial_values_bytes =
            checked_f32_byte_len(partial_count, "resident lm_head top1 partial values")?;
        let partial_indices_bytes = partial_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "resident lm_head top1 partial index byte size overflows".to_string())?;
        if top1_partial_values_host.len() < partial_values_bytes {
            return Err(format!(
                "resident lm_head top1 value host buffer is too small: got {} bytes expected at least {partial_values_bytes}",
                top1_partial_values_host.len()
            ));
        }
        if top1_partial_indices_host.len() < partial_indices_bytes {
            return Err(format!(
                "resident lm_head top1 index host buffer is too small: got {} bytes expected at least {partial_indices_bytes}",
                top1_partial_indices_host.len()
            ));
        }
        top1_partial_values_buffer
            .copy_to_host(
                0,
                &mut top1_partial_values_host[..partial_values_bytes],
                Some(stream),
            )
            .map_err(|err| format!("failed to copy resident lm_head top1 partial values: {err}"))?;
        top1_partial_indices_buffer
            .copy_to_host(
                0,
                &mut top1_partial_indices_host[..partial_indices_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident lm_head top1 partial indices: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize resident lm_head top1 partials: {err}")
        })?;
        return package_top1_from_partial_bytes(
            &top1_partial_values_host[..partial_values_bytes],
            &top1_partial_indices_host[..partial_indices_bytes],
        );
    }
    logits_buffer
        .copy_to_host(0, logits_host, Some(stream))
        .map_err(|err| format!("failed to copy resident lm_head logits: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize resident lm_head logits: {err}"))?;
    package_top_logits_from_f32_bytes(logits_host, top_k)
}

fn copy_f32_values_to_runtime_buffer_chunked(
    buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    values: &[f32],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
) -> Result<(), String> {
    const COPY_CHUNK_F32: usize = 1 << 20;
    for (chunk_index, chunk) in values.chunks(COPY_CHUNK_F32).enumerate() {
        let offset_elements = chunk_index
            .checked_mul(COPY_CHUNK_F32)
            .ok_or_else(|| format!("{label} copy offset overflows"))?;
        let offset_bytes = offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} copy byte offset overflows"))?;
        let bytes = encode_f32_to_bytes(chunk);
        buffer
            .copy_from_host(offset_bytes, &bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} chunk {chunk_index}: {err}"))?;
    }
    Ok(())
}

fn copy_file_to_runtime_buffer_chunked(
    buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    path: &std::path::Path,
    bytes: usize,
    chunk_bytes: usize,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
) -> Result<(), String> {
    if chunk_bytes == 0 {
        return Err(format!("{label} chunk bytes must be greater than zero"));
    }
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {label} {}: {err}", path.display()))?;
    let mut chunk = vec![0_u8; chunk_bytes.min(bytes.max(1))];
    let mut offset = 0_usize;
    while offset < bytes {
        let remaining = bytes - offset;
        let read_len = remaining.min(chunk.len());
        file.read_exact(&mut chunk[..read_len])
            .map_err(|err| format!("failed to read {label} at byte offset {offset}: {err}"))?;
        buffer
            .copy_from_host(offset, &chunk[..read_len], Some(stream))
            .map_err(|err| {
                format!("failed to copy {label} chunk at byte offset {offset}: {err}")
            })?;
        offset = offset
            .checked_add(read_len)
            .ok_or_else(|| format!("{label} copy offset overflows"))?;
    }
    Ok(())
}

fn package_logit_precedes(left: &PackageTokenLogit, right: &PackageTokenLogit) -> bool {
    left.logit
        .total_cmp(&right.logit)
        .reverse()
        .then_with(|| left.token_id.cmp(&right.token_id))
        .is_lt()
}

fn push_package_top_logit(
    top_logits: &mut Vec<PackageTokenLogit>,
    top_k: usize,
    candidate: PackageTokenLogit,
) {
    if top_logits.len() < top_k {
        top_logits.push(candidate);
        top_logits.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        return;
    }
    if let Some(last) = top_logits.last() {
        if !package_logit_precedes(&candidate, last) {
            return;
        }
    }
    if let Some(last) = top_logits.last_mut() {
        *last = candidate;
    }
    top_logits.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

fn package_top_logits_from_f32_bytes(
    logits_bytes: &[u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    if top_k == 0 {
        return Err("top k must be greater than zero".to_string());
    }
    if !logits_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<f32>())
    {
        return Err("resident lm_head logits byte length is not f32-aligned".to_string());
    }
    let mut top_logits = Vec::with_capacity(top_k);
    for (token_id, chunk) in logits_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .enumerate()
    {
        let logit = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
        if !logit.is_finite() {
            return Err(format!("lm_head logit for token {token_id} is not finite"));
        }
        push_package_top_logit(
            &mut top_logits,
            top_k,
            PackageTokenLogit { token_id, logit },
        );
    }
    Ok(top_logits)
}

fn package_top1_from_partial_bytes(
    partial_value_bytes: &[u8],
    partial_index_bytes: &[u8],
) -> Result<Vec<PackageTokenLogit>, String> {
    if !partial_value_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<f32>())
    {
        return Err("resident lm_head top1 partial value byte length is not f32-aligned".into());
    }
    if !partial_index_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<u32>())
    {
        return Err("resident lm_head top1 partial index byte length is not u32-aligned".into());
    }
    let partial_count = partial_value_bytes.len() / std::mem::size_of::<f32>();
    if partial_count == 0 || partial_index_bytes.len() / std::mem::size_of::<u32>() != partial_count
    {
        return Err("resident lm_head top1 partial value/index length mismatch".into());
    }
    let mut best: Option<PackageTokenLogit> = None;
    for partial in 0..partial_count {
        let value_offset = partial * std::mem::size_of::<f32>();
        let index_offset = partial * std::mem::size_of::<u32>();
        let logit = f32::from_le_bytes(
            partial_value_bytes[value_offset..value_offset + std::mem::size_of::<f32>()]
                .try_into()
                .expect("f32 partial"),
        );
        let token_id_u32 = u32::from_le_bytes(
            partial_index_bytes[index_offset..index_offset + std::mem::size_of::<u32>()]
                .try_into()
                .expect("u32 partial"),
        );
        if !logit.is_finite() {
            return Err(format!(
                "resident lm_head top1 partial {partial} is not finite"
            ));
        }
        let token_id = usize::try_from(token_id_u32)
            .map_err(|_| format!("resident lm_head top1 token id {token_id_u32} is too large"))?;
        let candidate = PackageTokenLogit { token_id, logit };
        if best
            .as_ref()
            .map(|current| package_logit_precedes(&candidate, current))
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.map(|entry| vec![entry])
        .ok_or_else(|| "resident lm_head top1 produced no partials".into())
}

enum PackageTokenIdsIncrementalLayer {
    LinearAttention(PackageLinearAttnResidentStepLayer),
    SelfAttention(PackageSelfAttnResidentStepLayer),
}

#[derive(Clone, Copy, Default)]
struct PackageLinearAttnComponentStepMs {
    input_rmsnorm_ms: f64,
    qkv_projection_ms: f64,
    z_projection_ms: f64,
    qkv_prepare_ms: f64,
    gate_beta_projection_ms: f64,
    recurrent_ms: f64,
    attention_post_ms: f64,
    out_projection_residual_ms: f64,
    post_rmsnorm_ms: f64,
    mlp_gate_up_activation_ms: f64,
    mlp_down_residual_ms: f64,
}

impl PackageLinearAttnComponentStepMs {
    fn add_assign(&mut self, other: Self) {
        self.input_rmsnorm_ms += other.input_rmsnorm_ms;
        self.qkv_projection_ms += other.qkv_projection_ms;
        self.z_projection_ms += other.z_projection_ms;
        self.qkv_prepare_ms += other.qkv_prepare_ms;
        self.gate_beta_projection_ms += other.gate_beta_projection_ms;
        self.recurrent_ms += other.recurrent_ms;
        self.attention_post_ms += other.attention_post_ms;
        self.out_projection_residual_ms += other.out_projection_residual_ms;
        self.post_rmsnorm_ms += other.post_rmsnorm_ms;
        self.mlp_gate_up_activation_ms += other.mlp_gate_up_activation_ms;
        self.mlp_down_residual_ms += other.mlp_down_residual_ms;
    }

    fn total_ms(&self) -> f64 {
        self.input_rmsnorm_ms
            + self.qkv_projection_ms
            + self.z_projection_ms
            + self.qkv_prepare_ms
            + self.gate_beta_projection_ms
            + self.recurrent_ms
            + self.attention_post_ms
            + self.out_projection_residual_ms
            + self.post_rmsnorm_ms
            + self.mlp_gate_up_activation_ms
            + self.mlp_down_residual_ms
    }

    fn report_json(self) -> serde_json::Value {
        serde_json::json!({
            "input_rmsnorm_ms": self.input_rmsnorm_ms,
            "qkv_projection_ms": self.qkv_projection_ms,
            "z_projection_ms": self.z_projection_ms,
            "qkv_prepare_ms": self.qkv_prepare_ms,
            "gate_beta_projection_ms": self.gate_beta_projection_ms,
            "recurrent_ms": self.recurrent_ms,
            "attention_post_ms": self.attention_post_ms,
            "out_projection_residual_ms": self.out_projection_residual_ms,
            "post_rmsnorm_ms": self.post_rmsnorm_ms,
            "mlp_gate_up_activation_ms": self.mlp_gate_up_activation_ms,
            "mlp_down_residual_ms": self.mlp_down_residual_ms,
            "total_ms": self.total_ms(),
        })
    }

    fn report_summary_json(self, count: usize) -> serde_json::Value {
        serde_json::json!({
            "count": count,
            "input_rmsnorm_ms": component_total_mean_json(self.input_rmsnorm_ms, count),
            "qkv_projection_ms": component_total_mean_json(self.qkv_projection_ms, count),
            "z_projection_ms": component_total_mean_json(self.z_projection_ms, count),
            "qkv_prepare_ms": component_total_mean_json(self.qkv_prepare_ms, count),
            "gate_beta_projection_ms": component_total_mean_json(self.gate_beta_projection_ms, count),
            "recurrent_ms": component_total_mean_json(self.recurrent_ms, count),
            "attention_post_ms": component_total_mean_json(self.attention_post_ms, count),
            "out_projection_residual_ms": component_total_mean_json(self.out_projection_residual_ms, count),
            "post_rmsnorm_ms": component_total_mean_json(self.post_rmsnorm_ms, count),
            "mlp_gate_up_activation_ms": component_total_mean_json(self.mlp_gate_up_activation_ms, count),
            "mlp_down_residual_ms": component_total_mean_json(self.mlp_down_residual_ms, count),
            "total_ms": component_total_mean_json(self.total_ms(), count),
        })
    }
}

#[derive(Clone, Copy, Default)]
struct PackageSelfAttnComponentStepMs {
    input_rmsnorm_ms: f64,
    qkv_projection_ms: f64,
    qk_norm_rope_kv_write_ms: f64,
    paged_decode_ms: f64,
    output_gate_ms: f64,
    o_projection_residual_ms: f64,
    post_rmsnorm_ms: f64,
    mlp_gate_up_activation_ms: f64,
    mlp_down_residual_ms: f64,
}

impl PackageSelfAttnComponentStepMs {
    fn add_assign(&mut self, other: Self) {
        self.input_rmsnorm_ms += other.input_rmsnorm_ms;
        self.qkv_projection_ms += other.qkv_projection_ms;
        self.qk_norm_rope_kv_write_ms += other.qk_norm_rope_kv_write_ms;
        self.paged_decode_ms += other.paged_decode_ms;
        self.output_gate_ms += other.output_gate_ms;
        self.o_projection_residual_ms += other.o_projection_residual_ms;
        self.post_rmsnorm_ms += other.post_rmsnorm_ms;
        self.mlp_gate_up_activation_ms += other.mlp_gate_up_activation_ms;
        self.mlp_down_residual_ms += other.mlp_down_residual_ms;
    }

    fn total_ms(&self) -> f64 {
        self.input_rmsnorm_ms
            + self.qkv_projection_ms
            + self.qk_norm_rope_kv_write_ms
            + self.paged_decode_ms
            + self.output_gate_ms
            + self.o_projection_residual_ms
            + self.post_rmsnorm_ms
            + self.mlp_gate_up_activation_ms
            + self.mlp_down_residual_ms
    }

    fn report_json(self) -> serde_json::Value {
        serde_json::json!({
            "input_rmsnorm_ms": self.input_rmsnorm_ms,
            "qkv_projection_ms": self.qkv_projection_ms,
            "qk_norm_rope_kv_write_ms": self.qk_norm_rope_kv_write_ms,
            "paged_decode_ms": self.paged_decode_ms,
            "output_gate_ms": self.output_gate_ms,
            "o_projection_residual_ms": self.o_projection_residual_ms,
            "post_rmsnorm_ms": self.post_rmsnorm_ms,
            "mlp_gate_up_activation_ms": self.mlp_gate_up_activation_ms,
            "mlp_down_residual_ms": self.mlp_down_residual_ms,
            "total_ms": self.total_ms(),
        })
    }

    fn report_summary_json(self, count: usize) -> serde_json::Value {
        serde_json::json!({
            "count": count,
            "input_rmsnorm_ms": component_total_mean_json(self.input_rmsnorm_ms, count),
            "qkv_projection_ms": component_total_mean_json(self.qkv_projection_ms, count),
            "qk_norm_rope_kv_write_ms": component_total_mean_json(self.qk_norm_rope_kv_write_ms, count),
            "paged_decode_ms": component_total_mean_json(self.paged_decode_ms, count),
            "output_gate_ms": component_total_mean_json(self.output_gate_ms, count),
            "o_projection_residual_ms": component_total_mean_json(self.o_projection_residual_ms, count),
            "post_rmsnorm_ms": component_total_mean_json(self.post_rmsnorm_ms, count),
            "mlp_gate_up_activation_ms": component_total_mean_json(self.mlp_gate_up_activation_ms, count),
            "mlp_down_residual_ms": component_total_mean_json(self.mlp_down_residual_ms, count),
            "total_ms": component_total_mean_json(self.total_ms(), count),
        })
    }
}

fn component_total_mean_json(total_ms: f64, count: usize) -> serde_json::Value {
    serde_json::json!({
        "total_ms": total_ms,
        "mean_ms": if count > 0 {
            Some(total_ms / count as f64)
        } else {
            None
        },
    })
}

impl PackageTokenIdsIncrementalLayer {
    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(layer) => layer.output_buffer(),
            PackageTokenIdsIncrementalLayer::SelfAttention(layer) => layer.output_buffer(),
        }
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<Vec<f32>, String> {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(layer) => layer.read_output(stream),
            PackageTokenIdsIncrementalLayer::SelfAttention(layer) => layer.read_output(stream),
        }
    }

    fn take_linear_attn_component_step_ms(&mut self) -> serde_json::Value {
        self.take_linear_attn_component_step_ms_raw()
            .map(PackageLinearAttnComponentStepMs::report_json)
            .unwrap_or(serde_json::Value::Null)
    }

    fn take_linear_attn_component_step_ms_raw(
        &mut self,
    ) -> Option<PackageLinearAttnComponentStepMs> {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(layer) => {
                layer.take_last_component_step_ms()
            }
            PackageTokenIdsIncrementalLayer::SelfAttention(_) => None,
        }
    }

    fn take_self_attn_component_step_ms(&mut self) -> serde_json::Value {
        self.take_self_attn_component_step_ms_raw()
            .map(PackageSelfAttnComponentStepMs::report_json)
            .unwrap_or(serde_json::Value::Null)
    }

    fn take_self_attn_component_step_ms_raw(&mut self) -> Option<PackageSelfAttnComponentStepMs> {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(_) => None,
            PackageTokenIdsIncrementalLayer::SelfAttention(layer) => {
                layer.take_last_component_step_ms()
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        residual: &[f32],
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(layer) => {
                layer.step_from_host_to_device(stream, residual, label)
            }
            PackageTokenIdsIncrementalLayer::SelfAttention(layer) => layer
                .step_from_host_to_device(
                    stream,
                    residual,
                    rotary_dim,
                    rope_base,
                    rope_position,
                    cache_position,
                    label,
                ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        match self {
            PackageTokenIdsIncrementalLayer::LinearAttention(layer) => {
                layer.step_from_device_to_device(stream, residual_buffer, label)
            }
            PackageTokenIdsIncrementalLayer::SelfAttention(layer) => layer
                .step_from_device_to_device(
                    stream,
                    residual_buffer,
                    rotary_dim,
                    rope_base,
                    rope_position,
                    cache_position,
                    label,
                ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_incremental_layers_step(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    layers: &mut [PackageTokenIdsIncrementalLayer],
    rotary_dim: usize,
    rope_base: f32,
    rope_position: usize,
    cache_position: usize,
    residual: Vec<f32>,
    hidden: usize,
    label: &str,
) -> Result<(Vec<f32>, Vec<f64>), String> {
    if residual.len() != hidden {
        return Err(format!(
            "{label} residual length {} does not match hidden {hidden}",
            residual.len()
        ));
    }
    let mut residual_host = Some(residual);
    let mut residual_device_layer: Option<usize> = None;
    let mut layer_step_ms = Vec::with_capacity(layers.len());

    for layer_position in 0..layers.len() {
        let layer_step_started = Instant::now();
        let layer_label = format!("{label} layer {layer_position} position {rope_position}");
        if let Some(previous_position) = residual_device_layer {
            let (previous_layers, current_layers) = layers.split_at_mut(layer_position);
            let previous = previous_layers.get(previous_position).ok_or_else(|| {
                format!(
                    "{layer_label} previous device residual layer {previous_position} is missing"
                )
            })?;
            current_layers[0].step_from_device_to_device(
                stream,
                previous.output_buffer(),
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        } else {
            let residual = residual_host
                .take()
                .ok_or_else(|| format!("{layer_label} missing host residual"))?;
            if residual.len() != hidden {
                return Err(format!(
                    "{layer_label} input length {} does not match hidden {hidden}",
                    residual.len()
                ));
            }
            layers[layer_position].step_from_host_to_device(
                stream,
                &residual,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        }
        residual_device_layer = Some(layer_position);
        layer_step_ms.push(layer_step_started.elapsed().as_secs_f64() * 1000.0);
    }

    let output = if let Some(previous_position) = residual_device_layer {
        layers
            .get(previous_position)
            .ok_or_else(|| {
                format!("{label} final device residual layer {previous_position} is missing")
            })?
            .read_output(stream)?
    } else {
        residual_host.ok_or_else(|| format!("{label} missing final host residual"))?
    };
    if output.len() != hidden {
        return Err(format!(
            "{label} output length {} does not match hidden {hidden}",
            output.len()
        ));
    }
    Ok((output, layer_step_ms))
}

fn package_token_ids_incremental_layers_step_device(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    layers: &mut [PackageTokenIdsIncrementalLayer],
    rotary_dim: usize,
    rope_base: f32,
    rope_position: usize,
    cache_position: usize,
    residual: Vec<f32>,
    hidden: usize,
    label: &str,
    sync_each_layer_for_timing: bool,
) -> Result<
    (
        usize,
        Vec<f64>,
        Vec<serde_json::Value>,
        Vec<serde_json::Value>,
    ),
    String,
> {
    if residual.len() != hidden {
        return Err(format!(
            "{label} residual length {} does not match hidden {hidden}",
            residual.len()
        ));
    }
    if layers.is_empty() {
        return Err(format!(
            "{label} device layer step requires at least one layer"
        ));
    }
    let mut residual_host = Some(residual);
    let mut residual_device_layer: Option<usize> = None;
    let mut layer_step_ms = Vec::with_capacity(layers.len());
    let mut linear_attn_component_step_ms = Vec::with_capacity(layers.len());
    let mut self_attn_component_step_ms = Vec::with_capacity(layers.len());

    for layer_position in 0..layers.len() {
        let layer_step_started = Instant::now();
        let layer_label = format!("{label} layer {layer_position} position {rope_position}");
        if let Some(previous_position) = residual_device_layer {
            let (previous_layers, current_layers) = layers.split_at_mut(layer_position);
            let previous = previous_layers.get(previous_position).ok_or_else(|| {
                format!(
                    "{layer_label} previous device residual layer {previous_position} is missing"
                )
            })?;
            current_layers[0].step_from_device_to_device(
                stream,
                previous.output_buffer(),
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        } else {
            let residual = residual_host
                .take()
                .ok_or_else(|| format!("{layer_label} missing host residual"))?;
            layers[layer_position].step_from_host_to_device(
                stream,
                &residual,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        }
        residual_device_layer = Some(layer_position);
        if sync_each_layer_for_timing {
            stream
                .synchronize()
                .map_err(|err| format!("failed to synchronize {layer_label}: {err}"))?;
        }
        layer_step_ms.push(layer_step_started.elapsed().as_secs_f64() * 1000.0);
        linear_attn_component_step_ms
            .push(layers[layer_position].take_linear_attn_component_step_ms());
        self_attn_component_step_ms.push(layers[layer_position].take_self_attn_component_step_ms());
    }

    residual_device_layer
        .map(|layer_position| {
            (
                layer_position,
                layer_step_ms,
                linear_attn_component_step_ms,
                self_attn_component_step_ms,
            )
        })
        .ok_or_else(|| format!("{label} missing final device residual"))
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_incremental_layers_step_device_input(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    layers: &mut [PackageTokenIdsIncrementalLayer],
    rotary_dim: usize,
    rope_base: f32,
    rope_position: usize,
    cache_position: usize,
    residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
    hidden: usize,
    label: &str,
    sync_each_layer_for_timing: bool,
) -> Result<
    (
        usize,
        Vec<f64>,
        Vec<serde_json::Value>,
        Vec<serde_json::Value>,
    ),
    String,
> {
    if layers.is_empty() {
        return Err(format!(
            "{label} device layer step requires at least one layer"
        ));
    }
    let required_bytes = checked_f32_byte_len(hidden, "incremental decode device residual")?;
    let actual_bytes = residual_buffer
        .size()
        .map_err(|err| format!("failed to query {label} residual buffer size: {err}"))?;
    if actual_bytes < required_bytes {
        return Err(format!(
            "{label} residual buffer is too small: got {actual_bytes} bytes expected at least {required_bytes}"
        ));
    }

    let mut layer_step_ms = Vec::with_capacity(layers.len());
    let mut linear_attn_component_step_ms = Vec::with_capacity(layers.len());
    let mut self_attn_component_step_ms = Vec::with_capacity(layers.len());
    let first_label = format!("{label} layer 0 position {rope_position}");
    let first_started = Instant::now();
    layers[0].step_from_device_to_device(
        stream,
        residual_buffer,
        rotary_dim,
        rope_base,
        rope_position,
        cache_position,
        &first_label,
    )?;
    if sync_each_layer_for_timing {
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize {first_label}: {err}"))?;
    }
    layer_step_ms.push(first_started.elapsed().as_secs_f64() * 1000.0);
    linear_attn_component_step_ms.push(layers[0].take_linear_attn_component_step_ms());
    self_attn_component_step_ms.push(layers[0].take_self_attn_component_step_ms());

    for layer_position in 1..layers.len() {
        let layer_step_started = Instant::now();
        let layer_label = format!("{label} layer {layer_position} position {rope_position}");
        let (previous_layers, current_layers) = layers.split_at_mut(layer_position);
        let previous = previous_layers.get(layer_position - 1).ok_or_else(|| {
            format!(
                "{layer_label} previous device residual layer {} is missing",
                layer_position - 1
            )
        })?;
        current_layers[0].step_from_device_to_device(
            stream,
            previous.output_buffer(),
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            &layer_label,
        )?;
        if sync_each_layer_for_timing {
            stream
                .synchronize()
                .map_err(|err| format!("failed to synchronize {layer_label}: {err}"))?;
        }
        layer_step_ms.push(layer_step_started.elapsed().as_secs_f64() * 1000.0);
        linear_attn_component_step_ms
            .push(layers[layer_position].take_linear_attn_component_step_ms());
        self_attn_component_step_ms.push(layers[layer_position].take_self_attn_component_step_ms());
    }

    Ok((
        layers.len() - 1,
        layer_step_ms,
        linear_attn_component_step_ms,
        self_attn_component_step_ms,
    ))
}

fn package_token_ids_top_logits_result(
    top_logits: Vec<PackageTokenLogit>,
) -> Result<(serde_json::Value, usize), String> {
    let next = top_logits
        .first()
        .map(|entry| entry.token_id)
        .ok_or_else(|| "incremental lm_head returned no top logits".to_string())?;
    let top_logits_json = top_logits
        .iter()
        .map(|entry| {
            serde_json::json!({
                "token_id": entry.token_id,
                "logit": entry.logit,
            })
        })
        .collect::<Vec<_>>();
    Ok((serde_json::Value::Array(top_logits_json), next))
}

fn package_token_ids_incremental_final_logits(
    path: &str,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    lm_head_runtime: &mut PackageLmHeadRuntime,
    final_hidden: &[f32],
    final_norm: &PassthroughF32Data,
    top_k: usize,
) -> Result<(serde_json::Value, usize), String> {
    let final_normed = runtime_host_rmsnorm_f32(final_hidden, &final_norm.values, 1e-6_f32);
    if final_normed.len() != final_hidden.len()
        || final_normed.iter().any(|value| !value.is_finite())
    {
        return Err(
            "incremental final normalized hidden state contains invalid values".to_string(),
        );
    }
    package_token_ids_top_logits_result(lm_head_runtime.top_logits(
        path,
        stream,
        &final_normed,
        top_k,
    )?)
}

fn package_token_ids_incremental_final_logits_device(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    lm_head_runtime: &mut PackageLmHeadRuntime,
    final_norm_runtime: &mut PackageFinalNormRuntime,
    final_hidden_buffer: &ullm_runtime_sys::RuntimeBuffer,
    top_k: usize,
    label: &str,
) -> Result<(serde_json::Value, usize), String> {
    final_norm_runtime.normalize_device(stream, final_hidden_buffer, label)?;
    package_token_ids_top_logits_result(lm_head_runtime.top_logits_from_device_buffer(
        stream,
        final_norm_runtime.output_buffer(),
        top_k,
    )?)
}

fn checked_product_u64(values: &[usize], label: &str) -> Result<u64, String> {
    let mut product = 1_u64;
    for &value in values {
        let value =
            u64::try_from(value).map_err(|_| format!("{label} value {value} exceeds u64 range"))?;
        product = product
            .checked_mul(value)
            .ok_or_else(|| format!("{label} byte count overflows u64"))?;
    }
    Ok(product)
}

fn package_token_ids_incremental_kv_cache_layer_bytes(
    cache_blocks: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
) -> Result<u64, String> {
    let kv_width = head_dim
        .checked_add(value_dim)
        .ok_or_else(|| "incremental KV width overflows".to_string())?;
    checked_product_u64(
        &[
            cache_blocks,
            block_size,
            kv_heads,
            kv_width,
            std::mem::size_of::<f32>(),
        ],
        "incremental KV cache",
    )
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_generate_incremental_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    prompt_token_ids: Vec<usize>,
    generated_tokens: usize,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    lm_head_mode: PackageLmHeadMode,
    stop_token_ids: Vec<usize>,
    stop_token_sequences: Vec<Vec<usize>>,
    sync_decode_layers_for_timing: bool,
    sync_decode_each_layer_for_timing: bool,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    if prompt_token_ids.is_empty() {
        return Err(
            "package token-id generate smoke requires at least one prompt token".to_string(),
        );
    }
    if layer_indices.is_empty() {
        return Err("package token-id generate smoke requires at least one layer".to_string());
    }
    let sync_linear_attn_components_for_timing =
        env_flag_enabled("ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING");
    let use_prefill_device_token_loop = env_flag_enabled("ULLM_PREFILL_DEVICE_TOKEN_LOOP");
    let sync_prefill_each_layer_for_timing =
        env_flag_enabled("ULLM_SYNC_PREFILL_EACH_LAYER_FOR_TIMING");
    let use_aq4_matvec_qkv_z_gate_beta_requested =
        !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA");
    let use_aq4_matvec_pair_qkv_z = !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_PAIR_QKV_Z");
    let use_aq4_matvec_triple_self_attn_qkv =
        !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV");
    let use_aq4_matvec_pair_self_attn_qk =
        !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_PAIR_SELF_ATTN_QK");

    let run_started = Instant::now();
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let use_aq4_matvec_qkv_z_gate_beta =
        use_aq4_matvec_qkv_z_gate_beta_requested && info.backend == "hip";
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let sq_overlay = sq_artifact.map(|artifact| Qwen3PackageSqOverlay {
        artifact,
        row_chunk: 256,
    });
    let sq_overlay_json = sq_artifact.map(|artifact| {
        let candidate_legacy = if artifact.manifest.candidate.id == FORMAT_SQ8_0 {
            None
        } else {
            Some(artifact.manifest.candidate.id.as_str())
        };
        let implementation_id = artifact
            .manifest
            .candidate
            .implementation_id
            .as_deref()
            .or(candidate_legacy)
            .unwrap_or("none");
        serde_json::json!({
            "artifact": artifact.artifact_dir,
            "candidate": FORMAT_SQ8_0,
            "candidate_legacy": candidate_legacy,
            "format_id": FORMAT_SQ8_0,
            "implementation_id": implementation_id,
            "schema_version": artifact.manifest.schema_version,
            "fp8_tensor_count": artifact.manifest.storage.fp8_tensor_count,
            "passthrough_tensor_count": artifact.manifest.storage.passthrough_tensor_count,
            "row_chunk": 256,
        })
    });

    let (embedding_vocab, hidden) = package_embedding_shape(path)?;
    if hidden == 0 {
        return Err("incremental prompt embedding hidden size is zero".to_string());
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id >= embedding_vocab)
    {
        return Err(format!(
            "incremental prompt token id {token_id} is out of embedding range 0..{embedding_vocab}"
        ));
    }

    let total_tokens = prompt_token_ids
        .len()
        .checked_add(generated_tokens)
        .ok_or_else(|| "incremental total token count overflows".to_string())?
        .max(1);
    let block_size = 256_usize.min(total_tokens.max(1));
    let cache_blocks = (total_tokens - 1) / block_size + 1;
    if cache_blocks > u32::MAX as usize {
        return Err(format!(
            "incremental cache block count {cache_blocks} exceeds u32 range"
        ));
    }
    let block_table = (0..cache_blocks)
        .map(|block| {
            u32::try_from(block)
                .map_err(|_| format!("incremental block index {block} exceeds u32 range"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let layer_load_started = Instant::now();
    let mut layer_kinds = Vec::with_capacity(layer_indices.len());
    let mut layers = Vec::with_capacity(layer_indices.len());
    let mut self_attn_layer_count = 0_usize;
    let mut kv_cache_bytes = 0_u64;
    let mut self_attn_shape = None;
    let mut resolved_rotary_dim = None;
    for &layer_index in &layer_indices {
        let layer_kind = package_decoder_layer_kind(path, layer_index)
            .map_err(|err| format!("failed to identify package layer {layer_index}: {err}"))?;
        layer_kinds.push(layer_kind.as_str());
        match layer_kind {
            PackageDecoderLayerKind::LinearAttention => {
                let layer = if sq_overlay.is_some() {
                    let mut registry = WeightRegistry::new();
                    PackageLinearAttnResidentStepLayer::load_with_registry(
                        &mut context,
                        &mut stream,
                        &mut registry,
                        None,
                        path,
                        chunk_bytes,
                        layer_index,
                        sq_overlay.as_ref(),
                    )
                } else {
                    PackageLinearAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        chunk_bytes,
                        layer_index,
                    )
                }
                .map_err(|err| {
                    format!("failed to load incremental linear-attn layer {layer_index}: {err}")
                })?;
                layers.push(PackageTokenIdsIncrementalLayer::LinearAttention(layer));
            }
            PackageDecoderLayerKind::SelfAttention => {
                let layer = if sq_overlay.is_some() {
                    let mut registry = WeightRegistry::new();
                    PackageSelfAttnResidentStepLayer::load_with_registry(
                        &mut context,
                        &mut stream,
                        &mut registry,
                        None,
                        path,
                        chunk_bytes,
                        layer_index,
                        &block_table,
                        block_size,
                        cache_blocks,
                        sq_overlay.as_ref(),
                    )
                } else {
                    PackageSelfAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        chunk_bytes,
                        layer_index,
                        &block_table,
                        block_size,
                        cache_blocks,
                    )
                }
                .map_err(|err| {
                    format!("failed to load incremental self-attn layer {layer_index}: {err}")
                })?;
                if layer.hidden != hidden {
                    return Err(format!(
                        "incremental self-attn layer {layer_index} hidden mismatch: layer_hidden={} embedding_hidden={hidden}",
                        layer.hidden
                    ));
                }
                let layer_rotary_dim =
                    parse_package_token_ids_rotary_dim(layer.head_dim, rotary_dim.as_deref())?;
                if let Some(previous) = resolved_rotary_dim {
                    if previous != layer_rotary_dim {
                        return Err(format!(
                            "incremental rotary dim changed: previous={previous} current={layer_rotary_dim}"
                        ));
                    }
                } else {
                    resolved_rotary_dim = Some(layer_rotary_dim);
                }
                if self_attn_shape.is_none() {
                    self_attn_shape = Some(serde_json::json!({
                        "q_heads": layer.q_heads,
                        "kv_heads": layer.kv_heads,
                        "head_dim": layer.head_dim,
                        "value_dim": layer.value_dim,
                        "rotary_dim": layer_rotary_dim,
                        "q_projection_layout": layer.q_projection_layout.as_str(),
                    }));
                }
                let layer_kv_bytes = package_token_ids_incremental_kv_cache_layer_bytes(
                    cache_blocks,
                    block_size,
                    layer.kv_heads,
                    layer.head_dim,
                    layer.value_dim,
                )?;
                kv_cache_bytes = kv_cache_bytes
                    .checked_add(layer_kv_bytes)
                    .ok_or_else(|| "incremental KV cache total bytes overflow u64".to_string())?;
                self_attn_layer_count = self_attn_layer_count
                    .checked_add(1)
                    .ok_or_else(|| "incremental self-attn layer count overflows".to_string())?;
                layers.push(PackageTokenIdsIncrementalLayer::SelfAttention(layer));
            }
        }
    }
    let rotary_dim_value = resolved_rotary_dim.unwrap_or(0);
    let kv_cache_allocated_blocks = cache_blocks
        .checked_mul(self_attn_layer_count)
        .ok_or_else(|| "incremental KV cache allocated block count overflows".to_string())?;
    let layer_load_ms = layer_load_started.elapsed().as_secs_f64() * 1000.0;

    let mut final_norm = read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, chunk_bytes)
        .map_err(|err| format!("failed to read final RMSNorm tensor: {err}"))?;
    final_norm.values =
        effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
    if final_norm.values.len() != hidden {
        return Err(format!(
            "incremental final RMSNorm length mismatch: len={} hidden={hidden}",
            final_norm.values.len()
        ));
    }

    let lm_head_load_started = Instant::now();
    let mut lm_head_runtime = PackageLmHeadRuntime::load(
        lm_head_mode,
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        hidden,
        lm_head_chunk_rows,
    )?;
    let mut final_norm_runtime = if lm_head_runtime.supports_device_input() {
        Some(PackageFinalNormRuntime::load(
            &mut context,
            &mut stream,
            &final_norm,
            hidden,
        )?)
    } else {
        None
    };
    let lm_head_load_ms = lm_head_load_started.elapsed().as_secs_f64() * 1000.0;
    let embedding_runtime_load_started = Instant::now();
    let mut embedding_runtime = if lm_head_runtime.supports_device_input() {
        PackageEmbeddingRuntime::load_if_available(
            &mut context,
            &mut stream,
            path,
            chunk_bytes,
            hidden,
        )?
    } else {
        None
    };
    let embedding_runtime_load_ms = embedding_runtime_load_started.elapsed().as_secs_f64() * 1000.0;

    let embed_started = Instant::now();
    let mut residual_sequence = if use_prefill_device_token_loop && embedding_runtime.is_some() {
        Vec::new()
    } else if let Some(runtime) = embedding_runtime.as_mut() {
        let mut values = Vec::with_capacity(prompt_token_ids.len() * hidden);
        for (index, &token_id) in prompt_token_ids.iter().enumerate() {
            let row = runtime.gather_token_values(
                &mut stream,
                token_id,
                &format!("incremental prompt embedding token {index}"),
            )?;
            if row.len() != hidden {
                return Err(format!(
                    "incremental prompt resident embedding length {} does not match hidden {hidden}",
                    row.len()
                ));
            }
            values.extend(row);
        }
        values
    } else {
        let embedding_rows =
            read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &prompt_token_ids)
                .map_err(|err| format!("failed to read prompt embedding rows: {err}"))?;
        if embedding_rows.columns != hidden
            || embedding_rows.values.len() != prompt_token_ids.len() * hidden
        {
            return Err(format!(
                "incremental prompt embedding shape mismatch: hidden={} columns={} values={} prompt_tokens={}",
                hidden,
                embedding_rows.columns,
                embedding_rows.values.len(),
                prompt_token_ids.len()
            ));
        }
        embedding_rows.values
    };
    let embed_ms = embed_started.elapsed().as_secs_f64() * 1000.0;

    let prefill_started = Instant::now();
    let prefill_layers_started = Instant::now();
    let mut prefill_layer_step_ms = (0..layers.len())
        .map(|_| Vec::with_capacity(prompt_token_ids.len()))
        .collect::<Vec<_>>();
    let mut prefill_linear_attn_component_sums =
        vec![PackageLinearAttnComponentStepMs::default(); layers.len()];
    let mut prefill_linear_attn_component_counts = vec![0_usize; layers.len()];
    let mut prefill_self_attn_component_sums =
        vec![PackageSelfAttnComponentStepMs::default(); layers.len()];
    let mut prefill_self_attn_component_counts = vec![0_usize; layers.len()];
    if use_prefill_device_token_loop {
        for (timestep, &token_id) in prompt_token_ids.iter().enumerate() {
            let position = position_offset
                .checked_add(timestep)
                .ok_or_else(|| "incremental prefill position overflows".to_string())?;
            let mut first_residual_host = None;
            if let Some(runtime) = embedding_runtime.as_mut() {
                runtime.gather_token(
                    &mut stream,
                    token_id,
                    &format!("incremental device prefill embedding token {timestep}"),
                )?;
            } else {
                let start = timestep * hidden;
                let end = start + hidden;
                if end > residual_sequence.len() {
                    return Err(format!(
                        "incremental device prefill host embedding slice {start}..{end} exceeds {} values",
                        residual_sequence.len()
                    ));
                }
                first_residual_host = Some(residual_sequence[start..end].to_vec());
            }

            for layer_position in 0..layers.len() {
                let prefill_layer_step_started = Instant::now();
                let layer_label = format!(
                    "incremental device prefill layer {layer_position} timestep {timestep}"
                );
                if layer_position == 0 {
                    if let Some(runtime) = embedding_runtime.as_ref() {
                        layers[0].step_from_device_to_device(
                            &mut stream,
                            runtime.output_buffer(),
                            rotary_dim_value,
                            rope_base,
                            position,
                            timestep,
                            &layer_label,
                        )?;
                    } else {
                        let residual = first_residual_host
                            .take()
                            .ok_or_else(|| format!("{layer_label} missing host residual"))?;
                        layers[0].step_from_host_to_device(
                            &mut stream,
                            &residual,
                            rotary_dim_value,
                            rope_base,
                            position,
                            timestep,
                            &layer_label,
                        )?;
                    }
                } else {
                    let (previous_layers, current_layers) = layers.split_at_mut(layer_position);
                    let previous = previous_layers.get(layer_position - 1).ok_or_else(|| {
                        format!(
                            "{layer_label} previous device residual layer {} is missing",
                            layer_position - 1
                        )
                    })?;
                    current_layers[0].step_from_device_to_device(
                        &mut stream,
                        previous.output_buffer(),
                        rotary_dim_value,
                        rope_base,
                        position,
                        timestep,
                        &layer_label,
                    )?;
                }
                if sync_prefill_each_layer_for_timing {
                    stream
                        .synchronize()
                        .map_err(|err| format!("failed to synchronize {layer_label}: {err}"))?;
                }
                prefill_layer_step_ms[layer_position]
                    .push(prefill_layer_step_started.elapsed().as_secs_f64() * 1000.0);
                if let Some(component_ms) =
                    layers[layer_position].take_linear_attn_component_step_ms_raw()
                {
                    prefill_linear_attn_component_sums[layer_position].add_assign(component_ms);
                    prefill_linear_attn_component_counts[layer_position] += 1;
                }
                if let Some(component_ms) =
                    layers[layer_position].take_self_attn_component_step_ms_raw()
                {
                    prefill_self_attn_component_sums[layer_position].add_assign(component_ms);
                    prefill_self_attn_component_counts[layer_position] += 1;
                }
            }
        }
    } else {
        for (layer_position, layer) in layers.iter_mut().enumerate() {
            let mut next_sequence = Vec::with_capacity(residual_sequence.len());
            for timestep in 0..prompt_token_ids.len() {
                let prefill_layer_step_started = Instant::now();
                let start = timestep * hidden;
                let end = start + hidden;
                let position = position_offset
                    .checked_add(timestep)
                    .ok_or_else(|| "incremental prefill position overflows".to_string())?;
                let layer_label =
                    format!("incremental prefill layer {layer_position} timestep {timestep}");
                layer.step_from_host_to_device(
                    &mut stream,
                    &residual_sequence[start..end],
                    rotary_dim_value,
                    rope_base,
                    position,
                    timestep,
                    &layer_label,
                )?;
                let output = layer.read_output(&mut stream)?;
                if output.len() != hidden {
                    return Err(format!(
                        "incremental prefill layer {layer_position} output length {} does not match hidden {hidden}",
                        output.len()
                    ));
                }
                prefill_layer_step_ms[layer_position]
                    .push(prefill_layer_step_started.elapsed().as_secs_f64() * 1000.0);
                if let Some(component_ms) = layer.take_linear_attn_component_step_ms_raw() {
                    prefill_linear_attn_component_sums[layer_position].add_assign(component_ms);
                    prefill_linear_attn_component_counts[layer_position] += 1;
                }
                if let Some(component_ms) = layer.take_self_attn_component_step_ms_raw() {
                    prefill_self_attn_component_sums[layer_position].add_assign(component_ms);
                    prefill_self_attn_component_counts[layer_position] += 1;
                }
                next_sequence.extend(output);
            }
            residual_sequence = next_sequence;
        }
    }
    let prefill_layers_ms = prefill_layers_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_lm_head_started = Instant::now();
    let (prefill_top_logits, first_generated) =
        if use_prefill_device_token_loop && lm_head_runtime.supports_device_input() {
            let final_norm_runtime = final_norm_runtime.as_mut().ok_or_else(|| {
                "incremental prefill final RMSNorm runtime is missing".to_string()
            })?;
            let final_layer = layers
                .last()
                .ok_or_else(|| "incremental device prefill has no final layer".to_string())?;
            package_token_ids_incremental_final_logits_device(
                &mut stream,
                &mut lm_head_runtime,
                final_norm_runtime,
                final_layer.output_buffer(),
                top_k,
                "incremental device prefill",
            )?
        } else if use_prefill_device_token_loop {
            let final_layer = layers
                .last()
                .ok_or_else(|| "incremental device prefill has no final layer".to_string())?;
            let final_hidden = final_layer.read_output(&mut stream)?;
            package_token_ids_incremental_final_logits(
                path,
                &mut stream,
                &mut lm_head_runtime,
                &final_hidden,
                &final_norm,
                top_k,
            )?
        } else {
            let final_hidden_start = (prompt_token_ids.len() - 1) * hidden;
            package_token_ids_incremental_final_logits(
                path,
                &mut stream,
                &mut lm_head_runtime,
                &residual_sequence[final_hidden_start..final_hidden_start + hidden],
                &final_norm,
                top_k,
            )?
        };
    let prefill_lm_head_ms = prefill_lm_head_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;

    let mut sequence_ids = prompt_token_ids.clone();
    let mut generated_token_ids = Vec::with_capacity(generated_tokens);
    let mut decode_step_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_embedding_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_layers_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_layer_step_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_linear_attn_component_step_ms =
        Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_self_attn_component_step_ms =
        Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_lm_head_ms = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut decode_positions = Vec::with_capacity(generated_tokens.saturating_sub(1));
    let mut last_top_logits = prefill_top_logits.clone();
    let mut stopped_on_token_id = None;
    let mut stopped_on_token_sequence = None;
    if generated_tokens > 0 {
        generated_token_ids.push(first_generated);
        sequence_ids.push(first_generated);
        if stop_token_ids.contains(&first_generated) {
            stopped_on_token_id = Some(first_generated);
        } else if let Some(sequence) =
            matched_stop_token_sequence(&generated_token_ids, &stop_token_sequences)
        {
            stopped_on_token_sequence = Some(sequence);
        }
    }

    while generated_token_ids.len() < generated_tokens
        && stopped_on_token_id.is_none()
        && stopped_on_token_sequence.is_none()
    {
        let decode_started = Instant::now();
        let token_id = *sequence_ids
            .last()
            .ok_or_else(|| "incremental sequence unexpectedly empty".to_string())?;
        let decode_embedding_started = Instant::now();
        let mut embedding_values = None;
        if let Some(runtime) = embedding_runtime.as_mut() {
            runtime.gather_token(&mut stream, token_id, "incremental decode")?;
        } else {
            let embedding =
                read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &[token_id])
                    .map_err(|err| format!("failed to read incremental decode embedding: {err}"))?;
            if embedding.values.len() != hidden {
                return Err(format!(
                    "incremental decode embedding length {} does not match hidden {hidden}",
                    embedding.values.len()
                ));
            }
            embedding_values = Some(embedding.values);
        }
        let embedding_step_ms = decode_embedding_started.elapsed().as_secs_f64() * 1000.0;
        let decode_index = generated_token_ids.len() - 1;
        let position = position_offset
            .checked_add(prompt_token_ids.len())
            .and_then(|value| value.checked_add(decode_index))
            .ok_or_else(|| "incremental decode position overflows".to_string())?;
        let cache_position = prompt_token_ids
            .len()
            .checked_add(decode_index)
            .ok_or_else(|| "incremental decode cache position overflows".to_string())?;
        let decode_layers_started = Instant::now();
        let (
            top_logits,
            next,
            layers_step_ms,
            layer_step_ms,
            linear_attn_component_step_ms,
            self_attn_component_step_ms,
        ) = if lm_head_runtime.supports_device_input() {
            let final_norm_runtime = final_norm_runtime
                .as_mut()
                .ok_or_else(|| "incremental decode final RMSNorm runtime is missing".to_string())?;
            let (
                final_layer_position,
                layer_step_ms,
                linear_attn_component_step_ms,
                self_attn_component_step_ms,
            ) = if let Some(runtime) = embedding_runtime.as_ref() {
                package_token_ids_incremental_layers_step_device_input(
                    &mut stream,
                    &mut layers,
                    rotary_dim_value,
                    rope_base,
                    position,
                    cache_position,
                    runtime.output_buffer(),
                    hidden,
                    "incremental decode",
                    sync_decode_each_layer_for_timing,
                )?
            } else {
                let embedding_values = embedding_values
                    .take()
                    .ok_or_else(|| "incremental decode missing host embedding".to_string())?;
                package_token_ids_incremental_layers_step_device(
                    &mut stream,
                    &mut layers,
                    rotary_dim_value,
                    rope_base,
                    position,
                    cache_position,
                    embedding_values,
                    hidden,
                    "incremental decode",
                    sync_decode_each_layer_for_timing,
                )?
            };
            if sync_decode_layers_for_timing && !sync_decode_each_layer_for_timing {
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize incremental decode layers: {err}")
                })?;
            }
            let layers_step_ms = decode_layers_started.elapsed().as_secs_f64() * 1000.0;
            let final_hidden_buffer = layers
                .get(final_layer_position)
                .ok_or_else(|| {
                    format!("incremental decode final layer {final_layer_position} is missing")
                })?
                .output_buffer();
            let decode_lm_head_started = Instant::now();
            let (top_logits, next) = package_token_ids_incremental_final_logits_device(
                &mut stream,
                &mut lm_head_runtime,
                final_norm_runtime,
                final_hidden_buffer,
                top_k,
                "incremental decode",
            )?;
            let lm_head_step_ms = decode_lm_head_started.elapsed().as_secs_f64() * 1000.0;
            decode_lm_head_ms.push(lm_head_step_ms);
            (
                top_logits,
                next,
                layers_step_ms,
                layer_step_ms,
                linear_attn_component_step_ms,
                self_attn_component_step_ms,
            )
        } else {
            let embedding_values = embedding_values
                .take()
                .ok_or_else(|| "incremental decode missing host embedding".to_string())?;
            let (residual, layer_step_ms) = package_token_ids_incremental_layers_step(
                &mut stream,
                &mut layers,
                rotary_dim_value,
                rope_base,
                position,
                cache_position,
                embedding_values,
                hidden,
                "incremental decode",
            )?;
            let layers_step_ms = decode_layers_started.elapsed().as_secs_f64() * 1000.0;
            let decode_lm_head_started = Instant::now();
            let (top_logits, next) = package_token_ids_incremental_final_logits(
                path,
                &mut stream,
                &mut lm_head_runtime,
                &residual,
                &final_norm,
                top_k,
            )?;
            let lm_head_step_ms = decode_lm_head_started.elapsed().as_secs_f64() * 1000.0;
            decode_lm_head_ms.push(lm_head_step_ms);
            let linear_attn_component_step_ms = vec![serde_json::Value::Null; layer_step_ms.len()];
            let self_attn_component_step_ms = vec![serde_json::Value::Null; layer_step_ms.len()];
            (
                top_logits,
                next,
                layers_step_ms,
                layer_step_ms,
                linear_attn_component_step_ms,
                self_attn_component_step_ms,
            )
        };
        last_top_logits = top_logits;
        decode_step_ms.push(decode_started.elapsed().as_secs_f64() * 1000.0);
        decode_embedding_ms.push(embedding_step_ms);
        decode_layers_ms.push(layers_step_ms);
        decode_layer_step_ms.push(layer_step_ms);
        decode_linear_attn_component_step_ms.push(linear_attn_component_step_ms);
        decode_self_attn_component_step_ms.push(self_attn_component_step_ms);
        decode_positions.push(position);
        generated_token_ids.push(next);
        sequence_ids.push(next);
        if stop_token_ids.contains(&next) {
            stopped_on_token_id = Some(next);
        } else if let Some(sequence) =
            matched_stop_token_sequence(&generated_token_ids, &stop_token_sequences)
        {
            stopped_on_token_sequence = Some(sequence);
        }
    }

    let decode_ms = decode_step_ms.iter().sum::<f64>();
    let total_ms = run_started.elapsed().as_secs_f64() * 1000.0;
    let timed_decode_tokens = decode_step_ms.len();
    let decode_step_summary = timed_step_summary_json(&decode_step_ms);
    let prompt_token_count = prompt_token_ids.len();
    let prefill_layer_step_summary = prefill_layer_step_ms
        .iter()
        .enumerate()
        .map(|(layer_position, step_ms)| {
            let wall_ms = step_ms.iter().sum::<f64>();
            let linear_attn_components = if prefill_linear_attn_component_counts[layer_position] > 0
            {
                prefill_linear_attn_component_sums[layer_position]
                    .report_summary_json(prefill_linear_attn_component_counts[layer_position])
            } else {
                serde_json::Value::Null
            };
            let self_attn_components = if prefill_self_attn_component_counts[layer_position] > 0 {
                prefill_self_attn_component_sums[layer_position]
                    .report_summary_json(prefill_self_attn_component_counts[layer_position])
            } else {
                serde_json::Value::Null
            };
            serde_json::json!({
                "layer_position": layer_position,
                "layer_index": layer_indices[layer_position],
                "kind": layer_kinds[layer_position],
                "prompt_tokens": prompt_token_count,
                "wall_ms": wall_ms,
                "token_tps": tps(prompt_token_count, wall_ms),
                "step_wall_summary": timed_step_summary_json(step_ms),
                "linear_attn_component_summary": linear_attn_components,
                "self_attn_component_summary": self_attn_components,
            })
        })
        .collect::<Vec<_>>();
    let stop_reason = if stopped_on_token_id.is_some() {
        "stop_token"
    } else if stopped_on_token_sequence.is_some() {
        "stop_sequence"
    } else {
        "max_generated_tokens"
    };
    let stopped = stopped_on_token_id.is_some() || stopped_on_token_sequence.is_some();
    let report = serde_json::json!({
        "schema_version": "package-token-ids-generate-smoke-v0.1",
        "package": path,
        "sq_overlay": sq_overlay_json,
        "git_commit": current_git_commit(),
        "backend": info.backend.to_string(),
        "device_index": device_index,
        "device_name": info.name,
        "device_total_global_mem": info.total_global_mem,
        "layers": layer_indices,
        "layer_kinds": layer_kinds,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "final_sequence_len": sequence_ids.len(),
        "hidden": hidden,
        "self_attn": self_attn_shape,
        "top_k": top_k,
        "lm_head_chunk_rows": lm_head_chunk_rows,
        "lm_head_runtime": lm_head_runtime.report_json(lm_head_load_ms),
        "embedding_runtime": embedding_runtime
            .as_ref()
            .map(PackageEmbeddingRuntime::report_json)
            .unwrap_or(serde_json::Value::Null),
        "final_norm_runtime": final_norm_runtime
            .as_ref()
            .map(PackageFinalNormRuntime::report_json)
            .unwrap_or(serde_json::Value::Null),
        "decode_embedding_device": embedding_runtime.is_some(),
        "decode_final_logits_device": final_norm_runtime.is_some()
            && lm_head_runtime.supports_device_input(),
        "rotary_dim": resolved_rotary_dim,
        "rope_base": rope_base,
        "position_offset": position_offset,
        "stop": {
            "token_ids": stop_token_ids,
            "token_sequences": stop_token_sequences,
            "stopped": stopped,
            "stopped_on_token_id": stopped_on_token_id,
            "stopped_on_token_sequence": stopped_on_token_sequence,
            "reason": stop_reason,
        },
        "decode_mode": "hybrid_incremental_greedy",
        "incremental_decode": true,
        "prefill": {
            "executor": if use_prefill_device_token_loop {
                "device_token_loop"
            } else {
                "layer_major_host_token_loop"
            },
            "real_batch": false,
            "token_parallelism": 1,
            "request_parallelism": 1,
            "projection_executor": "single_token_matvec",
            "mlp_executor": "single_token_matvec",
            "attention_executor": "single_token_decode_step",
            "device_resident": use_prefill_device_token_loop,
            "sync_each_layer_for_timing": sync_prefill_each_layer_for_timing,
            "prompt_tokens": prompt_token_count,
            "wall_ms": prefill_ms,
            "layers_wall_ms": prefill_layers_ms,
            "lm_head_wall_ms": prefill_lm_head_ms,
            "layer_step_summary": prefill_layer_step_summary,
            "tps": tps(prompt_token_count, prefill_ms),
            "top_logits": prefill_top_logits,
        },
        "decode": {
            "requested_generated_tokens": generated_tokens,
            "timed_incremental_steps": timed_decode_tokens,
            "sync_layers_for_timing": sync_decode_layers_for_timing,
            "sync_each_layer_for_timing": sync_decode_each_layer_for_timing,
            "sync_linear_attn_components_for_timing": sync_linear_attn_components_for_timing,
            "sync_self_attn_components_for_timing": env_flag_enabled("ULLM_SYNC_SELF_ATTN_COMPONENTS_FOR_TIMING"),
            "use_aq4_matvec_qkv_z_gate_beta": use_aq4_matvec_qkv_z_gate_beta,
            "use_aq4_matvec_pair_qkv_z": use_aq4_matvec_pair_qkv_z,
            "use_aq4_matvec_triple_self_attn_qkv": use_aq4_matvec_triple_self_attn_qkv,
            "use_aq4_matvec_pair_self_attn_qk": use_aq4_matvec_pair_self_attn_qk,
            "use_paged_decode_sigmoid_gate_self_attn": !env_flag_enabled("ULLM_DISABLE_PAGED_DECODE_SIGMOID_GATE_SELF_ATTN"),
            "positions": decode_positions,
            "step_wall_ms": decode_step_ms,
            "step_wall_summary": decode_step_summary,
            "embedding_step_ms": decode_embedding_ms,
            "layers_step_ms": decode_layers_ms,
            "layer_step_ms": decode_layer_step_ms,
            "linear_attn_component_step_ms": decode_linear_attn_component_step_ms,
            "self_attn_component_step_ms": decode_self_attn_component_step_ms,
            "lm_head_step_ms": decode_lm_head_ms,
            "wall_ms": decode_ms,
            "timed_step_tps": tps(timed_decode_tokens, decode_ms),
            "end_to_end_generated_tps": tps(generated_tokens, total_ms),
            "last_top_logits": last_top_logits,
        },
        "throughput": {
            "total_model_input_tokens": prompt_token_ids.len() + timed_decode_tokens,
            "model_input_tps": tps(prompt_token_ids.len() + timed_decode_tokens, prefill_ms + decode_ms),
            "total_wall_ms": total_ms,
        },
        "timing_ms": {
            "embedding_read": embed_ms,
            "embedding_runtime_load": embedding_runtime_load_ms,
            "layer_load": layer_load_ms,
            "lm_head_load": lm_head_load_ms,
            "prefill": prefill_ms,
            "decode": decode_ms,
            "total": total_ms,
        },
        "memory": {
            "vram_baseline_bytes": serde_json::Value::Null,
            "vram_peak_bytes": serde_json::Value::Null,
            "vram_consumed_bytes": serde_json::Value::Null,
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_allocated_blocks": kv_cache_allocated_blocks,
            "kv_cache_free_blocks": 0,
            "kv_cache_block_size": block_size,
            "kv_cache_self_attn_layers": self_attn_layer_count,
            "kv_cache_value_dtype": "f32",
            "cache_blocks": cache_blocks,
            "block_size": block_size,
        },
        "correctness": {
            "verified": true,
            "nan_or_inf_detected": false,
        },
        "notes": [
            "This path keeps selected linear-attn and self-attn weights resident as compact AQ4 payloads and runs direct AQ4 matvec without full f32 materialization.",
            "Decode layer_step_ms records CPU launch timing and can shift deferred GPU work to later host-boundary reads; use layers_step_ms and step_wall_summary for throughput comparisons."
        ],
        "verified": true,
    });
    serde_json::to_string_pretty(&report).map_err(|err| {
        format!("failed to encode incremental token-id generate smoke report: {err}")
    })
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_logits_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    token_ids: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    package_token_ids_logits_smoke_impl_with_sq_overlay(
        path,
        device_index,
        chunk_bytes,
        layer_indices,
        token_ids,
        top_k,
        lm_head_chunk_rows,
        rotary_dim,
        rope_base,
        position_offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_token_ids_logits_smoke_impl_with_sq_overlay(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    token_ids: Vec<usize>,
    top_k: usize,
    lm_head_chunk_rows: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    sq_artifact: Option<&ullm_engine::sq::SqFp8Artifact>,
) -> Result<String, String> {
    if token_ids.is_empty() {
        return Err("package token-id logits smoke requires at least one token ID".to_string());
    }
    if layer_indices.is_empty() {
        return Err("package token-id logits smoke requires at least one layer".to_string());
    }

    let run_started = Instant::now();
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let sq_overlay = sq_artifact.map(|artifact| Qwen3PackageSqOverlay {
        artifact,
        row_chunk: 256,
    });
    let sq_overlay_json = sq_artifact.map(|artifact| {
        let candidate_legacy = if artifact.manifest.candidate.id == FORMAT_SQ8_0 {
            None
        } else {
            Some(artifact.manifest.candidate.id.as_str())
        };
        let implementation_id = artifact
            .manifest
            .candidate
            .implementation_id
            .as_deref()
            .or(candidate_legacy)
            .unwrap_or("none");
        serde_json::json!({
            "artifact": artifact.artifact_dir,
            "candidate": FORMAT_SQ8_0,
            "candidate_legacy": candidate_legacy,
            "format_id": FORMAT_SQ8_0,
            "implementation_id": implementation_id,
            "schema_version": artifact.manifest.schema_version,
            "fp8_tensor_count": artifact.manifest.storage.fp8_tensor_count,
            "passthrough_tensor_count": artifact.manifest.storage.passthrough_tensor_count,
            "row_chunk": 256,
        })
    });

    let embed_started = Instant::now();
    let embedding_rows =
        read_named_passthrough_f32_rows(path, QWEN3_EMBED_TOKENS_TENSOR, &token_ids)
            .map_err(|err| format!("failed to read token embedding rows: {err}"))?;
    let hidden = embedding_rows.columns;
    if hidden == 0 {
        return Err("embedding hidden size must be greater than zero".to_string());
    }
    if embedding_rows.shape.len() != 2 {
        return Err(format!(
            "embedding tensor must be 2D, got shape {:?}",
            embedding_rows.shape
        ));
    }
    if embedding_rows.columns != hidden {
        return Err(format!(
            "embedding hidden size mismatch: columns={} hidden={hidden}",
            embedding_rows.columns
        ));
    }
    let sequence_len = token_ids.len();
    let expected_embedding_values = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "embedding output size overflows".to_string())?;
    if embedding_rows.values.len() != expected_embedding_values {
        return Err(format!(
            "embedding output length mismatch: expected {} got {}",
            expected_embedding_values,
            embedding_rows.values.len()
        ));
    }
    let embed_ms = embed_started.elapsed().as_secs_f64() * 1000.0;

    let block_size = sequence_len;
    let cache_blocks = 1_usize;
    let block_table = vec![0_u32];
    let mut residual_sequence = embedding_rows.values;

    let mut layer_kinds = Vec::with_capacity(layer_indices.len());
    let mut self_attn_shape = None;
    let mut self_attn_rotary_dim = None;
    let mut layer_load_ms = 0.0_f64;
    let layer_started = Instant::now();
    for (layer_position, &layer_index) in layer_indices.iter().enumerate() {
        let layer_kind = package_decoder_layer_kind(path, layer_index)
            .map_err(|err| format!("failed to identify package layer {layer_index}: {err}"))?;
        layer_kinds.push(layer_kind.as_str());
        match layer_kind {
            PackageDecoderLayerKind::SelfAttention => {
                let layer_load_started = Instant::now();
                let layer = if let Some(overlay) = sq_overlay.as_ref() {
                    qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay(
                        &mut context,
                        &mut stream,
                        path,
                        chunk_bytes,
                        layer_index,
                        Some(overlay),
                    )
                } else {
                    qwen3_package_decoder_layer_runtime_from_package(
                        &mut context,
                        &mut stream,
                        path,
                        chunk_bytes,
                        layer_index,
                    )
                }
                .map_err(|err| {
                    format!("failed to load self-attn package layer {layer_index}: {err}")
                })?;
                layer_load_ms += layer_load_started.elapsed().as_secs_f64() * 1000.0;
                if layer.runtime_shape.hidden != hidden {
                    return Err(format!(
                        "self-attn layer {layer_index} hidden mismatch: layer_hidden={} embedding_hidden={hidden}",
                        layer.runtime_shape.hidden
                    ));
                }
                let rotary_dim = parse_package_token_ids_rotary_dim(
                    layer.runtime_shape.head_dim,
                    rotary_dim.as_deref(),
                )?;
                if let Some(previous) = self_attn_rotary_dim {
                    if previous != rotary_dim {
                        return Err(format!(
                            "self-attn rotary dim changed: previous={previous} current={rotary_dim}"
                        ));
                    }
                } else {
                    self_attn_rotary_dim = Some(rotary_dim);
                }
                if self_attn_shape.is_none() {
                    self_attn_shape = Some(serde_json::json!({
                        "q_heads": layer.runtime_shape.q_heads,
                        "kv_heads": layer.runtime_shape.kv_heads,
                        "head_dim": layer.runtime_shape.head_dim,
                        "value_dim": layer.runtime_shape.value_dim,
                        "rotary_dim": rotary_dim,
                    }));
                }
                let decode_shape = PagedDecodeShape {
                    block_size,
                    cache_blocks,
                    q_heads: layer.runtime_shape.q_heads,
                    kv_heads: layer.runtime_shape.kv_heads,
                    head_dim: layer.runtime_shape.head_dim,
                    value_dim: layer.runtime_shape.value_dim,
                };
                let prepared = qwen3_self_attn_prepare_model_loop_sequence_smoke(
                    &mut context,
                    &mut stream,
                    &layer.weights.self_attn,
                    residual_sequence,
                    sequence_len,
                    rotary_dim,
                    rope_base,
                    position_offset,
                    &layer.input_norm,
                    &layer.q_norm,
                    &layer.k_norm,
                    &block_table,
                    block_size,
                    cache_blocks,
                    &format!(
                        "package-token-ids-logits-smoke layer {} position {}",
                        layer.layer_index, layer_position
                    ),
                )?;
                let layer_output = qwen3_decoder_layer_sequence_to_host_f32(
                    &layer.weights,
                    &mut context,
                    &mut stream,
                    decode_shape,
                    &block_table,
                    prepared.softmax_scale,
                    1e-5_f32,
                    &prepared.q_rope,
                    &prepared.k_rope,
                    &prepared.v_projected,
                    prepared.q_gate.as_deref(),
                    &prepared.residual_sequence,
                    sequence_len,
                )
                .map_err(|err| {
                    format!("failed to run package token-id logits layer {layer_index}: {err}")
                })?;
                residual_sequence = layer_output.layer_output;
            }
            PackageDecoderLayerKind::LinearAttention => {
                let run = package_linear_attn_mlp_block_sequence_run(
                    path,
                    device_index,
                    chunk_bytes,
                    layer_index,
                    sequence_len,
                    residual_sequence,
                    None,
                    None,
                )
                .map_err(|err| {
                    format!("failed to run linear-attn package layer {layer_index}: {err}")
                })?;
                if run.layer_output.len() != expected_embedding_values {
                    return Err(format!(
                        "linear-attn layer {layer_index} output length mismatch: got {} expected {}",
                        run.layer_output.len(),
                        expected_embedding_values
                    ));
                }
                residual_sequence = run.layer_output;
            }
        }
    }
    let layer_ms = layer_started.elapsed().as_secs_f64() * 1000.0;

    let final_started = Instant::now();
    let mut final_norm = read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, chunk_bytes)
        .map_err(|err| format!("failed to read final RMSNorm tensor: {err}"))?;
    final_norm.values =
        effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
    if final_norm.values.len() != hidden {
        return Err(format!(
            "final RMSNorm length mismatch: len={} hidden={}",
            final_norm.values.len(),
            hidden
        ));
    }
    let final_token_start = (sequence_len - 1)
        .checked_mul(hidden)
        .ok_or_else(|| "final token slice start overflows".to_string())?;
    let final_token_end = final_token_start
        .checked_add(hidden)
        .ok_or_else(|| "final token slice end overflows".to_string())?;
    let final_hidden = &residual_sequence[final_token_start..final_token_end];
    let final_normed = runtime_host_rmsnorm_f32(final_hidden, &final_norm.values, 1e-6_f32);
    if final_normed.len() != hidden || final_normed.iter().any(|value| !value.is_finite()) {
        return Err("final normalized hidden state contains invalid values".to_string());
    }
    let final_norm_ms = final_started.elapsed().as_secs_f64() * 1000.0;

    let lm_head_started = Instant::now();
    let (lm_head_report, top_logits) = match package_lm_head_top_k_from_rows(
        path,
        &final_normed,
        top_k,
        lm_head_chunk_rows,
    ) {
        Ok((lm_head_vocab, lm_head_dtype, lm_head_shape, top_logits)) => (
            serde_json::json!({
                "tensor": QWEN3_LM_HEAD_TENSOR,
                "mode": PackageLmHeadMode::CpuChunked.as_str(),
                "dtype": lm_head_dtype,
                "shape": lm_head_shape,
                "vocab": lm_head_vocab,
                "chunk_rows": lm_head_chunk_rows,
            }),
            top_logits,
        ),
        Err(cpu_err) => {
            let resident_load_started = Instant::now();
            let mut lm_head_runtime = PackageLmHeadRuntime::load(
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
                        "failed to compute lm_head top-k: cpu_chunked_error={cpu_err}; resident_error={resident_err}"
                    )
                })?;
            let resident_load_ms = resident_load_started.elapsed().as_secs_f64() * 1000.0;
            let top_logits = lm_head_runtime.top_logits(path, &mut stream, &final_normed, top_k)?;
            (lm_head_runtime.report_json(resident_load_ms), top_logits)
        }
    };
    let lm_head_ms = lm_head_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = run_started.elapsed().as_secs_f64() * 1000.0;

    let top_logits_json = top_logits
        .iter()
        .map(|entry| {
            serde_json::json!({
                "token_id": entry.token_id,
                "logit": entry.logit,
            })
        })
        .collect::<Vec<_>>();
    let report = serde_json::json!({
        "schema_version": "package-token-ids-logits-smoke-v0.1",
        "package": path,
        "sq_overlay": sq_overlay_json,
        "backend": info.backend.to_string(),
        "device_index": device_index,
        "device_name": info.name,
        "layers": layer_indices,
        "layer_kinds": layer_kinds,
        "token_ids": token_ids,
        "sequence_len": sequence_len,
        "hidden": hidden,
        "self_attn": self_attn_shape,
        "rotary_dim": self_attn_rotary_dim,
        "rope_base": rope_base,
        "position_offset": position_offset,
        "embedding": {
            "tensor": QWEN3_EMBED_TOKENS_TENSOR,
            "dtype": embedding_rows.dtype,
            "shape": embedding_rows.shape,
        },
        "final_norm": {
            "tensor": QWEN3_FINAL_NORM_TENSOR,
            "dtype": final_norm.dtype,
            "shape": final_norm.shape,
        },
        "lm_head": lm_head_report,
        "top_k": top_k,
        "top_logits": top_logits_json,
        "timing_ms": {
            "layer_load": layer_load_ms,
            "embedding_read": embed_ms,
            "layers": layer_ms,
            "final_norm": final_norm_ms,
            "lm_head_top_k": lm_head_ms,
            "total": total_ms,
        },
        "verified": true,
    });
    serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to encode token-id logits smoke report: {err}"))
}
