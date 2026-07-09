fn package_lm_head_top_k_from_rows(
    path: &str,
    hidden: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<(usize, String, Vec<u64>, Vec<PackageTokenLogit>), String> {
    if hidden.is_empty() {
        return Err("lm_head top-k hidden vector must not be empty".to_string());
    }
    if top_k == 0 || chunk_rows == 0 {
        return Err("lm_head top-k and chunk_rows must be greater than zero".to_string());
    }
    let first_row = read_named_passthrough_f32_row_range(path, QWEN3_LM_HEAD_TENSOR, 0, 1)
        .map_err(|err| format!("failed to read lm_head first row: {err}"))?;
    if first_row.shape.len() != 2 {
        return Err(format!(
            "lm_head must be 2D, got shape {:?}",
            first_row.shape
        ));
    }
    let vocab = usize::try_from(first_row.shape[0])
        .map_err(|_| "lm_head vocab size is too large for this host".to_string())?;
    if first_row.columns != hidden.len() {
        return Err(format!(
            "lm_head hidden size mismatch: columns={} hidden={}",
            first_row.columns,
            hidden.len()
        ));
    }

    let mut top_logits = Vec::new();
    let mut start = 0_usize;
    while start < vocab {
        let end = start
            .checked_add(chunk_rows)
            .map(|candidate| candidate.min(vocab))
            .ok_or_else(|| "lm_head chunk end overflows".to_string())?;
        let rows =
            read_named_passthrough_f32_row_range(path, QWEN3_LM_HEAD_TENSOR, start, end - start)
                .map_err(|err| format!("failed to read lm_head rows {start}..{end}: {err}"))?;
        if rows.columns != hidden.len() || rows.shape != first_row.shape {
            return Err(format!(
                "lm_head chunk shape changed for rows {start}..{end}: columns={} shape={:?}",
                rows.columns, rows.shape
            ));
        }
        for (offset, row) in rows.values.chunks_exact(hidden.len()).enumerate() {
            let mut logit = 0.0_f32;
            for (weight, value) in row.iter().zip(hidden.iter()) {
                logit += weight * value;
            }
            if !logit.is_finite() {
                return Err(format!(
                    "lm_head logit for token {} is not finite",
                    start + offset
                ));
            }
            top_logits.push(PackageTokenLogit {
                token_id: start + offset,
                logit,
            });
        }
        top_logits.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        top_logits.truncate(top_k);
        start = end;
    }

    Ok((vocab, first_row.dtype, first_row.shape, top_logits))
}

#[cfg(test)]
mod package_token_ids_logits_tests {
    use super::*;

    #[test]
    fn package_token_ids_layer_indices_default_to_qwen35_layers() {
        let layers = parse_package_token_ids_layer_indices(None).unwrap();
        assert_eq!(layers.len(), QWEN35_9B_DEFAULT_LAYER_COUNT);
        assert_eq!(layers.first().copied(), Some(0));
        assert_eq!(
            layers.last().copied(),
            Some(QWEN35_9B_DEFAULT_LAYER_COUNT - 1)
        );

        let layers = parse_package_token_ids_layer_indices(Some("all".to_string())).unwrap();
        assert_eq!(layers.len(), QWEN35_9B_DEFAULT_LAYER_COUNT);

        let layers = parse_package_token_ids_layer_indices(Some("0,2,4".to_string())).unwrap();
        assert_eq!(layers, vec![0, 2, 4]);
    }

    #[test]
    fn package_prompt_token_ids_accepts_len_form() {
        let tokens = parse_package_prompt_token_ids(Some("len:5".to_string())).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);

        let tokens = parse_package_prompt_token_ids(Some("len=3".to_string())).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);

        assert!(parse_package_prompt_token_ids(Some("len:0".to_string())).is_err());
    }

    #[test]
    fn package_prompt_token_ids_batch_accepts_len_count_or_semicolon_lists() {
        let batch = parse_package_prompt_token_ids_batch(Some("len:3x2".to_string())).unwrap();
        assert_eq!(batch, vec![vec![1, 2, 3], vec![1, 2, 3]]);

        let batch =
            parse_package_prompt_token_ids_batch(Some("1,2,3; len:2; 7,8".to_string())).unwrap();
        assert_eq!(batch, vec![vec![1, 2, 3], vec![1, 2], vec![7, 8]]);

        assert!(parse_package_prompt_token_ids_batch(Some("len:0x2".to_string())).is_err());
        assert!(parse_package_prompt_token_ids_batch(Some("1,2;;3,4".to_string())).is_err());
    }

    #[test]
    fn package_manifest_layer_entries_detects_mixed_layer_order() {
        let root = std::env::temp_dir().join(format!(
            "ullm-manifest-layer-entries-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("payload")).unwrap();
        for name in ["idx.bin", "scale.bin", "codebook.bin"] {
            std::fs::write(root.join("payload").join(name), [0_u8; 4]).unwrap();
        }
        std::fs::write(
            root.join("manifest.json"),
            r#"{
              "schema_version": "test",
              "tensors": [
                {
                  "name": "model.language_model.layers.2.linear_attn.in_proj_qkv.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                },
                {
                  "name": "model.language_model.layers.0.self_attn.q_proj.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                },
                {
                  "name": "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                }
              ]
            }"#,
        )
        .unwrap();

        let entries = package_manifest_layer_entries(root.to_str().unwrap()).unwrap();
        assert_eq!(
            entries,
            vec![
                PackageManifestLayerEntry {
                    layer_index: 0,
                    kind: PackageDecoderLayerKind::SelfAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 1,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 2,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
            ]
        );
        assert!(package_layer_entries_are_contiguous(&entries));

        let layers = parse_package_token_ids_layer_indices_for_package(
            root.to_str().unwrap(),
            Some("manifest-all".to_string()),
        )
        .unwrap();
        assert_eq!(layers, vec![0, 1, 2]);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_manifest_layer_entries_detects_qwen3_namespace() {
        let root = std::env::temp_dir().join(format!(
            "ullm-manifest-layer-entries-qwen3-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("payload")).unwrap();
        for name in ["idx.bin", "scale.bin", "codebook.bin"] {
            std::fs::write(root.join("payload").join(name), [0_u8; 4]).unwrap();
        }
        std::fs::write(
            root.join("manifest.json"),
            r#"{
              "schema_version": "test",
              "tensors": [
                {
                  "name": "model.layers.2.linear_attn.in_proj_qkv.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                },
                {
                  "name": "model.layers.0.self_attn.q_proj.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                },
                {
                  "name": "model.layers.1.linear_attn.in_proj_qkv.weight",
                  "shape": [1, 1],
                  "elements": 1,
                  "groups": 1,
                  "index_file": "payload/idx.bin",
                  "scale_file": "payload/scale.bin",
                  "codebook_file": "payload/codebook.bin"
                }
              ]
            }"#,
        )
        .unwrap();

        let entries = package_manifest_layer_entries(root.to_str().unwrap()).unwrap();
        assert_eq!(
            entries,
            vec![
                PackageManifestLayerEntry {
                    layer_index: 0,
                    kind: PackageDecoderLayerKind::SelfAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 1,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 2,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
            ]
        );
        assert!(package_layer_entries_are_contiguous(&entries));

        let layers = parse_package_token_ids_layer_indices_for_package(
            root.to_str().unwrap(),
            Some("manifest-all".to_string()),
        )
        .unwrap();
        assert_eq!(layers, vec![0, 1, 2]);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_generated_tokens_batch_accepts_scalar_or_csv() {
        assert_eq!(
            parse_package_generated_tokens_batch(Some("8".to_string()), 3).unwrap(),
            vec![8, 8, 8]
        );
        assert_eq!(
            parse_package_generated_tokens_batch(Some("1,2,3".to_string()), 3).unwrap(),
            vec![1, 2, 3]
        );
        assert!(parse_package_generated_tokens_batch(Some("1,2".to_string()), 3).is_err());
        assert!(parse_package_generated_tokens_batch(Some("0".to_string()), 3).is_err());
    }

    #[test]
    fn cold_prefill_attention_work_counts_triangular_prompt_pairs() {
        assert_eq!(
            cold_prefill_attention_work_tokens_from_lengths(&[3, 5]).unwrap(),
            21
        );
        assert_eq!(
            cold_prefill_attention_work_tokens_from_lengths(&[512, 512]).unwrap(),
            262656
        );
    }

    #[test]
    fn package_stop_token_ids_accepts_none_or_csv() {
        assert_eq!(
            parse_package_stop_token_ids(None).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            parse_package_stop_token_ids(Some("none".to_string())).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            parse_package_stop_token_ids(Some("1, 2,3".to_string())).unwrap(),
            vec![1, 2, 3]
        );
        assert!(parse_package_stop_token_ids(Some("1,,2".to_string())).is_err());
    }

    #[test]
    fn package_stop_token_sequences_accepts_none_or_semicolon_csv() {
        assert_eq!(
            parse_package_stop_token_sequences(None).unwrap(),
            Vec::<Vec<usize>>::new()
        );
        assert_eq!(
            parse_package_stop_token_sequences(Some("none".to_string())).unwrap(),
            Vec::<Vec<usize>>::new()
        );
        assert_eq!(
            parse_package_stop_token_sequences(Some("1, 2; 3,4,5".to_string())).unwrap(),
            vec![vec![1, 2], vec![3, 4, 5]]
        );
        assert!(parse_package_stop_token_sequences(Some("1,2;;3".to_string())).is_err());
    }

    #[test]
    fn matched_stop_token_sequence_matches_only_suffix() {
        assert_eq!(
            matched_stop_token_sequence(&[9, 1, 2], &[vec![1, 2]]),
            Some(vec![1, 2])
        );
        assert_eq!(matched_stop_token_sequence(&[9, 1], &[vec![1, 2]]), None);
        assert_eq!(matched_stop_token_sequence(&[1, 2, 9], &[vec![1, 2]]), None);
    }

    #[test]
    fn package_report_helpers_read_top_token_and_timing() {
        let report = serde_json::json!({
            "timing_ms": {
                "total": 12.5
            },
            "top_logits": [
                {
                    "token_id": 42,
                    "logit": 3.25
                }
            ]
        });
        assert_eq!(package_report_total_ms(&report, "test").unwrap(), 12.5);
        assert_eq!(package_report_top_token_id(&report, "test").unwrap(), 42);
        assert!(package_report_top_logits_json(&report, "test").is_ok());
    }

    #[test]
    fn package_lm_head_top_k_from_rows_reads_chunked_passthrough() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-package-token-lm-head-top-k-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(
            root.join("passthrough/lm_head.raw"),
            [
                1.0_f32.to_le_bytes(),
                0.0_f32.to_le_bytes(),
                0.0_f32.to_le_bytes(),
                3.0_f32.to_le_bytes(),
                2.0_f32.to_le_bytes(),
                2.0_f32.to_le_bytes(),
                (-1.0_f32).to_le_bytes(),
                10.0_f32.to_le_bytes(),
            ]
            .concat(),
        )
        .unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "lm_head.weight",
                "dtype": "F32",
                "shape": [4, 2],
                "elements": 8,
                "payload_bytes": 32,
                "payload_encoding": "raw_safetensors_payload",
                "payload_file": "passthrough/lm_head.raw"
              }]
            }"#,
        )
        .unwrap();

        let (vocab, dtype, shape, top_logits) =
            package_lm_head_top_k_from_rows(root.to_str().unwrap(), &[1.0, 1.0], 2, 2).unwrap();
        assert_eq!(vocab, 4);
        assert_eq!(dtype, "F32");
        assert_eq!(shape, vec![4, 2]);
        assert_eq!(top_logits.len(), 2);
        assert_eq!(top_logits[0].token_id, 3);
        assert_eq!(top_logits[0].logit, 9.0);
        assert_eq!(top_logits[1].token_id, 2);
        assert_eq!(top_logits[1].logit, 4.0);

        fs::remove_dir_all(root).unwrap();
    }
}

fn package_linear_attn_aux_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    aux: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-aux-smoke requires a .ullm.d path");
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
    let aux = match parse_linear_attn_aux(aux.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_aux = match aux {
        LinearAttnAux::ALog => vec![(
            "a-log",
            format!("model.language_model.layers.{layer_index}.linear_attn.A_log"),
        )],
        LinearAttnAux::DtBias => vec![(
            "dt-bias",
            format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias"),
        )],
        LinearAttnAux::Conv1d => vec![(
            "conv1d",
            format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight"),
        )],
        LinearAttnAux::Norm => vec![(
            "norm",
            format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight"),
        )],
        LinearAttnAux::All => vec![
            (
                "a-log",
                format!("model.language_model.layers.{layer_index}.linear_attn.A_log"),
            ),
            (
                "dt-bias",
                format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias"),
            ),
            (
                "conv1d",
                format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight"),
            ),
            (
                "norm",
                format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight"),
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

    for (aux_name, tensor_name) in requested_aux {
        let selector = TensorSelector::Name(tensor_name.clone());
        let bundle = match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector)
        {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package passthrough tensor {tensor_name}: {err}");
                return ExitCode::from(1);
            }
        };

        if let Err(err) = validate_passthrough_shape_elements(&bundle) {
            eprintln!("invalid passthrough shape for {tensor_name}: {err}");
            return ExitCode::from(1);
        }

        let elements = match usize::try_from(bundle.elements) {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("passthrough tensor {tensor_name} has zero elements");
                return ExitCode::from(1);
            }
            Err(_) => {
                eprintln!(
                    "passthrough tensor {tensor_name} element count is too large for this host"
                );
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
        let payload_bytes = if bundle.payload_bytes == 0 {
            bundle.payload_file.bytes
        } else {
            bundle.payload_bytes
        };
        if payload.len() != elements {
            eprintln!(
                "passthrough tensor element count mismatch for {tensor_name}: expected {elements} got {}",
                payload.len()
            );
            return ExitCode::from(1);
        }

        let payload_f32_bytes = encode_f32_to_bytes(&payload);
        let mut buffer = match context.alloc_buffer(payload_f32_bytes.len()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate runtime buffer for {tensor_name}: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = buffer.copy_from_host(0, &payload_f32_bytes, Some(&mut stream)) {
            eprintln!("failed to copy payload for {tensor_name} into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after payload copy: {err}");
            return ExitCode::from(1);
        }

        let mut output = vec![0_u8; payload_f32_bytes.len()];
        if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
            eprintln!("failed to copy payload back for {tensor_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after payload readback: {err}");
            return ExitCode::from(1);
        }
        if payload_f32_bytes != output {
            eprintln!("runtime roundtrip mismatch for {tensor_name}");
            return ExitCode::from(1);
        }

        let preview = decode_f32_le_values(&output);
        let preview_count = preview.len().min(8);
        println!(
            "package-linear-attn-aux-smoke package={} layer={} aux={} tensor=\"{}\" dtype={} elements={} shape={} payload_bytes={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            aux_name,
            tensor_name,
            dtype,
            elements,
            format_u64_shape(&bundle.shape),
            payload_bytes,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&preview[..preview_count])
        );
    }
    ExitCode::SUCCESS
}

fn package_linear_attn_qkv_norm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-qkv-norm-smoke requires a .ullm.d path");
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

    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");

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
    if norm_elements != 128 {
        eprintln!("RMSNorm tensor must have 128 elements, got {norm_elements}");
        return ExitCode::from(1);
    }
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
            "passthrough tensor element count mismatch for {norm_tensor}: expected {norm_elements} got {}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows < 128 {
        eprintln!(
            "qkv tensor {qkv_tensor} has too few rows for preview validation: rows={qkv_rows}, expected at least 128"
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(qkv_cols);
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic qkv input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let qkv_output_bytes = match qkv_rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut qkv_output = match context.alloc_buffer(qkv_output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &qkv_matrix,
        &input_buffer,
        qkv_rows,
        qkv_cols,
        &mut qkv_output,
        Some(&mut stream),
    ) {
        eprintln!("failed to run qkv matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv matvec: {err}");
        return ExitCode::from(1);
    }

    let qkv_preview_count = 128_usize;
    let mut qkv_preview_bytes = vec![0_u8; qkv_preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = qkv_output.copy_to_host(0, &mut qkv_preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy qkv preview output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv preview copy: {err}");
        return ExitCode::from(1);
    }
    let qkv_output_preview = decode_f32_le_values(&qkv_preview_bytes);
    let norm_input = qkv_output_preview;

    let epsilon = 1e-5_f32;
    let expected = runtime_host_rmsnorm_f32(&norm_input, &norm_weight, epsilon);
    if expected.len() != norm_elements {
        eprintln!("failed to build deterministic RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_input_bytes = encode_f32_to_bytes(&norm_input);
    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_input_buffer = match context.alloc_buffer(norm_input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv-rmsnorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_input_buffer.copy_from_host(0, &norm_input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy qkv output preview to qkv-rmsnorm input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv-rmsnorm input copy: {err}");
        return ExitCode::from(1);
    }
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv-rmsnorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &norm_input_buffer,
        &norm_weight_buffer,
        norm_elements,
        epsilon,
        &mut norm_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut norm_output_bytes = vec![0_u8; norm_weight_bytes.len()];
    if let Err(err) = norm_output_buffer.copy_to_host(0, &mut norm_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy qkv-rmsnorm output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv-rmsnorm output copy: {err}");
        return ExitCode::from(1);
    }
    let norm_output = decode_f32_le_values(&norm_output_bytes);

    if norm_output.len() != expected.len() {
        eprintln!(
            "runtime RMSNorm output size mismatch: expected {} got {}",
            expected.len(),
            norm_output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in norm_output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-qkv-norm-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    let qkv_preview = &norm_input[..8.min(norm_input.len())];
    let norm_preview = &norm_output[..8.min(norm_output.len())];
    println!(
        "package-linear-attn-qkv-norm-smoke package={} layer={} qkv_tensor=\"{}\" norm_tensor=\"{}\" hidden={} qkv_rows={} norm_elements={} backend={} device_index={} name=\"{}\" qkv_preview={} norm_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        norm_tensor,
        qkv_cols,
        qkv_rows,
        norm_elements,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(qkv_preview),
        format_f32_preview(norm_preview),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_conv1d_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-conv1d-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");

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

    let selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight = match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let expected_conv_elements = match conv_channels.checked_mul(kernel_size) {
        Some(value) => value,
        None => {
            eprintln!("conv1d weight element count overflows");
            return ExitCode::from(1);
        }
    };
    if conv_weight.len() != expected_conv_elements {
        eprintln!(
            "conv1d weight element count mismatch: expected {expected_conv_elements} got {}",
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = match qkv_rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv step byte size overflows");
            return ExitCode::from(1);
        }
    };
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match qkv_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(qkv_cols);
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = base_input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = (index % 17) as f32 - 8.0_f32;
                *value + (timestep as f32) * phase * 0.00025_f32
            })
            .collect::<Vec<_>>();
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy qkv input timestep {timestep} into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after qkv timestep {timestep}: {err}");
            return ExitCode::from(1);
        }

        let start = timestep * qkv_step_bytes;
        let end = start + qkv_step_bytes;
        if let Err(err) =
            qkv_step_buffer.copy_to_host(0, &mut qkv_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after qkv timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected.len() != qkv_sequence.len() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);

    if conv_output.len() != expected.len() {
        eprintln!(
            "runtime depthwise conv1d output size mismatch: expected {} got {}",
            expected.len(),
            conv_output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-conv1d-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    let qkv_preview = &qkv_sequence[..8.min(qkv_sequence.len())];
    let conv_preview = &conv_output[..8.min(conv_output.len())];
    println!(
        "package-linear-attn-conv1d-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" hidden={} channels={} sequence_len={} kernel_size={} dtype={} backend={} device_index={} name=\"{}\" qkv_preview={} conv_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        qkv_cols,
        qkv_rows,
        sequence_len,
        kernel_size,
        dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(qkv_preview),
        format_f32_preview(conv_preview),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_gate_beta_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-gate-beta-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");

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

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };

    let mut registry = WeightRegistry::new();
    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows == 0 || b_rows == 0 || a_cols == 0 || b_cols == 0 {
        eprintln!("linear attention a/b projection matrix has zero dimension");
        return ExitCode::from(1);
    }
    if a_rows != b_rows || a_cols != b_cols {
        eprintln!(
            "linear attention a/b projection shapes differ: a=[{a_rows},{a_cols}] b=[{b_rows},{b_cols}]"
        );
        return ExitCode::from(1);
    }
    let heads = a_rows;
    let hidden = a_cols;
    if a_log.len() != heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match heads: len={} heads={heads}",
            a_log.len()
        );
        return ExitCode::from(1);
    }
    if dt_bias.len() != heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match heads: len={} heads={heads}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let step_bytes = match heads.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta step byte size overflows");
            return ExitCode::from(1);
        }
    };
    let sequence_bytes_len = match step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate beta input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(hidden);
    let mut a_sequence_bytes = vec![0_u8; sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = base_input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = (index % 17) as f32 - 8.0_f32;
                *value + (timestep as f32) * phase * 0.00025_f32
            })
            .collect::<Vec<_>>();
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy gate beta input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_buffer,
            heads,
            hidden,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_buffer,
            heads,
            hidden,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after gate beta timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let start = timestep * step_bytes;
        let end = start + step_bytes;
        if let Err(err) =
            a_step_buffer.copy_to_host(0, &mut a_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) =
            b_step_buffer.copy_to_host(0, &mut b_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after gate beta timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_bytes = vec![0_u8; sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);

    let mut max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-gate-beta-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-gate-beta-smoke package={} layer={} a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" hidden={} heads={} sequence_len={} a_log_dtype={} dt_bias_dtype={} backend={} device_index={} name=\"{}\" a_preview={} b_preview={} gate_preview={} beta_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        hidden,
        heads,
        sequence_len,
        a_log_dtype,
        dt_bias_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&a_sequence[..8.min(a_sequence.len())]),
        format_f32_preview(&b_sequence[..8.min(b_sequence.len())]),
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&beta_output[..8.min(beta_output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_recurrent_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-recurrent-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = value_heads * value_dim;
    let recurrent_channels = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;

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

    let conv_selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &conv_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    if conv_channels != recurrent_channels {
        eprintln!(
            "conv1d channels must match Qwen3.5 linear attention q/k/v layout: conv_channels={conv_channels}, expected={recurrent_channels}"
        );
        return ExitCode::from(1);
    }
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let conv_dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight =
        match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, conv_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_weight.len() != conv_channels * kernel_size {
        eprintln!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_log.len() != value_heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match value heads: len={} value_heads={value_heads}",
            a_log.len()
        );
        return ExitCode::from(1);
    }

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if dt_bias.len() != value_heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match value heads: len={} value_heads={value_heads}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }

    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows != value_heads || b_rows != value_heads {
        eprintln!(
            "linear attention a/b rows must match value_heads={value_heads}: a_rows={a_rows}, b_rows={b_rows}"
        );
        return ExitCode::from(1);
    }
    if a_cols != qkv_cols || b_cols != qkv_cols {
        eprintln!(
            "linear attention a/b hidden sizes must match qkv hidden={qkv_cols}: a_cols={a_cols}, b_cols={b_cols}"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = qkv_rows * std::mem::size_of::<f32>();
    let gate_beta_step_bytes = value_heads * std::mem::size_of::<f32>();
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let gate_beta_sequence_bytes_len = match gate_beta_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match qkv_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention input byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(qkv_cols);
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_buffer,
            value_heads,
            qkv_cols,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_buffer,
            value_heads,
            qkv_cols,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after linear attention timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let qkv_start = timestep * qkv_step_bytes;
        let qkv_end = qkv_start + qkv_step_bytes;
        if let Err(err) = qkv_step_buffer.copy_to_host(
            0,
            &mut qkv_sequence_bytes[qkv_start..qkv_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let gate_beta_start = timestep * gate_beta_step_bytes;
        let gate_beta_end = gate_beta_start + gate_beta_step_bytes;
        if let Err(err) = a_step_buffer.copy_to_host(
            0,
            &mut a_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = b_step_buffer.copy_to_host(
            0,
            &mut b_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected_conv = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected_conv.is_empty() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);
    let mut conv_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected_conv.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-recurrent-smoke conv1d mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > conv_max_abs_diff {
            conv_max_abs_diff = diff;
        }
    }

    let conv_activated = runtime_host_silu_f32(&conv_output);
    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_activated,
        sequence_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to split linear attention qkv: {err}");
            return ExitCode::from(1);
        }
    };

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        value_heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        value_heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);
    let mut gate_beta_max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-recurrent-smoke gate/beta mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > gate_beta_max_abs_diff {
            gate_beta_max_abs_diff = diff;
        }
    }

    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent state element count overflows");
            return ExitCode::from(1);
        }
    };
    let output_elements = match sequence_len.checked_mul(v_elements_per_step) {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent output element count overflows");
            return ExitCode::from(1);
        }
    };
    let initial_state = vec![0.0_f32; state_elements];
    let mut expected_state = initial_state.clone();
    let expected_recurrent_output = runtime_host_linear_attn_recurrent_f32(
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
    if expected_recurrent_output.len() != output_elements {
        eprintln!("failed to build package linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&qkv_split.q);
    let k_bytes = encode_f32_to_bytes(&qkv_split.k);
    let v_bytes = encode_f32_to_bytes(&qkv_split.v);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let output_bytes_len = output_elements * std::mem::size_of::<f32>();
    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut recurrent_output_buffer = match context.alloc_buffer(output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate recurrent output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_output_buffer,
        &beta_output_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut recurrent_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }

    let mut recurrent_output_bytes = vec![0_u8; output_bytes_len];
    let mut final_state_bytes = vec![0_u8; state_bytes.len()];
    if let Err(err) =
        recurrent_output_buffer.copy_to_host(0, &mut recurrent_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let recurrent_output = decode_f32_le_values(&recurrent_output_bytes);
    let final_state = decode_f32_le_values(&final_state_bytes);
    let mut recurrent_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in recurrent_output
        .iter()
        .zip(expected_recurrent_output.iter())
    {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-recurrent-smoke output mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-recurrent-smoke state mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-recurrent-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} conv_dtype={} a_log_dtype={} dt_bias_dtype={} backend={} device_index={} name=\"{}\" q_preview={} k_preview={} v_preview={} gate_preview={} output_preview={} state_preview={} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        qkv_cols,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        conv_dtype,
        a_log_dtype,
        dt_bias_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&qkv_split.q[..8.min(qkv_split.q.len())]),
        format_f32_preview(&qkv_split.k[..8.min(qkv_split.k.len())]),
        format_f32_preview(&qkv_split.v[..8.min(qkv_split.v.len())]),
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&recurrent_output[..8.min(recurrent_output.len())]),
        format_f32_preview(&final_state[..8.min(final_state.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_post_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-post-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let value_heads = 32_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let epsilon = 1e-6_f32;

    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");

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

    let norm_selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &norm_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package linear attention norm tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&norm_bundle) {
        eprintln!("invalid linear attention norm shape for {norm_tensor}: {err}");
        return ExitCode::from(1);
    }
    let norm_dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight =
        match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, norm_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if norm_weight.len() != value_dim {
        eprintln!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (z_rows, z_cols, z_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &z_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {z_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if z_rows != hidden {
        eprintln!("z projection rows must match hidden={hidden}: z_rows={z_rows}");
        return ExitCode::from(1);
    }

    let (out_rows, out_cols, out_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &out_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {out_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if out_rows != z_cols || out_cols != hidden {
        eprintln!("out projection shape must be [{z_cols},{hidden}], got [{out_rows},{out_cols}]");
        return ExitCode::from(1);
    }

    let hidden_bytes = hidden * std::mem::size_of::<f32>();
    let sequence_elements = match sequence_len.checked_mul(hidden) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post sequence element count overflows");
            return ExitCode::from(1);
        }
    };
    let sequence_bytes_len = match sequence_elements.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match z_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post input byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut hidden_input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate hidden input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut z_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(z_cols);
    let mut z_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) =
            hidden_input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream))
        {
            eprintln!("failed to copy z input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &z_matrix,
            &hidden_input_buffer,
            z_rows,
            z_cols,
            &mut z_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run z matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after z timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) =
            z_step_buffer.copy_to_host(0, &mut z_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy z timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after z timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let z_sequence = decode_f32_le_values(&z_sequence_bytes);

    let core_output = deterministic_linear_attn_core_output(sequence_len, value_heads, value_dim);
    let mut expected_normed = vec![0.0_f32; sequence_elements];
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let start = (timestep * value_heads + value_head) * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(&core_output[start..end], &norm_weight, epsilon);
            if normed.len() != value_dim {
                eprintln!("failed to build linear attention post RMSNorm reference");
                return ExitCode::from(1);
            }
            expected_normed[start..end].copy_from_slice(&normed);
        }
    }
    let expected_activated = runtime_host_silu_mul_f32(&z_sequence, &expected_normed);
    if expected_activated.len() != sequence_elements {
        eprintln!("failed to build linear attention gated RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy linear attention norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_input_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for row in 0..(sequence_len * value_heads) {
        let start = row * value_dim;
        let end = start + value_dim;
        let input_bytes = encode_f32_to_bytes(&core_output[start..end]);
        if let Err(err) = norm_input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
            &norm_input_buffer,
            &norm_weight_buffer,
            value_dim,
            epsilon,
            &mut norm_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run linear attention post rmsnorm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after norm row {row}: {err}");
            return ExitCode::from(1);
        }
        let byte_start = start * std::mem::size_of::<f32>();
        let byte_end = end * std::mem::size_of::<f32>();
        if let Err(err) = norm_output_buffer.copy_to_host(
            0,
            &mut normed_sequence_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy linear attention norm row {row} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after norm row {row} host copy: {err}");
            return ExitCode::from(1);
        }
    }
    let normed_sequence = decode_f32_le_values(&normed_sequence_bytes);
    let mut norm_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in normed_sequence.iter().zip(expected_normed.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!("package-linear-attn-post-smoke RMSNorm mismatch: max_abs_diff={diff}");
            return ExitCode::from(1);
        }
        if diff > norm_max_abs_diff {
            norm_max_abs_diff = diff;
        }
    }

    let mut z_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut activated_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = z_sequence_buffer.copy_from_host(0, &z_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy z sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) =
        normed_sequence_buffer.copy_from_host(0, &normed_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy normed sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &z_sequence_buffer,
        &normed_sequence_buffer,
        sequence_elements,
        &mut activated_sequence_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run linear attention post silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm silu_mul: {err}");
        return ExitCode::from(1);
    }
    let mut activated_sequence_bytes = vec![0_u8; sequence_bytes_len];
    if let Err(err) =
        activated_sequence_buffer.copy_to_host(0, &mut activated_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy activated sequence back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after activated sequence copy: {err}");
        return ExitCode::from(1);
    }
    let activated_sequence = decode_f32_le_values(&activated_sequence_bytes);
    let mut activation_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in activated_sequence.iter().zip(expected_activated.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!("package-linear-attn-post-smoke gated RMSNorm mismatch: max_abs_diff={diff}");
            return ExitCode::from(1);
        }
        if diff > activation_max_abs_diff {
            activation_max_abs_diff = diff;
        }
    }

    let out_matrix_bytes_len = match out_rows
        .checked_mul(out_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("out projection matrix byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
    if let Err(err) = out_matrix.copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy out projection matrix back to host for reference: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after out matrix host copy: {err}");
        return ExitCode::from(1);
    }
    let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
    let mut expected_output = Vec::with_capacity(sequence_len * out_rows);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_activated[start..end],
            out_rows,
            out_cols,
        );
        if output.len() != out_rows {
            eprintln!("failed to build linear attention post out projection reference");
            return ExitCode::from(1);
        }
        expected_output.extend_from_slice(&output);
    }

    let output_sequence_bytes_len = match sequence_len
        .checked_mul(out_rows)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention post output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut out_step_buffer = match context.alloc_buffer(out_rows * std::mem::size_of::<f32>()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection step buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_sequence_bytes = vec![0_u8; output_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) = out_input_buffer.copy_from_host(
            0,
            &activated_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &out_input_buffer,
            out_rows,
            out_cols,
            &mut out_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run out projection matvec timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }
        let output_start = timestep * out_rows * std::mem::size_of::<f32>();
        let output_end = output_start + out_rows * std::mem::size_of::<f32>();
        if let Err(err) = out_step_buffer.copy_to_host(
            0,
            &mut output_sequence_bytes[output_start..output_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let output_sequence = decode_f32_le_values(&output_sequence_bytes);
    let mut output_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output_sequence.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 2e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-post-smoke out projection mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > output_max_abs_diff {
            output_max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-post-smoke package={} layer={} z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} value_heads={} value_dim={} sequence_len={} norm_dtype={} backend={} device_index={} name=\"{}\" core_preview={} z_preview={} normed_preview={} activated_preview={} output_preview={} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        z_tensor,
        norm_tensor,
        out_tensor,
        hidden,
        value_heads,
        value_dim,
        sequence_len,
        norm_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&core_output[..8.min(core_output.len())]),
        format_f32_preview(&z_sequence[..8.min(z_sequence.len())]),
        format_f32_preview(&normed_sequence[..8.min(normed_sequence.len())]),
        format_f32_preview(&activated_sequence[..8.min(activated_sequence.len())]),
        format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_workflow_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    package_linear_attn_workflow_smoke_impl(
        "package-linear-attn-workflow-smoke",
        false,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
    )
}

fn package_linear_attn_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    package_linear_attn_workflow_smoke_impl(
        "package-linear-attn-block-smoke",
        true,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
    )
}

fn package_linear_attn_workflow_smoke_impl(
    command_name: &str,
    include_block: bool,
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden_size = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden_size;
    let recurrent_channels = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let epsilon = 1e-6_f32;

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

    let mut input_norm_dtype = String::new();
    let mut input_norm_weight = Vec::new();
    if include_block {
        let input_norm_selector = TensorSelector::Name(input_norm_tensor.clone());
        let input_norm_bundle = match ullm_engine::package::select_passthrough_payload_bundle(
            &path,
            &input_norm_selector,
        ) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package input RMSNorm tensor: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = validate_passthrough_shape_elements(&input_norm_bundle) {
            eprintln!("invalid input RMSNorm shape for {input_norm_tensor}: {err}");
            return ExitCode::from(1);
        }
        input_norm_dtype = match resolve_passthrough_dtype(&input_norm_bundle, &input_norm_tensor) {
            Ok(value) => value.to_string(),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        input_norm_weight = match read_passthrough_payload_f32_bytes(
            &input_norm_bundle,
            chunk_bytes,
            &input_norm_dtype,
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {input_norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
        if input_norm_weight.len() != hidden_size {
            eprintln!(
                "input RMSNorm length must match hidden_size={hidden_size}: len={}",
                input_norm_weight.len()
            );
            return ExitCode::from(1);
        }
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

    let conv_selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &conv_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    if conv_channels != recurrent_channels {
        eprintln!(
            "conv1d channels must match Qwen3.5 linear attention q/k/v layout: conv_channels={conv_channels}, expected={recurrent_channels}"
        );
        return ExitCode::from(1);
    }
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let conv_dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight =
        match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, conv_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_weight.len() != conv_channels * kernel_size {
        eprintln!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_log.len() != value_heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match value_heads={value_heads}: len={}",
            a_log.len()
        );
        return ExitCode::from(1);
    }

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if dt_bias.len() != value_heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match value_heads={value_heads}: len={}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let norm_selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &norm_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package linear attention norm tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&norm_bundle) {
        eprintln!("invalid linear attention norm shape for {norm_tensor}: {err}");
        return ExitCode::from(1);
    }
    let norm_dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight =
        match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, norm_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if norm_weight.len() != value_dim {
        eprintln!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }
    if qkv_cols != hidden_size {
        eprintln!("qkv input cols must match hidden_size={hidden_size}: qkv_cols={qkv_cols}");
        return ExitCode::from(1);
    }

    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (z_rows, z_cols, z_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &z_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {z_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (out_rows, out_cols, out_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &out_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {out_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows != value_heads || b_rows != value_heads {
        eprintln!(
            "linear attention a/b rows must match value_heads={value_heads}: a_rows={a_rows}, b_rows={b_rows}"
        );
        return ExitCode::from(1);
    }
    if a_cols != hidden_size || b_cols != hidden_size {
        eprintln!(
            "linear attention a/b hidden sizes must match hidden_size={hidden_size}: a_cols={a_cols}, b_cols={b_cols}"
        );
        return ExitCode::from(1);
    }
    if z_rows != hidden_size || z_cols != hidden_size {
        eprintln!(
            "z projection shape must be [{hidden_size},{hidden_size}], got [{z_rows},{z_cols}]"
        );
        return ExitCode::from(1);
    }
    if out_rows != hidden_size || out_cols != hidden_size {
        eprintln!(
            "out projection shape must be [{hidden_size},{hidden_size}], got [{out_rows},{out_cols}]"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = qkv_rows * std::mem::size_of::<f32>();
    let gate_beta_step_bytes = value_heads * std::mem::size_of::<f32>();
    let hidden_bytes = hidden_size * std::mem::size_of::<f32>();
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let gate_beta_sequence_bytes_len = match gate_beta_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let hidden_sequence_bytes_len = match hidden_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention hidden sequence byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input_norm_weight_bytes = if include_block {
        encode_f32_to_bytes(&input_norm_weight)
    } else {
        Vec::new()
    };
    let mut input_norm_weight_buffer = if include_block {
        Some(match context.alloc_buffer(input_norm_weight_bytes.len()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate input RMSNorm weight buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    if let Some(buffer) = input_norm_weight_buffer.as_mut() {
        if let Err(err) = buffer.copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream)) {
            eprintln!("failed to copy input RMSNorm weight into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after input RMSNorm weight copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let mut input_norm_output_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate input RMSNorm output buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut z_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(hidden_size);
    let mut residual_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    let mut input_norm_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    let mut expected_input_norm = if include_block {
        Vec::with_capacity(sequence_len * hidden_size)
    } else {
        Vec::new()
    };
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut z_sequence_bytes = vec![0_u8; hidden_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if include_block {
            let residual_start = timestep * hidden_bytes;
            let residual_end = residual_start + hidden_bytes;
            residual_sequence_bytes[residual_start..residual_end]
                .copy_from_slice(&step_input_bytes);
            let expected_normed =
                runtime_host_rmsnorm_f32(&step_input, &input_norm_weight, epsilon);
            if expected_normed.len() != hidden_size {
                eprintln!("failed to build input RMSNorm reference for timestep {timestep}");
                return ExitCode::from(1);
            }
            expected_input_norm.extend_from_slice(&expected_normed);
        }
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if include_block {
            let input_norm_weight_buffer = input_norm_weight_buffer
                .as_ref()
                .expect("input RMSNorm weight buffer exists in block mode");
            let input_norm_output_buffer = input_norm_output_buffer
                .as_mut()
                .expect("input RMSNorm output buffer exists in block mode");
            if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
                &input_buffer,
                input_norm_weight_buffer,
                hidden_size,
                epsilon,
                input_norm_output_buffer,
                Some(&mut stream),
            ) {
                eprintln!("failed to run input RMSNorm timestep {timestep}: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after input RMSNorm timestep {timestep}: {err}"
                );
                return ExitCode::from(1);
            }
            let norm_start = timestep * hidden_bytes;
            let norm_end = norm_start + hidden_bytes;
            if let Err(err) = input_norm_output_buffer.copy_to_host(
                0,
                &mut input_norm_sequence_bytes[norm_start..norm_end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy input RMSNorm timestep {timestep} back to host: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after input RMSNorm timestep {timestep} host copy: {err}"
                );
                return ExitCode::from(1);
            }
        }
        let projection_input_buffer = if include_block {
            input_norm_output_buffer
                .as_ref()
                .expect("input RMSNorm output buffer exists in block mode")
        } else {
            &input_buffer
        };
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            projection_input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            projection_input_buffer,
            value_heads,
            hidden_size,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            projection_input_buffer,
            value_heads,
            hidden_size,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &z_matrix,
            projection_input_buffer,
            hidden_size,
            hidden_size,
            &mut z_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run z matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after linear attention timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let qkv_start = timestep * qkv_step_bytes;
        let qkv_end = qkv_start + qkv_step_bytes;
        if let Err(err) = qkv_step_buffer.copy_to_host(
            0,
            &mut qkv_sequence_bytes[qkv_start..qkv_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let gate_beta_start = timestep * gate_beta_step_bytes;
        let gate_beta_end = gate_beta_start + gate_beta_step_bytes;
        if let Err(err) = a_step_buffer.copy_to_host(
            0,
            &mut a_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = b_step_buffer.copy_to_host(
            0,
            &mut b_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let z_start = timestep * hidden_bytes;
        let z_end = z_start + hidden_bytes;
        if let Err(err) =
            z_step_buffer.copy_to_host(0, &mut z_sequence_bytes[z_start..z_end], Some(&mut stream))
        {
            eprintln!("failed to copy z timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let input_norm_sequence = if include_block {
        decode_f32_le_values(&input_norm_sequence_bytes)
    } else {
        Vec::new()
    };
    let mut input_norm_max_abs_diff = 0.0_f32;
    if include_block {
        if input_norm_sequence.len() != expected_input_norm.len() {
            eprintln!(
                "{command_name} input RMSNorm output size mismatch: expected {} got {}",
                expected_input_norm.len(),
                input_norm_sequence.len()
            );
            return ExitCode::from(1);
        }
        for (lhs, rhs) in input_norm_sequence.iter().zip(expected_input_norm.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 1e-4_f32.max(rhs.abs() * 1e-5_f32);
            if diff > tolerance {
                eprintln!(
                    "{command_name} input RMSNorm mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
                );
                return ExitCode::from(1);
            }
            if diff > input_norm_max_abs_diff {
                input_norm_max_abs_diff = diff;
            }
        }
    }

    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected_conv = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected_conv.is_empty() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }
    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);
    let mut conv_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected_conv.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-workflow-smoke conv1d mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > conv_max_abs_diff {
            conv_max_abs_diff = diff;
        }
    }

    let conv_activated = runtime_host_silu_f32(&conv_output);
    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_activated,
        sequence_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to split linear attention qkv: {err}");
            return ExitCode::from(1);
        }
    };

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let z_sequence = decode_f32_le_values(&z_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        value_heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        value_heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }
    let mut gate_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);
    let mut gate_beta_max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-workflow-smoke gate/beta mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > gate_beta_max_abs_diff {
            gate_beta_max_abs_diff = diff;
        }
    }

    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent state element count overflows");
            return ExitCode::from(1);
        }
    };
    let recurrent_output_elements = match sequence_len.checked_mul(hidden_size) {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent output element count overflows");
            return ExitCode::from(1);
        }
    };
    let initial_state = vec![0.0_f32; state_elements];
    let mut expected_state = initial_state.clone();
    let expected_recurrent_output = runtime_host_linear_attn_recurrent_f32(
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
    if expected_recurrent_output.len() != recurrent_output_elements {
        eprintln!("failed to build package linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&qkv_split.q);
    let k_bytes = encode_f32_to_bytes(&qkv_split.k);
    let v_bytes = encode_f32_to_bytes(&qkv_split.v);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let recurrent_output_bytes_len = recurrent_output_elements * std::mem::size_of::<f32>();
    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut recurrent_output_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate recurrent output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_output_buffer,
        &beta_output_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut recurrent_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }
    let mut recurrent_output_bytes = vec![0_u8; recurrent_output_bytes_len];
    let mut final_state_bytes = vec![0_u8; state_bytes.len()];
    if let Err(err) =
        recurrent_output_buffer.copy_to_host(0, &mut recurrent_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let recurrent_output = decode_f32_le_values(&recurrent_output_bytes);
    let final_state = decode_f32_le_values(&final_state_bytes);
    let mut recurrent_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in recurrent_output
        .iter()
        .zip(expected_recurrent_output.iter())
    {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke recurrent output mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke recurrent state mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }

    let mut expected_normed = vec![0.0_f32; recurrent_output_elements];
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let start = (timestep * value_heads + value_head) * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(
                &expected_recurrent_output[start..end],
                &norm_weight,
                epsilon,
            );
            if normed.len() != value_dim {
                eprintln!("failed to build linear attention workflow RMSNorm reference");
                return ExitCode::from(1);
            }
            expected_normed[start..end].copy_from_slice(&normed);
        }
    }
    let expected_activated = runtime_host_silu_mul_f32(&z_sequence, &expected_normed);
    if expected_activated.len() != recurrent_output_elements {
        eprintln!("failed to build linear attention workflow gated RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy linear attention norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }
    let mut norm_input_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_bytes = vec![0_u8; recurrent_output_bytes_len];
    for row in 0..(sequence_len * value_heads) {
        let start = row * value_dim;
        let end = start + value_dim;
        let byte_start = start * std::mem::size_of::<f32>();
        let byte_end = end * std::mem::size_of::<f32>();
        if let Err(err) = norm_input_buffer.copy_from_host(
            0,
            &recurrent_output_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy linear attention workflow norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
            &norm_input_buffer,
            &norm_weight_buffer,
            value_dim,
            epsilon,
            &mut norm_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run linear attention workflow rmsnorm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after workflow norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = norm_output_buffer.copy_to_host(
            0,
            &mut normed_sequence_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!(
                "failed to copy linear attention workflow norm row {row} back to host: {err}"
            );
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after workflow norm row {row} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let normed_sequence = decode_f32_le_values(&normed_sequence_bytes);
    let mut norm_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in normed_sequence.iter().zip(expected_normed.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke RMSNorm mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > norm_max_abs_diff {
            norm_max_abs_diff = diff;
        }
    }

    let mut z_sequence_buffer = match context.alloc_buffer(hidden_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut activated_sequence_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = z_sequence_buffer.copy_from_host(0, &z_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy z sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) =
        normed_sequence_buffer.copy_from_host(0, &normed_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy normed sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &z_sequence_buffer,
        &normed_sequence_buffer,
        recurrent_output_elements,
        &mut activated_sequence_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run linear attention workflow silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after workflow silu_mul: {err}");
        return ExitCode::from(1);
    }
    let mut activated_sequence_bytes = vec![0_u8; recurrent_output_bytes_len];
    if let Err(err) =
        activated_sequence_buffer.copy_to_host(0, &mut activated_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy activated sequence back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after activated sequence copy: {err}");
        return ExitCode::from(1);
    }
    let activated_sequence = decode_f32_le_values(&activated_sequence_bytes);
    let mut activation_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in activated_sequence.iter().zip(expected_activated.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke gated RMSNorm mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > activation_max_abs_diff {
            activation_max_abs_diff = diff;
        }
    }

    let out_matrix_bytes_len = match out_rows
        .checked_mul(out_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("out projection matrix byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
    if let Err(err) = out_matrix.copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy out projection matrix back to host for reference: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after out matrix host copy: {err}");
        return ExitCode::from(1);
    }
    let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
    let mut expected_output = Vec::with_capacity(sequence_len * hidden_size);
    for timestep in 0..sequence_len {
        let start = timestep * hidden_size;
        let end = start + hidden_size;
        let output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_activated[start..end],
            out_rows,
            out_cols,
        );
        if output.len() != out_rows {
            eprintln!("failed to build linear attention workflow out projection reference");
            return ExitCode::from(1);
        }
        expected_output.extend_from_slice(&output);
    }
    let residual_sequence = if include_block {
        decode_f32_le_values(&residual_sequence_bytes)
    } else {
        Vec::new()
    };
    let expected_block_output = if include_block {
        let output = runtime_host_add_f32(&residual_sequence, &expected_output);
        if output.len() != expected_output.len() {
            eprintln!("failed to build {command_name} residual add reference");
            return ExitCode::from(1);
        }
        output
    } else {
        Vec::new()
    };

    let mut out_input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut out_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection step buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut residual_step_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate residual step buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut block_step_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate block step output buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut output_sequence_bytes = vec![0_u8; hidden_sequence_bytes_len];
    let mut block_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    for timestep in 0..sequence_len {
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) = out_input_buffer.copy_from_host(
            0,
            &activated_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &out_input_buffer,
            out_rows,
            out_cols,
            &mut out_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run out projection matvec timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }
        if let Err(err) = out_step_buffer.copy_to_host(
            0,
            &mut output_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
        if include_block {
            let residual_step_buffer = residual_step_buffer
                .as_mut()
                .expect("residual step buffer exists in block mode");
            let block_step_buffer = block_step_buffer
                .as_mut()
                .expect("block step output buffer exists in block mode");
            if let Err(err) = residual_step_buffer.copy_from_host(
                0,
                &residual_sequence_bytes[start..end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy residual timestep {timestep} into runtime buffer: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = ullm_runtime_sys::add_f32(
                residual_step_buffer,
                &out_step_buffer,
                hidden_size,
                block_step_buffer,
                Some(&mut stream),
            ) {
                eprintln!("failed to run runtime residual add timestep {timestep}: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after residual add timestep {timestep}: {err}"
                );
                return ExitCode::from(1);
            }
            if let Err(err) = block_step_buffer.copy_to_host(
                0,
                &mut block_sequence_bytes[start..end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy block output timestep {timestep} back to host: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after block output timestep {timestep} host copy: {err}"
                );
                return ExitCode::from(1);
            }
        }
    }
    let output_sequence = decode_f32_le_values(&output_sequence_bytes);
    let mut output_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output_sequence.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 3e-3_f32.max(rhs.abs() * 2e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke out projection mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > output_max_abs_diff {
            output_max_abs_diff = diff;
        }
    }

    let block_output = if include_block {
        decode_f32_le_values(&block_sequence_bytes)
    } else {
        Vec::new()
    };
    let mut block_max_abs_diff = 0.0_f32;
    if include_block {
        if block_output.len() != expected_block_output.len() {
            eprintln!(
                "{command_name} output size mismatch: expected {} got {}",
                expected_block_output.len(),
                block_output.len()
            );
            return ExitCode::from(1);
        }
        for (lhs, rhs) in block_output.iter().zip(expected_block_output.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 3e-3_f32.max(rhs.abs() * 2e-5_f32);
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
    }

    if include_block {
        println!(
            "package-linear-attn-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} input_norm_preview={} workflow_output_preview={} block_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} verified=true",
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
            hidden_size,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            sequence_len,
            kernel_size,
            qk_l2_norm,
            input_norm_dtype,
            conv_dtype,
            a_log_dtype,
            dt_bias_dtype,
            norm_dtype,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&residual_sequence[..8.min(residual_sequence.len())]),
            format_f32_preview(&input_norm_sequence[..8.min(input_norm_sequence.len())]),
            format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
            format_f32_preview(&block_output[..8.min(block_output.len())]),
        );
        return ExitCode::SUCCESS;
    }

    println!(
        "package-linear-attn-workflow-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} backend={} device_index={} name=\"{}\" recurrent_preview={} z_preview={} activated_preview={} output_preview={} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        hidden_size,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        conv_dtype,
        a_log_dtype,
        dt_bias_dtype,
        norm_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&recurrent_output[..8.min(recurrent_output.len())]),
        format_f32_preview(&z_sequence[..8.min(z_sequence.len())]),
        format_f32_preview(&activated_sequence[..8.min(activated_sequence.len())]),
        format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-mlp-block-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 1, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let result = if sequence_len == 1 {
        package_linear_attn_mlp_block_smoke_impl(&path, device_index, chunk_bytes, layer_index)
    } else {
        package_linear_attn_mlp_block_sequence_smoke_impl(
            &path,
            device_index,
            chunk_bytes,
            layer_index,
            sequence_len,
        )
    };

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

fn package_linear_attn_mlp_block_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
) -> Result<String, String> {
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
    let sequence_len = 1_usize;
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
    let qkv_bytes = qkv_rows_expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "qkv byte size overflows".to_string())?;
    let gate_beta_bytes = value_heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "gate/beta byte size overflows".to_string())?;

    let residual = deterministic_f32_vector(hidden);
    let residual_bytes = encode_f32_to_bytes(&residual);
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

    let mut input_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(input_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate input RMSNorm weight buffer: {err}"))?;
    let mut input_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input RMSNorm output buffer: {err}"))?;
    input_buffer
        .copy_from_host(0, &residual_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy residual input into runtime buffer: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input copy: {err}"))?;
    ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &input_norm_weight_buffer,
        hidden,
        input_epsilon,
        &mut input_normed_buffer,
        Some(&mut stream),
    )
    .map_err(|err| format!("failed to run input RMSNorm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input RMSNorm: {err}"))?;
    let mut input_normed_bytes = vec![0_u8; hidden_bytes];
    input_normed_buffer
        .copy_to_host(0, &mut input_normed_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input RMSNorm copy: {err}"))?;
    let input_normed = decode_f32_le_values(&input_normed_bytes);
    let expected_input_normed =
        runtime_host_rmsnorm_f32(&residual, &input_norm_weight_values, input_epsilon);
    let input_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke input RMSNorm",
        &input_normed,
        &expected_input_normed,
        1e-4,
        1e-5,
    )?;

    let (
        attention_block_output,
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
        let (qkv_rows, qkv_cols, qkv_matrix) = materialize_selected_aq4_matrix(
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
        let (out_rows, out_cols, out_matrix) = materialize_selected_aq4_matrix(
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

        let mut qkv_buffer = context
            .alloc_buffer(qkv_bytes)
            .map_err(|err| format!("failed to allocate qkv output buffer: {err}"))?;
        let mut a_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate a output buffer: {err}"))?;
        let mut b_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate b output buffer: {err}"))?;
        let mut z_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate z output buffer: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_normed_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run qkv matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_normed_buffer,
            a_rows,
            a_cols,
            &mut a_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run a matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_normed_buffer,
            b_rows,
            b_cols,
            &mut b_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run b matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &z_matrix,
            &input_normed_buffer,
            z_rows,
            z_cols,
            &mut z_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run z matvec: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention projections: {err}")
        })?;

        let mut qkv_bytes_host = vec![0_u8; qkv_bytes];
        let mut a_bytes_host = vec![0_u8; gate_beta_bytes];
        let mut b_bytes_host = vec![0_u8; gate_beta_bytes];
        let mut z_bytes_host = vec![0_u8; hidden_bytes];
        qkv_buffer
            .copy_to_host(0, &mut qkv_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy qkv output to host: {err}"))?;
        a_buffer
            .copy_to_host(0, &mut a_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy a output to host: {err}"))?;
        b_buffer
            .copy_to_host(0, &mut b_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy b output to host: {err}"))?;
        z_buffer
            .copy_to_host(0, &mut z_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy z output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after projection host copies: {err}"))?;
        let qkv_output = decode_f32_le_values(&qkv_bytes_host);
        let a_output = decode_f32_le_values(&a_bytes_host);
        let b_output = decode_f32_le_values(&b_bytes_host);
        let z_output = decode_f32_le_values(&z_bytes_host);

        let mut conv_weight_buffer = context
            .alloc_buffer(conv_weight_bytes.len())
            .map_err(|err| format!("failed to allocate conv1d weight buffer: {err}"))?;
        let mut conv_output_buffer = context
            .alloc_buffer(qkv_bytes)
            .map_err(|err| format!("failed to allocate conv1d output buffer: {err}"))?;
        conv_weight_buffer
            .copy_from_host(0, &conv_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d weight into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d weight copy: {err}"))?;
        ullm_runtime_sys::depthwise_conv1d_f32(
            &qkv_buffer,
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
        let mut conv_output_bytes = vec![0_u8; qkv_bytes];
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

        let mut a_log_buffer = context
            .alloc_buffer(a_log_bytes.len())
            .map_err(|err| format!("failed to allocate A_log buffer: {err}"))?;
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias_bytes.len())
            .map_err(|err| format!("failed to allocate dt_bias buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate gate output buffer: {err}"))?;
        let mut beta_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate beta output buffer: {err}"))?;
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
            &a_buffer,
            &b_buffer,
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
        let mut gate_bytes = vec![0_u8; gate_beta_bytes];
        let mut beta_bytes = vec![0_u8; gate_beta_bytes];
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
            .alloc_buffer(hidden_bytes)
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
        let mut recurrent_bytes = vec![0_u8; hidden_bytes];
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

        let mut expected_attn_normed = vec![0.0_f32; hidden];
        for value_head in 0..value_heads {
            let start = value_head * value_dim;
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
        let mut attn_normed_bytes = vec![0_u8; hidden_bytes];
        for value_head in 0..value_heads {
            let start = value_head * value_dim;
            let byte_start = start * std::mem::size_of::<f32>();
            let byte_end = byte_start + attn_norm_weight_bytes.len();
            attn_norm_input_buffer
                .copy_from_host(0, &recurrent_bytes[byte_start..byte_end], Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy linear attention norm row {value_head}: {err}")
                })?;
            ullm_runtime_sys::rmsnorm_f32(
                &attn_norm_input_buffer,
                &attn_norm_weight_buffer,
                value_dim,
                input_epsilon,
                &mut attn_norm_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to run linear attention norm row {value_head}: {err}")
            })?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm row {value_head}: {err}")
            })?;
            attn_norm_output_buffer
                .copy_to_host(
                    0,
                    &mut attn_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy linear attention norm row {value_head}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize after linear attention norm row copy {value_head}: {err}"
                )
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
        let mut attn_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention normed buffer: {err}"))?;
        let mut attn_activated_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear attention activated buffer: {err}")
        })?;
        attn_normed_buffer
            .copy_from_host(0, &attn_normed_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention normed values: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention normed copy: {err}")
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &z_buffer,
            &attn_normed_buffer,
            hidden,
            &mut attn_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention SiLU-mul: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention SiLU-mul: {err}")
        })?;
        let mut attn_activated_bytes = vec![0_u8; hidden_bytes];
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
        let expected_attn_output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_attn_activated,
            out_rows,
            out_cols,
        );
        let mut attn_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention output buffer: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &attn_activated_buffer,
            out_rows,
            out_cols,
            &mut attn_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run out projection matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after out projection: {err}"))?;
        let mut attn_output_bytes = vec![0_u8; hidden_bytes];
        attn_output_buffer
            .copy_to_host(0, &mut attn_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after attention output copy: {err}"))?;
        let attn_output = decode_f32_le_values(&attn_output_bytes);
        let attn_output_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke out projection",
            &attn_output,
            &expected_attn_output,
            3e-3,
            2e-5,
        )?;

        let mut attn_block_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention block output buffer: {err}"))?;
        ullm_runtime_sys::add_f32(
            &input_buffer,
            &attn_output_buffer,
            hidden,
            &mut attn_block_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run attention residual add: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after attention residual add: {err}"))?;
        let mut attn_block_bytes = vec![0_u8; hidden_bytes];
        attn_block_buffer
            .copy_to_host(0, &mut attn_block_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy attention block output to host: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after attention block host copy: {err}")
        })?;
        let attention_block_output = decode_f32_le_values(&attn_block_bytes);
        let expected_attention_block = runtime_host_add_f32(&residual, &attn_output);
        let attn_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke attention residual",
            &attention_block_output,
            &expected_attention_block,
            1e-5,
            1e-6,
        )?;
        (
            attention_block_output,
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

    let post_normed_expected = runtime_host_rmsnorm_f32(
        &attention_block_output,
        &post_norm_weight_values,
        mlp_epsilon,
    );
    let mut attn_block_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate retained attention block buffer: {err}"))?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    let mut post_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate post RMSNorm output buffer: {err}"))?;
    let attention_block_bytes = encode_f32_to_bytes(&attention_block_output);
    attn_block_buffer
        .copy_from_host(0, &attention_block_bytes, Some(&mut stream))
        .map_err(|err| {
            format!("failed to copy attention block output into runtime buffer: {err}")
        })?;
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm input copy: {err}"))?;
    ullm_runtime_sys::rmsnorm_f32(
        &attn_block_buffer,
        &post_norm_weight_buffer,
        hidden,
        mlp_epsilon,
        &mut post_normed_buffer,
        Some(&mut stream),
    )
    .map_err(|err| format!("failed to run post RMSNorm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm: {err}"))?;
    let mut post_normed_bytes = vec![0_u8; hidden_bytes];
    post_normed_buffer
        .copy_to_host(0, &mut post_normed_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm host copy: {err}"))?;
    let post_normed = decode_f32_le_values(&post_normed_bytes);
    let post_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke post RMSNorm",
        &post_normed,
        &post_normed_expected,
        1e-4,
        1e-5,
    )?;

    let (mlp_output, layer_output, layer_block_max_abs_diff) = {
        let mut registry = WeightRegistry::new();
        let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &gate_tensor,
            chunk_bytes,
        )?;
        let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &up_tensor,
            chunk_bytes,
        )?;
        let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
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
        let intermediate = gate_rows;
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "MLP intermediate byte size overflows".to_string())?;
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
        ullm_runtime_sys::matvec_f32(
            &gate_matrix,
            &post_normed_buffer,
            gate_rows,
            gate_cols,
            &mut gate_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP gate matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &up_matrix,
            &post_normed_buffer,
            up_rows,
            up_cols,
            &mut up_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP up matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP gate/up: {err}"))?;
        ullm_runtime_sys::silu_mul_f32(
            &gate_buffer,
            &up_buffer,
            intermediate,
            &mut mlp_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP SiLU-mul: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP SiLU-mul: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &down_matrix,
            &mlp_activated_buffer,
            down_rows,
            down_cols,
            &mut mlp_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP down matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP down: {err}"))?;
        let mut mlp_output_bytes = vec![0_u8; hidden_bytes];
        mlp_output_buffer
            .copy_to_host(0, &mut mlp_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy MLP output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP output copy: {err}"))?;
        let mlp_output = decode_f32_le_values(&mlp_output_bytes);

        let mut layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate layer output buffer: {err}"))?;
        ullm_runtime_sys::add_f32(
            &attn_block_buffer,
            &mlp_output_buffer,
            hidden,
            &mut layer_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP residual add: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP residual add: {err}"))?;
        let mut layer_output_bytes = vec![0_u8; hidden_bytes];
        layer_output_buffer
            .copy_to_host(0, &mut layer_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy layer output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after layer output copy: {err}"))?;
        let layer_output = decode_f32_le_values(&layer_output_bytes);
        let expected_layer_output = runtime_host_add_f32(&attention_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (mlp_output, layer_output, layer_block_max_abs_diff)
    };

    Ok(format!(
        "package-linear-attn-mlp-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_output_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} attn_norm_max_abs_diff={attn_norm_max_abs_diff:.9} attn_activation_max_abs_diff={attn_activation_max_abs_diff:.9} attn_output_max_abs_diff={attn_output_max_abs_diff:.9} attn_block_max_abs_diff={attn_block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} layer_block_max_abs_diff={layer_block_max_abs_diff:.9} verified=true",
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
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&residual[..8.min(residual.len())]),
        format_f32_preview(&attn_output[..8.min(attn_output.len())]),
        format_f32_preview(&attention_block_output[..8.min(attention_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
    ))
}

fn package_linear_attn_mlp_block_sequence_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
) -> Result<String, String> {
    let hidden = 32_usize * 128_usize;
    let base_residual = deterministic_f32_vector(hidden);
    let mut residual_sequence = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        residual_sequence.extend(linear_attn_step_input(&base_residual, timestep));
    }
    let run = package_linear_attn_mlp_block_sequence_run(
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        residual_sequence,
        None,
        None,
    )?;
    Ok(run.line)
}

fn package_linear_attn_stateful_step_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-stateful-step-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 3, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    match package_linear_attn_stateful_step_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
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

fn package_linear_attn_stateful_step_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
) -> Result<String, String> {
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden;
    let qkv_step_elements = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let mlp_epsilon = 1e-5_f32;

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
    if conv_channels != qkv_step_elements {
        return Err(format!(
            "conv1d channels must match q/k/v layout: conv_channels={conv_channels}, expected={qkv_step_elements}"
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

    let base_residual = deterministic_f32_vector(hidden);
    let mut residual_sequence = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        residual_sequence.extend(linear_attn_step_input(&base_residual, timestep));
    }
    let run = package_linear_attn_mlp_block_sequence_run(
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        residual_sequence,
        None,
        None,
    )?;
    if run.attention_qkv_projection_dim != qkv_step_elements
        || run.attention_gate_dim != value_heads
        || run.attention_recurrent_qk_dim != key_heads * key_dim
    {
        return Err(format!(
            "linear attention stateful step metadata mismatch: qkv_dim={} gate_dim={} qk_dim={}",
            run.attention_qkv_projection_dim,
            run.attention_gate_dim,
            run.attention_recurrent_qk_dim
        ));
    }

    let mut conv_state = LinearAttnConv1dStepState::new(qkv_step_elements, kernel_size)?;
    let mut stepped_conv = Vec::with_capacity(run.attention_conv_pre_silu.len());
    for qkv_step in run.attention_qkv_projection.chunks_exact(qkv_step_elements) {
        stepped_conv.extend(conv_state.step(qkv_step, &conv.values)?);
    }
    let conv_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful conv1d step",
        &stepped_conv,
        &run.attention_conv_pre_silu,
        1e-4,
        1e-5,
    )?;
    let mut conv_context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime conv context: {err}"))?;
    let mut conv_stream = conv_context
        .create_stream()
        .map_err(|err| format!("failed to create runtime conv stream: {err}"))?;
    let qkv_step_bytes = qkv_step_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention qkv step byte size overflows".to_string())?;
    let conv_history_elements = qkv_step_elements
        .checked_mul(kernel_size)
        .ok_or_else(|| "linear attention conv history element count overflows".to_string())?;
    let conv_history_bytes = conv_history_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention conv history byte size overflows".to_string())?;
    let mut conv_history_buffer = conv_context
        .alloc_buffer(conv_history_bytes)
        .map_err(|err| format!("failed to allocate runtime conv history buffer: {err}"))?;
    let mut conv_weight_buffer = conv_context
        .alloc_buffer(
            conv.values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "linear attention conv weight byte size overflows".to_string())?,
        )
        .map_err(|err| format!("failed to allocate runtime conv weight buffer: {err}"))?;
    let mut conv_output_buffer = conv_context
        .alloc_buffer(conv_history_bytes)
        .map_err(|err| format!("failed to allocate runtime conv output buffer: {err}"))?;
    conv_weight_buffer
        .copy_from_host(
            0,
            &encode_f32_to_bytes(&conv.values),
            Some(&mut conv_stream),
        )
        .map_err(|err| format!("failed to copy runtime conv weight: {err}"))?;
    conv_stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize runtime conv weight copy: {err}"))?;
    let mut runtime_conv_history = vec![0.0_f32; conv_history_elements];
    let mut runtime_stepped_conv = Vec::with_capacity(run.attention_conv_pre_silu.len());
    for qkv_step in run.attention_qkv_projection.chunks_exact(qkv_step_elements) {
        if kernel_size > 1 {
            runtime_conv_history.rotate_left(qkv_step_elements);
        }
        let latest_start = (kernel_size - 1) * qkv_step_elements;
        runtime_conv_history[latest_start..latest_start + qkv_step_elements]
            .copy_from_slice(qkv_step);
        conv_history_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&runtime_conv_history),
                Some(&mut conv_stream),
            )
            .map_err(|err| format!("failed to copy runtime conv history: {err}"))?;
        conv_stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize runtime conv history copy: {err}"))?;
        ullm_runtime_sys::depthwise_conv1d_f32(
            &conv_history_buffer,
            &conv_weight_buffer,
            qkv_step_elements,
            kernel_size,
            kernel_size,
            &mut conv_output_buffer,
            Some(&mut conv_stream),
        )
        .map_err(|err| format!("failed to run runtime conv step: {err}"))?;
        conv_stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize runtime conv step: {err}"))?;
        let mut conv_step_bytes = vec![0_u8; qkv_step_bytes];
        let latest_byte_start = latest_start
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "linear attention conv latest byte offset overflows".to_string())?;
        conv_output_buffer
            .copy_to_host(
                latest_byte_start,
                &mut conv_step_bytes,
                Some(&mut conv_stream),
            )
            .map_err(|err| format!("failed to copy runtime conv step output: {err}"))?;
        conv_stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize runtime conv output copy: {err}"))?;
        runtime_stepped_conv.extend(decode_f32_le_values(&conv_step_bytes));
    }
    let runtime_conv_step_max_abs_diff = verify_f32_close(
        "package linear attention runtime conv1d step",
        &runtime_stepped_conv,
        &run.attention_conv_pre_silu,
        1e-4,
        1e-5,
    )?;
    let stepped_conv_activated = runtime_host_silu_f32(&stepped_conv);
    let conv_activation_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful conv activation",
        &stepped_conv_activated,
        &run.attention_conv,
        1e-4,
        1e-5,
    )?;

    let split_full = split_linear_attn_qkv_for_recurrent(
        &stepped_conv_activated,
        sequence_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    )?;
    let q_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful recurrent q",
        &split_full.q,
        &run.attention_recurrent_q,
        1e-4,
        1e-5,
    )?;
    let k_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful recurrent k",
        &split_full.k,
        &run.attention_recurrent_k,
        1e-4,
        1e-5,
    )?;
    let v_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful recurrent v",
        &split_full.v,
        &run.attention_recurrent_v,
        1e-4,
        1e-5,
    )?;
    let (gate, beta) = runtime_host_linear_attn_gate_beta_f32(
        &run.attention_a_projection,
        &run.attention_b_projection,
        &a_log.values,
        &dt_bias.values,
        value_heads,
        sequence_len,
    );
    let gate_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful gate",
        &gate,
        &run.attention_gate,
        1e-4,
        1e-5,
    )?;
    let beta_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful beta",
        &beta,
        &run.attention_beta,
        1e-4,
        1e-5,
    )?;

    let mut recurrent_state = vec![0.0_f32; value_heads * key_dim * value_dim];
    let mut stepped_recurrent = Vec::with_capacity(run.attention_recurrent.len());
    for timestep in 0..sequence_len {
        let conv_start = timestep * qkv_step_elements;
        let conv_end = conv_start + qkv_step_elements;
        let split = split_linear_attn_qkv_for_recurrent(
            &stepped_conv_activated[conv_start..conv_end],
            1,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            qk_l2_norm,
            q_scale,
        )?;
        let gate_start = timestep * value_heads;
        let gate_end = gate_start + value_heads;
        let recurrent_step = runtime_host_linear_attn_recurrent_f32(
            &split.q,
            &split.k,
            &split.v,
            &gate[gate_start..gate_end],
            &beta[gate_start..gate_end],
            key_heads,
            value_heads,
            1,
            key_dim,
            value_dim,
            &mut recurrent_state,
        );
        stepped_recurrent.extend(recurrent_step);
    }
    let recurrent_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful recurrent",
        &stepped_recurrent,
        &run.attention_recurrent,
        1e-3,
        1e-5,
    )?;

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let q_step_bytes = q_elements_per_step
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention q step byte size overflows".to_string())?;
    let k_step_bytes = q_step_bytes;
    let v_step_bytes = v_elements_per_step
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention v step byte size overflows".to_string())?;
    let gate_beta_step_bytes = value_heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention gate/beta step byte size overflows".to_string())?;
    let state_elements = value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "linear attention recurrent state size overflows".to_string())?;
    let state_bytes = state_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention recurrent state byte size overflows".to_string())?;
    let mut q_buffer = context
        .alloc_buffer(q_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent q step buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent k step buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent v step buffer: {err}"))?;
    let mut gate_buffer = context
        .alloc_buffer(gate_beta_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent gate step buffer: {err}"))?;
    let mut beta_buffer = context
        .alloc_buffer(gate_beta_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent beta step buffer: {err}"))?;
    let mut state_buffer = context
        .alloc_buffer(state_bytes)
        .map_err(|err| format!("failed to allocate recurrent state buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(v_step_bytes)
        .map_err(|err| format!("failed to allocate recurrent output step buffer: {err}"))?;
    let zero_state = vec![0.0_f32; state_elements];
    let zero_state_bytes = encode_f32_to_bytes(&zero_state);
    state_buffer
        .copy_from_host(0, &zero_state_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to initialize recurrent runtime state: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after recurrent state init: {err}"))?;

    let mut runtime_recurrent_bytes = vec![
        0_u8;
        v_step_bytes.checked_mul(sequence_len).ok_or_else(
            || { "linear attention runtime recurrent output byte size overflows".to_string() }
        )?
    ];
    for timestep in 0..sequence_len {
        let q_start = timestep * q_elements_per_step;
        let q_end = q_start + q_elements_per_step;
        let v_start = timestep * v_elements_per_step;
        let v_end = v_start + v_elements_per_step;
        let gate_start = timestep * value_heads;
        let gate_end = gate_start + value_heads;
        q_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&split_full.q[q_start..q_end]),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent q step {timestep}: {err}"))?;
        k_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&split_full.k[q_start..q_end]),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent k step {timestep}: {err}"))?;
        v_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&split_full.v[v_start..v_end]),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent v step {timestep}: {err}"))?;
        gate_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&gate[gate_start..gate_end]),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent gate step {timestep}: {err}"))?;
        beta_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&beta[gate_start..gate_end]),
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent beta step {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize recurrent step {timestep} inputs: {err}")
        })?;
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            1,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run recurrent runtime step {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize recurrent runtime step {timestep}: {err}")
        })?;
        let byte_start = timestep * v_step_bytes;
        let byte_end = byte_start + v_step_bytes;
        output_buffer
            .copy_to_host(
                0,
                &mut runtime_recurrent_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to copy recurrent runtime step {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize recurrent runtime copy {timestep}: {err}")
        })?;
    }
    let runtime_recurrent = decode_f32_le_values(&runtime_recurrent_bytes);
    let runtime_recurrent_step_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime recurrent",
        &runtime_recurrent,
        &run.attention_recurrent,
        1e-3,
        1e-5,
    )?;

    let hidden_bytes = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention hidden byte size overflows".to_string())?;
    let value_dim_bytes = value_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention value dim byte size overflows".to_string())?;
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_values =
        effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm_weight_values);
    let mut registry = WeightRegistry::new();
    let (out_rows, out_cols, out_matrix) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        path,
        &out_tensor,
        chunk_bytes,
    )?;
    if out_rows != hidden || out_cols != hidden {
        return Err(format!(
            "linear attention out projection shape must be [{hidden},{hidden}], got [{out_rows},{out_cols}]"
        ));
    }

    let mut attn_norm_weight_buffer =
        context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate stateful linear attention norm weight buffer: {err}")
            })?;
    let mut attn_norm_input_buffer = context.alloc_buffer(value_dim_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention norm input buffer: {err}")
    })?;
    let mut attn_norm_output_buffer = context.alloc_buffer(value_dim_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention norm output buffer: {err}")
    })?;
    let mut z_step_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate stateful linear attention z buffer: {err}"))?;
    let mut attn_normed_step_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention normed step buffer: {err}")
    })?;
    let mut attn_projection_input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention projection input buffer: {err}")
    })?;
    let mut attn_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention output buffer: {err}")
    })?;
    let mut residual_step_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention residual buffer: {err}")
    })?;
    let mut attn_block_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention block buffer: {err}")
    })?;
    attn_norm_weight_buffer
        .copy_from_host(0, &attn_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy stateful linear attention norm weight: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize stateful linear attention norm weight copy: {err}")
    })?;

    let mut runtime_attention_normed = Vec::with_capacity(run.attention_normed.len());
    let mut runtime_attention_projection_input =
        Vec::with_capacity(run.attention_projection_input.len());
    let mut runtime_attention_output = Vec::with_capacity(run.attention_output.len());
    let mut runtime_attention_block_output = Vec::with_capacity(run.attention_block_output.len());
    for timestep in 0..sequence_len {
        let hidden_start = timestep * hidden;
        let hidden_end = hidden_start + hidden;
        let recurrent_step_bytes =
            encode_f32_to_bytes(&runtime_recurrent[hidden_start..hidden_end]);
        let mut normed_step_bytes = vec![0_u8; hidden_bytes];
        for value_head in 0..value_heads {
            let row_element_start = value_head * value_dim;
            let row_byte_start = row_element_start * std::mem::size_of::<f32>();
            let row_byte_end = row_byte_start + value_dim_bytes;
            attn_norm_input_buffer
                .copy_from_host(
                    0,
                    &recurrent_step_bytes[row_byte_start..row_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!(
                        "failed to copy stateful linear attention norm row {value_head} timestep {timestep}: {err}"
                    )
                })?;
            ullm_runtime_sys::rmsnorm_f32(
                &attn_norm_input_buffer,
                &attn_norm_weight_buffer,
                value_dim,
                1e-6_f32,
                &mut attn_norm_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| {
                format!(
                    "failed to run stateful linear attention norm row {value_head} timestep {timestep}: {err}"
                )
            })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize stateful linear attention norm row {value_head} timestep {timestep}: {err}"
                )
            })?;
            attn_norm_output_buffer
                .copy_to_host(
                    0,
                    &mut normed_step_bytes[row_byte_start..row_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!(
                        "failed to copy stateful linear attention norm row {value_head} timestep {timestep}: {err}"
                    )
                })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize stateful linear attention norm copy row {value_head} timestep {timestep}: {err}"
                )
            })?;
        }
        runtime_attention_normed.extend(decode_f32_le_values(&normed_step_bytes));

        z_step_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&run.attention_z_projection[hidden_start..hidden_end]),
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy stateful linear attention z timestep {timestep}: {err}")
            })?;
        attn_normed_step_buffer
            .copy_from_host(0, &normed_step_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention normed timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention projection input timestep {timestep}: {err}"
            )
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &z_step_buffer,
            &attn_normed_step_buffer,
            hidden,
            &mut attn_projection_input_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention SiLU-mul timestep {timestep}: {err}")
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention SiLU-mul timestep {timestep}: {err}"
            )
        })?;
        let mut projection_input_bytes = vec![0_u8; hidden_bytes];
        attn_projection_input_buffer
            .copy_to_host(0, &mut projection_input_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention projection input timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention projection input copy timestep {timestep}: {err}"
            )
        })?;
        runtime_attention_projection_input.extend(decode_f32_le_values(&projection_input_bytes));

        ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &attn_projection_input_buffer,
            out_rows,
            out_cols,
            &mut attn_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!(
                "failed to run stateful linear attention out projection timestep {timestep}: {err}"
            )
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention out projection timestep {timestep}: {err}"
            )
        })?;
        let mut attention_output_bytes = vec![0_u8; hidden_bytes];
        attn_output_buffer
            .copy_to_host(0, &mut attention_output_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention output timestep {timestep}: {err}"
                )
            })?;
        runtime_attention_output.extend(decode_f32_le_values(&attention_output_bytes));

        let residual_step = linear_attn_step_input(&base_residual, timestep);
        residual_step_buffer
            .copy_from_host(0, &encode_f32_to_bytes(&residual_step), Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention residual timestep {timestep}: {err}"
                )
            })?;
        ullm_runtime_sys::add_f32(
            &residual_step_buffer,
            &attn_output_buffer,
            hidden,
            &mut attn_block_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!(
                "failed to run stateful linear attention residual add timestep {timestep}: {err}"
            )
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention residual add timestep {timestep}: {err}"
            )
        })?;
        let mut attention_block_bytes = vec![0_u8; hidden_bytes];
        attn_block_output_buffer
            .copy_to_host(0, &mut attention_block_bytes, Some(&mut stream))
            .map_err(|err| {
                format!("failed to copy stateful linear attention block timestep {timestep}: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention block copy timestep {timestep}: {err}"
            )
        })?;
        runtime_attention_block_output.extend(decode_f32_le_values(&attention_block_bytes));
    }
    let runtime_attention_normed_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime attention norm",
        &runtime_attention_normed,
        &run.attention_normed,
        1e-3,
        1e-5,
    )?;
    let runtime_attention_projection_input_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime projection input",
        &runtime_attention_projection_input,
        &run.attention_projection_input,
        1e-3,
        1e-5,
    )?;
    let runtime_attention_output_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime attention output",
        &runtime_attention_output,
        &run.attention_output,
        3e-3,
        2e-5,
    )?;
    let runtime_attention_block_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime attention block",
        &runtime_attention_block_output,
        &run.attention_block_output,
        3e-3,
        2e-5,
    )?;
    drop(out_matrix);
    drop(registry);

    let mut mlp_registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut mlp_registry,
        path,
        &gate_tensor,
        chunk_bytes,
    )?;
    let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut mlp_registry,
        path,
        &up_tensor,
        chunk_bytes,
    )?;
    let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut mlp_registry,
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
    let intermediate = gate_rows;
    if run.mlp_intermediate != intermediate {
        return Err(format!(
            "MLP intermediate mismatch: run={} materialized={intermediate}",
            run.mlp_intermediate
        ));
    }
    let intermediate_bytes = intermediate
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "MLP intermediate byte size overflows".to_string())?;

    let mut post_norm_weight_buffer =
        context
            .alloc_buffer(post_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate stateful linear attention post norm weight: {err}")
            })?;
    let mut post_normed_step_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention post normed buffer: {err}")
    })?;
    let mut mlp_gate_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention MLP gate buffer: {err}")
    })?;
    let mut mlp_up_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention MLP up buffer: {err}")
    })?;
    let mut mlp_activation_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention MLP activation buffer: {err}")
    })?;
    let mut mlp_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention MLP output buffer: {err}")
    })?;
    let mut layer_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate stateful linear attention layer output buffer: {err}")
    })?;
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| {
            format!("failed to copy stateful linear attention post norm weight: {err}")
        })?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize stateful linear attention post norm weight copy: {err}")
    })?;

    let mut runtime_post_normed = Vec::with_capacity(run.post_normed.len());
    let mut runtime_mlp_gate_projection = Vec::with_capacity(run.mlp_gate_projection.len());
    let mut runtime_mlp_up_projection = Vec::with_capacity(run.mlp_up_projection.len());
    let mut runtime_mlp_activation = Vec::with_capacity(run.mlp_activation.len());
    let mut runtime_mlp_output = Vec::with_capacity(run.mlp_output.len());
    let mut runtime_layer_output = Vec::with_capacity(run.layer_output.len());
    for timestep in 0..sequence_len {
        let hidden_start = timestep * hidden;
        let hidden_end = hidden_start + hidden;
        let attention_block_bytes =
            encode_f32_to_bytes(&runtime_attention_block_output[hidden_start..hidden_end]);
        attn_block_output_buffer
            .copy_from_host(0, &attention_block_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention block for MLP timestep {timestep}: {err}"
                )
            })?;
        ullm_runtime_sys::rmsnorm_f32(
            &attn_block_output_buffer,
            &post_norm_weight_buffer,
            hidden,
            mlp_epsilon,
            &mut post_normed_step_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention post norm timestep {timestep}: {err}")
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention post norm timestep {timestep}: {err}"
            )
        })?;
        let mut post_normed_bytes = vec![0_u8; hidden_bytes];
        post_normed_step_buffer
            .copy_to_host(0, &mut post_normed_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention post norm timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention post norm copy timestep {timestep}: {err}"
            )
        })?;
        runtime_post_normed.extend(decode_f32_le_values(&post_normed_bytes));

        ullm_runtime_sys::matvec_f32(
            &gate_matrix,
            &post_normed_step_buffer,
            gate_rows,
            gate_cols,
            &mut mlp_gate_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention MLP gate timestep {timestep}: {err}")
        })?;
        ullm_runtime_sys::matvec_f32(
            &up_matrix,
            &post_normed_step_buffer,
            up_rows,
            up_cols,
            &mut mlp_up_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention MLP up timestep {timestep}: {err}")
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP gate/up timestep {timestep}: {err}"
            )
        })?;
        let mut mlp_gate_bytes = vec![0_u8; intermediate_bytes];
        let mut mlp_up_bytes = vec![0_u8; intermediate_bytes];
        mlp_gate_buffer
            .copy_to_host(0, &mut mlp_gate_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention MLP gate timestep {timestep}: {err}"
                )
            })?;
        mlp_up_buffer
            .copy_to_host(0, &mut mlp_up_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention MLP up timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP gate/up copy timestep {timestep}: {err}"
            )
        })?;
        runtime_mlp_gate_projection.extend(decode_f32_le_values(&mlp_gate_bytes));
        runtime_mlp_up_projection.extend(decode_f32_le_values(&mlp_up_bytes));

        ullm_runtime_sys::silu_mul_f32(
            &mlp_gate_buffer,
            &mlp_up_buffer,
            intermediate,
            &mut mlp_activation_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!(
                "failed to run stateful linear attention MLP SiLU-mul timestep {timestep}: {err}"
            )
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP SiLU-mul timestep {timestep}: {err}"
            )
        })?;
        let mut mlp_activation_bytes = vec![0_u8; intermediate_bytes];
        mlp_activation_buffer
            .copy_to_host(0, &mut mlp_activation_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention MLP activation timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP activation copy timestep {timestep}: {err}"
            )
        })?;
        runtime_mlp_activation.extend(decode_f32_le_values(&mlp_activation_bytes));

        ullm_runtime_sys::matvec_f32(
            &down_matrix,
            &mlp_activation_buffer,
            down_rows,
            down_cols,
            &mut mlp_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention MLP down timestep {timestep}: {err}")
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP down timestep {timestep}: {err}"
            )
        })?;
        let mut mlp_output_bytes = vec![0_u8; hidden_bytes];
        mlp_output_buffer
            .copy_to_host(0, &mut mlp_output_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention MLP output timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention MLP output copy timestep {timestep}: {err}"
            )
        })?;
        runtime_mlp_output.extend(decode_f32_le_values(&mlp_output_bytes));

        ullm_runtime_sys::add_f32(
            &attn_block_output_buffer,
            &mlp_output_buffer,
            hidden,
            &mut layer_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| {
            format!("failed to run stateful linear attention layer add timestep {timestep}: {err}")
        })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention layer add timestep {timestep}: {err}"
            )
        })?;
        let mut layer_output_bytes = vec![0_u8; hidden_bytes];
        layer_output_buffer
            .copy_to_host(0, &mut layer_output_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy stateful linear attention layer output timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize stateful linear attention layer output copy timestep {timestep}: {err}"
            )
        })?;
        runtime_layer_output.extend(decode_f32_le_values(&layer_output_bytes));
    }
    let runtime_post_normed_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime post norm",
        &runtime_post_normed,
        &run.post_normed,
        1e-3,
        1e-5,
    )?;
    let runtime_mlp_gate_projection_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime MLP gate",
        &runtime_mlp_gate_projection,
        &run.mlp_gate_projection,
        3e-3,
        2e-5,
    )?;
    let runtime_mlp_up_projection_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime MLP up",
        &runtime_mlp_up_projection,
        &run.mlp_up_projection,
        3e-3,
        2e-5,
    )?;
    let runtime_mlp_activation_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime MLP activation",
        &runtime_mlp_activation,
        &run.mlp_activation,
        3e-3,
        2e-5,
    )?;
    let runtime_mlp_output_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime MLP output",
        &runtime_mlp_output,
        &run.mlp_output,
        3e-3,
        2e-5,
    )?;
    let runtime_layer_output_max_abs_diff = verify_f32_close(
        "package linear attention stateful runtime layer output",
        &runtime_layer_output,
        &run.layer_output,
        3e-3,
        2e-5,
    )?;
    let mut resident_step_layer = PackageLinearAttnResidentStepLayer::load(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        layer_index,
    )?;
    let mut resident_layer_output = Vec::with_capacity(run.layer_output.len());
    for timestep in 0..sequence_len {
        let step_output = resident_step_layer.step(
            &mut stream,
            &linear_attn_step_input(&base_residual, timestep),
        )?;
        resident_layer_output.extend(step_output);
    }
    let resident_step_layer_output_max_abs_diff = verify_f32_close(
        "package linear attention resident stateful layer output",
        &resident_layer_output,
        &run.layer_output,
        3e-3,
        2e-5,
    )?;

    Ok(format!(
        "package-linear-attn-stateful-step-smoke package={} layer={} sequence_len={} kernel_size={} hidden={} key_heads={} value_heads={} key_dim={} value_dim={} qkv_step_elements={} mlp_intermediate={} backend={} device_index={} name=\"{}\" conv_step_max_abs_diff={conv_step_max_abs_diff:.9} runtime_conv_step_max_abs_diff={runtime_conv_step_max_abs_diff:.9} conv_activation_step_max_abs_diff={conv_activation_step_max_abs_diff:.9} q_step_max_abs_diff={q_step_max_abs_diff:.9} k_step_max_abs_diff={k_step_max_abs_diff:.9} v_step_max_abs_diff={v_step_max_abs_diff:.9} gate_step_max_abs_diff={gate_step_max_abs_diff:.9} beta_step_max_abs_diff={beta_step_max_abs_diff:.9} recurrent_step_max_abs_diff={recurrent_step_max_abs_diff:.9} runtime_recurrent_step_max_abs_diff={runtime_recurrent_step_max_abs_diff:.9} runtime_attention_normed_max_abs_diff={runtime_attention_normed_max_abs_diff:.9} runtime_attention_projection_input_max_abs_diff={runtime_attention_projection_input_max_abs_diff:.9} runtime_attention_output_max_abs_diff={runtime_attention_output_max_abs_diff:.9} runtime_attention_block_max_abs_diff={runtime_attention_block_max_abs_diff:.9} runtime_post_normed_max_abs_diff={runtime_post_normed_max_abs_diff:.9} runtime_mlp_gate_projection_max_abs_diff={runtime_mlp_gate_projection_max_abs_diff:.9} runtime_mlp_up_projection_max_abs_diff={runtime_mlp_up_projection_max_abs_diff:.9} runtime_mlp_activation_max_abs_diff={runtime_mlp_activation_max_abs_diff:.9} runtime_mlp_output_max_abs_diff={runtime_mlp_output_max_abs_diff:.9} runtime_layer_output_max_abs_diff={runtime_layer_output_max_abs_diff:.9} resident_step_layer_output_max_abs_diff={resident_step_layer_output_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        sequence_len,
        kernel_size,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qkv_step_elements,
        intermediate,
        info.backend,
        device_index,
        info.name,
    ))
}

fn package_linear_attn_request_state_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    request_count: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-request-state-smoke requires a .ullm.d path");
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
    let request_count = match parse_optional_usize(request_count, 2, "request count") {
        Ok(value) if (1..=4).contains(&value) => value,
        Ok(_) => {
            eprintln!(
                "request count must be between 1 and 4 for package-linear-attn-request-state-smoke"
            );
            return ExitCode::from(2);
        }
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

    match package_linear_attn_request_state_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        request_count,
        sequence_len,
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

fn package_linear_attn_request_state_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    request_count: usize,
    sequence_len: usize,
) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create linear-attn request-state context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query linear-attn request-state device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create linear-attn request-state stream: {err}"))?;

    let mut request_ids = Vec::with_capacity(request_count);
    for request_index in 0..request_count {
        let offset = u64::try_from(request_index)
            .map_err(|_| "request index is too large for RequestId".to_string())?;
        request_ids.push(RequestId(501 + offset));
    }
    let mut batch_layer = PackageLinearAttnResidentStepBatchLayer::load(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        layer_index,
        request_ids.clone(),
        None,
    )?;
    if batch_layer.layer_index() != layer_index {
        return Err(format!(
            "linear-attn request-state layer index mismatch: got {} expected {layer_index}",
            batch_layer.layer_index()
        ));
    }
    if batch_layer.request_count() != request_count {
        return Err(format!(
            "linear-attn request-state request count mismatch: got {} expected {request_count}",
            batch_layer.request_count()
        ));
    }
    if batch_layer.request_ids() != request_ids.as_slice() {
        return Err("linear-attn request-state request id order changed".to_string());
    }
    let hidden = batch_layer.hidden();
    let mut request_slots = Vec::with_capacity(request_count);
    for request_id in &request_ids {
        request_slots.push(batch_layer.request_slot(*request_id)?);
    }
    let unknown_request_rejected = batch_layer.request_slot(RequestId(u64::MAX)).is_err();
    if !unknown_request_rejected {
        return Err("linear-attn request-state accepted an unknown request id".to_string());
    }

    let mut base_inputs = Vec::with_capacity(request_count);
    for request_index in 0..request_count {
        let mut base = deterministic_f32_vector(hidden);
        let request_scale = (request_index as f32 + 1.0_f32) * 0.0005_f32;
        for (feature_index, value) in base.iter_mut().enumerate() {
            let phase = (feature_index % 31) as f32 - 15.0_f32;
            *value += phase * request_scale;
        }
        base_inputs.push(base);
    }

    let mut batch_outputs_by_request = (0..request_count)
        .map(|_| Vec::with_capacity(sequence_len * hidden))
        .collect::<Vec<_>>();
    let interleaved_started = Instant::now();
    let mut output_max_abs = 0.0_f32;
    let mut nonfinite_count = 0_usize;
    let mut first_output_preview = None;
    let mut last_output_preview = String::new();
    for timestep in 0..sequence_len {
        for (request_index, &request_id) in request_ids.iter().enumerate() {
            let residual = linear_attn_step_input(&base_inputs[request_index], timestep);
            let label = format!(
                "linear-attn request-state request={} timestep={timestep}",
                request_id.0
            );
            batch_layer.step_from_host_to_device(&mut stream, request_id, &residual, &label)?;
            let output = batch_layer.read_output(&mut stream, request_id)?;
            if output.len() != hidden {
                return Err(format!(
                    "linear-attn request-state output length mismatch: request={} timestep={timestep} got {} expected {hidden}",
                    request_id.0,
                    output.len()
                ));
            }
            for &value in &output {
                if value.is_finite() {
                    output_max_abs = output_max_abs.max(value.abs());
                } else {
                    nonfinite_count += 1;
                }
            }
            let preview_len = output.len().min(6);
            let preview = format_f32_preview(&output[..preview_len]);
            if first_output_preview.is_none() {
                first_output_preview = Some(preview.clone());
            }
            last_output_preview = preview;
            batch_outputs_by_request[request_index].extend_from_slice(&output);
        }
    }
    let interleaved_wall_ms = interleaved_started.elapsed().as_secs_f64() * 1000.0;
    let interleaved_steps = request_count
        .checked_mul(sequence_len)
        .ok_or_else(|| "linear-attn request-state step count overflows".to_string())?;
    if nonfinite_count != 0 {
        return Err(format!(
            "linear-attn request-state produced {nonfinite_count} non-finite output values"
        ));
    }
    if output_max_abs == 0.0 {
        return Err("linear-attn request-state produced only zero outputs".to_string());
    }

    drop(batch_layer);

    let mut serial_max_abs_diff = 0.0_f32;
    let serial_started = Instant::now();
    for (request_index, request_id) in request_ids.iter().enumerate() {
        let mut serial_layer = PackageLinearAttnResidentStepLayer::load(
            &mut context,
            &mut stream,
            path,
            chunk_bytes,
            layer_index,
        )
        .map_err(|err| {
            format!(
                "failed to load linear-attn serial reference layer {layer_index} for request {}: {err}",
                request_id.0
            )
        })?;
        let mut serial_outputs = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let output = serial_layer.step(
                &mut stream,
                &linear_attn_step_input(&base_inputs[request_index], timestep),
            )?;
            serial_outputs.extend(output);
        }
        let diff = verify_f32_close(
            &format!(
                "linear-attn request-state serial reference request {}",
                request_id.0
            ),
            &batch_outputs_by_request[request_index],
            &serial_outputs,
            3e-3,
            2e-5,
        )?;
        serial_max_abs_diff = serial_max_abs_diff.max(diff);
    }
    let serial_reference_wall_ms = serial_started.elapsed().as_secs_f64() * 1000.0;
    let interleaved_step_tps = if interleaved_wall_ms > 0.0 {
        interleaved_steps as f64 / (interleaved_wall_ms / 1000.0)
    } else {
        0.0
    };
    let request_ids_u64 = request_ids
        .iter()
        .map(|request_id| request_id.0)
        .collect::<Vec<_>>();
    let first_output_preview = first_output_preview.unwrap_or_else(|| "[]".to_string());

    Ok(format!(
        "package-linear-attn-request-state-smoke package={} layer={} request_count={} sequence_len={} request_ids={:?} request_slots={:?} hidden={} interleaved_steps={} state_owner=PackageLinearAttnResidentStepBatchLayer state_isolated_by_request=true shared_weight_final_design=false full_mixed_runner=false backend={} device_index={} name=\"{}\" interleaved_wall_ms={interleaved_wall_ms:.6} serial_reference_wall_ms={serial_reference_wall_ms:.6} interleaved_step_tps={interleaved_step_tps:.6} output_max_abs={output_max_abs:.9} serial_reference_max_abs_diff={serial_max_abs_diff:.9} nonfinite_count={} unknown_request_rejected={} first_output_preview={} last_output_preview={} verified=true",
        path,
        layer_index,
        request_count,
        sequence_len,
        request_ids_u64,
        request_slots,
        hidden,
        interleaved_steps,
        info.backend,
        device_index,
        info.name,
        nonfinite_count,
        unknown_request_rejected,
        first_output_preview,
        last_output_preview,
    ))
}
