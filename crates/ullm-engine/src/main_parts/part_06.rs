#[derive(Clone, Copy)]
enum PackageSelfAttnResidentStepInput<'a> {
    InternalInputBuffer,
    ExternalBuffer(&'a ullm_runtime_sys::RuntimeBuffer),
}

struct PackageSelfAttnResidentStepWeights {
    sync_component_timing: bool,
    use_paged_decode_sigmoid_gate: bool,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    attention_elements: usize,
    block_size: usize,
    cache_blocks: usize,
    q_projection_layout: PackageSelfAttnQProjectionLayout,
    input_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    q_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    k_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    post_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    q_matrix: PackageAq4ResidentMatvec,
    k_matrix: PackageAq4ResidentMatvec,
    v_matrix: PackageAq4ResidentMatvec,
    o_matrix: PackageAq4ResidentMatvec,
    mlp_gate_matrix: PackageAq4ResidentMatvec,
    mlp_up_matrix: PackageAq4ResidentMatvec,
    mlp_down_matrix: PackageAq4ResidentMatvec,
}

struct PackageSelfAttnResidentStepLayer {
    weights: std::sync::Arc<PackageSelfAttnResidentStepWeights>,
    last_component_step_ms: Option<PackageSelfAttnComponentStepMs>,
    written_len: usize,
    block_table_buffer: ullm_runtime_sys::RuntimeBuffer,
    input_buffer: ullm_runtime_sys::RuntimeBuffer,
    input_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    q_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
    q_gate_buffer: ullm_runtime_sys::RuntimeBuffer,
    k_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
    v_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
    q_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    k_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    q_rope_buffer: ullm_runtime_sys::RuntimeBuffer,
    k_rope_buffer: ullm_runtime_sys::RuntimeBuffer,
    k_cache_buffer: ullm_runtime_sys::RuntimeBuffer,
    v_cache_buffer: ullm_runtime_sys::RuntimeBuffer,
    attention_output_buffer: ullm_runtime_sys::RuntimeBuffer,
    attention_projection_input_buffer: ullm_runtime_sys::RuntimeBuffer,
    attention_block_output_buffer: ullm_runtime_sys::RuntimeBuffer,
    post_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    mlp_activation_buffer: ullm_runtime_sys::RuntimeBuffer,
    layer_output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

impl std::ops::Deref for PackageSelfAttnResidentStepLayer {
    type Target = PackageSelfAttnResidentStepWeights;

    fn deref(&self) -> &Self::Target {
        self.weights.as_ref()
    }
}

impl PackageSelfAttnResidentStepLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
        block_table: &[u32],
        block_size: usize,
        cache_blocks: usize,
    ) -> Result<Self, String> {
        let mut registry = WeightRegistry::new();
        Self::load_with_registry(
            context,
            stream,
            &mut registry,
            None,
            path,
            chunk_bytes,
            layer_index,
            block_table,
            block_size,
            cache_blocks,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_with_registry(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        mut shared_buffers: Option<&mut PackageResidentSharedBufferRegistry>,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
        block_table: &[u32],
        block_size: usize,
        cache_blocks: usize,
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        if block_table.len() != cache_blocks {
            return Err(format!(
                "self-attn resident block table length {} does not match cache blocks {cache_blocks}",
                block_table.len()
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

        let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
        input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
        let mut q_norm = read_named_passthrough_f32(path, &q_norm_tensor, chunk_bytes)?;
        q_norm.values = effective_rmsnorm_weight_values(&q_norm_tensor, &q_norm.values);
        let mut k_norm = read_named_passthrough_f32(path, &k_norm_tensor, chunk_bytes)?;
        k_norm.values = effective_rmsnorm_weight_values(&k_norm_tensor, &k_norm.values);
        let mut post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
        post_norm.values = effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);

        let q_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &q_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let k_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &k_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let v_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &v_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let o_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &o_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_gate_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &gate_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_up_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &up_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_down_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &down_tensor,
            chunk_bytes,
            sq_overlay,
        )?;

        let hidden = q_matrix.cols;
        let head_dim = q_norm.values.len();
        if hidden == 0 || input_norm.values.len() != hidden || post_norm.values.len() != hidden {
            return Err(format!(
                "self-attn resident hidden/norm mismatch: hidden={hidden} input_norm={} post_norm={}",
                input_norm.values.len(),
                post_norm.values.len()
            ));
        }
        if head_dim == 0 || k_norm.values.len() != head_dim {
            return Err(format!(
                "self-attn resident q/k norm mismatch: q_norm={} k_norm={}",
                head_dim,
                k_norm.values.len()
            ));
        }
        if k_matrix.cols != hidden || v_matrix.cols != hidden {
            return Err(format!(
                "self-attn resident q/k/v hidden mismatch: q_cols={} k_cols={} v_cols={}",
                q_matrix.cols, k_matrix.cols, v_matrix.cols
            ));
        }
        if !k_matrix.rows.is_multiple_of(head_dim) {
            return Err(format!(
                "self-attn resident k rows {} are not a multiple of head_dim {head_dim}",
                k_matrix.rows
            ));
        }
        let kv_heads = k_matrix.rows / head_dim;
        if kv_heads == 0 || !v_matrix.rows.is_multiple_of(kv_heads) {
            return Err(format!(
                "self-attn resident v rows {} are not compatible with kv_heads {kv_heads}",
                v_matrix.rows
            ));
        }
        let value_dim = v_matrix.rows / kv_heads;
        let two_hidden = hidden
            .checked_mul(2)
            .ok_or_else(|| "self-attn resident hidden*2 overflows".to_string())?;
        let two_head_dim = head_dim
            .checked_mul(2)
            .ok_or_else(|| "self-attn resident head_dim*2 overflows".to_string())?;
        let (q_projection_layout, q_heads) =
            if q_matrix.rows == two_hidden && q_matrix.rows.is_multiple_of(two_head_dim) {
                (
                    PackageSelfAttnQProjectionLayout::Qwen35Gated,
                    q_matrix.rows / two_head_dim,
                )
            } else if q_matrix.rows.is_multiple_of(head_dim) {
                (
                    PackageSelfAttnQProjectionLayout::Plain,
                    q_matrix.rows / head_dim,
                )
            } else {
                return Err(format!(
                    "self-attn resident q rows {} do not match plain or Qwen3.5 gated layout",
                    q_matrix.rows
                ));
            };
        if q_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "self-attn resident q_heads {q_heads} must be nonzero and a multiple of kv_heads {kv_heads}"
            ));
        }
        let attention_elements = q_heads
            .checked_mul(value_dim)
            .ok_or_else(|| "self-attn resident attention element count overflows".to_string())?;
        if o_matrix.rows != hidden || o_matrix.cols != attention_elements {
            return Err(format!(
                "self-attn resident o shape mismatch: got [{},{}] expected [{hidden},{attention_elements}]",
                o_matrix.rows, o_matrix.cols
            ));
        }
        if mlp_gate_matrix.rows != mlp_up_matrix.rows
            || mlp_gate_matrix.cols != mlp_up_matrix.cols
            || mlp_gate_matrix.cols != hidden
        {
            return Err(format!(
                "self-attn resident MLP gate/up shape mismatch: gate=[{},{}] up=[{},{}] hidden={hidden}",
                mlp_gate_matrix.rows, mlp_gate_matrix.cols, mlp_up_matrix.rows, mlp_up_matrix.cols
            ));
        }
        if mlp_down_matrix.rows != hidden || mlp_down_matrix.cols != mlp_gate_matrix.rows {
            return Err(format!(
                "self-attn resident MLP down shape mismatch: got [{},{}] expected [{hidden},{}]",
                mlp_down_matrix.rows, mlp_down_matrix.cols, mlp_gate_matrix.rows
            ));
        }
        let intermediate = mlp_gate_matrix.rows;

        let decode_shape = PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        };
        decode_shape.validate()?;

        let hidden_bytes = checked_f32_byte_len(hidden, "self-attn resident hidden")?;
        let q_projected_bytes =
            checked_f32_byte_len(q_matrix.rows, "self-attn resident q projected")?;
        let q_elements = decode_shape.q_elements()?;
        let q_bytes = checked_f32_byte_len(q_elements, "self-attn resident q")?;
        let k_bytes =
            checked_f32_byte_len(decode_shape.k_token_elements()?, "self-attn resident k")?;
        let v_bytes =
            checked_f32_byte_len(decode_shape.v_token_elements()?, "self-attn resident v")?;
        let attention_bytes =
            checked_f32_byte_len(attention_elements, "self-attn resident attention")?;
        let k_cache_elements = decode_shape.k_cache_elements()?;
        let v_cache_elements = decode_shape.v_cache_elements()?;
        let intermediate_bytes =
            checked_f32_byte_len(intermediate, "self-attn resident intermediate")?;

        let input_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("self-attn-input-norm:{input_norm_tensor}"),
            &input_norm.values,
            "self-attn resident input norm weight",
        )?;
        let q_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("self-attn-q-norm:{q_norm_tensor}"),
            &q_norm.values,
            "self-attn resident q norm weight",
        )?;
        let k_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("self-attn-k-norm:{k_norm_tensor}"),
            &k_norm.values,
            "self-attn resident k norm weight",
        )?;
        let post_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("self-attn-post-norm:{post_norm_tensor}"),
            &post_norm.values,
            "self-attn resident post norm weight",
        )?;
        let mut block_table_buffer = context
            .alloc_buffer(block_table.len() * std::mem::size_of::<u32>())
            .map_err(|err| format!("failed to allocate self-attn resident block table: {err}"))?;

        let mut input_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident input: {err}"))?;
        let input_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident input normed: {err}"))?;
        let mut q_projected_buffer = context
            .alloc_buffer(q_projected_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q projected: {err}"))?;
        let mut q_gate_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q gate: {err}"))?;
        let mut k_projected_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k projected: {err}"))?;
        let mut v_projected_buffer = context
            .alloc_buffer(v_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident v projected: {err}"))?;
        let q_normed_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q normed: {err}"))?;
        let k_normed_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k normed: {err}"))?;
        let mut q_rope_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q RoPE: {err}"))?;
        let k_rope_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k RoPE: {err}"))?;
        let mut k_cache_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                k_cache_elements,
                "self-attn resident k cache",
            )?)
            .map_err(|err| format!("failed to allocate self-attn resident k cache: {err}"))?;
        let mut v_cache_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                v_cache_elements,
                "self-attn resident v cache",
            )?)
            .map_err(|err| format!("failed to allocate self-attn resident v cache: {err}"))?;
        let attention_output_buffer = context.alloc_buffer(attention_bytes).map_err(|err| {
            format!("failed to allocate self-attn resident attention output: {err}")
        })?;
        let mut attention_projection_input_buffer =
            context.alloc_buffer(attention_bytes).map_err(|err| {
                format!("failed to allocate self-attn resident attention projection input: {err}")
            })?;
        let mut attention_block_output_buffer =
            context.alloc_buffer(hidden_bytes).map_err(|err| {
                format!("failed to allocate self-attn resident attention block output: {err}")
            })?;
        let post_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident post normed: {err}"))?;
        let mlp_activation_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
            format!("failed to allocate self-attn resident MLP activation: {err}")
        })?;
        let layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident layer output: {err}"))?;

        block_table_buffer
            .copy_from_host(0, &encode_u32_to_bytes(block_table), Some(stream))
            .map_err(|err| format!("failed to copy self-attn resident block table: {err}"))?;
        k_cache_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; k_cache_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize self-attn resident k cache: {err}"))?;
        v_cache_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; v_cache_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize self-attn resident v cache: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize self-attn resident layer setup: {err}")
        })?;
        prewarm_aq4_matvec_add_once(
            stream,
            &o_matrix,
            &mut attention_projection_input_buffer,
            &mut input_buffer,
            &mut attention_block_output_buffer,
            "self-attn resident AQ4 matvec add",
        )?;
        if !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV") {
            prewarm_aq4_matvec_triple_once(
                stream,
                &q_matrix,
                &k_matrix,
                &v_matrix,
                &mut input_buffer,
                &mut q_projected_buffer,
                &mut k_projected_buffer,
                &mut v_projected_buffer,
                "self-attn resident AQ4 q/k/v triple projection",
            )?;
        } else if !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_PAIR_SELF_ATTN_QK") {
            prewarm_aq4_matvec_pair_once(
                stream,
                &q_matrix,
                &k_matrix,
                &mut input_buffer,
                &mut q_projected_buffer,
                &mut k_projected_buffer,
                "self-attn resident AQ4 q/k pair projection",
            )?;
        }
        if matches!(
            q_projection_layout,
            PackageSelfAttnQProjectionLayout::Qwen35Gated
        ) {
            prewarm_qwen35_qk_norm_rope_paged_kv_write_once(
                stream,
                &mut q_projected_buffer,
                &mut k_projected_buffer,
                &mut v_projected_buffer,
                &q_norm_weight_buffer,
                &k_norm_weight_buffer,
                &block_table_buffer,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                block_size,
                cache_blocks,
                &mut q_gate_buffer,
                &mut q_rope_buffer,
                &mut k_cache_buffer,
                &mut v_cache_buffer,
                "self-attn resident Qwen3.5 q/k norm RoPE paged KV write",
            )?;
        }

        let weights = std::sync::Arc::new(PackageSelfAttnResidentStepWeights {
            sync_component_timing: env_flag_enabled("ULLM_SYNC_SELF_ATTN_COMPONENTS_FOR_TIMING"),
            use_paged_decode_sigmoid_gate: matches!(
                q_projection_layout,
                PackageSelfAttnQProjectionLayout::Qwen35Gated
            ) && !env_flag_enabled(
                "ULLM_DISABLE_PAGED_DECODE_SIGMOID_GATE_SELF_ATTN",
            ),
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            attention_elements,
            block_size,
            cache_blocks,
            q_projection_layout,
            input_norm_weight_buffer,
            q_norm_weight_buffer,
            k_norm_weight_buffer,
            post_norm_weight_buffer,
            q_matrix,
            k_matrix,
            v_matrix,
            o_matrix,
            mlp_gate_matrix,
            mlp_up_matrix,
            mlp_down_matrix,
        });

        Ok(Self {
            weights,
            last_component_step_ms: None,
            written_len: 0,
            block_table_buffer,
            input_buffer,
            input_normed_buffer,
            q_projected_buffer,
            q_gate_buffer,
            k_projected_buffer,
            v_projected_buffer,
            q_normed_buffer,
            k_normed_buffer,
            q_rope_buffer,
            k_rope_buffer,
            k_cache_buffer,
            v_cache_buffer,
            attention_output_buffer,
            attention_projection_input_buffer,
            attention_block_output_buffer,
            post_normed_buffer,
            mlp_activation_buffer,
            layer_output_buffer,
        })
    }

    fn load_state_with_weights(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        weights: std::sync::Arc<PackageSelfAttnResidentStepWeights>,
        block_table: &[u32],
    ) -> Result<Self, String> {
        if block_table.len() != weights.cache_blocks {
            return Err(format!(
                "self-attn resident shared-weight block table length {} does not match cache blocks {}",
                block_table.len(),
                weights.cache_blocks
            ));
        }
        let decode_shape = PagedDecodeShape {
            block_size: weights.block_size,
            cache_blocks: weights.cache_blocks,
            q_heads: weights.q_heads,
            kv_heads: weights.kv_heads,
            head_dim: weights.head_dim,
            value_dim: weights.value_dim,
        };
        decode_shape.validate()?;

        let hidden_bytes = checked_f32_byte_len(weights.hidden, "self-attn resident hidden")?;
        let q_projected_bytes =
            checked_f32_byte_len(weights.q_matrix.rows, "self-attn resident q projected")?;
        let q_elements = decode_shape.q_elements()?;
        let q_bytes = checked_f32_byte_len(q_elements, "self-attn resident q")?;
        let k_bytes =
            checked_f32_byte_len(decode_shape.k_token_elements()?, "self-attn resident k")?;
        let v_bytes =
            checked_f32_byte_len(decode_shape.v_token_elements()?, "self-attn resident v")?;
        let attention_bytes =
            checked_f32_byte_len(weights.attention_elements, "self-attn resident attention")?;
        let k_cache_elements = decode_shape.k_cache_elements()?;
        let v_cache_elements = decode_shape.v_cache_elements()?;
        let intermediate_bytes = checked_f32_byte_len(
            weights.mlp_gate_matrix.rows,
            "self-attn resident intermediate",
        )?;

        let mut block_table_buffer = context
            .alloc_buffer(block_table.len() * std::mem::size_of::<u32>())
            .map_err(|err| {
                format!("failed to allocate self-attn resident shared-weight block table: {err}")
            })?;
        let input_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident input: {err}"))?;
        let input_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident input normed: {err}"))?;
        let q_projected_buffer = context
            .alloc_buffer(q_projected_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q projected: {err}"))?;
        let q_gate_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q gate: {err}"))?;
        let k_projected_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k projected: {err}"))?;
        let v_projected_buffer = context
            .alloc_buffer(v_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident v projected: {err}"))?;
        let q_normed_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q normed: {err}"))?;
        let k_normed_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k normed: {err}"))?;
        let q_rope_buffer = context
            .alloc_buffer(q_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident q RoPE: {err}"))?;
        let k_rope_buffer = context
            .alloc_buffer(k_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident k RoPE: {err}"))?;
        let mut k_cache_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                k_cache_elements,
                "self-attn resident k cache",
            )?)
            .map_err(|err| format!("failed to allocate self-attn resident k cache: {err}"))?;
        let mut v_cache_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                v_cache_elements,
                "self-attn resident v cache",
            )?)
            .map_err(|err| format!("failed to allocate self-attn resident v cache: {err}"))?;
        let attention_output_buffer = context.alloc_buffer(attention_bytes).map_err(|err| {
            format!("failed to allocate self-attn resident attention output: {err}")
        })?;
        let attention_projection_input_buffer =
            context.alloc_buffer(attention_bytes).map_err(|err| {
                format!("failed to allocate self-attn resident attention projection input: {err}")
            })?;
        let attention_block_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate self-attn resident attention block output: {err}")
        })?;
        let post_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident post normed: {err}"))?;
        let mlp_activation_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
            format!("failed to allocate self-attn resident MLP activation: {err}")
        })?;
        let layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate self-attn resident layer output: {err}"))?;

        block_table_buffer
            .copy_from_host(0, &encode_u32_to_bytes(block_table), Some(stream))
            .map_err(|err| {
                format!("failed to copy self-attn resident shared-weight block table: {err}")
            })?;
        k_cache_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; k_cache_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize self-attn resident k cache: {err}"))?;
        v_cache_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; v_cache_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize self-attn resident v cache: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize self-attn resident shared-weight state setup: {err}")
        })?;

        Ok(Self {
            weights,
            last_component_step_ms: None,
            written_len: 0,
            block_table_buffer,
            input_buffer,
            input_normed_buffer,
            q_projected_buffer,
            q_gate_buffer,
            k_projected_buffer,
            v_projected_buffer,
            q_normed_buffer,
            k_normed_buffer,
            q_rope_buffer,
            k_rope_buffer,
            k_cache_buffer,
            v_cache_buffer,
            attention_output_buffer,
            attention_projection_input_buffer,
            attention_block_output_buffer,
            post_normed_buffer,
            mlp_activation_buffer,
            layer_output_buffer,
        })
    }

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
        if residual.len() != self.hidden {
            return Err(format!(
                "{label} self-attn resident residual length mismatch: got {} expected {}",
                residual.len(),
                self.hidden
            ));
        }
        self.input_buffer
            .copy_from_host(0, &encode_f32_to_bytes(residual), Some(stream))
            .map_err(|err| format!("failed to copy self-attn resident residual: {err}"))?;
        self.run_device_step(
            stream,
            PackageSelfAttnResidentStepInput::InternalInputBuffer,
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            label,
        )
    }

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
        let expected_bytes = checked_f32_byte_len(self.hidden, "self-attn resident input")?;
        let actual_bytes = residual_buffer
            .size()
            .map_err(|err| format!("failed to query {label} residual buffer size: {err}"))?;
        if actual_bytes < expected_bytes {
            return Err(format!(
                "{label} residual buffer is too small: got {actual_bytes} bytes expected at least {expected_bytes}"
            ));
        }
        self.run_device_step(
            stream,
            PackageSelfAttnResidentStepInput::ExternalBuffer(residual_buffer),
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            label,
        )
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<Vec<f32>, String> {
        read_runtime_buffer_f32(
            &self.layer_output_buffer,
            stream,
            self.hidden,
            "self-attn resident layer output",
        )
    }

    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.layer_output_buffer
    }

    fn take_last_component_step_ms(&mut self) -> Option<PackageSelfAttnComponentStepMs> {
        self.last_component_step_ms.take()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_device_step_input(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: PackageSelfAttnResidentStepInput<'_>,
        sync_component_timing: bool,
        component_step_ms: &mut PackageSelfAttnComponentStepMs,
        label: &str,
    ) -> Result<(), String> {
        let hidden = self.weights.hidden;
        let finish_component = |stream: &mut ullm_runtime_sys::RuntimeStream,
                                started: Instant,
                                component_label: &str|
         -> Result<f64, String> {
            if sync_component_timing {
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize {label} self-attn {component_label}: {err}")
                })?;
                Ok(started.elapsed().as_secs_f64() * 1000.0)
            } else {
                Ok(0.0)
            }
        };
        let component_started = Instant::now();
        match input {
            PackageSelfAttnResidentStepInput::InternalInputBuffer => ullm_runtime_sys::rmsnorm_f32(
                &self.input_buffer,
                self.weights.input_norm_weight_buffer.as_ref(),
                hidden,
                1e-6_f32,
                &mut self.input_normed_buffer,
                Some(stream),
            ),
            PackageSelfAttnResidentStepInput::ExternalBuffer(buffer) => {
                ullm_runtime_sys::rmsnorm_f32(
                    buffer,
                    self.weights.input_norm_weight_buffer.as_ref(),
                    hidden,
                    1e-6_f32,
                    &mut self.input_normed_buffer,
                    Some(stream),
                )
            }
        }
        .map_err(|err| format!("failed to run {label} self-attn input RMSNorm: {err}"))?;
        component_step_ms.input_rmsnorm_ms =
            finish_component(stream, component_started, "input RMSNorm")?;
        Ok(())
    }

    fn run_device_step_qkv_projection(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        sync_component_timing: bool,
        component_step_ms: &mut PackageSelfAttnComponentStepMs,
        label: &str,
    ) -> Result<(), String> {
        let component_started = Instant::now();
        let finish_component = |stream: &mut ullm_runtime_sys::RuntimeStream,
                                started: Instant,
                                component_label: &str|
         -> Result<f64, String> {
            if sync_component_timing {
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize {label} self-attn {component_label}: {err}")
                })?;
                Ok(started.elapsed().as_secs_f64() * 1000.0)
            } else {
                Ok(0.0)
            }
        };
        if !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV") {
            self.weights.q_matrix.matvec_triple_with(
                &self.weights.k_matrix,
                &self.weights.v_matrix,
                &self.input_normed_buffer,
                &mut self.q_projected_buffer,
                &mut self.k_projected_buffer,
                &mut self.v_projected_buffer,
                stream,
                "self-attn resident q/k/v projection",
            )?;
        } else if env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_PAIR_SELF_ATTN_QK") {
            self.weights.q_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.q_projected_buffer,
                stream,
                "self-attn resident q projection",
            )?;
            self.weights.k_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.k_projected_buffer,
                stream,
                "self-attn resident k projection",
            )?;
            self.weights.v_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.v_projected_buffer,
                stream,
                "self-attn resident v projection",
            )?;
        } else {
            self.weights.q_matrix.matvec_pair_with(
                &self.weights.k_matrix,
                &self.input_normed_buffer,
                &mut self.q_projected_buffer,
                &mut self.k_projected_buffer,
                stream,
                "self-attn resident q/k projection",
            )?;
            self.weights.v_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.v_projected_buffer,
                stream,
                "self-attn resident v projection",
            )?;
        }
        component_step_ms.qkv_projection_ms =
            finish_component(stream, component_started, "q/k/v projection")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_device_step_after_qkv_projection(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: PackageSelfAttnResidentStepInput<'_>,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        sync_component_timing: bool,
        component_step_ms: &mut PackageSelfAttnComponentStepMs,
        label: &str,
    ) -> Result<(), String> {
        let q_projection_layout = self.weights.q_projection_layout;
        let q_heads = self.weights.q_heads;
        let kv_heads = self.weights.kv_heads;
        let head_dim = self.weights.head_dim;
        let value_dim = self.weights.value_dim;
        let block_size = self.weights.block_size;
        let cache_blocks = self.weights.cache_blocks;
        let hidden = self.weights.hidden;
        let attention_elements = self.weights.attention_elements;
        let finish_component = |stream: &mut ullm_runtime_sys::RuntimeStream,
                                started: Instant,
                                component_label: &str|
         -> Result<f64, String> {
            if sync_component_timing {
                stream.synchronize().map_err(|err| {
                    format!("failed to synchronize {label} self-attn {component_label}: {err}")
                })?;
                Ok(started.elapsed().as_secs_f64() * 1000.0)
            } else {
                Ok(0.0)
            }
        };

        let component_started = Instant::now();
        match self.weights.q_projection_layout {
            PackageSelfAttnQProjectionLayout::Plain => {
                ullm_runtime_sys::segmented_rmsnorm_f32(
                    &self.q_projected_buffer,
                    self.weights.q_norm_weight_buffer.as_ref(),
                    q_heads,
                    head_dim,
                    1e-5_f32,
                    &mut self.q_normed_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn q RMSNorm: {err}"))?;
                ullm_runtime_sys::segmented_rmsnorm_f32(
                    &self.k_projected_buffer,
                    self.weights.k_norm_weight_buffer.as_ref(),
                    kv_heads,
                    head_dim,
                    1e-5_f32,
                    &mut self.k_normed_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn k RMSNorm: {err}"))?;
                ullm_runtime_sys::rope_f32(
                    &self.q_normed_buffer,
                    1,
                    q_heads,
                    head_dim,
                    rotary_dim,
                    rope_position,
                    rope_base,
                    &mut self.q_rope_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn q RoPE: {err}"))?;
                ullm_runtime_sys::rope_f32(
                    &self.k_normed_buffer,
                    1,
                    kv_heads,
                    head_dim,
                    rotary_dim,
                    rope_position,
                    rope_base,
                    &mut self.k_rope_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn k RoPE: {err}"))?;
                ullm_runtime_sys::paged_kv_write_f32(
                    &self.k_rope_buffer,
                    &self.v_projected_buffer,
                    &self.block_table_buffer,
                    cache_position,
                    block_size,
                    cache_blocks,
                    kv_heads,
                    head_dim,
                    value_dim,
                    &mut self.k_cache_buffer,
                    &mut self.v_cache_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn paged KV write: {err}"))?;
            }
            PackageSelfAttnQProjectionLayout::Qwen35Gated => {
                ullm_runtime_sys::qwen35_qk_norm_rope_paged_kv_write_f32(
                    &self.q_projected_buffer,
                    &self.k_projected_buffer,
                    &self.v_projected_buffer,
                    self.weights.q_norm_weight_buffer.as_ref(),
                    self.weights.k_norm_weight_buffer.as_ref(),
                    &self.block_table_buffer,
                    q_heads,
                    kv_heads,
                    head_dim,
                    value_dim,
                    rotary_dim,
                    rope_position,
                    rope_base,
                    1e-5_f32,
                    cache_position,
                    block_size,
                    cache_blocks,
                    &mut self.q_gate_buffer,
                    &mut self.q_rope_buffer,
                    &mut self.k_cache_buffer,
                    &mut self.v_cache_buffer,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run {label} self-attn q/k norm RoPE paged KV write: {err}")
                })?;
            }
        }
        component_step_ms.qk_norm_rope_kv_write_ms =
            finish_component(stream, component_started, "q/k norm RoPE paged KV write")?;

        self.written_len = self
            .written_len
            .checked_add(1)
            .ok_or_else(|| format!("{label} self-attn written length overflows"))?;

        let component_started = Instant::now();
        if self.weights.use_paged_decode_sigmoid_gate {
            ullm_runtime_sys::paged_decode_attn_sigmoid_gate_f32(
                &self.q_rope_buffer,
                &self.q_gate_buffer,
                &self.k_cache_buffer,
                &self.v_cache_buffer,
                &self.block_table_buffer,
                self.written_len,
                block_size,
                cache_blocks,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                1.0_f32 / (head_dim as f32).sqrt(),
                &mut self.attention_output_buffer,
                Some(stream),
            )
        } else {
            ullm_runtime_sys::paged_decode_attn_f32(
                &self.q_rope_buffer,
                &self.k_cache_buffer,
                &self.v_cache_buffer,
                &self.block_table_buffer,
                self.written_len,
                block_size,
                cache_blocks,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                1.0_f32 / (head_dim as f32).sqrt(),
                &mut self.attention_output_buffer,
                Some(stream),
            )
        }
        .map_err(|err| format!("failed to run {label} self-attn paged decode: {err}"))?;
        component_step_ms.paged_decode_ms =
            finish_component(stream, component_started, "paged decode")?;

        let component_started = Instant::now();
        let projection_input_buffer = match q_projection_layout {
            PackageSelfAttnQProjectionLayout::Plain => &self.attention_output_buffer,
            PackageSelfAttnQProjectionLayout::Qwen35Gated
                if self.weights.use_paged_decode_sigmoid_gate =>
            {
                &self.attention_output_buffer
            }
            PackageSelfAttnQProjectionLayout::Qwen35Gated
                if !env_flag_enabled("ULLM_DISABLE_SIGMOID_MUL_IN_PLACE") =>
            {
                ullm_runtime_sys::sigmoid_mul_f32_in_place(
                    &self.q_gate_buffer,
                    &mut self.attention_output_buffer,
                    attention_elements,
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to run {label} self-attn output gate in-place: {err}")
                })?;
                &self.attention_output_buffer
            }
            PackageSelfAttnQProjectionLayout::Qwen35Gated => {
                ullm_runtime_sys::sigmoid_mul_f32(
                    &self.q_gate_buffer,
                    &self.attention_output_buffer,
                    attention_elements,
                    &mut self.attention_projection_input_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} self-attn output gate: {err}"))?;
                &self.attention_projection_input_buffer
            }
        };
        component_step_ms.output_gate_ms =
            finish_component(stream, component_started, "output gate")?;

        let component_started = Instant::now();
        self.weights
            .o_matrix
            .matvec_add(
                projection_input_buffer,
                match input {
                    PackageSelfAttnResidentStepInput::InternalInputBuffer => &self.input_buffer,
                    PackageSelfAttnResidentStepInput::ExternalBuffer(buffer) => buffer,
                },
                &mut self.attention_block_output_buffer,
                stream,
                "self-attn resident o projection residual",
            )
            .map_err(|err| {
                format!("failed to run {label} self-attn o projection residual: {err}")
            })?;
        component_step_ms.o_projection_residual_ms =
            finish_component(stream, component_started, "o projection residual")?;

        let component_started = Instant::now();
        ullm_runtime_sys::rmsnorm_f32(
            &self.attention_block_output_buffer,
            self.weights.post_norm_weight_buffer.as_ref(),
            hidden,
            1e-5_f32,
            &mut self.post_normed_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} self-attn post RMSNorm: {err}"))?;
        component_step_ms.post_rmsnorm_ms =
            finish_component(stream, component_started, "post RMSNorm")?;

        let component_started = Instant::now();
        self.weights.mlp_gate_matrix.matvec_silu_mul_with(
            &self.weights.mlp_up_matrix,
            &self.post_normed_buffer,
            &mut self.mlp_activation_buffer,
            stream,
            "self-attn resident MLP gate/up activation",
        )?;
        component_step_ms.mlp_gate_up_activation_ms =
            finish_component(stream, component_started, "MLP gate/up activation")?;

        let component_started = Instant::now();
        self.weights
            .mlp_down_matrix
            .matvec_add(
                &self.mlp_activation_buffer,
                &self.attention_block_output_buffer,
                &mut self.layer_output_buffer,
                stream,
                "self-attn resident MLP down residual",
            )
            .map_err(|err| format!("failed to run {label} self-attn MLP down residual: {err}"))?;
        component_step_ms.mlp_down_residual_ms =
            finish_component(stream, component_started, "MLP down residual")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_device_step(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: PackageSelfAttnResidentStepInput<'_>,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        if cache_position != self.written_len {
            return Err(format!(
                "{label} self-attn resident cache_position {cache_position} does not match written_len {}",
                self.written_len
            ));
        }
        let sync_component_timing = self.weights.sync_component_timing;
        let mut component_step_ms = PackageSelfAttnComponentStepMs::default();
        self.last_component_step_ms = None;
        self.run_device_step_input(stream, input, sync_component_timing, &mut component_step_ms, label)?;
        self.run_device_step_qkv_projection(stream, sync_component_timing, &mut component_step_ms, label)?;
        self.run_device_step_after_qkv_projection(
            stream,
            input,
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            sync_component_timing,
            &mut component_step_ms,
            label,
        )?;
        if sync_component_timing {
            self.last_component_step_ms = Some(component_step_ms);
        }
        Ok(())
    }
}

struct PackageLinearAttnResidentStepWeights {
    layer_index: usize,
    sync_component_timing: bool,
    use_qkv_z_gate_beta_fusion: bool,
    use_qkv_z_pair: bool,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    hidden: usize,
    q_scale: f32,
    qk_l2_norm: bool,
    kernel_size: usize,
    input_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    conv_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    a_log_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    dt_bias_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    attn_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    post_norm_weight_buffer: std::sync::Arc<ullm_runtime_sys::RuntimeBuffer>,
    qkv_matrix: PackageAq4ResidentMatvec,
    a_matrix: PackageAq4ResidentMatvec,
    b_matrix: PackageAq4ResidentMatvec,
    z_matrix: PackageAq4ResidentMatvec,
    out_matrix: PackageAq4ResidentMatvec,
    mlp_gate_matrix: PackageAq4ResidentMatvec,
    mlp_up_matrix: PackageAq4ResidentMatvec,
    mlp_down_matrix: PackageAq4ResidentMatvec,
}

struct PackageLinearAttnResidentStepLayer {
    weights: std::sync::Arc<PackageLinearAttnResidentStepWeights>,
    last_component_step_ms: Option<PackageLinearAttnComponentStepMs>,
    conv_history_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_state_buffer: ullm_runtime_sys::RuntimeBuffer,
    input_buffer: ullm_runtime_sys::RuntimeBuffer,
    input_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    qkv_buffer: ullm_runtime_sys::RuntimeBuffer,
    qkv_conv_output_buffer: ullm_runtime_sys::RuntimeBuffer,
    z_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_q_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_k_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_v_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_gate_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_beta_buffer: ullm_runtime_sys::RuntimeBuffer,
    recurrent_output_buffer: ullm_runtime_sys::RuntimeBuffer,
    attn_projection_input_buffer: ullm_runtime_sys::RuntimeBuffer,
    attn_block_output_buffer: ullm_runtime_sys::RuntimeBuffer,
    post_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    mlp_activation_buffer: ullm_runtime_sys::RuntimeBuffer,
    layer_output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

impl std::ops::Deref for PackageLinearAttnResidentStepLayer {
    type Target = PackageLinearAttnResidentStepWeights;

    fn deref(&self) -> &Self::Target {
        self.weights.as_ref()
    }
}

#[derive(Clone, Copy)]
enum PackageLinearAttnResidentStepInput<'a> {
    InternalInputBuffer,
    ExternalBuffer(&'a ullm_runtime_sys::RuntimeBuffer),
}

impl PackageLinearAttnResidentStepLayer {
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
    ) -> Result<Self, String> {
        let mut registry = WeightRegistry::new();
        Self::load_with_registry(
            context,
            stream,
            &mut registry,
            None,
            path,
            chunk_bytes,
            layer_index,
            None,
        )
    }

    fn load_with_registry(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        mut shared_buffers: Option<&mut PackageResidentSharedBufferRegistry>,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        let key_heads = 16_usize;
        let value_heads = 32_usize;
        let key_dim = 128_usize;
        let value_dim = 128_usize;
        let hidden = value_heads * value_dim;
        let sync_component_timing = env_flag_enabled("ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING");
        let use_qkv_z_gate_beta_fusion_requested =
            !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA");
        let use_qkv_z_pair = !env_flag_enabled("ULLM_DISABLE_AQ4_MATVEC_PAIR_QKV_Z");
        let q_elements_per_step = key_heads * key_dim;
        let k_elements_per_step = q_elements_per_step;
        let v_elements_per_step = hidden;
        let qkv_step_elements = q_elements_per_step + k_elements_per_step + v_elements_per_step;
        let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
        let qk_l2_norm = true;

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
        let dt_bias_tensor =
            format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
        let z_tensor =
            format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
        let norm_tensor =
            format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
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
                "linear-attn resident input norm length mismatch: got {} expected {hidden}",
                input_norm.values.len()
            ));
        }
        let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)?;
        if conv.shape.len() != 3 || conv.shape[1] != 1 {
            return Err(format!(
                "linear-attn resident conv1d tensor shape must be [channels,1,kernel], got {}",
                format_u64_shape(&conv.shape)
            ));
        }
        let conv_channels = usize::try_from(conv.shape[0])
            .map_err(|_| "linear-attn resident conv channels are too large".to_string())?;
        let kernel_size = usize::try_from(conv.shape[2])
            .map_err(|_| "linear-attn resident conv kernel is too large".to_string())?;
        if conv_channels != qkv_step_elements {
            return Err(format!(
                "linear-attn resident conv channels mismatch: got {conv_channels} expected {qkv_step_elements}"
            ));
        }
        if conv.values.len() != conv_channels * kernel_size {
            return Err(format!(
                "linear-attn resident conv element count mismatch: got {} expected {}",
                conv.values.len(),
                conv_channels * kernel_size
            ));
        }
        let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)?;
        if a_log.values.len() != value_heads {
            return Err(format!(
                "linear-attn resident A_log length mismatch: got {} expected {value_heads}",
                a_log.values.len()
            ));
        }
        let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)?;
        if dt_bias.values.len() != value_heads {
            return Err(format!(
                "linear-attn resident dt_bias length mismatch: got {} expected {value_heads}",
                dt_bias.values.len()
            ));
        }
        let attn_norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)?;
        if attn_norm.values.len() != value_dim {
            return Err(format!(
                "linear-attn resident norm length mismatch: got {} expected {value_dim}",
                attn_norm.values.len()
            ));
        }
        let post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
        if post_norm.values.len() != hidden {
            return Err(format!(
                "linear-attn resident post norm length mismatch: got {} expected {hidden}",
                post_norm.values.len()
            ));
        }
        let input_norm_weight_values =
            effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
        let post_norm_weight_values =
            effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);

        let qkv_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &qkv_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let a_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &a_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let b_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &b_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let z_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &z_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let out_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &out_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_gate_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &gate_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_up_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &up_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        let mlp_down_matrix = PackageAq4ResidentMatvec::load_with_sq_overlay(
            context,
            stream,
            registry,
            shared_buffers.as_deref_mut(),
            path,
            &down_tensor,
            chunk_bytes,
            sq_overlay,
        )?;
        if qkv_matrix.rows != qkv_step_elements || qkv_matrix.cols != hidden {
            return Err(format!(
                "linear-attn resident qkv shape mismatch: got [{},{}] expected [{qkv_step_elements},{hidden}]",
                qkv_matrix.rows, qkv_matrix.cols
            ));
        }
        if a_matrix.rows != value_heads
            || b_matrix.rows != value_heads
            || a_matrix.cols != hidden
            || b_matrix.cols != hidden
        {
            return Err(format!(
                "linear-attn resident a/b shape mismatch: a=[{},{}] b=[{},{}] expected [{value_heads},{hidden}]",
                a_matrix.rows, a_matrix.cols, b_matrix.rows, b_matrix.cols
            ));
        }
        if z_matrix.rows != hidden
            || z_matrix.cols != hidden
            || out_matrix.rows != hidden
            || out_matrix.cols != hidden
        {
            return Err(format!(
                "linear-attn resident z/out shape mismatch: z=[{},{}] out=[{},{}] expected [{hidden},{hidden}]",
                z_matrix.rows, z_matrix.cols, out_matrix.rows, out_matrix.cols
            ));
        }
        if mlp_gate_matrix.rows != mlp_up_matrix.rows
            || mlp_gate_matrix.cols != mlp_up_matrix.cols
            || mlp_gate_matrix.cols != hidden
        {
            return Err(format!(
                "linear-attn resident MLP gate/up shape mismatch: gate=[{},{}] up=[{},{}] hidden={hidden}",
                mlp_gate_matrix.rows, mlp_gate_matrix.cols, mlp_up_matrix.rows, mlp_up_matrix.cols
            ));
        }
        if mlp_down_matrix.rows != hidden || mlp_down_matrix.cols != mlp_gate_matrix.rows {
            return Err(format!(
                "linear-attn resident MLP down shape mismatch: got [{},{}] expected [{hidden},{}]",
                mlp_down_matrix.rows, mlp_down_matrix.cols, mlp_gate_matrix.rows
            ));
        }
        let intermediate = mlp_gate_matrix.rows;

        let hidden_bytes = checked_f32_byte_len(hidden, "linear-attn resident hidden")?;
        let qkv_step_bytes =
            checked_f32_byte_len(qkv_step_elements, "linear-attn resident qkv step")?;
        let gate_beta_step_bytes =
            checked_f32_byte_len(value_heads, "linear-attn resident gate/beta step")?;
        let intermediate_bytes =
            checked_f32_byte_len(intermediate, "linear-attn resident intermediate")?;
        let conv_history_elements =
            qkv_step_elements.checked_mul(kernel_size).ok_or_else(|| {
                "linear-attn resident conv history element count overflows".to_string()
            })?;
        let conv_history_bytes =
            checked_f32_byte_len(conv_history_elements, "linear-attn resident conv history")?;
        let state_elements = value_heads
            .checked_mul(key_dim)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| {
                "linear-attn resident recurrent state element count overflows".to_string()
            })?;
        let state_bytes = checked_f32_byte_len(state_elements, "linear-attn resident state")?;

        let a_log_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-a-log:{a_log_tensor}"),
            &a_log.values,
            "linear-attn resident A_log",
        )?;
        let dt_bias_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-dt-bias:{dt_bias_tensor}"),
            &dt_bias.values,
            "linear-attn resident dt_bias",
        )?;
        let input_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-input-norm:{input_norm_tensor}"),
            &input_norm_weight_values,
            "linear-attn resident input norm weight",
        )?;
        let post_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-post-norm:{post_norm_tensor}"),
            &post_norm_weight_values,
            "linear-attn resident post norm weight",
        )?;
        let attn_norm_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-attn-norm:{norm_tensor}"),
            &attn_norm.values,
            "linear-attn resident attention norm weight",
        )?;
        let conv_weight_buffer = package_resident_f32_buffer(
            context,
            stream,
            &mut shared_buffers,
            format!("linear-attn-conv-weight:{conv_tensor}"),
            &conv.values,
            "linear-attn resident conv weight",
        )?;
        let mut conv_history_buffer = context.alloc_buffer(conv_history_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident conv history: {err}")
        })?;

        let mut input_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident input: {err}"))?;
        let mut input_normed_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident input normed: {err}")
        })?;
        let mut qkv_buffer = context
            .alloc_buffer(qkv_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident qkv: {err}"))?;
        let mut qkv_conv_output_buffer = context.alloc_buffer(qkv_step_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident qkv conv output: {err}")
        })?;
        let mut z_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident z: {err}"))?;
        let mut recurrent_q_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                q_elements_per_step,
                "linear-attn resident recurrent q",
            )?)
            .map_err(|err| format!("failed to allocate linear-attn resident q: {err}"))?;
        let mut recurrent_k_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                k_elements_per_step,
                "linear-attn resident recurrent k",
            )?)
            .map_err(|err| format!("failed to allocate linear-attn resident k: {err}"))?;
        let mut recurrent_v_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident v: {err}"))?;
        let mut recurrent_gate_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident gate: {err}"))?;
        let mut recurrent_beta_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident beta: {err}"))?;
        let mut recurrent_state_buffer = context
            .alloc_buffer(state_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident state: {err}"))?;
        let mut recurrent_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident recurrent output: {err}")
        })?;
        let mut attn_projection_input_buffer =
            context.alloc_buffer(hidden_bytes).map_err(|err| {
                format!("failed to allocate linear-attn resident attention projection input: {err}")
            })?;
        let mut attn_block_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident attention block: {err}")
        })?;
        let mut post_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident post normed: {err}"))?;
        let mut mlp_activation_buffer =
            context.alloc_buffer(intermediate_bytes).map_err(|err| {
                format!("failed to allocate linear-attn resident MLP activation: {err}")
            })?;
        let layer_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident layer output: {err}")
        })?;

        conv_history_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; conv_history_elements]),
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to initialize linear-attn resident conv history: {err}")
            })?;
        recurrent_state_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; state_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize linear-attn resident state: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn resident layer setup: {err}")
        })?;
        let device_info = context
            .device_info()
            .map_err(|err| format!("failed to get linear-attn resident device info: {err}"))?;
        let use_qkv_z_gate_beta_fusion =
            use_qkv_z_gate_beta_fusion_requested && device_info.backend == "hip";
        if device_info.backend == "hip" {
            prewarm_aq4_matvec_once(
                stream,
                &qkv_matrix,
                &mut input_normed_buffer,
                &mut qkv_buffer,
                "linear-attn resident AQ4 matvec",
            )?;
            if use_qkv_z_pair {
                prewarm_aq4_matvec_pair_once(
                    stream,
                    &qkv_matrix,
                    &z_matrix,
                    &mut input_normed_buffer,
                    &mut qkv_buffer,
                    &mut z_buffer,
                    "linear-attn resident AQ4 qkv/z pair",
                )?;
            }
            if use_qkv_z_gate_beta_fusion {
                prewarm_aq4_matvec_qkv_z_gate_beta_once(
                    stream,
                    &qkv_matrix,
                    &z_matrix,
                    &a_matrix,
                    &b_matrix,
                    &mut input_normed_buffer,
                    &a_log_buffer,
                    &dt_bias_buffer,
                    &mut qkv_buffer,
                    &mut z_buffer,
                    &mut recurrent_gate_buffer,
                    &mut recurrent_beta_buffer,
                    "linear-attn resident AQ4 qkv/z gate-beta",
                )?;
            }
            prewarm_aq4_matvec_gate_beta_once(
                stream,
                &a_matrix,
                &b_matrix,
                &mut input_normed_buffer,
                &a_log_buffer,
                &dt_bias_buffer,
                &mut recurrent_gate_buffer,
                &mut recurrent_beta_buffer,
                "linear-attn resident AQ4 gate-beta",
            )?;
            prewarm_aq4_matvec_silu_mul_once(
                stream,
                &mlp_gate_matrix,
                &mlp_up_matrix,
                &mut post_normed_buffer,
                &mut mlp_activation_buffer,
                "linear-attn resident AQ4 SiLU-mul",
            )?;
            prewarm_linear_attn_qkv_prepare_once(
                device_info.device_id,
                stream,
                &mut qkv_buffer,
                &conv_weight_buffer,
                &mut conv_history_buffer,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                q_scale,
                qk_l2_norm,
                &mut qkv_conv_output_buffer,
                &mut recurrent_q_buffer,
                &mut recurrent_k_buffer,
                &mut recurrent_v_buffer,
                "linear-attn resident qkv prepare",
            )?;
            prewarm_linear_attn_post_once(
                device_info.device_id,
                stream,
                &mut recurrent_output_buffer,
                &attn_norm_weight_buffer,
                &mut z_buffer,
                value_heads,
                value_dim,
                &mut attn_projection_input_buffer,
                "linear-attn resident post RMSNorm SiLU-mul",
            )?;
        }
        prewarm_aq4_matvec_add_once(
            stream,
            &out_matrix,
            &mut attn_projection_input_buffer,
            &mut input_buffer,
            &mut attn_block_output_buffer,
            "linear-attn resident AQ4 matvec add",
        )?;

        let weights = std::sync::Arc::new(PackageLinearAttnResidentStepWeights {
            layer_index,
            sync_component_timing,
            use_qkv_z_gate_beta_fusion,
            use_qkv_z_pair,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            hidden,
            q_scale,
            qk_l2_norm,
            kernel_size,
            input_norm_weight_buffer,
            conv_weight_buffer,
            a_log_buffer,
            dt_bias_buffer,
            attn_norm_weight_buffer,
            post_norm_weight_buffer,
            qkv_matrix,
            a_matrix,
            b_matrix,
            z_matrix,
            out_matrix,
            mlp_gate_matrix,
            mlp_up_matrix,
            mlp_down_matrix,
        });

        Ok(Self {
            weights,
            last_component_step_ms: None,
            conv_history_buffer,
            recurrent_state_buffer,
            input_buffer,
            input_normed_buffer,
            qkv_buffer,
            qkv_conv_output_buffer,
            z_buffer,
            recurrent_q_buffer,
            recurrent_k_buffer,
            recurrent_v_buffer,
            recurrent_gate_buffer,
            recurrent_beta_buffer,
            recurrent_output_buffer,
            attn_projection_input_buffer,
            attn_block_output_buffer,
            post_normed_buffer,
            mlp_activation_buffer,
            layer_output_buffer,
        })
    }

    fn load_state_with_weights(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        weights: std::sync::Arc<PackageLinearAttnResidentStepWeights>,
    ) -> Result<Self, String> {
        let q_elements_per_step = weights
            .key_heads
            .checked_mul(weights.key_dim)
            .ok_or_else(|| "linear-attn resident q element count overflows".to_string())?;
        let k_elements_per_step = q_elements_per_step;
        let v_elements_per_step = weights.hidden;
        let qkv_step_elements = q_elements_per_step
            .checked_add(k_elements_per_step)
            .and_then(|value| value.checked_add(v_elements_per_step))
            .ok_or_else(|| "linear-attn resident qkv step element count overflows".to_string())?;
        let hidden_bytes = checked_f32_byte_len(weights.hidden, "linear-attn resident hidden")?;
        let qkv_step_bytes =
            checked_f32_byte_len(qkv_step_elements, "linear-attn resident qkv step")?;
        let gate_beta_step_bytes =
            checked_f32_byte_len(weights.value_heads, "linear-attn resident gate/beta step")?;
        let intermediate_bytes = checked_f32_byte_len(
            weights.mlp_gate_matrix.rows,
            "linear-attn resident intermediate",
        )?;
        let conv_history_elements = qkv_step_elements
            .checked_mul(weights.kernel_size)
            .ok_or_else(|| {
                "linear-attn resident conv history element count overflows".to_string()
            })?;
        let conv_history_bytes =
            checked_f32_byte_len(conv_history_elements, "linear-attn resident conv history")?;
        let state_elements = weights
            .value_heads
            .checked_mul(weights.key_dim)
            .and_then(|value| value.checked_mul(weights.value_dim))
            .ok_or_else(|| {
                "linear-attn resident recurrent state element count overflows".to_string()
            })?;
        let state_bytes = checked_f32_byte_len(state_elements, "linear-attn resident state")?;

        let mut conv_history_buffer = context.alloc_buffer(conv_history_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident conv history: {err}")
        })?;
        let input_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident input: {err}"))?;
        let input_normed_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident input normed: {err}")
        })?;
        let qkv_buffer = context
            .alloc_buffer(qkv_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident qkv: {err}"))?;
        let qkv_conv_output_buffer = context.alloc_buffer(qkv_step_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident qkv conv output: {err}")
        })?;
        let z_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident z: {err}"))?;
        let recurrent_q_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                q_elements_per_step,
                "linear-attn resident recurrent q",
            )?)
            .map_err(|err| format!("failed to allocate linear-attn resident q: {err}"))?;
        let recurrent_k_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                k_elements_per_step,
                "linear-attn resident recurrent k",
            )?)
            .map_err(|err| format!("failed to allocate linear-attn resident k: {err}"))?;
        let recurrent_v_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident v: {err}"))?;
        let recurrent_gate_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident gate: {err}"))?;
        let recurrent_beta_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident beta: {err}"))?;
        let mut recurrent_state_buffer = context
            .alloc_buffer(state_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident state: {err}"))?;
        let recurrent_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident recurrent output: {err}")
        })?;
        let attn_projection_input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident attention projection input: {err}")
        })?;
        let attn_block_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident attention block: {err}")
        })?;
        let post_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear-attn resident post normed: {err}"))?;
        let mlp_activation_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident MLP activation: {err}")
        })?;
        let layer_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear-attn resident layer output: {err}")
        })?;

        conv_history_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; conv_history_elements]),
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to initialize linear-attn resident conv history: {err}")
            })?;
        recurrent_state_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&vec![0.0_f32; state_elements]),
                Some(stream),
            )
            .map_err(|err| format!("failed to initialize linear-attn resident state: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize linear-attn resident shared-weight state setup: {err}")
        })?;

        Ok(Self {
            weights,
            last_component_step_ms: None,
            conv_history_buffer,
            recurrent_state_buffer,
            input_buffer,
            input_normed_buffer,
            qkv_buffer,
            qkv_conv_output_buffer,
            z_buffer,
            recurrent_q_buffer,
            recurrent_k_buffer,
            recurrent_v_buffer,
            recurrent_gate_buffer,
            recurrent_beta_buffer,
            recurrent_output_buffer,
            attn_projection_input_buffer,
            attn_block_output_buffer,
            post_normed_buffer,
            mlp_activation_buffer,
            layer_output_buffer,
        })
    }

    fn step(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        residual: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.step_from_host_to_device(stream, residual, "linear-attn resident layer")?;
        self.read_output(stream)
    }

    fn step_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        residual: &[f32],
        label: &str,
    ) -> Result<(), String> {
        if residual.len() != self.hidden {
            return Err(format!(
                "linear-attn resident layer {} residual length mismatch: got {} expected {}",
                self.layer_index,
                residual.len(),
                self.hidden
            ));
        }
        self.input_buffer
            .copy_from_host(0, &encode_f32_to_bytes(residual), Some(stream))
            .map_err(|err| format!("failed to copy linear-attn resident residual: {err}"))?;
        self.run_device_step(
            stream,
            PackageLinearAttnResidentStepInput::InternalInputBuffer,
            label,
        )
    }

    fn step_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        let expected_bytes = checked_f32_byte_len(self.hidden, "linear-attn resident input")?;
        let actual_bytes = residual_buffer
            .size()
            .map_err(|err| format!("failed to query {label} residual buffer size: {err}"))?;
        if actual_bytes < expected_bytes {
            return Err(format!(
                "{label} residual buffer is too small: got {actual_bytes} bytes expected at least {expected_bytes}"
            ));
        }
        self.run_device_step(
            stream,
            PackageLinearAttnResidentStepInput::ExternalBuffer(residual_buffer),
            label,
        )
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<Vec<f32>, String> {
        read_runtime_buffer_f32(
            &self.layer_output_buffer,
            stream,
            self.hidden,
            "linear-attn resident layer output",
        )
    }

    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.layer_output_buffer
    }

    fn take_last_component_step_ms(&mut self) -> Option<PackageLinearAttnComponentStepMs> {
        self.last_component_step_ms.take()
    }

    fn run_device_step(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: PackageLinearAttnResidentStepInput<'_>,
        label: &str,
    ) -> Result<(), String> {
        self.last_component_step_ms = None;
        let weights = self.weights.as_ref();
        let hidden = weights.hidden;
        let key_heads = weights.key_heads;
        let value_heads = weights.value_heads;
        let key_dim = weights.key_dim;
        let value_dim = weights.value_dim;
        let sync_component_timing = weights.sync_component_timing;
        let mut component_step_ms = PackageLinearAttnComponentStepMs::default();
        macro_rules! component_started {
            () => {
                if sync_component_timing {
                    Some(Instant::now())
                } else {
                    None
                }
            };
        }
        macro_rules! finish_component {
            ($started:expr, $field:ident, $component:expr) => {
                if let Some(component_started) = $started {
                    stream.synchronize().map_err(|err| {
                        format!("failed to synchronize {label} {}: {err}", $component)
                    })?;
                    component_step_ms.$field = component_started.elapsed().as_secs_f64() * 1000.0;
                }
            };
        }

        let component_started = component_started!();
        match input {
            PackageLinearAttnResidentStepInput::InternalInputBuffer => {
                ullm_runtime_sys::rmsnorm_f32(
                    &self.input_buffer,
                    weights.input_norm_weight_buffer.as_ref(),
                    hidden,
                    1e-6_f32,
                    &mut self.input_normed_buffer,
                    Some(stream),
                )
            }
            PackageLinearAttnResidentStepInput::ExternalBuffer(buffer) => {
                ullm_runtime_sys::rmsnorm_f32(
                    buffer,
                    weights.input_norm_weight_buffer.as_ref(),
                    hidden,
                    1e-6_f32,
                    &mut self.input_normed_buffer,
                    Some(stream),
                )
            }
        }
        .map_err(|err| format!("failed to run {label} input RMSNorm: {err}"))?;
        finish_component!(component_started, input_rmsnorm_ms, "input RMSNorm");

        let use_qkv_z_gate_beta_fusion =
            weights.use_qkv_z_gate_beta_fusion && !sync_component_timing;
        if use_qkv_z_gate_beta_fusion {
            weights.qkv_matrix.matvec_qkv_z_gate_beta_with(
                &weights.z_matrix,
                &weights.a_matrix,
                &weights.b_matrix,
                &self.input_normed_buffer,
                weights.a_log_buffer.as_ref(),
                weights.dt_bias_buffer.as_ref(),
                &mut self.qkv_buffer,
                &mut self.z_buffer,
                &mut self.recurrent_gate_buffer,
                &mut self.recurrent_beta_buffer,
                stream,
                "linear-attn resident qkv/z gate-beta projection",
            )?;
        } else if weights.use_qkv_z_pair && !sync_component_timing {
            weights.qkv_matrix.matvec_pair_with(
                &weights.z_matrix,
                &self.input_normed_buffer,
                &mut self.qkv_buffer,
                &mut self.z_buffer,
                stream,
                "linear-attn resident qkv/z projection",
            )?;
        } else {
            let component_started = component_started!();
            weights.qkv_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.qkv_buffer,
                stream,
                "linear-attn resident qkv projection",
            )?;
            finish_component!(component_started, qkv_projection_ms, "qkv projection");
            let component_started = component_started!();
            weights.z_matrix.matvec(
                &self.input_normed_buffer,
                &mut self.z_buffer,
                stream,
                "linear-attn resident z projection",
            )?;
            finish_component!(component_started, z_projection_ms, "z projection");
        }

        let component_started = component_started!();
        ullm_runtime_sys::linear_attn_qkv_prepare_f32(
            &self.qkv_buffer,
            weights.conv_weight_buffer.as_ref(),
            &mut self.conv_history_buffer,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            weights.kernel_size,
            weights.q_scale,
            weights.qk_l2_norm,
            &mut self.qkv_conv_output_buffer,
            &mut self.recurrent_q_buffer,
            &mut self.recurrent_k_buffer,
            &mut self.recurrent_v_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run linear-attn resident qkv prepare: {err}"))?;
        finish_component!(component_started, qkv_prepare_ms, "qkv prepare");
        if !use_qkv_z_gate_beta_fusion {
            let component_started = component_started!();
            weights.a_matrix.matvec_gate_beta_with(
                &weights.b_matrix,
                &self.input_normed_buffer,
                weights.a_log_buffer.as_ref(),
                weights.dt_bias_buffer.as_ref(),
                &mut self.recurrent_gate_buffer,
                &mut self.recurrent_beta_buffer,
                stream,
                "linear-attn resident a/b gate-beta",
            )?;
            finish_component!(component_started, gate_beta_projection_ms, "a/b gate-beta");
        }
        let component_started = component_started!();
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &self.recurrent_q_buffer,
            &self.recurrent_k_buffer,
            &self.recurrent_v_buffer,
            &self.recurrent_gate_buffer,
            &self.recurrent_beta_buffer,
            key_heads,
            value_heads,
            1,
            key_dim,
            value_dim,
            &mut self.recurrent_state_buffer,
            &mut self.recurrent_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run linear-attn resident recurrent step: {err}"))?;
        finish_component!(component_started, recurrent_ms, "recurrent step");

        let component_started = component_started!();
        ullm_runtime_sys::segmented_rmsnorm_silu_mul_f32(
            &self.recurrent_output_buffer,
            weights.attn_norm_weight_buffer.as_ref(),
            &self.z_buffer,
            value_heads,
            value_dim,
            1e-6_f32,
            &mut self.attn_projection_input_buffer,
            Some(stream),
        )
        .map_err(|err| {
            format!("failed to run linear-attn resident attention RMSNorm SiLU-mul: {err}")
        })?;
        finish_component!(
            component_started,
            attention_post_ms,
            "attention RMSNorm SiLU-mul"
        );
        let component_started = component_started!();
        weights
            .out_matrix
            .matvec_add(
                &self.attn_projection_input_buffer,
                match input {
                    PackageLinearAttnResidentStepInput::InternalInputBuffer => &self.input_buffer,
                    PackageLinearAttnResidentStepInput::ExternalBuffer(buffer) => buffer,
                },
                &mut self.attn_block_output_buffer,
                stream,
                "linear-attn resident out projection residual",
            )
            .map_err(|err| {
                format!("failed to run linear-attn resident attention residual: {err}")
            })?;
        finish_component!(
            component_started,
            out_projection_residual_ms,
            "out projection residual"
        );

        let component_started = component_started!();
        ullm_runtime_sys::rmsnorm_f32(
            &self.attn_block_output_buffer,
            weights.post_norm_weight_buffer.as_ref(),
            hidden,
            1e-5_f32,
            &mut self.post_normed_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run linear-attn resident post RMSNorm: {err}"))?;
        finish_component!(component_started, post_rmsnorm_ms, "post RMSNorm");
        let component_started = component_started!();
        weights.mlp_gate_matrix.matvec_silu_mul_with(
            &weights.mlp_up_matrix,
            &self.post_normed_buffer,
            &mut self.mlp_activation_buffer,
            stream,
            "linear-attn resident MLP gate/up activation",
        )?;
        finish_component!(
            component_started,
            mlp_gate_up_activation_ms,
            "MLP gate/up activation"
        );
        let component_started = component_started!();
        weights
            .mlp_down_matrix
            .matvec_add(
                &self.mlp_activation_buffer,
                &self.attn_block_output_buffer,
                &mut self.layer_output_buffer,
                stream,
                "linear-attn resident MLP down residual",
            )
            .map_err(|err| format!("failed to run linear-attn resident layer residual: {err}"))?;
        finish_component!(component_started, mlp_down_residual_ms, "MLP down residual");
        if sync_component_timing {
            self.last_component_step_ms = Some(component_step_ms);
        }
        Ok(())
    }
}

#[allow(dead_code)]
struct PackageSelfAttnResidentStepBatchLayer {
    layer_index: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    block_size: usize,
    cache_blocks: usize,
    request_index: std::collections::BTreeMap<RequestId, usize>,
    request_ids: Vec<RequestId>,
    layers: Vec<PackageSelfAttnResidentStepLayer>,
    batch_input_normed_buffer: ullm_runtime_sys::RuntimeBuffer,
    batch_q_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
    batch_k_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
    batch_v_projected_buffer: ullm_runtime_sys::RuntimeBuffer,
}

#[allow(dead_code)]
impl PackageSelfAttnResidentStepBatchLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
        request_ids: Vec<RequestId>,
        block_size: usize,
        cache_blocks: usize,
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err(format!(
                "self-attn resident batch layer {layer_index} block_size must be greater than zero"
            ));
        }
        if cache_blocks == 0 {
            return Err(format!(
                "self-attn resident batch layer {layer_index} cache_blocks must be greater than zero"
            ));
        }
        if cache_blocks > u32::MAX as usize {
            return Err(format!(
                "self-attn resident batch layer {layer_index} cache_blocks {cache_blocks} exceeds u32 range"
            ));
        }
        let request_index =
            package_self_attn_request_slot_index(&request_ids, "self-attn resident batch")?;
        let block_table = (0..cache_blocks)
            .map(|block| {
                u32::try_from(block).map_err(|_| {
                    format!("self-attn resident batch block index {block} exceeds u32 range")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut layers = Vec::with_capacity(request_ids.len());
        let mut hidden = None;
        let mut q_heads = None;
        let mut kv_heads = None;
        let mut head_dim = None;
        let mut value_dim = None;
        let mut registry = WeightRegistry::new();
        let mut shared_buffers = PackageResidentSharedBufferRegistry::new();
        let mut shared_weights: Option<std::sync::Arc<PackageSelfAttnResidentStepWeights>> = None;
        for request_id in &request_ids {
            let layer = if let Some(weights) = shared_weights.clone() {
                PackageSelfAttnResidentStepLayer::load_state_with_weights(
                    context,
                    stream,
                    weights,
                    &block_table,
                )
            } else {
                let layer = PackageSelfAttnResidentStepLayer::load_with_registry(
                    context,
                    stream,
                    &mut registry,
                    Some(&mut shared_buffers),
                    path,
                    chunk_bytes,
                    layer_index,
                    &block_table,
                    block_size,
                    cache_blocks,
                    sq_overlay,
                )?;
                shared_weights = Some(layer.weights.clone());
                Ok(layer)
            }
            .map_err(|err| {
                format!(
                    "failed to load self-attn resident batch layer {layer_index} for request {request_id:?}: {err}"
                )
            })?;
            if let Some(previous) = hidden {
                if previous != layer.hidden {
                    return Err(format!(
                        "self-attn resident batch layer {layer_index} hidden changed: previous={previous} current={}",
                        layer.hidden
                    ));
                }
            } else {
                hidden = Some(layer.hidden);
            }
            if let Some(previous) = q_heads {
                if previous != layer.q_heads {
                    return Err(format!(
                        "self-attn resident batch layer {layer_index} q_heads changed: previous={previous} current={}",
                        layer.q_heads
                    ));
                }
            } else {
                q_heads = Some(layer.q_heads);
            }
            if let Some(previous) = kv_heads {
                if previous != layer.kv_heads {
                    return Err(format!(
                        "self-attn resident batch layer {layer_index} kv_heads changed: previous={previous} current={}",
                        layer.kv_heads
                    ));
                }
            } else {
                kv_heads = Some(layer.kv_heads);
            }
            if let Some(previous) = head_dim {
                if previous != layer.head_dim {
                    return Err(format!(
                        "self-attn resident batch layer {layer_index} head_dim changed: previous={previous} current={}",
                        layer.head_dim
                    ));
                }
            } else {
                head_dim = Some(layer.head_dim);
            }
            if let Some(previous) = value_dim {
                if previous != layer.value_dim {
                    return Err(format!(
                        "self-attn resident batch layer {layer_index} value_dim changed: previous={previous} current={}",
                        layer.value_dim
                    ));
                }
            } else {
                value_dim = Some(layer.value_dim);
            }
            if layer.block_size != block_size || layer.cache_blocks != cache_blocks {
                return Err(format!(
                    "self-attn resident batch layer {layer_index} cache shape changed: block_size={} cache_blocks={}",
                    layer.block_size, layer.cache_blocks
                ));
            }
            layers.push(layer);
        }
        let hidden = hidden.ok_or_else(|| {
            format!("self-attn resident batch layer {layer_index} has no request slots")
        })?;
        let max_batch_count = request_ids.len();
        let batch_input_normed_elements = hidden
            .checked_mul(max_batch_count)
            .ok_or_else(|| {
                format!(
                    "self-attn resident batch layer {layer_index} input normed batch overflows"
                )
            })?;
        let first_weights = layers
            .first()
            .ok_or_else(|| format!("self-attn resident batch layer {layer_index} has no states"))?
            .weights
            .as_ref();
        let q_projected_rows = first_weights.q_matrix.rows;
        let k_projected_rows = first_weights.k_matrix.rows;
        let v_projected_rows = first_weights.v_matrix.rows;
        let batch_q_projected_elements =
            q_projected_rows
                .checked_mul(max_batch_count)
                .ok_or_else(|| {
                    format!(
                        "self-attn resident batch layer {layer_index} q projected batch overflows"
                    )
                })?;
        let batch_k_projected_elements =
            k_projected_rows
                .checked_mul(max_batch_count)
                .ok_or_else(|| {
                    format!(
                        "self-attn resident batch layer {layer_index} k projected batch overflows"
                    )
                })?;
        let batch_v_projected_elements =
            v_projected_rows
                .checked_mul(max_batch_count)
                .ok_or_else(|| {
                    format!(
                        "self-attn resident batch layer {layer_index} v projected batch overflows"
                    )
                })?;
        let batch_input_normed_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                batch_input_normed_elements,
                "self-attn resident batch input normed",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn resident batch input normed: {err}")
            })?;
        let batch_q_projected_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                batch_q_projected_elements,
                "self-attn resident batch q projected",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn resident batch q projected: {err}")
            })?;
        let batch_k_projected_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                batch_k_projected_elements,
                "self-attn resident batch k projected",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn resident batch k projected: {err}")
            })?;
        let batch_v_projected_buffer = context
            .alloc_buffer(checked_f32_byte_len(
                batch_v_projected_elements,
                "self-attn resident batch v projected",
            )?)
            .map_err(|err| {
                format!("failed to allocate self-attn resident batch v projected: {err}")
            })?;
        Ok(Self {
            layer_index,
            hidden,
            q_heads: q_heads.ok_or_else(|| {
                format!("self-attn resident batch layer {layer_index} has no q_heads")
            })?,
            kv_heads: kv_heads.ok_or_else(|| {
                format!("self-attn resident batch layer {layer_index} has no kv_heads")
            })?,
            head_dim: head_dim.ok_or_else(|| {
                format!("self-attn resident batch layer {layer_index} has no head_dim")
            })?,
            value_dim: value_dim.ok_or_else(|| {
                format!("self-attn resident batch layer {layer_index} has no value_dim")
            })?,
            block_size,
            cache_blocks,
            request_index,
            request_ids,
            layers,
            batch_input_normed_buffer,
            batch_q_projected_buffer,
            batch_k_projected_buffer,
            batch_v_projected_buffer,
        })
    }

    fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    fn request_count(&self) -> usize {
        self.request_ids.len()
    }

    fn layer_index(&self) -> usize {
        self.layer_index
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn cache_blocks(&self) -> usize {
        self.cache_blocks
    }

    fn q_heads(&self) -> usize {
        self.q_heads
    }

    fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn value_dim(&self) -> usize {
        self.value_dim
    }

    fn request_slot(&self, request_id: RequestId) -> Result<usize, String> {
        self.request_index.get(&request_id).copied().ok_or_else(|| {
            format!(
                "self-attn resident batch layer {} has no state slot for request {request_id:?}",
                self.layer_index
            )
        })
    }

    fn layer_for_request_mut(
        &mut self,
        request_id: RequestId,
    ) -> Result<&mut PackageSelfAttnResidentStepLayer, String> {
        let slot = self.request_slot(request_id)?;
        self.layers.get_mut(slot).ok_or_else(|| {
            format!(
                "self-attn resident batch layer {} missing state slot {slot} for request {request_id:?}",
                self.layer_index
            )
        })
    }

    fn layer_for_request(
        &self,
        request_id: RequestId,
    ) -> Result<&PackageSelfAttnResidentStepLayer, String> {
        let slot = self.request_slot(request_id)?;
        self.layers.get(slot).ok_or_else(|| {
            format!(
                "self-attn resident batch layer {} missing state slot {slot} for request {request_id:?}",
                self.layer_index
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn step_batch_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        items: &[MixedRequestStateBatchStepItem],
        rotary_dim: usize,
        rope_base: f32,
        label: &str,
    ) -> Result<(), String> {
        if items.is_empty() {
            return Ok(());
        }
        if items.len() == 1 {
            let item = &items[0];
            return self.step_from_host_to_device(
                stream,
                item.request_id,
                &item.residual,
                rotary_dim,
                rope_base,
                item.rope_position,
                item.cache_position,
                &format!(
                    "{label} request={} position={}",
                    item.request_id.0, item.rope_position
                ),
            );
        }
        if items.len() > self.request_count() {
            return Err(format!(
                "{label} self-attn resident batch item count {} exceeds request slots {}",
                items.len(),
                self.request_count()
            ));
        }

        let weights = self
            .layers
            .first()
            .ok_or_else(|| format!("{label} self-attn resident batch has no states"))?
            .weights
            .clone();
        let sync_component_timing = weights.sync_component_timing;
        let q_projected_rows = weights.q_matrix.rows;
        let k_projected_rows = weights.k_matrix.rows;
        let v_projected_rows = weights.v_matrix.rows;
        let batch_count = items.len();
        let input_normed_elements = self.hidden.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch input normed elements overflow")
        })?;
        let q_projected_elements = q_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch q projected elements overflow")
        })?;
        let k_projected_elements = k_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch k projected elements overflow")
        })?;
        let v_projected_elements = v_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch v projected elements overflow")
        })?;

        let mut slots = Vec::with_capacity(batch_count);
        let mut component_step_ms = Vec::with_capacity(batch_count);
        let mut input_normed_values = vec![0.0_f32; input_normed_elements];
        for (batch_index, item) in items.iter().enumerate() {
            if item.residual.len() != self.hidden {
                return Err(format!(
                    "{label} request {:?} residual length {} does not match hidden {}",
                    item.request_id,
                    item.residual.len(),
                    self.hidden
                ));
            }
            let slot = self.request_slot(item.request_id)?;
            let item_label = format!(
                "{label} request={} position={}",
                item.request_id.0, item.rope_position
            );
            let layer = self.layers.get_mut(slot).ok_or_else(|| {
                format!(
                    "{label} self-attn resident batch missing state slot {slot} for request {:?}",
                    item.request_id
                )
            })?;
            if item.cache_position != layer.written_len {
                return Err(format!(
                    "{item_label} self-attn resident cache_position {} does not match written_len {}",
                    item.cache_position, layer.written_len
                ));
            }
            layer.last_component_step_ms = None;
            layer
                .input_buffer
                .copy_from_host(0, &encode_f32_to_bytes(&item.residual), Some(stream))
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident residual: {err}")
                })?;
            let mut step_ms = PackageSelfAttnComponentStepMs::default();
            layer.run_device_step_input(
                stream,
                PackageSelfAttnResidentStepInput::InternalInputBuffer,
                sync_component_timing,
                &mut step_ms,
                &item_label,
            )?;
            let normed_values = read_runtime_buffer_f32(
                &layer.input_normed_buffer,
                stream,
                self.hidden,
                &format!("{item_label} self-attn resident input normed"),
            )?;
            let start = batch_index.checked_mul(self.hidden).ok_or_else(|| {
                format!("{label} self-attn resident batch input normed offset overflows")
            })?;
            input_normed_values[start..start + self.hidden].copy_from_slice(&normed_values);
            slots.push(slot);
            component_step_ms.push(step_ms);
        }

        self.batch_input_normed_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&input_normed_values),
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy {label} self-attn resident batch input normed: {err}")
            })?;

        let component_started = Instant::now();
        weights.q_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_q_projected_buffer,
            stream,
            "self-attn resident batch q projection",
        )?;
        weights.k_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_k_projected_buffer,
            stream,
            "self-attn resident batch k projection",
        )?;
        weights.v_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_v_projected_buffer,
            stream,
            "self-attn resident batch v projection",
        )?;
        let qkv_projection_ms = if sync_component_timing {
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize {label} self-attn batch q/k/v projection: {err}")
            })?;
            component_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        for step_ms in &mut component_step_ms {
            step_ms.qkv_projection_ms = qkv_projection_ms;
        }

        let q_projected_values = read_runtime_buffer_f32(
            &self.batch_q_projected_buffer,
            stream,
            q_projected_elements,
            &format!("{label} self-attn resident batch q projected"),
        )?;
        let k_projected_values = read_runtime_buffer_f32(
            &self.batch_k_projected_buffer,
            stream,
            k_projected_elements,
            &format!("{label} self-attn resident batch k projected"),
        )?;
        let v_projected_values = read_runtime_buffer_f32(
            &self.batch_v_projected_buffer,
            stream,
            v_projected_elements,
            &format!("{label} self-attn resident batch v projected"),
        )?;

        for (batch_index, item) in items.iter().enumerate() {
            let item_label = format!(
                "{label} request={} position={}",
                item.request_id.0, item.rope_position
            );
            let layer = self.layers.get_mut(slots[batch_index]).ok_or_else(|| {
                format!(
                    "{label} self-attn resident batch missing state slot {} for request {:?}",
                    slots[batch_index], item.request_id
                )
            })?;
            let q_start = batch_index.checked_mul(q_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch q projected offset overflows")
            })?;
            let k_start = batch_index.checked_mul(k_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch k projected offset overflows")
            })?;
            let v_start = batch_index.checked_mul(v_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch v projected offset overflows")
            })?;
            layer
                .q_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&q_projected_values[q_start..q_start + q_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident q projected: {err}")
                })?;
            layer
                .k_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&k_projected_values[k_start..k_start + k_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident k projected: {err}")
                })?;
            layer
                .v_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&v_projected_values[v_start..v_start + v_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident v projected: {err}")
                })?;
            layer.run_device_step_after_qkv_projection(
                stream,
                PackageSelfAttnResidentStepInput::InternalInputBuffer,
                rotary_dim,
                rope_base,
                item.rope_position,
                item.cache_position,
                sync_component_timing,
                component_step_ms
                    .get_mut(batch_index)
                    .ok_or_else(|| {
                        format!("{label} self-attn resident batch component timing is missing")
                    })?,
                &item_label,
            )?;
            if sync_component_timing {
                layer.last_component_step_ms = Some(component_step_ms[batch_index]);
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_batch_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        items: &[(RequestId, &ullm_runtime_sys::RuntimeBuffer, usize, usize)],
        rotary_dim: usize,
        rope_base: f32,
        label: &str,
    ) -> Result<(), String> {
        if items.is_empty() {
            return Ok(());
        }
        if items.len() == 1 {
            let &(request_id, residual_buffer, rope_position, cache_position) = &items[0];
            return self.step_from_device_to_device(
                stream,
                request_id,
                residual_buffer,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &format!(
                    "{label} request={} position={}",
                    request_id.0, rope_position
                ),
            );
        }
        if items.len() > self.request_count() {
            return Err(format!(
                "{label} self-attn resident batch item count {} exceeds request slots {}",
                items.len(),
                self.request_count()
            ));
        }

        let weights = self
            .layers
            .first()
            .ok_or_else(|| format!("{label} self-attn resident batch has no states"))?
            .weights
            .clone();
        let sync_component_timing = weights.sync_component_timing;
        let q_projected_rows = weights.q_matrix.rows;
        let k_projected_rows = weights.k_matrix.rows;
        let v_projected_rows = weights.v_matrix.rows;
        let batch_count = items.len();
        let input_normed_elements = self.hidden.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch input normed elements overflow")
        })?;
        let q_projected_elements = q_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch q projected elements overflow")
        })?;
        let k_projected_elements = k_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch k projected elements overflow")
        })?;
        let v_projected_elements = v_projected_rows.checked_mul(batch_count).ok_or_else(|| {
            format!("{label} self-attn resident batch v projected elements overflow")
        })?;

        let expected_input_bytes = checked_f32_byte_len(self.hidden, "self-attn resident batch input")?;
        let mut slots = Vec::with_capacity(batch_count);
        let mut component_step_ms = Vec::with_capacity(batch_count);
        let mut input_normed_values = vec![0.0_f32; input_normed_elements];
        for (batch_index, item) in items.iter().enumerate() {
            let &(request_id, residual_buffer, rope_position, cache_position) = item;
            let item_label = format!(
                "{label} request={} position={}",
                request_id.0, rope_position
            );
            let actual_bytes = residual_buffer
                .size()
                .map_err(|err| format!("failed to query {item_label} self-attn residual size: {err}"))?;
            if actual_bytes < expected_input_bytes {
                return Err(format!(
                    "{item_label} self-attn resident residual buffer too small: got {actual_bytes} bytes expected at least {expected_input_bytes}"
                ));
            }
            let slot = self.request_slot(request_id)?;
            let layer = self.layers.get_mut(slot).ok_or_else(|| {
                format!(
                    "{label} self-attn resident batch missing state slot {slot} for request {request_id:?}"
                )
            })?;
            if cache_position != layer.written_len {
                return Err(format!(
                    "{item_label} self-attn resident cache_position {} does not match written_len {}",
                    cache_position, layer.written_len
                ));
            }
            layer.last_component_step_ms = None;
            let mut step_ms = PackageSelfAttnComponentStepMs::default();
            layer.run_device_step_input(
                stream,
                PackageSelfAttnResidentStepInput::ExternalBuffer(residual_buffer),
                sync_component_timing,
                &mut step_ms,
                &item_label,
            )?;
            let normed_values = read_runtime_buffer_f32(
                &layer.input_normed_buffer,
                stream,
                self.hidden,
                &format!("{item_label} self-attn resident input normed"),
            )?;
            let start = batch_index.checked_mul(self.hidden).ok_or_else(|| {
                format!("{label} self-attn resident batch input normed offset overflows")
            })?;
            input_normed_values[start..start + self.hidden].copy_from_slice(&normed_values);
            slots.push(slot);
            component_step_ms.push(step_ms);
        }

        self.batch_input_normed_buffer
            .copy_from_host(
                0,
                &encode_f32_to_bytes(&input_normed_values),
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy {label} self-attn resident batch input normed: {err}")
            })?;

        let component_started = Instant::now();
        weights.q_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_q_projected_buffer,
            stream,
            "self-attn resident batch q projection",
        )?;
        weights.k_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_k_projected_buffer,
            stream,
            "self-attn resident batch k projection",
        )?;
        weights.v_matrix.matvec_batch(
            &self.batch_input_normed_buffer,
            batch_count,
            &mut self.batch_v_projected_buffer,
            stream,
            "self-attn resident batch v projection",
        )?;
        let qkv_projection_ms = if sync_component_timing {
            stream
                .synchronize()
                .map_err(|err| {
                    format!(
                        "failed to synchronize {label} self-attn batch q/k/v projection: {err}"
                    )
                })?;
            component_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        for step_ms in &mut component_step_ms {
            step_ms.qkv_projection_ms = qkv_projection_ms;
        }

        let q_projected_values = read_runtime_buffer_f32(
            &self.batch_q_projected_buffer,
            stream,
            q_projected_elements,
            &format!("{label} self-attn resident batch q projected"),
        )?;
        let k_projected_values = read_runtime_buffer_f32(
            &self.batch_k_projected_buffer,
            stream,
            k_projected_elements,
            &format!("{label} self-attn resident batch k projected"),
        )?;
        let v_projected_values = read_runtime_buffer_f32(
            &self.batch_v_projected_buffer,
            stream,
            v_projected_elements,
            &format!("{label} self-attn resident batch v projected"),
        )?;

        for (batch_index, item) in items.iter().enumerate() {
            let &(request_id, residual_buffer, rope_position, cache_position) = item;
            let item_label = format!(
                "{label} request={} position={}",
                request_id.0, rope_position
            );
            let layer = self.layers.get_mut(slots[batch_index]).ok_or_else(|| {
                format!(
                    "{label} self-attn resident batch missing state slot {} for request {request_id:?}",
                    slots[batch_index]
                )
            })?;
            let q_start = batch_index.checked_mul(q_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch q projected offset overflows")
            })?;
            let k_start = batch_index.checked_mul(k_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch k projected offset overflows")
            })?;
            let v_start = batch_index.checked_mul(v_projected_rows).ok_or_else(|| {
                format!("{label} self-attn resident batch v projected offset overflows")
            })?;
            layer
                .q_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&q_projected_values[q_start..q_start + q_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident q projected: {err}")
                })?;
            layer
                .k_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&k_projected_values[k_start..k_start + k_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident k projected: {err}")
                })?;
            layer
                .v_projected_buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&v_projected_values[v_start..v_start + v_projected_rows]),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy {item_label} self-attn resident v projected: {err}")
                })?;
            layer.run_device_step_after_qkv_projection(
                stream,
                PackageSelfAttnResidentStepInput::ExternalBuffer(residual_buffer),
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                sync_component_timing,
                component_step_ms
                    .get_mut(batch_index)
                    .ok_or_else(|| {
                        format!("{label} self-attn resident batch component timing is missing")
                    })?,
                &item_label,
            )?;
            if sync_component_timing {
                layer.last_component_step_ms = Some(component_step_ms[batch_index]);
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual: &[f32],
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        self.layer_for_request_mut(request_id)?
            .step_from_host_to_device(
                stream,
                residual,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                label,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        self.layer_for_request_mut(request_id)?
            .step_from_device_to_device(
                stream,
                residual_buffer,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                label,
            )
    }

    fn output_buffer(
        &self,
        request_id: RequestId,
    ) -> Result<&ullm_runtime_sys::RuntimeBuffer, String> {
        Ok(self.layer_for_request(request_id)?.output_buffer())
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
    ) -> Result<Vec<f32>, String> {
        self.layer_for_request(request_id)?.read_output(stream)
    }

    fn take_last_component_step_ms(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<PackageSelfAttnComponentStepMs>, String> {
        Ok(self
            .layer_for_request_mut(request_id)?
            .take_last_component_step_ms())
    }
}

#[allow(dead_code)]
struct PackageLinearAttnResidentStepBatchLayer {
    layer_index: usize,
    hidden: usize,
    request_index: std::collections::BTreeMap<RequestId, usize>,
    request_ids: Vec<RequestId>,
    layers: Vec<PackageLinearAttnResidentStepLayer>,
}

#[allow(dead_code)]
impl PackageLinearAttnResidentStepBatchLayer {
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_index: usize,
        request_ids: Vec<RequestId>,
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        let request_index =
            package_linear_attn_request_slot_index(&request_ids, "linear-attn resident batch")?;
        let mut layers = Vec::with_capacity(request_ids.len());
        let mut hidden = None;
        let mut registry = WeightRegistry::new();
        let mut shared_buffers = PackageResidentSharedBufferRegistry::new();
        let mut shared_weights: Option<std::sync::Arc<PackageLinearAttnResidentStepWeights>> = None;
        for request_id in &request_ids {
            let layer = if let Some(weights) = shared_weights.clone() {
                PackageLinearAttnResidentStepLayer::load_state_with_weights(context, stream, weights)
            } else {
                let layer = PackageLinearAttnResidentStepLayer::load_with_registry(
                    context,
                    stream,
                    &mut registry,
                    Some(&mut shared_buffers),
                    path,
                    chunk_bytes,
                    layer_index,
                    sq_overlay,
                )?;
                shared_weights = Some(layer.weights.clone());
                Ok(layer)
            }
            .map_err(|err| {
                format!(
                    "failed to load linear-attn resident batch layer {layer_index} for request {request_id:?}: {err}"
                )
            })?;
            if let Some(previous) = hidden {
                if previous != layer.hidden {
                    return Err(format!(
                        "linear-attn resident batch layer {layer_index} hidden changed: previous={previous} current={}",
                        layer.hidden
                    ));
                }
            } else {
                hidden = Some(layer.hidden);
            }
            layers.push(layer);
        }
        let hidden = hidden.ok_or_else(|| {
            format!("linear-attn resident batch layer {layer_index} has no request slots")
        })?;
        Ok(Self {
            layer_index,
            hidden,
            request_index,
            request_ids,
            layers,
        })
    }

    fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    fn request_count(&self) -> usize {
        self.request_ids.len()
    }

    fn layer_index(&self) -> usize {
        self.layer_index
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn request_slot(&self, request_id: RequestId) -> Result<usize, String> {
        self.request_index.get(&request_id).copied().ok_or_else(|| {
            format!(
                "linear-attn resident batch layer {} has no state slot for request {request_id:?}",
                self.layer_index
            )
        })
    }

    fn layer_for_request_mut(
        &mut self,
        request_id: RequestId,
    ) -> Result<&mut PackageLinearAttnResidentStepLayer, String> {
        let slot = self.request_slot(request_id)?;
        self.layers.get_mut(slot).ok_or_else(|| {
            format!(
                "linear-attn resident batch layer {} missing state slot {slot} for request {request_id:?}",
                self.layer_index
            )
        })
    }

    fn layer_for_request(
        &self,
        request_id: RequestId,
    ) -> Result<&PackageLinearAttnResidentStepLayer, String> {
        let slot = self.request_slot(request_id)?;
        self.layers.get(slot).ok_or_else(|| {
            format!(
                "linear-attn resident batch layer {} missing state slot {slot} for request {request_id:?}",
                self.layer_index
            )
        })
    }

    fn step_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual: &[f32],
        label: &str,
    ) -> Result<(), String> {
        self.layer_for_request_mut(request_id)?
            .step_from_host_to_device(stream, residual, label)
    }

    fn step_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        self.layer_for_request_mut(request_id)?
            .step_from_device_to_device(stream, residual_buffer, label)
    }

    fn output_buffer(
        &self,
        request_id: RequestId,
    ) -> Result<&ullm_runtime_sys::RuntimeBuffer, String> {
        Ok(self.layer_for_request(request_id)?.output_buffer())
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
    ) -> Result<Vec<f32>, String> {
        self.layer_for_request(request_id)?.read_output(stream)
    }

    fn take_last_component_step_ms(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<PackageLinearAttnComponentStepMs>, String> {
        Ok(self
            .layer_for_request_mut(request_id)?
            .take_last_component_step_ms())
    }
}

#[allow(dead_code)]
enum PackageMixedRequestStateLayer {
    LinearAttention(PackageLinearAttnResidentStepBatchLayer),
    SelfAttention(PackageSelfAttnResidentStepBatchLayer),
}

#[allow(dead_code)]
impl PackageMixedRequestStateLayer {
    fn kind(&self) -> &'static str {
        match self {
            Self::LinearAttention(_) => PackageDecoderLayerKind::LinearAttention.as_str(),
            Self::SelfAttention(_) => PackageDecoderLayerKind::SelfAttention.as_str(),
        }
    }

    fn layer_index(&self) -> usize {
        match self {
            Self::LinearAttention(layer) => layer.layer_index(),
            Self::SelfAttention(layer) => layer.layer_index(),
        }
    }

    fn hidden(&self) -> usize {
        match self {
            Self::LinearAttention(layer) => layer.hidden(),
            Self::SelfAttention(layer) => layer.hidden(),
        }
    }

    fn self_attn_head_dim(&self) -> Option<usize> {
        match self {
            Self::LinearAttention(_) => None,
            Self::SelfAttention(layer) => Some(layer.head_dim()),
        }
    }

    fn self_attn_shape_json(&self) -> Option<serde_json::Value> {
        match self {
            Self::LinearAttention(_) => None,
            Self::SelfAttention(layer) => Some(serde_json::json!({
                "layer_index": layer.layer_index(),
                "q_heads": layer.q_heads(),
                "kv_heads": layer.kv_heads(),
                "head_dim": layer.head_dim(),
                "value_dim": layer.value_dim(),
                "block_size": layer.block_size(),
                "cache_blocks": layer.cache_blocks(),
            })),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual: &[f32],
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => {
                layer.step_from_host_to_device(stream, request_id, residual, label)
            }
            Self::SelfAttention(layer) => layer.step_from_host_to_device(
                stream,
                request_id,
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
    fn step_batch_from_host_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        items: &[MixedRequestStateBatchStepItem],
        rotary_dim: usize,
        rope_base: f32,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => {
                for item in items {
                    layer.step_from_host_to_device(
                        stream,
                        item.request_id,
                        &item.residual,
                        &format!(
                            "{label} request={} position={}",
                            item.request_id.0, item.rope_position
                        ),
                    )?;
                }
                Ok(())
            }
            Self::SelfAttention(layer) => layer.step_batch_from_host_to_device(
                stream, items, rotary_dim, rope_base, label,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_batch_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        items: &[(RequestId, &ullm_runtime_sys::RuntimeBuffer, usize, usize)],
        rotary_dim: usize,
        rope_base: f32,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => {
                for &(request_id, residual_buffer, rope_position, _cache_position) in items {
                    layer.step_from_device_to_device(
                        stream,
                        request_id,
                        residual_buffer,
                        &format!(
                            "{label} request={} position={}",
                            request_id.0, rope_position
                        ),
                    )?;
                }
                Ok(())
            }
            Self::SelfAttention(layer) => layer.step_batch_from_device_to_device(
                stream,
                items,
                rotary_dim,
                rope_base,
                label,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_from_device_to_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => {
                layer.step_from_device_to_device(stream, request_id, residual_buffer, label)
            }
            Self::SelfAttention(layer) => layer.step_from_device_to_device(
                stream,
                request_id,
                residual_buffer,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                label,
            ),
        }
    }

    fn output_buffer(
        &self,
        request_id: RequestId,
    ) -> Result<&ullm_runtime_sys::RuntimeBuffer, String> {
        match self {
            Self::LinearAttention(layer) => layer.output_buffer(request_id),
            Self::SelfAttention(layer) => layer.output_buffer(request_id),
        }
    }

    fn read_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_id: RequestId,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::LinearAttention(layer) => layer.read_output(stream, request_id),
            Self::SelfAttention(layer) => layer.read_output(stream, request_id),
        }
    }
}

fn package_linear_attn_request_slot_index(
    request_ids: &[RequestId],
    label: &str,
) -> Result<std::collections::BTreeMap<RequestId, usize>, String> {
    package_request_slot_index(request_ids, label)
}

fn package_self_attn_request_slot_index(
    request_ids: &[RequestId],
    label: &str,
) -> Result<std::collections::BTreeMap<RequestId, usize>, String> {
    package_request_slot_index(request_ids, label)
}

fn package_request_slot_index(
    request_ids: &[RequestId],
    label: &str,
) -> Result<std::collections::BTreeMap<RequestId, usize>, String> {
    if request_ids.is_empty() {
        return Err(format!("{label} requires at least one request id"));
    }
    let mut index = std::collections::BTreeMap::new();
    for (slot, &request_id) in request_ids.iter().enumerate() {
        if index.insert(request_id, slot).is_some() {
            return Err(format!("{label} has duplicate request id {request_id:?}"));
        }
    }
    Ok(index)
}

fn runtime_host_linear_attn_gate_beta_f32(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    heads: usize,
    sequence_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let elements = match heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return (Vec::new(), Vec::new()),
    };
    if a.len() != elements || b.len() != elements || a_log.len() != heads || dt_bias.len() != heads
    {
        return (Vec::new(), Vec::new());
    }

    let mut gate = vec![0.0_f32; elements];
    let mut beta = vec![0.0_f32; elements];
    for index in 0..elements {
        let head = index % heads;
        let x = a[index] + dt_bias[head];
        let softplus = if x <= 20.0_f32 {
            (1.0_f32 + x.exp()).ln()
        } else {
            x
        };
        gate[index] = -a_log[head].exp() * softplus;
        beta[index] = 1.0_f32 / (1.0_f32 + (-b[index]).exp());
    }
    (gate, beta)
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_linear_attn_recurrent_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    key_heads: usize,
    value_heads: usize,
    sequence_len: usize,
    key_dim: usize,
    value_dim: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let key_head_sequence_elements = match key_heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    let value_head_sequence_elements = match value_heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    let qk_elements = match key_head_sequence_elements.checked_mul(key_dim) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let v_elements = match value_head_sequence_elements.checked_mul(value_dim) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => return Vec::new(),
    };
    if key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || key_dim == 0
        || value_dim == 0
        || q.len() != qk_elements
        || k.len() != qk_elements
        || v.len() != v_elements
        || gate.len() != value_head_sequence_elements
        || beta.len() != value_head_sequence_elements
        || state.len() != state_elements
    {
        return Vec::new();
    }

    let mut output = vec![0.0_f32; v_elements];
    let key_head_group = value_heads / key_heads;
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let key_head = value_head / key_head_group;
            let value_head_index = timestep * value_heads + value_head;
            let key_head_index = timestep * key_heads + key_head;
            let qk_base = key_head_index * key_dim;
            let v_base = value_head_index * value_dim;
            let state_head_offset = value_head * key_dim * value_dim;
            let decay = gate[value_head_index].exp();
            let beta_value = beta[value_head_index];

            for key in 0..key_dim {
                let state_key_offset = state_head_offset + key * value_dim;
                for value in 0..value_dim {
                    state[state_key_offset + value] *= decay;
                }
            }

            for value in 0..value_dim {
                let mut current = 0.0_f32;
                for key in 0..key_dim {
                    current +=
                        state[state_head_offset + key * value_dim + value] * k[qk_base + key];
                }
                let v_prime = (v[v_base + value] - current) * beta_value;
                for key in 0..key_dim {
                    state[state_head_offset + key * value_dim + value] +=
                        k[qk_base + key] * v_prime;
                }
            }

            for value in 0..value_dim {
                let mut sum = 0.0_f32;
                for key in 0..key_dim {
                    sum += state[state_head_offset + key * value_dim + value] * q[qk_base + key];
                }
                output[v_base + value] = sum;
            }
        }
    }
    output
}

#[cfg(test)]
mod linear_attn_step_state_tests {
    use super::*;

    #[test]
    fn linear_attn_request_slot_index_rejects_empty_and_duplicate_ids() {
        assert!(package_linear_attn_request_slot_index(&[], "test batch").is_err());
        assert!(
            package_linear_attn_request_slot_index(
                &[RequestId(10), RequestId(11), RequestId(10)],
                "test batch"
            )
            .is_err()
        );
    }

    #[test]
    fn linear_attn_request_slot_index_preserves_request_order() {
        let index = package_linear_attn_request_slot_index(
            &[RequestId(42), RequestId(7), RequestId(99)],
            "test batch",
        )
        .unwrap();

        assert_eq!(index.get(&RequestId(42)), Some(&0));
        assert_eq!(index.get(&RequestId(7)), Some(&1));
        assert_eq!(index.get(&RequestId(99)), Some(&2));
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn self_attn_request_slot_index_rejects_empty_and_duplicate_ids() {
        assert!(package_self_attn_request_slot_index(&[], "test batch").is_err());
        assert!(
            package_self_attn_request_slot_index(
                &[RequestId(20), RequestId(21), RequestId(20)],
                "test batch"
            )
            .is_err()
        );
    }

    #[test]
    fn self_attn_request_slot_index_preserves_request_order() {
        let index = package_self_attn_request_slot_index(
            &[RequestId(142), RequestId(107), RequestId(199)],
            "test batch",
        )
        .unwrap();

        assert_eq!(index.get(&RequestId(142)), Some(&0));
        assert_eq!(index.get(&RequestId(107)), Some(&1));
        assert_eq!(index.get(&RequestId(199)), Some(&2));
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn linear_attn_conv1d_step_matches_full_causal_conv() {
        let channels = 5_usize;
        let sequence_len = 6_usize;
        let kernel_size = 3_usize;
        let input = (0..channels * sequence_len)
            .map(|index| ((index as f32) + 1.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..channels * kernel_size)
            .map(|index| ((index as f32) - 3.0) * 0.0625)
            .collect::<Vec<_>>();
        let expected =
            runtime_host_depthwise_conv1d_f32(&input, &weight, channels, sequence_len, kernel_size);

        let mut state = LinearAttnConv1dStepState::new(channels, kernel_size).unwrap();
        let mut stepped = Vec::with_capacity(input.len());
        for current in input.chunks_exact(channels) {
            stepped.extend(state.step(current, &weight).unwrap());
        }

        let diff = verify_f32_close(
            "linear attention conv1d step",
            &stepped,
            &expected,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert_eq!(diff, 0.0);
        assert_eq!(state.seen_tokens, sequence_len);
    }

    #[test]
    fn linear_attn_stateful_host_steps_match_full_recurrent() {
        let key_heads = 2_usize;
        let value_heads = 4_usize;
        let key_dim = 3_usize;
        let value_dim = 2_usize;
        let sequence_len = 5_usize;
        let kernel_size = 3_usize;
        let q_elements_per_step = key_heads * key_dim;
        let k_elements_per_step = key_heads * key_dim;
        let v_elements_per_step = value_heads * value_dim;
        let qkv_step_elements = q_elements_per_step + k_elements_per_step + v_elements_per_step;
        let q_scale = 1.0_f32 / (key_dim as f32).sqrt();

        let qkv_input = (0..sequence_len * qkv_step_elements)
            .map(|index| ((index % 17) as f32 + 1.0) * 0.03125)
            .collect::<Vec<_>>();
        let conv_weight = (0..qkv_step_elements * kernel_size)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.015625)
            .collect::<Vec<_>>();
        let a = (0..sequence_len * value_heads)
            .map(|index| ((index % 7) as f32 - 2.0) * 0.125)
            .collect::<Vec<_>>();
        let b = (0..sequence_len * value_heads)
            .map(|index| ((index % 5) as f32 - 1.0) * 0.09375)
            .collect::<Vec<_>>();
        let a_log = (0..value_heads)
            .map(|index| -2.0 + (index as f32) * 0.125)
            .collect::<Vec<_>>();
        let dt_bias = (0..value_heads)
            .map(|index| -0.25 + (index as f32) * 0.0625)
            .collect::<Vec<_>>();

        let full_conv = runtime_host_depthwise_conv1d_f32(
            &qkv_input,
            &conv_weight,
            qkv_step_elements,
            sequence_len,
            kernel_size,
        );
        let full_conv_activated = runtime_host_silu_f32(&full_conv);
        let full_qkv = split_linear_attn_qkv_for_recurrent(
            &full_conv_activated,
            sequence_len,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            true,
            q_scale,
        )
        .unwrap();
        let (full_gate, full_beta) = runtime_host_linear_attn_gate_beta_f32(
            &a,
            &b,
            &a_log,
            &dt_bias,
            value_heads,
            sequence_len,
        );
        let mut full_state = vec![0.0_f32; value_heads * key_dim * value_dim];
        let full_recurrent = runtime_host_linear_attn_recurrent_f32(
            &full_qkv.q,
            &full_qkv.k,
            &full_qkv.v,
            &full_gate,
            &full_beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut full_state,
        );

        let mut conv_state =
            LinearAttnConv1dStepState::new(qkv_step_elements, kernel_size).unwrap();
        let mut recurrent_state = vec![0.0_f32; value_heads * key_dim * value_dim];
        let mut stepped_recurrent = Vec::with_capacity(full_recurrent.len());
        for timestep in 0..sequence_len {
            let qkv_start = timestep * qkv_step_elements;
            let qkv_end = qkv_start + qkv_step_elements;
            let conv_step = conv_state
                .step(&qkv_input[qkv_start..qkv_end], &conv_weight)
                .unwrap();
            let conv_step_activated = runtime_host_silu_f32(&conv_step);
            let split = split_linear_attn_qkv_for_recurrent(
                &conv_step_activated,
                1,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                true,
                q_scale,
            )
            .unwrap();
            let gate_start = timestep * value_heads;
            let gate_end = gate_start + value_heads;
            let (gate_step, beta_step) = runtime_host_linear_attn_gate_beta_f32(
                &a[gate_start..gate_end],
                &b[gate_start..gate_end],
                &a_log,
                &dt_bias,
                value_heads,
                1,
            );
            let recurrent_step = runtime_host_linear_attn_recurrent_f32(
                &split.q,
                &split.k,
                &split.v,
                &gate_step,
                &beta_step,
                key_heads,
                value_heads,
                1,
                key_dim,
                value_dim,
                &mut recurrent_state,
            );
            stepped_recurrent.extend(recurrent_step);
        }

        verify_f32_close(
            "linear attention recurrent step output",
            &stepped_recurrent,
            &full_recurrent,
            1e-6,
            1e-6,
        )
        .unwrap();
        verify_f32_close(
            "linear attention recurrent step state",
            &recurrent_state,
            &full_state,
            1e-6,
            1e-6,
        )
        .unwrap();
    }
}

fn format_f32_preview(values: &[f32]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("{value:.7}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{joined}]")
}
