    fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f32s_to_bf16_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values) / 2);
        for value in values {
            let bits = value.to_bits();
            let bf16 = (bits >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }

    fn fp8_e4m3_to_f32_unscaled(value: u8) -> f32 {
        let sign = value >> 7;
        let exponent = (value >> 3) & 0x0f;
        let mantissa = value & 0x07;
        let magnitude = if exponent == 0 {
            f32::from(mantissa) * 0.001953125
        } else {
            (1.0 + f32::from(mantissa) * 0.125) * 2.0_f32.powi(i32::from(exponent) - 7)
        };
        if sign == 0 { magnitude } else { -magnitude }
    }

    fn fp8_e4m3_encode_scaled(value: f32, scale: f32) -> u8 {
        if value == 0.0 || !value.is_finite() {
            return 0;
        }
        let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
        let magnitude = (value.abs() / scale).min(240.0);
        if magnitude < 0.001953125 {
            return 0;
        }
        if magnitude < 0.015625 {
            let mantissa = (magnitude / 0.001953125).round().clamp(0.0, 7.0) as u8;
            if mantissa == 0 {
                return 0;
            }
            return sign | mantissa;
        }
        let mut exponent = magnitude.log2().floor() as i32;
        let mut mantissa = ((magnitude / 2.0_f32.powi(exponent) - 1.0) * 8.0).round() as i32;
        if mantissa == 8 {
            exponent += 1;
            mantissa = 0;
        }
        if exponent > 7 {
            return sign | 0x77;
        }
        let biased_exponent = (exponent + 7).clamp(1, 14) as u8;
        sign | (biased_exponent << 3) | (mantissa.clamp(0, 7) as u8)
    }

    fn fp8_e4m3_quantize(values: &[f32]) -> (Vec<u8>, f32, Vec<f32>) {
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 240.0 };
        let encoded = values
            .iter()
            .copied()
            .map(|value| fp8_e4m3_encode_scaled(value, scale))
            .collect::<Vec<_>>();
        let decoded = encoded
            .iter()
            .copied()
            .map(|value| fp8_e4m3_to_f32_unscaled(value) * scale)
            .collect::<Vec<_>>();
        (encoded, scale, decoded)
    }

    fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn le_bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn u32s_to_le_bytes(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn expected_rmsnorm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
        let sum_squares = input.iter().map(|value| value * value).sum::<f32>();
        let inv_rms = 1.0 / (sum_squares / input.len() as f32 + epsilon).sqrt();
        input
            .iter()
            .zip(weight)
            .map(|(input, weight)| input * inv_rms * weight)
            .collect()
    }

    fn expected_silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = *gate;
                let sigmoid = 1.0 / (1.0 + (-gate).exp());
                gate * sigmoid * *up
            })
            .collect()
    }

    fn expected_sigmoid_mul(gate: &[f32], input: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(input)
            .map(|(gate, input)| {
                let sigmoid = 1.0 / (1.0 + (-*gate).exp());
                sigmoid * *input
            })
            .collect()
    }

    fn expected_rope(
        input: &[f32],
        sequence_len: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
    ) -> Vec<f32> {
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
    fn expected_causal_attn(
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

    fn expected_decode_attn(
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
        let mut output = vec![0.0_f32; q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = q_head * head_dim;
            let mut scores = Vec::with_capacity(cache_len);
            for cache_t in 0..cache_len {
                let k_base = (cache_t * kv_heads + kv_head) * head_dim;
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
                for (cache_t, weight) in weights.iter().enumerate() {
                    let v_index = (cache_t * kv_heads + kv_head) * value_dim + value;
                    weighted += *weight * v_cache[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_cached_prefix_attn(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; new_tokens * q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for token_index in 0..new_tokens {
            let cache_len = cached_prefix_len + token_index + 1;
            for q_head in 0..q_heads {
                let kv_head = q_head / q_per_kv;
                let q_base = (token_index * q_heads + q_head) * head_dim;
                let mut scores = Vec::with_capacity(cache_len);
                for cache_t in 0..cache_len {
                    let k_base = (cache_t * kv_heads + kv_head) * head_dim;
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
                let output_base = (token_index * q_heads + q_head) * value_dim;
                for value in 0..value_dim {
                    let mut weighted = 0.0_f32;
                    for (cache_t, weight) in weights.iter().enumerate() {
                        let v_index = (cache_t * kv_heads + kv_head) * value_dim + value;
                        weighted += *weight * v_cache[v_index];
                    }
                    output[output_base + value] = weighted / denominator;
                }
            }
        }
        output
    }

    fn expected_paged_decode_attn(
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
        let mut output = vec![0.0_f32; q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = q_head * head_dim;
            let mut scores = Vec::with_capacity(cache_len);
            for cache_t in 0..cache_len {
                let block = block_table[cache_t / block_size] as usize;
                let offset = cache_t % block_size;
                let k_base = ((block * block_size + offset) * kv_heads + kv_head) * head_dim;
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
                for (cache_t, weight) in weights.iter().enumerate() {
                    let block = block_table[cache_t / block_size] as usize;
                    let offset = cache_t % block_size;
                    let v_base = ((block * block_size + offset) * kv_heads + kv_head) * value_dim;
                    weighted += *weight * v_cache[v_base + value];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
        output
    }

    fn expected_depthwise_conv1d(
        input: &[f32],
        weight: &[f32],
        channels: usize,
        sequence_len: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; channels * sequence_len];
        for t in 0..sequence_len {
            for c in 0..channels {
                let mut value = 0.0_f32;
                for k in 0..kernel_size {
                    let left_padding = kernel_size - 1 - k;
                    if t >= left_padding {
                        value +=
                            input[(t - left_padding) * channels + c] * weight[c * kernel_size + k];
                    }
                }
                output[t * channels + c] = value;
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_linear_attn_qkv_prepare(
        qkv: &[f32],
        conv_weight: &[f32],
        conv_history: &mut [f32],
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        q_scale: f32,
        qk_l2_norm: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_elements = key_heads * key_dim;
        let v_elements = value_heads * value_dim;
        let channels = q_elements * 2 + v_elements;
        assert_eq!(qkv.len(), channels);
        assert_eq!(conv_weight.len(), channels * kernel_size);
        assert_eq!(conv_history.len(), channels * kernel_size);

        let mut conv_output = vec![0.0_f32; channels];
        for channel in 0..channels {
            for kernel in 0..kernel_size - 1 {
                conv_history[kernel * channels + channel] =
                    conv_history[(kernel + 1) * channels + channel];
            }
            conv_history[(kernel_size - 1) * channels + channel] = qkv[channel];
            let mut sum = 0.0_f32;
            for kernel in 0..kernel_size {
                sum += conv_history[kernel * channels + channel]
                    * conv_weight[channel * kernel_size + kernel];
            }
            let sigmoid = 1.0 / (1.0 + (-sum).exp());
            conv_output[channel] = sum * sigmoid;
        }

        let mut q = vec![0.0_f32; q_elements];
        let mut k = vec![0.0_f32; q_elements];
        let mut v = vec![0.0_f32; v_elements];
        for head in 0..key_heads {
            let q_base = head * key_dim;
            let k_base = q_elements + head * key_dim;
            let target = head * key_dim;
            let q_norm = (conv_output[q_base..q_base + key_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            let k_norm = (conv_output[k_base..k_base + key_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            for dim in 0..key_dim {
                let q_value = conv_output[q_base + dim];
                let k_value = conv_output[k_base + dim];
                q[target + dim] = if qk_l2_norm {
                    q_value / q_norm * q_scale
                } else {
                    q_value * q_scale
                };
                k[target + dim] = if qk_l2_norm {
                    k_value / k_norm
                } else {
                    k_value
                };
            }
        }
        let v_base = q_elements * 2;
        v.copy_from_slice(&conv_output[v_base..v_base + v_elements]);
        (conv_output, q, k, v)
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_linear_attn_qkv_prepare_batch(
        qkv: &[f32],
        conv_weight: &[f32],
        conv_history: &mut [f32],
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        sequence_len: usize,
        q_scale: f32,
        qk_l2_norm: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_elements = key_heads * key_dim;
        let v_elements = value_heads * value_dim;
        let channels = q_elements * 2 + v_elements;
        assert_eq!(qkv.len(), channels * sequence_len);
        let mut conv = Vec::with_capacity(channels * sequence_len);
        let mut q = Vec::with_capacity(q_elements * sequence_len);
        let mut k = Vec::with_capacity(q_elements * sequence_len);
        let mut v = Vec::with_capacity(v_elements * sequence_len);
        for token in 0..sequence_len {
            let token_base = token * channels;
            let (token_conv, token_q, token_k, token_v) = expected_linear_attn_qkv_prepare(
                &qkv[token_base..token_base + channels],
                conv_weight,
                conv_history,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                q_scale,
                qk_l2_norm,
            );
            conv.extend(token_conv);
            q.extend(token_q);
            k.extend(token_k);
            v.extend(token_v);
        }
        (conv, q, k, v)
    }

    fn assert_linear_attn_qkv_prepare_batch_matches_expected(
        context: &mut RuntimeContext,
        tolerance: f32,
    ) {
        let mut stream = context.create_stream().unwrap();
        let key_heads = 2;
        let value_heads = 2;
        let key_dim = 2;
        let value_dim = 2;
        let kernel_size = 3;
        let sequence_len = 4;
        let q_scale = 0.5;
        let channels = key_heads * key_dim * 2 + value_heads * value_dim;
        let qkv_values: Vec<f32> = (0..sequence_len * channels)
            .map(|index| index as f32 * 0.07 - 0.35)
            .collect();
        let conv_weight_values: Vec<f32> = (0..channels)
            .flat_map(|channel| {
                let offset = channel as f32 * 0.001;
                [0.25_f32 + offset, -0.5 + offset, 1.0 - offset]
            })
            .collect();
        let history_values: Vec<f32> = (0..channels * kernel_size)
            .map(|index| index as f32 * 0.03 - 0.4)
            .collect();
        let mut expected_history = history_values.clone();
        let (expected_conv, expected_q, expected_k, expected_v) =
            expected_linear_attn_qkv_prepare_batch(
                &qkv_values,
                &conv_weight_values,
                &mut expected_history,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                sequence_len,
                q_scale,
                true,
            );

        let mut qkv = context
            .alloc_buffer(qkv_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_weight = context
            .alloc_buffer(conv_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_history = context
            .alloc_buffer(history_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_output = context
            .alloc_buffer(expected_conv.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_output = context
            .alloc_buffer(expected_q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_output = context
            .alloc_buffer(expected_k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_output = context
            .alloc_buffer(expected_v.len() * std::mem::size_of::<f32>())
            .unwrap();

        qkv.copy_from_host(0, &f32s_to_le_bytes(&qkv_values), Some(&mut stream))
            .unwrap();
        conv_weight
            .copy_from_host(0, &f32s_to_le_bytes(&conv_weight_values), Some(&mut stream))
            .unwrap();
        conv_history
            .copy_from_host(0, &f32s_to_le_bytes(&history_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_qkv_prepare_batch_f32(
            &qkv,
            &conv_weight,
            &mut conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            true,
            &mut conv_output,
            &mut q_output,
            &mut k_output,
            &mut v_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut conv_bytes = vec![0_u8; expected_conv.len() * std::mem::size_of::<f32>()];
        let mut q_bytes = vec![0_u8; expected_q.len() * std::mem::size_of::<f32>()];
        let mut k_bytes = vec![0_u8; expected_k.len() * std::mem::size_of::<f32>()];
        let mut v_bytes = vec![0_u8; expected_v.len() * std::mem::size_of::<f32>()];
        let mut history_bytes = vec![0_u8; expected_history.len() * std::mem::size_of::<f32>()];
        conv_output
            .copy_to_host(0, &mut conv_bytes, Some(&mut stream))
            .unwrap();
        q_output
            .copy_to_host(0, &mut q_bytes, Some(&mut stream))
            .unwrap();
        k_output
            .copy_to_host(0, &mut k_bytes, Some(&mut stream))
            .unwrap();
        v_output
            .copy_to_host(0, &mut v_bytes, Some(&mut stream))
            .unwrap();
        conv_history
            .copy_to_host(0, &mut history_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        assert_f32s_close(&le_bytes_to_f32s(&conv_bytes), &expected_conv, tolerance);
        assert_f32s_close(&le_bytes_to_f32s(&q_bytes), &expected_q, tolerance);
        assert_f32s_close(&le_bytes_to_f32s(&k_bytes), &expected_k, tolerance);
        assert_f32s_close(&le_bytes_to_f32s(&v_bytes), &expected_v, tolerance);
        assert_f32s_close(
            &le_bytes_to_f32s(&history_bytes),
            &expected_history,
            tolerance,
        );
    }

    fn expected_linear_attn_gate_beta(
        a: &[f32],
        b: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        heads: usize,
        sequence_len: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut gate = Vec::with_capacity(heads * sequence_len);
        let mut beta = Vec::with_capacity(heads * sequence_len);
        for t in 0..sequence_len {
            for h in 0..heads {
                let index = t * heads + h;
                let x = a[index] + dt_bias[h];
                let softplus = if x <= 20.0 { (1.0 + x.exp()).ln() } else { x };
                gate.push(-a_log[h].exp() * softplus);
                beta.push(1.0 / (1.0 + (-b[index]).exp()));
            }
        }
        (gate, beta)
    }

    fn expected_linear_attn_recurrent_f32(
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
        initial_state: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial_state.to_vec();
        let mut output = vec![0.0_f32; sequence_len * value_heads * value_dim];
        let state_row_size = key_dim * value_dim;
        let key_head_group = value_heads / key_heads;
        for t in 0..sequence_len {
            for value_head in 0..value_heads {
                let key_head = value_head / key_head_group;
                let gate_index = t * value_heads + value_head;
                let factor = gate[gate_index].exp();
                for key in 0..key_dim {
                    for value in 0..value_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        state[state_index] *= factor;
                    }
                }
                for value in 0..value_dim {
                    let mut current = 0.0_f32;
                    for key in 0..key_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        let k_index = (t * key_heads + key_head) * key_dim + key;
                        current += state[state_index] * k[k_index];
                    }
                    let v_index = (t * value_heads + value_head) * value_dim + value;
                    let v_prime = (v[v_index] - current) * beta[gate_index];
                    for key in 0..key_dim {
                        let k_index = (t * key_heads + key_head) * key_dim + key;
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        state[state_index] += k[k_index] * v_prime;
                    }
                }
                for value in 0..value_dim {
                    let mut value_output = 0.0_f32;
                    for key in 0..key_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        let q_index = (t * key_heads + key_head) * key_dim + key;
                        value_output += state[state_index] * q[q_index];
                    }
                    let output_index = (t * value_heads + value_head) * value_dim + value;
                    output[output_index] = value_output;
                }
            }
        }
        (output, state)
    }

    fn assert_f32s_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected}"
            );
        }
    }
