    #[test]
    fn first_hip_paged_decode_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 5_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_cache_values = (0..cache_blocks * block_size * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_cache_values = (0..cache_blocks * block_size * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let block_table_values = vec![2_u32, 0_u32, 3_u32];
        let expected = expected_paged_decode_attn(
            &q_values,
            &k_cache_values,
            &v_cache_values,
            &block_table_values,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(k_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(v_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k_cache
            .copy_from_host(0, &f32s_to_le_bytes(&k_cache_values), Some(&mut stream))
            .unwrap();
        v_cache
            .copy_from_host(0, &f32s_to_le_bytes(&v_cache_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn hip_paged_decode_attn_f32_wave_softmax_long_context_matches_cpu_when_available() {
        let hip_devices: Vec<u32> = (1..device_count().unwrap())
            .filter(|&device_index| {
                device_info(device_index)
                    .map(|info| info.backend == "hip")
                    .unwrap_or(false)
            })
            .collect();
        if hip_devices.is_empty() {
            return;
        }

        let cache_len = 513_usize;
        let block_size = 7_usize;
        let cache_blocks = 80_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 128_usize;
        let value_dim = 256_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let physical_tokens = cache_blocks * block_size;
        let block_table_entries = (cache_len - 1) / block_size + 1;
        let q_values = (0..q_heads * head_dim)
            .map(|index| {
                if index % 97 == 0 {
                    8.0_f32
                } else if index % 89 == 0 {
                    -8.0_f32
                } else {
                    ((index % 29) as f32 - 14.0) * 0.03125
                }
            })
            .collect::<Vec<_>>();
        let k_cache_values = (0..physical_tokens * kv_heads * head_dim)
            .map(|index| {
                if index % 7919 == 0 {
                    6.0_f32
                } else if index % 7919 == 1 {
                    -6.0_f32
                } else {
                    ((index % 37) as f32 - 18.0) * 0.015625
                }
            })
            .collect::<Vec<_>>();
        let v_cache_values = (0..physical_tokens * kv_heads * value_dim)
            .map(|index| {
                if index % 12289 == 0 {
                    64.0_f32
                } else if index % 12289 == 1 {
                    -64.0_f32
                } else {
                    ((index % 41) as f32 - 20.0) * 0.0625
                }
            })
            .collect::<Vec<_>>();
        let gate_values = (0..q_heads * value_dim)
            .map(|index| {
                if index % 127 == 0 {
                    80.0_f32
                } else if index % 131 == 0 {
                    -80.0_f32
                } else {
                    ((index % 17) as f32 - 8.0) * 0.25
                }
            })
            .collect::<Vec<_>>();
        let block_table_values = (0..block_table_entries)
            .map(|index| ((index * 37 + 11) % cache_blocks) as u32)
            .collect::<Vec<_>>();
        let expected_plain = expected_paged_decode_attn(
            &q_values,
            &k_cache_values,
            &v_cache_values,
            &block_table_values,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let expected_gated = expected_sigmoid_mul(&gate_values, &expected_plain);
        let q_bytes = f32s_to_le_bytes(&q_values);
        let k_cache_bytes = f32s_to_le_bytes(&k_cache_values);
        let v_cache_bytes = f32s_to_le_bytes(&v_cache_values);
        let gate_bytes = f32s_to_le_bytes(&gate_values);
        let block_table_bytes = u32s_to_le_bytes(&block_table_values);

        for device_index in hip_devices {
            let mut context = RuntimeContext::create(device_index).unwrap();
            let mut stream = context.create_stream().unwrap();
            let mut q = context.alloc_buffer(q_bytes.len()).unwrap();
            let mut gate = context.alloc_buffer(gate_bytes.len()).unwrap();
            let mut k_cache = context.alloc_buffer(k_cache_bytes.len()).unwrap();
            let mut v_cache = context.alloc_buffer(v_cache_bytes.len()).unwrap();
            let mut block_table = context.alloc_buffer(block_table_bytes.len()).unwrap();
            let mut plain_output = context
                .alloc_buffer(expected_plain.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut gated_output = context
                .alloc_buffer(expected_gated.len() * std::mem::size_of::<f32>())
                .unwrap();

            q.copy_from_host(0, &q_bytes, Some(&mut stream)).unwrap();
            gate.copy_from_host(0, &gate_bytes, Some(&mut stream)).unwrap();
            k_cache
                .copy_from_host(0, &k_cache_bytes, Some(&mut stream))
                .unwrap();
            v_cache
                .copy_from_host(0, &v_cache_bytes, Some(&mut stream))
                .unwrap();
            block_table
                .copy_from_host(0, &block_table_bytes, Some(&mut stream))
                .unwrap();
            stream.synchronize().unwrap();

            paged_decode_attn_f32(
                &q,
                &k_cache,
                &v_cache,
                &block_table,
                cache_len,
                block_size,
                cache_blocks,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                &mut plain_output,
                Some(&mut stream),
            )
            .unwrap();
            paged_decode_attn_sigmoid_gate_f32(
                &q,
                &gate,
                &k_cache,
                &v_cache,
                &block_table,
                cache_len,
                block_size,
                cache_blocks,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                &mut gated_output,
                Some(&mut stream),
            )
            .unwrap();
            stream.synchronize().unwrap();

            let mut plain_bytes = vec![0_u8; expected_plain.len() * std::mem::size_of::<f32>()];
            let mut gated_bytes = vec![0_u8; expected_gated.len() * std::mem::size_of::<f32>()];
            plain_output
                .copy_to_host(0, &mut plain_bytes, Some(&mut stream))
                .unwrap();
            gated_output
                .copy_to_host(0, &mut gated_bytes, Some(&mut stream))
                .unwrap();
            stream.synchronize().unwrap();

            let plain = le_bytes_to_f32s(&plain_bytes);
            let gated = le_bytes_to_f32s(&gated_bytes);
            assert!(plain.iter().all(|value| value.is_finite()));
            assert!(gated.iter().all(|value| value.is_finite()));
            assert_f32s_close(&plain, &expected_plain, 1e-3);
            assert_f32s_close(&gated, &expected_gated, 1e-3);
        }
    }

    #[test]
    fn hip_paged_decode_attn_split_f32_context_matrix_matches_cpu_when_available() {
        let hip_devices: Vec<u32> = (1..device_count().unwrap())
            .filter(|&device_index| {
                device_info(device_index)
                    .map(|info| info.backend == "hip")
                    .unwrap_or(false)
            })
            .collect();
        for device_index in hip_devices {
            let mut context = RuntimeContext::create(device_index).unwrap();
            for cache_len in [1_usize, 255, 256, 257, 513, 1339] {
                for source_tile in [128_usize, 256] {
                    assert_paged_decode_split_matches_expected(
                        &mut context,
                        cache_len,
                        source_tile,
                    );
                }
            }
        }
    }

    #[test]
    fn first_hip_paged_decode_attn_split_f32_invalid_table_zeros_output_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 4_usize;
        let value_dim = 4_usize;
        let workspace_bytes =
            paged_decode_attn_split_workspace_bytes(q_heads, value_dim, 1, 1).unwrap();
        let mut q = context.alloc_buffer(q_heads * head_dim * 4).unwrap();
        let mut k_cache = context.alloc_buffer(kv_heads * head_dim * 4).unwrap();
        let mut v_cache = context.alloc_buffer(kv_heads * value_dim * 4).unwrap();
        let mut block_table = context.alloc_buffer(4).unwrap();
        let mut workspace = context.alloc_buffer(workspace_bytes).unwrap();
        let mut output = context.alloc_buffer(q_heads * value_dim * 4).unwrap();
        q.copy_from_host(
            0,
            &f32s_to_le_bytes(&vec![0.25; q_heads * head_dim]),
            Some(&mut stream),
        )
        .unwrap();
        k_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.5; kv_heads * head_dim]),
                Some(&mut stream),
            )
            .unwrap();
        v_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.75; kv_heads * value_dim]),
                Some(&mut stream),
            )
            .unwrap();
        block_table
            .copy_from_host(0, &u32s_to_le_bytes(&[u32::MAX]), Some(&mut stream))
            .unwrap();
        output
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![42.0; q_heads * value_dim]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();
        paged_decode_attn_split_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            1,
            1,
            1,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            0.5,
            1,
            &mut workspace,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut output_bytes = vec![0_u8; q_heads * value_dim * 4];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![0.0; q_heads * value_dim]
        );
    }

    #[test]
    fn first_hip_paged_decode_attn_split_f32_rejects_cpu_workspace_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut hip = RuntimeContext::create(1).unwrap();
        let mut cpu = RuntimeContext::create(0).unwrap();
        let q = hip.alloc_buffer(4).unwrap();
        let k_cache = hip.alloc_buffer(4).unwrap();
        let v_cache = hip.alloc_buffer(4).unwrap();
        let block_table = hip.alloc_buffer(4).unwrap();
        let mut workspace = cpu.alloc_buffer(12).unwrap();
        let mut output = hip.alloc_buffer(4).unwrap();
        let error = paged_decode_attn_split_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1.0,
            1,
            &mut workspace,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(error.contains("different backends") || error.contains("devices"), "{error}");
    }

    #[test]
    fn first_hip_paged_kv_write_f32_writes_expected_physical_slot_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_position = 3_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let k_values = vec![0.25_f32, -0.5, 1.25, 2.0, -3.0, 4.0];
        let v_values = vec![-0.75_f32, 0.5, 1.5, -2.5];
        let block_table_values = vec![2_u32, 0_u32];
        let physical_tokens = cache_blocks * block_size;
        let mut expected_k_cache = vec![0.0_f32; physical_tokens * kv_heads * head_dim];
        let mut expected_v_cache = vec![0.0_f32; physical_tokens * kv_heads * value_dim];
        let physical_timestep = block_table_values[cache_position / block_size] as usize
            * block_size
            + (cache_position % block_size);
        for kv_head in 0..kv_heads {
            let k_src = kv_head * head_dim;
            let k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            expected_k_cache[k_dst..k_dst + head_dim]
                .copy_from_slice(&k_values[k_src..k_src + head_dim]);

            let v_src = kv_head * value_dim;
            let v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            expected_v_cache[v_dst..v_dst + value_dim]
                .copy_from_slice(&v_values[v_src..v_src + value_dim]);
        }

        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(expected_k_cache.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(expected_v_cache.len() * std::mem::size_of::<f32>())
            .unwrap();

        k_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_k_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        v_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_v_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_kv_write_f32(
            &k,
            &v,
            &block_table,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            &mut k_cache,
            &mut v_cache,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut k_cache_bytes = vec![0_u8; expected_k_cache.len() * std::mem::size_of::<f32>()];
        let mut v_cache_bytes = vec![0_u8; expected_v_cache.len() * std::mem::size_of::<f32>()];
        k_cache
            .copy_to_host(0, &mut k_cache_bytes, Some(&mut stream))
            .unwrap();
        v_cache
            .copy_to_host(0, &mut v_cache_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_gate_beta_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let heads = 3_usize;
        let sequence_len = 4_usize;
        let a = [
            0.1_f32, -0.2, 1.2, 0.9, 0.8, -1.1, -0.7, 0.5, 1.4, -0.3, 0.2, -0.6,
        ];
        let b = [
            1.0_f32, -1.2, 0.3, -0.8, 0.6, 1.1, -0.5, 0.9, 0.0, -0.4, 1.3, -0.7,
        ];
        let a_log = [-1.0_f32, 0.25, -0.5];
        let dt_bias = [0.3_f32, -0.2, 0.4];
        let (expected_gate, expected_beta) =
            expected_linear_attn_gate_beta(&a, &b, &a_log, &dt_bias, heads, sequence_len);

        let mut a_buffer = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_buffer = context
            .alloc_buffer(b.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log_buffer = context
            .alloc_buffer(a_log.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();

        a_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a), Some(&mut stream))
            .unwrap();
        b_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&b), Some(&mut stream))
            .unwrap();
        a_log_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a_log), Some(&mut stream))
            .unwrap();
        dt_bias_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&dt_bias), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            heads,
            sequence_len,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_output_bytes = vec![0_u8; gate_output.size().unwrap()];
        gate_output
            .copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let mut beta_output_bytes = vec![0_u8; beta_output.size().unwrap()];
        beta_output
            .copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let gate_output_values = le_bytes_to_f32s(&gate_output_bytes);
        let beta_output_values = le_bytes_to_f32s(&beta_output_bytes);
        assert_f32s_close(&gate_output_values, &expected_gate, 1e-5);
        assert_f32s_close(&beta_output_values, &expected_beta, 1e-5);
    }

    struct LinearAttnRecurrentFixture {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        gate: Vec<f32>,
        beta: Vec<f32>,
        initial_state: Vec<f32>,
    }

    fn linear_attn_recurrent_fixture(
        key_heads: usize,
        value_heads: usize,
        sequence_len: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> LinearAttnRecurrentFixture {
        let qk_elements = sequence_len * key_heads * key_dim;
        let value_elements = sequence_len * value_heads * value_dim;
        let gate_beta_elements = sequence_len * value_heads;
        let state_elements = value_heads * key_dim * value_dim;
        LinearAttnRecurrentFixture {
            q: (0..qk_elements)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.003_906_25)
                .collect(),
            k: (0..qk_elements)
                .map(|index| ((index % 31) as f32 - 15.0) * 0.003_417_968_8)
                .collect(),
            v: (0..value_elements)
                .map(|index| ((index % 37) as f32 - 18.0) * 0.002_929_687_5)
                .collect(),
            gate: (0..gate_beta_elements)
                .map(|index| -0.125 - (index % 7) as f32 * 0.0625)
                .collect(),
            beta: (0..gate_beta_elements)
                .map(|index| 0.125 + (index % 5) as f32 * 0.0625)
                .collect(),
            initial_state: (0..state_elements)
                .map(|index| ((index % 41) as f32 - 20.0) * 0.001_953_125)
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_linear_attn_recurrent_fixture(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        fixture: &LinearAttnRecurrentFixture,
        key_heads: usize,
        value_heads: usize,
        sequence_len: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let f32_bytes = std::mem::size_of::<f32>();
        let mut q_buffer = context.alloc_buffer(fixture.q.len() * f32_bytes).unwrap();
        let mut k_buffer = context.alloc_buffer(fixture.k.len() * f32_bytes).unwrap();
        let mut v_buffer = context.alloc_buffer(fixture.v.len() * f32_bytes).unwrap();
        let mut gate_buffer = context.alloc_buffer(fixture.gate.len() * f32_bytes).unwrap();
        let mut beta_buffer = context.alloc_buffer(fixture.beta.len() * f32_bytes).unwrap();
        let mut state_buffer = context
            .alloc_buffer(fixture.initial_state.len() * f32_bytes)
            .unwrap();
        let mut output_buffer = context.alloc_buffer(fixture.v.len() * f32_bytes).unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.q), Some(&mut *stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.k), Some(&mut *stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.v), Some(&mut *stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.gate), Some(&mut *stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.beta), Some(&mut *stream))
            .unwrap();
        state_buffer
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&fixture.initial_state),
                Some(&mut *stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
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
            &mut output_buffer,
            Some(&mut *stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; fixture.v.len() * f32_bytes];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut *stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; fixture.initial_state.len() * f32_bytes];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut *stream))
            .unwrap();
        stream.synchronize().unwrap();
        (le_bytes_to_f32s(&output_bytes), le_bytes_to_f32s(&state_bytes))
    }

    #[test]
    fn first_hip_linear_attn_recurrent_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 3_usize;
        let key_dim = 2_usize;
        let value_dim = 2_usize;
        let q = [0.2_f32, 0.1, -0.4, 0.7, 0.8, 0.2];
        let k = [0.3_f32, 0.6, 0.7, -0.5, 0.4, 0.9];
        let v = [
            0.4_f32, -0.1, 0.6, 0.3, -0.2, 0.4, 0.1, -0.3, 0.5, 0.2, -0.4, 0.6,
        ];
        let gate = [0.05_f32, -0.1, 0.2, 0.15, -0.25, 0.3];
        let beta = [0.9_f32, 1.1, 0.7, 0.8, 0.6, 0.5];
        let initial_state = [0.1_f32, 0.2, 0.3, 0.4, -0.1, 0.0, 0.05, -0.05];
        let (expected_output, expected_state) = expected_linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &initial_state,
        );

        let mut q_buffer = context
            .alloc_buffer(q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_buffer = context
            .alloc_buffer(k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_buffer = context
            .alloc_buffer(gate.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_buffer = context
            .alloc_buffer(beta.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(initial_state.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
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
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; output_buffer.size().unwrap()];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; state_buffer.size().unwrap()];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected_output, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&state_bytes), &expected_state, 1e-5);
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1"]
    fn hip_linear_attn_recurrent_production_model_shapes_match_cpu_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1 before running this GPU differential test"
        );
        let device_index = (1..device_count().unwrap())
            .find(|&candidate| {
                device_info(candidate)
                    .map(|info| info.gcn_arch_name == "gfx1201")
                    .unwrap_or(false)
            })
            .expect("isolated gfx1201 HIP device");
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _block_size = ExperimentalEnvGuard::new("ULLM_LINEAR_ATTN_RECURRENT_BLOCK", Some("128"));
        let _require_hip =
            ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL", Some("1"));

        const KEY_DIM: usize = 128;
        const VALUE_DIM: usize = 128;
        let mut hip_context = RuntimeContext::create(device_index).unwrap();
        let mut hip_stream = hip_context.create_stream().unwrap();
        for &(case, key_heads, value_heads, sequence_len) in &[
            ("decode", 16_usize, 32_usize, 1_usize),
            ("short prefill", 16, 32, 7),
            ("M=128", 16, 32, 128),
        ] {
            let fixture = linear_attn_recurrent_fixture(
                key_heads,
                value_heads,
                sequence_len,
                KEY_DIM,
                VALUE_DIM,
            );
            let expected = {
                let mut cpu_context = RuntimeContext::create(0).unwrap();
                let mut cpu_stream = cpu_context.create_stream().unwrap();
                run_linear_attn_recurrent_fixture(
                    &mut cpu_context,
                    &mut cpu_stream,
                    &fixture,
                    key_heads,
                    value_heads,
                    sequence_len,
                    KEY_DIM,
                    VALUE_DIM,
                )
            };
            let actual = run_linear_attn_recurrent_fixture(
                &mut hip_context,
                &mut hip_stream,
                &fixture,
                key_heads,
                value_heads,
                sequence_len,
                KEY_DIM,
                VALUE_DIM,
            );
            assert_f32s_close(&actual.0, &expected.0, 1e-4);
            assert_f32s_close(&actual.1, &expected.1, 1e-4);
            eprintln!(
                "linear-attn recurrent production differential passed case={case} key_heads={key_heads} value_heads={value_heads} M={sequence_len} key_dim={KEY_DIM} value_dim={VALUE_DIM}"
            );
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1"]
    fn hip_linear_attn_recurrent_production_m2048_matches_generic_fallback_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1 before running this GPU differential test"
        );
        let device_index = (1..device_count().unwrap())
            .find(|&candidate| {
                device_info(candidate)
                    .map(|info| info.gcn_arch_name == "gfx1201")
                    .unwrap_or(false)
            })
            .expect("isolated gfx1201 HIP device");
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _require_hip =
            ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL", Some("1"));

        const KEY_HEADS: usize = 16;
        const VALUE_HEADS: usize = 32;
        const SEQUENCE_LEN: usize = 2_048;
        const KEY_DIM: usize = 128;
        const VALUE_DIM: usize = 128;
        const TOLERANCE: f32 = 1e-4;
        let fixture = linear_attn_recurrent_fixture(
            KEY_HEADS,
            VALUE_HEADS,
            SEQUENCE_LEN,
            KEY_DIM,
            VALUE_DIM,
        );
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let generic_reference = {
            // A 256-thread launch intentionally bypasses the exact 128-thread shuffle branch
            // while retaining the generic stable-ABI recurrence body as the reference.
            let _block_size =
                ExperimentalEnvGuard::new("ULLM_LINEAR_ATTN_RECURRENT_BLOCK", Some("256"));
            run_linear_attn_recurrent_fixture(
                &mut context,
                &mut stream,
                &fixture,
                KEY_HEADS,
                VALUE_HEADS,
                SEQUENCE_LEN,
                KEY_DIM,
                VALUE_DIM,
            )
        };
        let production = {
            let _block_size =
                ExperimentalEnvGuard::new("ULLM_LINEAR_ATTN_RECURRENT_BLOCK", Some("128"));
            run_linear_attn_recurrent_fixture(
                &mut context,
                &mut stream,
                &fixture,
                KEY_HEADS,
                VALUE_HEADS,
                SEQUENCE_LEN,
                KEY_DIM,
                VALUE_DIM,
            )
        };
        let output_max_abs_diff = production
            .0
            .iter()
            .zip(&generic_reference.0)
            .map(|(production, reference)| (production - reference).abs())
            .fold(0.0_f32, f32::max);
        let state_max_abs_diff = production
            .1
            .iter()
            .zip(&generic_reference.1)
            .map(|(production, reference)| (production - reference).abs())
            .fold(0.0_f32, f32::max);
        assert_f32s_close(&production.0, &generic_reference.0, TOLERANCE);
        assert_f32s_close(&production.1, &generic_reference.1, TOLERANCE);
        eprintln!(
            "linear-attn recurrent production differential passed case=M=2048 output_max_abs_diff={output_max_abs_diff:.9} state_max_abs_diff={state_max_abs_diff:.9} tolerance={TOLERANCE:.9}"
        );
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_TIMING=1"]
    fn hip_linear_attn_recurrent_production_m128_timing_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_TIMING").as_deref(),
            Ok("1"),
            "set ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_TIMING=1 before running this GPU timing test"
        );
        let device_index = (1..device_count().unwrap())
            .find(|&candidate| {
                device_info(candidate)
                    .map(|info| info.gcn_arch_name == "gfx1201")
                    .unwrap_or(false)
            })
            .expect("isolated gfx1201 HIP device");
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _block_size = ExperimentalEnvGuard::new("ULLM_LINEAR_ATTN_RECURRENT_BLOCK", Some("128"));
        let _require_hip =
            ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL", Some("1"));

        const KEY_HEADS: usize = 16;
        const VALUE_HEADS: usize = 32;
        const SEQUENCE_LEN: usize = 128;
        const KEY_DIM: usize = 128;
        const VALUE_DIM: usize = 128;
        const WARMUP_ITERATIONS: usize = 3;
        const TIMED_ITERATIONS: usize = 20;
        let fixture = linear_attn_recurrent_fixture(
            KEY_HEADS,
            VALUE_HEADS,
            SEQUENCE_LEN,
            KEY_DIM,
            VALUE_DIM,
        );
        let f32_bytes = std::mem::size_of::<f32>();
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut q_buffer = context.alloc_buffer(fixture.q.len() * f32_bytes).unwrap();
        let mut k_buffer = context.alloc_buffer(fixture.k.len() * f32_bytes).unwrap();
        let mut v_buffer = context.alloc_buffer(fixture.v.len() * f32_bytes).unwrap();
        let mut gate_buffer = context.alloc_buffer(fixture.gate.len() * f32_bytes).unwrap();
        let mut beta_buffer = context.alloc_buffer(fixture.beta.len() * f32_bytes).unwrap();
        let mut state_buffer = context
            .alloc_buffer(fixture.initial_state.len() * f32_bytes)
            .unwrap();
        let mut output_buffer = context.alloc_buffer(fixture.v.len() * f32_bytes).unwrap();
        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&fixture.initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        // Compile/load the HIPRTC kernel and reach steady state before measuring launches.
        for _ in 0..WARMUP_ITERATIONS {
            linear_attn_recurrent_f32(
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &gate_buffer,
                &beta_buffer,
                KEY_HEADS,
                VALUE_HEADS,
                SEQUENCE_LEN,
                KEY_DIM,
                VALUE_DIM,
                &mut state_buffer,
                &mut output_buffer,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let started = std::time::Instant::now();
        for _ in 0..TIMED_ITERATIONS {
            linear_attn_recurrent_f32(
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &gate_buffer,
                &beta_buffer,
                KEY_HEADS,
                VALUE_HEADS,
                SEQUENCE_LEN,
                KEY_DIM,
                VALUE_DIM,
                &mut state_buffer,
                &mut output_buffer,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let elapsed = started.elapsed();
        let milliseconds = elapsed.as_secs_f64() * 1_000.0 / TIMED_ITERATIONS as f64;
        assert!(milliseconds.is_finite() && milliseconds > 0.0);
        eprintln!(
            "linear-attn recurrent production timing key_heads={KEY_HEADS} value_heads={VALUE_HEADS} M={SEQUENCE_LEN} key_dim={KEY_DIM} value_dim={VALUE_DIM}: {milliseconds:.3} ms/launch ({:.3} launches/s)",
            1_000.0 / milliseconds
        );
    }

    #[test]
    fn first_hip_linear_attn_recurrent_f32_decode_step_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 1_usize;
        let key_dim = 4_usize;
        let value_dim = 3_usize;
        let q = [0.2_f32, -0.1, 0.4, 0.7];
        let k = [-0.3_f32, 0.6, 0.2, -0.5];
        let v = [0.4_f32, -0.1, 0.6, 0.3, -0.2, 0.4];
        let gate = [0.05_f32, -0.1];
        let beta = [0.9_f32, 1.1];
        let initial_state = [
            0.1_f32, 0.2, 0.3, 0.4, -0.1, 0.0, 0.05, -0.05, 0.2, 0.1, -0.2, 0.3, -0.3, 0.25, 0.15,
            -0.1, 0.05, 0.35, -0.15, 0.45, 0.2, -0.25, 0.1, -0.05,
        ];
        let (expected_output, expected_state) = expected_linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &initial_state,
        );

        let mut q_buffer = context
            .alloc_buffer(q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_buffer = context
            .alloc_buffer(k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_buffer = context
            .alloc_buffer(gate.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_buffer = context
            .alloc_buffer(beta.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(initial_state.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
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
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; output_buffer.size().unwrap()];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; state_buffer.size().unwrap()];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected_output, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&state_bytes), &expected_state, 1e-5);
    }

    #[test]
    fn first_hip_depthwise_conv1d_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let channels = 2_usize;
        let sequence_len = 6_usize;
        let kernel_size = 4_usize;
        let input_values = [
            0.5_f32, -1.0, 1.0, 2.0, -1.5, 0.75, -0.25, 3.5, 4.0, -2.0, 1.25, -0.5, 2.0, -3.0, 1.5,
            -0.75, 0.0, 0.5, 0.25, -1.25, 3.0, 1.0, -0.5, 2.5,
        ];
        let weight_values = [1.0_f32, 0.5, -1.0, 0.25, -0.5_f32, 1.0, -0.25, 2.0];

        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(sequence_len * channels * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        depthwise_conv1d_f32(
            &input,
            &weight,
            channels,
            sequence_len,
            kernel_size,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; sequence_len * channels * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected = expected_depthwise_conv1d(
            &input_values,
            &weight_values,
            channels,
            sequence_len,
            kernel_size,
        );
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_qkv_prepare_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 2;
        let value_heads = 2;
        let key_dim = 2;
        let value_dim = 2;
        let kernel_size = 3;
        let q_scale = 0.5;
        let channels = key_heads * key_dim * 2 + value_heads * value_dim;
        let qkv_values: Vec<f32> = (0..channels)
            .map(|index| index as f32 * 0.1 + 0.1)
            .collect();
        let conv_weight_values: Vec<f32> =
            (0..channels).flat_map(|_| [0.25_f32, 0.5, 1.0]).collect();
        let history_values = vec![0.0_f32; channels * kernel_size];
        let mut expected_history = history_values.clone();
        let (expected_conv, expected_q, expected_k, expected_v) = expected_linear_attn_qkv_prepare(
            &qkv_values,
            &conv_weight_values,
            &mut expected_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            true,
        );

        let mut qkv = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_weight = context
            .alloc_buffer(conv_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_history = context
            .alloc_buffer(history_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_output = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
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

        linear_attn_qkv_prepare_f32(
            &qkv,
            &conv_weight,
            &mut conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
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

        assert_f32s_close(&le_bytes_to_f32s(&conv_bytes), &expected_conv, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_bytes), &expected_q, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&k_bytes), &expected_k, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&v_bytes), &expected_v, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&history_bytes), &expected_history, 1e-6);
    }

    #[test]
    fn first_hip_linear_attn_qkv_prepare_batch_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        assert_linear_attn_qkv_prepare_batch_matches_expected(&mut context, 1e-5);
    }

    #[test]
    fn first_hip_context_allocates_runtime_buffer_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let info = context.device_info().unwrap();
        assert_eq!(info.backend, "hip");
        let buffer = context.alloc_buffer(4096).unwrap();
        assert_eq!(buffer.size().unwrap(), 4096);
    }

    #[test]
    fn first_hip_context_creates_stream_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn first_hip_buffer_roundtrips_host_data_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(4096).unwrap();
        let input: Vec<u8> = (0..4096).map(|value| (value * 31 + 7) as u8).collect();
        buffer.copy_from_host(0, &input, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; input.len()];
        buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn first_hip_wmma_fp8_probe_writes_nonzero_marker_when_available() {
        let rdna4_device = (1..device_count().unwrap()).find(|&index| {
            device_info(index)
                .map(|info| {
                    info.backend == "hip"
                        && (info.compute_major == 12 || info.gcn_arch_name.starts_with("gfx12"))
                })
                .unwrap_or(false)
        });
        let Some(device_index) = rdna4_device else {
            return;
        };
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(4).unwrap();
        wmma_fp8_probe(&mut buffer, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut raw = [0_u8; 4];
        buffer.copy_to_host(0, &mut raw, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let marker = u32::from_le_bytes(raw);
        assert_ne!(marker, 0_u32);
    }

    #[test]
    fn first_hip_wmma_fp8_qk_probe_outputs_finite_nonzero_values_when_available() {
        let rdna4_device = (1..device_count().unwrap()).find(|&index| {
            device_info(index)
                .map(|info| {
                    info.backend == "hip"
                        && (info.compute_major == 12 || info.gcn_arch_name.starts_with("gfx12"))
                })
                .unwrap_or(false)
        });
        let Some(device_index) = rdna4_device else {
            return;
        };
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let tile_bytes = 16_usize * 16_usize;
        let output_bytes = tile_bytes * std::mem::size_of::<f32>();
        let mut q_buffer = context.alloc_buffer(tile_bytes).unwrap();
        let mut k_buffer = context.alloc_buffer(tile_bytes).unwrap();
        let mut output_buffer = context.alloc_buffer(output_bytes).unwrap();

        let q_bytes = vec![0x38_u8; tile_bytes];
        let k_bytes = vec![0x38_u8; tile_bytes];

        q_buffer
            .copy_from_host(0, &q_bytes, Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &k_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        wmma_fp8_qk_probe(&q_buffer, &k_buffer, &mut output_buffer, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; output_bytes];
        output_buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let output = le_bytes_to_f32s(&output);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0_f32));
    }

    #[test]
    fn first_hip_rocwmma_fp8_qk_probe_outputs_finite_nonzero_values_when_available() {
        let rdna4_device = (1..device_count().unwrap()).find(|&index| {
            device_info(index)
                .map(|info| {
                    info.backend == "hip"
                        && (info.compute_major == 12 || info.gcn_arch_name.starts_with("gfx12"))
                })
                .unwrap_or(false)
        });
        let Some(device_index) = rdna4_device else {
            return;
        };
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let tile_bytes = 16_usize * 16_usize;
        let output_bytes = tile_bytes * std::mem::size_of::<f32>();
        let mut q_buffer = context.alloc_buffer(tile_bytes).unwrap();
        let mut k_buffer = context.alloc_buffer(tile_bytes).unwrap();
        let mut output_buffer = context.alloc_buffer(output_bytes).unwrap();

        let q_bytes = vec![0x38_u8; tile_bytes];
        let k_bytes = vec![0x38_u8; tile_bytes];

        q_buffer
            .copy_from_host(0, &q_bytes, Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &k_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rocwmma_fp8_qk_probe(&q_buffer, &k_buffer, &mut output_buffer, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; output_bytes];
        output_buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let output = le_bytes_to_f32s(&output);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0_f32));
    }

    #[test]
    fn first_hip_rocwmma_fp8_attn_probe_outputs_finite_nonzero_values_when_available() {
        let rdna4_device = (1..device_count().unwrap()).find(|&index| {
            device_info(index)
                .map(|info| {
                    info.backend == "hip"
                        && (info.compute_major == 12 || info.gcn_arch_name.starts_with("gfx12"))
                })
                .unwrap_or(false)
        });
        let Some(device_index) = rdna4_device else {
            return;
        };
        let mut context = RuntimeContext::create(device_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_bytes = 16_usize * 16_usize;
        let k_bytes = 32_usize * 16_usize;
        let v_elements = 32_usize * 16_usize;
        let output_bytes = 16_usize * 16_usize * std::mem::size_of::<f32>();
        let mut q_buffer = context.alloc_buffer(q_bytes).unwrap();
        let mut k_buffer = context.alloc_buffer(k_bytes).unwrap();
        let mut v_buffer = context
            .alloc_buffer(v_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context.alloc_buffer(output_bytes).unwrap();

        let q_input = vec![0x38_u8; q_bytes];
        let k_input = vec![0x38_u8; k_bytes];
        let v_input = (0..v_elements)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();

        q_buffer
            .copy_from_host(0, &q_input, Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &k_input, Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v_input), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rocwmma_fp8_attn_probe(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; output_bytes];
        output_buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let output = le_bytes_to_f32s(&output);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0_f32));
    }

    #[test]
    fn first_hip_paged_chunk_invalid_table_fails_closed_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let m = 2_usize;
        let block_size = 2_usize;
        let cache_blocks = 2_usize;
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 2_usize;
        let value_dim = 2_usize;
        let q_values = vec![0.1_f32; m * q_heads * head_dim];
        let k_values = vec![0.2_f32; m * kv_heads * head_dim];
        let v_values = vec![0.3_f32; m * kv_heads * value_dim];
        let cache_k = vec![0.4_f32; cache_blocks * block_size * kv_heads * head_dim];
        let cache_v = vec![0.5_f32; cache_blocks * block_size * kv_heads * value_dim];
        let mut q = context.alloc_buffer(q_values.len() * 4).unwrap();
        let mut k = context.alloc_buffer(k_values.len() * 4).unwrap();
        let mut v = context.alloc_buffer(v_values.len() * 4).unwrap();
        let mut table = context.alloc_buffer(4).unwrap();
        let mut k_cache = context.alloc_buffer(cache_k.len() * 4).unwrap();
        let mut v_cache = context.alloc_buffer(cache_v.len() * 4).unwrap();
        let mut output = context.alloc_buffer(m * q_heads * value_dim * 4).unwrap();
        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream)).unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream)).unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream)).unwrap();
        table.copy_from_host(0, &u32s_to_le_bytes(&[u32::MAX]), Some(&mut stream)).unwrap();
        k_cache.copy_from_host(0, &f32s_to_le_bytes(&cache_k), Some(&mut stream)).unwrap();
        v_cache.copy_from_host(0, &f32s_to_le_bytes(&cache_v), Some(&mut stream)).unwrap();
        output.copy_from_host(0, &f32s_to_le_bytes(&vec![42.0_f32; m * q_heads * value_dim]), Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();
        assert!(paged_kv_write_chunk_f32(&k, &v, &table, 0, m, block_size, cache_blocks, kv_heads, head_dim, value_dim, &mut k_cache, &mut v_cache, Some(&mut stream)).is_err());
        assert!(paged_causal_gqa_chunk_f32(&q, &k_cache, &v_cache, &table, 0, m, block_size, cache_blocks, q_heads, kv_heads, head_dim, value_dim, 1.0, &mut output, Some(&mut stream)).is_err());
        stream.synchronize().unwrap();
        let mut output_bytes = vec![0_u8; m * q_heads * value_dim * 4];
        output.copy_to_host(0, &mut output_bytes, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![42.0_f32; m * q_heads * value_dim]);
    }
