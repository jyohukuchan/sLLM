#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_reports_abi_version() {
        assert_eq!(abi_version(), 1);
    }

    #[test]
    fn runtime_has_at_least_cpu_device() {
        let count = device_count().unwrap();
        assert!(count >= 1);
        let info = device_info(0).unwrap();
        assert_eq!(info.backend, "cpu");
    }

    #[test]
    fn smoke_adds_f32_values() {
        let out = smoke_add_f32(&[1.0, 2.5, -3.0], &[4.0, -1.5, 3.5]).unwrap();
        assert_eq!(out, vec![5.0, 1.0, 0.5]);
    }

    #[test]
    fn cpu_context_allocates_runtime_buffer() {
        let mut context = RuntimeContext::create(0).unwrap();
        let info = context.device_info().unwrap();
        assert_eq!(info.backend, "cpu");
        let buffer = context.alloc_buffer(4096).unwrap();
        assert_eq!(buffer.size().unwrap(), 4096);
    }

    #[test]
    fn cpu_context_creates_and_synchronizes_stream() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn cpu_buffer_roundtrips_host_data() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(64).unwrap();
        let input: Vec<u8> = (0..48).map(|value| (value * 17 + 3) as u8).collect();
        buffer.copy_from_host(8, &input, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; input.len()];
        buffer
            .copy_to_host(8, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn zero_byte_buffer_copy_accepts_end_offset() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut buffer = context.alloc_buffer(8).unwrap();
        buffer.copy_from_host(8, &[], None).unwrap();
        let mut output = [];
        buffer.copy_to_host(8, &mut output, None).unwrap();
    }

    #[test]
    fn buffer_copy_rejects_out_of_bounds_range() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut buffer = context.alloc_buffer(4).unwrap();
        let err = buffer.copy_from_host(3, &[1_u8, 2], None).unwrap_err();
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn cpu_wmma_fp8_probe_writes_nonzero_marker() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_wmma_fp8_qk_probe_outputs_finite_nonzero_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_rocwmma_fp8_qk_probe_outputs_finite_nonzero_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_rocwmma_fp8_attn_probe_outputs_finite_nonzero_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_matvec_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn cpu_sq_fp8_matvec_f32_computes_expected_row_block_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload = context.alloc_buffer(6).unwrap();
        let mut scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        payload
            .copy_from_host(0, &[0x38, 0x40, 0xb8, 0x30, 0x00, 0x38], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_f32(
            &payload,
            &scales,
            &input,
            2,
            3,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![-11.0, 1.25]);
    }

    #[test]
    fn cpu_sq_fp8_matvec_batch_f32_computes_expected_row_block_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload = context.alloc_buffer(6).unwrap();
        let mut scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        payload
            .copy_from_host(0, &[0x38, 0x40, 0xb8, 0x30, 0x00, 0x38], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[0.5, -1.0, 2.0, 1.0, 2.0, -0.5]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_batch_f32(
            &payload,
            &scales,
            &input,
            2,
            3,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![-11.0, 1.25, 12.0, 0.25]
        );
    }

    #[test]
    fn cpu_sq_fp8_matvec_pair_f32_computes_expected_mixed_scale_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_payload = context.alloc_buffer(6).unwrap();
        let mut left_scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_payload = context.alloc_buffer(6).unwrap();
        let mut right_scales = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        let payload = [0x38, 0x40, 0xb8, 0x30, 0x00, 0x38];
        left_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        right_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        left_scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        right_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_pair_f32(
            &left_payload,
            &left_scales,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            &right_payload,
            &right_scales,
            SQ_FP8_SCALE_TENSOR,
            0,
            &input,
            2,
            2,
            3,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&left_bytes), vec![-11.0, 1.25]);
        assert_eq!(le_bytes_to_f32s(&right_bytes), vec![-3.5, 2.25]);
    }

    #[test]
    fn cpu_sq_fp8_matvec_triple_f32_computes_expected_mixed_scale_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_payload = context.alloc_buffer(6).unwrap();
        let mut first_scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_payload = context.alloc_buffer(6).unwrap();
        let mut second_scales = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_payload = context.alloc_buffer(6).unwrap();
        let mut third_scales = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        let payload = [0x38, 0x40, 0xb8, 0x30, 0x00, 0x38];
        first_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        second_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        third_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        first_scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        second_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        third_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 0.5]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_triple_f32(
            &first_payload,
            &first_scales,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            &second_payload,
            &second_scales,
            SQ_FP8_SCALE_TENSOR,
            0,
            &third_payload,
            &third_scales,
            SQ_FP8_SCALE_ROW,
            0,
            &input,
            2,
            2,
            2,
            3,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut third_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&first_bytes), vec![-11.0, 1.25]);
        assert_eq!(le_bytes_to_f32s(&second_bytes), vec![-3.5, 2.25]);
        assert_eq!(le_bytes_to_f32s(&third_bytes), vec![-7.0, 1.125]);
    }

    #[test]
    fn cpu_matvec_bf16_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<u16>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_bf16_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_bf16_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn cpu_bf16_row_f32_reads_selected_row() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<u16>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_bf16_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        bf16_row_f32(&matrix, 2, 3, 1, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 3 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn cpu_aq4_row_f32_reads_selected_row() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut indices = context.alloc_buffer(4).unwrap();
        let mut scales = context.alloc_buffer(4).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        indices
            .copy_from_host(0, &[0x21, 0x43, 0x65, 0x87], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(0, &[0, 1, 2, 3], Some(&mut stream))
            .unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0,
                ]),
                Some(&mut stream),
            )
            .unwrap();
        scale_values
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 10.0, 100.0, 1000.0]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        aq4_row_f32(
            &indices,
            &scales,
            &codebook,
            &scale_values,
            None,
            4,
            2,
            0.5,
            0,
            2,
            4,
            1,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![250.0, 300.0, 3500.0, 4000.0]
        );
    }

    #[test]
    fn cpu_matvec_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        let err = matvec_f32(&matrix, &input, 2, 3, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_top1_f32_writes_partial_maxima() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut values = vec![-1.0_f32; 300];
        values[123] = 8.0;
        values[259] = 9.0;
        values[260] = 9.0;
        let mut input = context
            .alloc_buffer(values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let partial_count = top1_partial_count(values.len()).unwrap();
        assert_eq!(partial_count, 2);
        let mut partial_values = context
            .alloc_buffer(partial_count * std::mem::size_of::<f32>())
            .unwrap();
        let mut partial_indices = context
            .alloc_buffer(partial_count * std::mem::size_of::<u32>())
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let written = top1_f32(
            &input,
            values.len(),
            &mut partial_values,
            &mut partial_indices,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(written, partial_count);
        stream.synchronize().unwrap();

        let mut value_bytes = vec![0_u8; partial_count * std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; partial_count * std::mem::size_of::<u32>()];
        partial_values
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        partial_indices
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![8.0, 9.0]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![123, 259]);
    }

    #[test]
    fn cpu_top1_pairs_f32_in_place_preserves_original_indices() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut values = vec![-1.0_f32; 300];
        let mut indices = vec![0_u32; 300];
        values[123] = 8.0;
        indices[123] = 900;
        values[259] = 9.0;
        indices[259] = 800;
        values[260] = 9.0;
        indices[260] = 700;
        let mut values_buffer = context
            .alloc_buffer(values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut indices_buffer = context
            .alloc_buffer(indices.len() * std::mem::size_of::<u32>())
            .unwrap();
        values_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut stream))
            .unwrap();
        indices_buffer
            .copy_from_host(0, &u32s_to_le_bytes(&indices), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let first_stage = top1_pairs_f32_in_place(
            &mut values_buffer,
            &mut indices_buffer,
            values.len(),
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(first_stage, 2);
        let second_stage = top1_pairs_f32_in_place(
            &mut values_buffer,
            &mut indices_buffer,
            first_stage,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(second_stage, 1);
        stream.synchronize().unwrap();

        let mut value_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; std::mem::size_of::<u32>()];
        values_buffer
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        indices_buffer
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![9.0]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![700]);
    }

    #[test]
    fn first_hip_top1_pairs_f32_in_place_preserves_original_indices_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut values = vec![-1.0_f32; 300];
        let mut indices = vec![0_u32; 300];
        values[123] = 8.0;
        indices[123] = 900;
        values[259] = 9.0;
        indices[259] = 800;
        values[260] = 9.0;
        indices[260] = 700;
        let mut values_buffer = context
            .alloc_buffer(values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut indices_buffer = context
            .alloc_buffer(indices.len() * std::mem::size_of::<u32>())
            .unwrap();
        values_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut stream))
            .unwrap();
        indices_buffer
            .copy_from_host(0, &u32s_to_le_bytes(&indices), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let first_stage = top1_pairs_f32_in_place(
            &mut values_buffer,
            &mut indices_buffer,
            values.len(),
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(first_stage, 2);
        let second_stage = top1_pairs_f32_in_place(
            &mut values_buffer,
            &mut indices_buffer,
            first_stage,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(second_stage, 1);
        stream.synchronize().unwrap();

        let mut value_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; std::mem::size_of::<u32>()];
        values_buffer
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        indices_buffer
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![9.0]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![700]);
    }

    #[test]
    fn cpu_rmsnorm_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0];
        let weight_values = [0.5_f32, 1.0, 1.5, -2.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rmsnorm_f32(
            &input,
            &weight,
            input_values.len(),
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_rmsnorm(&input_values, &weight_values, epsilon);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_segmented_rmsnorm_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_f32(
            &input,
            &weight,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_segmented_rmsnorm_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let gate_values = [-1.0_f32, 0.25, 1.0, -2.0, 0.5, 3.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_silu_mul_f32(
            &input,
            &weight,
            &gate,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let normed = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected = expected_silu_mul(&gate_values, &normed);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_rmsnorm_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let weight = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = rmsnorm_f32(&input, &weight, 4, 1e-5, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let up_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut up = context
            .alloc_buffer(up_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        up.copy_from_host(0, &f32s_to_le_bytes(&up_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        silu_mul_f32(
            &gate,
            &up,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&gate_values, &up_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_silu_mul_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let up = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = silu_mul_f32(&gate, &up, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_sigmoid_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32(
            &gate,
            &input,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_sigmoid_mul_f32_in_place_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input_output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input_output
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32_in_place(
            &gate,
            &mut input_output,
            gate_values.len(),
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        input_output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_sigmoid_mul_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let input = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = sigmoid_mul_f32(&gate, &input, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_qwen35_split_q_gate_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let projected_values = [1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let mut projected = context
            .alloc_buffer(projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut query = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        projected
            .copy_from_host(0, &f32s_to_le_bytes(&projected_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_split_q_gate_f32(&projected, 2, 2, &mut query, &mut gate, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let mut query_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        let mut gate_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        query
            .copy_to_host(0, &mut query_bytes, Some(&mut stream))
            .unwrap();
        gate.copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&query_bytes), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(le_bytes_to_f32s(&gate_bytes), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn cpu_qwen35_qk_norm_rope_f32_matches_split_norm_rope() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = q_heads * head_dim;
        let k_output_elements = kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-5);
    }

    #[test]
    fn cpu_qwen35_qk_norm_rope_batch_f32_matches_split_norm_rope() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let sequence_len = 3_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..sequence_len * q_heads * head_dim * 2)
            .map(|index| (index as f32 - 17.0) / 19.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| (index as f32 + 11.0) / -13.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = sequence_len * q_heads * head_dim;
        let k_output_elements = sequence_len * kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_batch_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            sequence_len,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let q_projected_stride = q_heads * head_dim * 2;
        let mut expected_q_gate = Vec::with_capacity(q_output_elements);
        let mut expected_q_query = Vec::with_capacity(q_output_elements);
        for token in 0..sequence_len {
            for head in 0..q_heads {
                let source_base = token * q_projected_stride + head * 2 * head_dim;
                expected_q_query
                    .extend_from_slice(&q_projected_values[source_base..source_base + head_dim]);
                expected_q_gate.extend_from_slice(
                    &q_projected_values[source_base + head_dim..source_base + 2 * head_dim],
                );
            }
        }
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            sequence_len,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            sequence_len,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-5);
    }

    #[test]
    fn cpu_qwen35_qk_norm_rope_paged_kv_write_f32_matches_split_norm_rope_and_paged_write() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let value_dim = 5_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let cache_position = 5_usize;
        let block_size = 4_usize;
        let cache_blocks = 2_usize;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let v_projected_values = (0..kv_heads * value_dim)
            .map(|index| (index as f32 - 2.0) / 7.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let block_table_values = vec![1_u32, 0_u32];
        let physical_tokens = cache_blocks * block_size;
        let q_output_elements = q_heads * head_dim;
        let k_cache_elements = physical_tokens * kv_heads * head_dim;
        let v_cache_elements = physical_tokens * kv_heads * value_dim;
        let initial_k_cache = (0..k_cache_elements)
            .map(|index| -10.0_f32 - index as f32)
            .collect::<Vec<_>>();
        let initial_v_cache = (0..v_cache_elements)
            .map(|index| 20.0_f32 + index as f32)
            .collect::<Vec<_>>();

        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_projected = context
            .alloc_buffer(v_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(k_cache_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(v_cache_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        v_projected
            .copy_from_host(0, &f32s_to_le_bytes(&v_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        k_cache
            .copy_from_host(0, &f32s_to_le_bytes(&initial_k_cache), Some(&mut stream))
            .unwrap();
        v_cache
            .copy_from_host(0, &f32s_to_le_bytes(&initial_v_cache), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_paged_kv_write_f32(
            &q_projected,
            &k_projected,
            &v_projected,
            &q_weight,
            &k_weight,
            &block_table,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            cache_blocks,
            &mut q_gate,
            &mut q_rope,
            &mut k_cache,
            &mut v_cache,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut expected_k_cache = initial_k_cache;
        let mut expected_v_cache = initial_v_cache;
        let physical_timestep = block_table_values[cache_position / block_size] as usize
            * block_size
            + (cache_position % block_size);
        for kv_head in 0..kv_heads {
            let k_src = kv_head * head_dim;
            let k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            expected_k_cache[k_dst..k_dst + head_dim]
                .copy_from_slice(&expected_k_rope[k_src..k_src + head_dim]);
            let v_src = kv_head * value_dim;
            let v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            expected_v_cache[v_dst..v_dst + value_dim]
                .copy_from_slice(&v_projected_values[v_src..v_src + value_dim]);
        }

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_cache_bytes = vec![0_u8; k_cache_elements * std::mem::size_of::<f32>()];
        let mut v_cache_bytes = vec![0_u8; v_cache_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_cache
            .copy_to_host(0, &mut k_cache_bytes, Some(&mut stream))
            .unwrap();
        v_cache
            .copy_to_host(0, &mut v_cache_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-6);
    }

    #[test]
    fn cpu_add_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let lhs_values = [-1.0_f32, 0.0, 1.0, 2.0, 8.5];
        let rhs_values = [3.0_f32, -4.0, 5.0, 6.0, -0.25];
        let mut lhs = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut rhs = context
            .alloc_buffer(rhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        lhs.copy_from_host(0, &f32s_to_le_bytes(&lhs_values), Some(&mut stream))
            .unwrap();
        rhs.copy_from_host(0, &f32s_to_le_bytes(&rhs_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        add_f32(&lhs, &rhs, lhs_values.len(), &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; lhs_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = lhs_values
            .iter()
            .zip(rhs_values.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn cpu_add_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let lhs = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let rhs = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = add_f32(&lhs, &rhs, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_rope_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let heads = 2_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 3_usize;
        let rope_base = 10000.0_f32;
        let elements = sequence_len * heads * head_dim;
        let input_values = (0..elements)
            .map(|index| (index as f32 - 11.0) / 7.0)
            .collect::<Vec<_>>();
        let expected = expected_rope(
            &input_values,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rope_f32(
            &input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_rope_f32_rejects_invalid_rotary_dim_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();

        let err = rope_f32(&input, 1, 1, 6, 5, 0, 10000.0, &mut output, None).unwrap_err();
        assert!(err.contains("rotary_dim"));

        let mut short_output = context
            .alloc_buffer(5 * std::mem::size_of::<f32>())
            .unwrap();
        let err = rope_f32(&input, 1, 1, 6, 4, 0, 10000.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_causal_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32(
            &q,
            &k,
            &v,
            sequence_len,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_causal_attn_f32_flash2_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32_flash2(
            &q,
            &k,
            &v,
            sequence_len,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_causal_attn_batch_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let batch_count = 2_usize;
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_elements_per_batch = sequence_len * q_heads * head_dim;
        let k_elements_per_batch = sequence_len * kv_heads * head_dim;
        let v_elements_per_batch = sequence_len * kv_heads * value_dim;
        let q_values = (0..batch_count * q_elements_per_batch)
            .map(|index| (index as f32 - 19.0) / 23.0)
            .collect::<Vec<_>>();
        let k_values = (0..batch_count * k_elements_per_batch)
            .map(|index| ((index * 3) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();
        let v_values = (0..batch_count * v_elements_per_batch)
            .map(|index| ((index * 5) as f32 - 11.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for batch in 0..batch_count {
            expected.extend(expected_causal_attn(
                &q_values[batch * q_elements_per_batch..(batch + 1) * q_elements_per_batch],
                &k_values[batch * k_elements_per_batch..(batch + 1) * k_elements_per_batch],
                &v_values[batch * v_elements_per_batch..(batch + 1) * v_elements_per_batch],
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            ));
        }
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_batch_f32(
            &q,
            &k,
            &v,
            batch_count,
            sequence_len,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_causal_attn_batch_f32_flash2_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let batch_count = 2_usize;
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_elements_per_batch = sequence_len * q_heads * head_dim;
        let k_elements_per_batch = sequence_len * kv_heads * head_dim;
        let v_elements_per_batch = sequence_len * kv_heads * value_dim;
        let q_values = (0..batch_count * q_elements_per_batch)
            .map(|index| (index as f32 - 19.0) / 23.0)
            .collect::<Vec<_>>();
        let k_values = (0..batch_count * k_elements_per_batch)
            .map(|index| ((index * 3) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();
        let v_values = (0..batch_count * v_elements_per_batch)
            .map(|index| ((index * 5) as f32 - 11.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for batch in 0..batch_count {
            expected.extend(expected_causal_attn(
                &q_values[batch * q_elements_per_batch..(batch + 1) * q_elements_per_batch],
                &k_values[batch * k_elements_per_batch..(batch + 1) * k_elements_per_batch],
                &v_values[batch * v_elements_per_batch..(batch + 1) * v_elements_per_batch],
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            ));
        }
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_batch_f32_flash2(
            &q,
            &k,
            &v,
            batch_count,
            sequence_len,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_causal_attn_f32_rejects_invalid_heads_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let k = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        let err = causal_attn_f32(&q, &k, &v, 1, 3, 2, 1, 1, 1.0, &mut output, None).unwrap_err();
        assert!(err.contains("q_heads"));

        let mut short_output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let err =
            causal_attn_f32(&q, &k, &v, 1, 4, 2, 1, 1, 1.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_decode_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..cache_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..cache_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_decode_attn(
            &q_values,
            &k_values,
            &v_values,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        decode_attn_f32(
            &q,
            &k,
            &v,
            cache_len,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_decode_attn_f32_rejects_invalid_shape_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let k = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(8 * std::mem::size_of::<f32>())
            .unwrap();

        let err = decode_attn_f32(&q, &k, &v, 3, 3, 2, 1, 2, 1.0, &mut output, None).unwrap_err();
        assert!(err.contains("q_heads"));

        let mut short_output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err =
            decode_attn_f32(&q, &k, &v, 3, 4, 2, 1, 2, 1.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_cached_prefix_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_values,
            &v_values,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_f32(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_f32_flash2_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_values,
            &v_values,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_f32_flash2(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_fp8_e4m3_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_fp8_e4m3_flash2_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_flash2(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_fp8_e4m3_flash2_fp8q_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (q_fp8, q_scale, q_expected) = fp8_e4m3_quantize(&q_values);
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_expected,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context.alloc_buffer(q_fp8.len()).unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &q_fp8, Some(&mut stream)).unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_flash2_fp8q(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_fp8_e4m3_rocwmma_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 3_usize;
        let new_tokens = 2_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 32_usize;
        let kv_heads = 2_usize;
        let head_dim = 32_usize;
        let value_dim = 32_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 97.0) / 211.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 113.0) / 257.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 131.0) / 293.0)
            .collect::<Vec<_>>();
        let (q_fp8, q_scale, q_expected) = fp8_e4m3_quantize(&q_values);
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_expected,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context.alloc_buffer(q_fp8.len()).unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &q_fp8, Some(&mut stream)).unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_rocwmma(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_cached_prefix_attn_f32_rejects_invalid_shape_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let k = context
            .alloc_buffer(10 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(10 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(8 * std::mem::size_of::<f32>())
            .unwrap();

        let err = cached_prefix_attn_f32(&q, &k, &v, 0, 1, 3, 2, 1, 2, 1.0, &mut output, None)
            .unwrap_err();
        assert!(err.contains("q_heads"));

        let mut short_output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err =
            cached_prefix_attn_f32(&q, &k, &v, 0, 1, 4, 2, 1, 2, 1.0, &mut short_output, None)
                .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_paged_decode_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_paged_decode_attn_sigmoid_gate_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
        let gate_values = (0..q_heads * value_dim)
            .map(|index| ((index * 7) as f32 - 5.0) / 9.0)
            .collect::<Vec<_>>();
        let k_cache_values = (0..cache_blocks * block_size * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_cache_values = (0..cache_blocks * block_size * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let block_table_values = vec![2_u32, 0_u32, 3_u32];
        let decoded = expected_paged_decode_attn(
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
        let expected = expected_sigmoid_mul(&gate_values, &decoded);

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
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
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_paged_decode_attn_f32_rejects_invalid_shape_short_output_or_short_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(12 * std::mem::size_of::<f32>())
            .unwrap();
        let k_cache = context
            .alloc_buffer(4 * 2 * 2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let v_cache = context
            .alloc_buffer(4 * 2 * 2 * 2 * std::mem::size_of::<f32>())
            .unwrap();
        let block_table = context
            .alloc_buffer(3 * std::mem::size_of::<u32>())
            .unwrap();

        let mut short_output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            5,
            2,
            4,
            3,
            2,
            3,
            2,
            1.0,
            &mut short_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("q_heads"));

        let mut output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            5,
            2,
            4,
            4,
            2,
            3,
            2,
            1.0,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));

        let short_block_table = context
            .alloc_buffer(2 * std::mem::size_of::<u32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &short_block_table,
            5,
            2,
            4,
            4,
            2,
            3,
            2,
            1.0,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("block"));
    }

    #[test]
    fn cpu_paged_kv_write_f32_writes_expected_physical_slot() {
        let mut context = RuntimeContext::create(0).unwrap();
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
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-6);
    }

    #[test]
    fn cpu_paged_kv_write_f32_rejects_short_cache_or_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let k = context
            .alloc_buffer(2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(2 * 2 * std::mem::size_of::<f32>())
            .unwrap();
        let short_block_table = context.alloc_buffer(std::mem::size_of::<u32>()).unwrap();
        let block_table = context
            .alloc_buffer(2 * std::mem::size_of::<u32>())
            .unwrap();
        let mut short_k_cache = context
            .alloc_buffer((4 * 2 * 2 * 3 - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(4 * 2 * 2 * 2 * std::mem::size_of::<f32>())
            .unwrap();

        let err = paged_kv_write_f32(
            &k,
            &v,
            &short_block_table,
            3,
            2,
            4,
            2,
            3,
            2,
            &mut short_k_cache,
            &mut v_cache,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("block_table"));

        let err = paged_kv_write_f32(
            &k,
            &v,
            &block_table,
            3,
            2,
            4,
            2,
            3,
            2,
            &mut short_k_cache,
            &mut v_cache,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("cache"));
    }

    #[test]
    fn cpu_linear_attn_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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

    #[test]
    fn cpu_linear_attn_gate_beta_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let a = [
            0.1_f32, -0.2, 1.2, 0.9, 0.8, -1.1, -0.7, 0.5, 1.4, -0.3, 0.2, -0.6,
        ];
        let b = [
            1.0_f32, -1.2, 0.3, -0.8, 0.6, 1.1, -0.5, 0.9, 0.0, -0.4, 1.3, -0.7,
        ];
        let a_log = [-1.0_f32, 0.25, -0.5];
        let dt_bias = [0.3_f32, -0.2, 0.4];
        let a_buffer = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let b_buffer = context
            .alloc_buffer(b.len() * std::mem::size_of::<f32>())
            .unwrap();
        let a_log_buffer = context
            .alloc_buffer(a_log.len() * std::mem::size_of::<f32>())
            .unwrap();
        let dt_bias_buffer = context
            .alloc_buffer(dt_bias.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer((a.len() - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();

        let err = linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            3,
            4,
            &mut gate_output,
            &mut beta_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("linear attention gate beta"));
        assert!(err.contains("gate_output"));
    }

    #[test]
    fn cpu_linear_attn_recurrent_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_linear_attn_recurrent_f32_rejects_short_output_or_state() {
        let mut context = RuntimeContext::create(0).unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 2_usize;
        let key_dim = 2_usize;
        let value_dim = 2_usize;
        let q_buffer = context
            .alloc_buffer(key_heads * sequence_len * key_dim * std::mem::size_of::<f32>())
            .unwrap();
        let k_buffer = context
            .alloc_buffer(key_heads * sequence_len * key_dim * std::mem::size_of::<f32>())
            .unwrap();
        let v_buffer = context
            .alloc_buffer(value_heads * sequence_len * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let gate_buffer = context
            .alloc_buffer(value_heads * sequence_len * std::mem::size_of::<f32>())
            .unwrap();
        let beta_buffer = context
            .alloc_buffer(value_heads * sequence_len * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(value_heads * key_dim * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let mut short_output = context
            .alloc_buffer((value_heads * sequence_len * value_dim - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut short_state = context
            .alloc_buffer((value_heads * key_dim * value_dim - 1) * std::mem::size_of::<f32>())
            .unwrap();

        let err = linear_attn_recurrent_f32(
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
            &mut short_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("linear attention recurrent"));
        assert!(err.contains("output"));

        let mut full_output = context
            .alloc_buffer(value_heads * sequence_len * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let state_error = linear_attn_recurrent_f32(
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
            &mut short_state,
            &mut full_output,
            None,
        )
        .unwrap_err();
        assert!(state_error.contains("linear attention recurrent"));
        assert!(state_error.contains("state"));
    }

    #[test]
    fn cpu_depthwise_conv1d_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let channels = 3_usize;
        let sequence_len = 5_usize;
        let kernel_size = 3_usize;
        let input_values = [
            1.0_f32, 0.5, -1.0, 2.0, 1.0, 0.5, 3.0, -0.5, 0.5, 4.0, -1.0, 1.5, 5.0, 0.0, -2.0,
        ];
        let weight_values = [1.0_f32, -1.0, 2.0, 0.5_f32, 1.0, -0.5, -1.0, 1.0, 1.5];
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
    fn cpu_depthwise_conv1d_f32_uses_causal_conv1d_weight_order() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, 3.0];
        let weight_values = [10.0_f32, 100.0, 1000.0];
        let expected = [1000.0_f32, 2100.0, 3210.0];
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        depthwise_conv1d_f32(&input, &weight, 1, 3, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_depthwise_conv1d_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let weight = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(5 * std::mem::size_of::<f32>())
            .unwrap();

        let err = depthwise_conv1d_f32(&input, &weight, 2, 3, 1, &mut output, None).unwrap_err();
        assert!(err.contains("depthwise conv1d"));
    }

    #[test]
    fn cpu_linear_attn_qkv_prepare_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
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
    fn cpu_linear_attn_qkv_prepare_batch_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        assert_linear_attn_qkv_prepare_batch_matches_expected(&mut context, 1e-6);
    }

    #[test]
    fn cpu_aq4_dequant_f32_materializes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x30], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![5.0, 10.0, 0.0, 60.0]);
    }

    #[test]
    fn cpu_aq4_dequant_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        index.copy_from_host(0, &[0x21_u8, 0x30], None).unwrap();
        scale.copy_from_host(0, &[0_u8, 1], None).unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), None)
            .unwrap();

        let err = aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_aq4_matvec_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![112.5, 30.0]);
    }

    #[test]
    fn cpu_aq4_matvec_batch_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * 2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[0.5, -1.0, 2.0, 1.0, 0.0, -0.5]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_batch_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![112.5, 30.0, -25.0, -12.5]
        );
    }

    #[test]
    fn cpu_aq4_matvec_top1_f32_writes_partial_maximum() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let partial_count = aq4_matvec_top1_partial_count(2).unwrap();
        let mut partial_values = context
            .alloc_buffer(partial_count * std::mem::size_of::<f32>())
            .unwrap();
        let mut partial_indices = context
            .alloc_buffer(partial_count * std::mem::size_of::<u32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let written = aq4_matvec_top1_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut partial_values,
            &mut partial_indices,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(written, partial_count);

        let mut value_bytes = vec![0_u8; partial_count * std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; partial_count * std::mem::size_of::<u32>()];
        partial_values
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        partial_indices
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![112.5]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![0]);
    }

    #[test]
    fn cpu_aq4_matvec_pair_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_index = context.alloc_buffer(2).unwrap();
        let mut left_scale = context.alloc_buffer(2).unwrap();
        let mut left_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut right_index = context.alloc_buffer(1).unwrap();
        let mut right_scale = context.alloc_buffer(1).unwrap();
        let mut right_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        left_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        left_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        right_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        right_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        left_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        right_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        left_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        right_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_pair_f32(
            &left_index,
            &left_scale,
            &left_codebook,
            &left_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &right_index,
            &right_scale,
            &right_codebook,
            &right_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            2,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_output_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&left_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&right_output_bytes), vec![28.0]);
    }

    #[test]
    fn cpu_aq4_matvec_triple_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_index = context.alloc_buffer(2).unwrap();
        let mut first_scale = context.alloc_buffer(2).unwrap();
        let mut first_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut second_index = context.alloc_buffer(1).unwrap();
        let mut second_scale = context.alloc_buffer(1).unwrap();
        let mut second_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_index = context.alloc_buffer(1).unwrap();
        let mut third_scale = context.alloc_buffer(1).unwrap();
        let mut third_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        first_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        first_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        second_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        second_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        third_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        third_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut first_codebook,
            &mut second_codebook,
            &mut third_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut first_scale_values,
            &mut second_scale_values,
            &mut third_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_triple_f32(
            &first_index,
            &first_scale,
            &first_codebook,
            &first_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &second_index,
            &second_scale,
            &second_codebook,
            &second_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &third_index,
            &third_scale,
            &third_codebook,
            &third_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            1,
            2,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut third_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_output_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_output_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&first_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&second_output_bytes), vec![28.0]);
        assert_eq!(le_bytes_to_f32s(&third_output_bytes), vec![38.0]);
    }

    #[test]
    fn cpu_aq4_matvec_qkv_z_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut qkv_index = context.alloc_buffer(2).unwrap();
        let mut qkv_scale = context.alloc_buffer(2).unwrap();
        let mut qkv_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut qkv_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut z_index = context.alloc_buffer(1).unwrap();
        let mut z_scale = context.alloc_buffer(1).unwrap();
        let mut z_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut z_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut a_index = context.alloc_buffer(1).unwrap();
        let mut a_scale = context.alloc_buffer(1).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut b_index = context.alloc_buffer(1).unwrap();
        let mut b_scale = context.alloc_buffer(1).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut dt_bias = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut qkv_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut z_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut gate_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut beta_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        qkv_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        qkv_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        z_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        z_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        a_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0xa9_u8], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut qkv_codebook,
            &mut z_codebook,
            &mut a_codebook,
            &mut b_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut qkv_scale_values,
            &mut z_scale_values,
            &mut a_scale_values,
            &mut b_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_qkv_z_gate_beta_f32(
            &qkv_index,
            &qkv_scale,
            &qkv_codebook,
            &qkv_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &z_index,
            &z_scale,
            &z_codebook,
            &z_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            1,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            1,
            2,
            0.1,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            1,
            1,
            2,
            &mut qkv_output,
            &mut z_output,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut qkv_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut z_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut gate_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut beta_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        qkv_output
            .copy_to_host(0, &mut qkv_output_bytes, Some(&mut stream))
            .unwrap();
        z_output
            .copy_to_host(0, &mut z_output_bytes, Some(&mut stream))
            .unwrap();
        gate_output
            .copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let x = 3.8_f32 + 0.5;
        let expected_gate = -x.exp().ln_1p();
        let expected_beta = 1.0_f32 / (1.0 + (-4.8_f32).exp());
        assert_eq!(le_bytes_to_f32s(&qkv_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&z_output_bytes), vec![28.0]);
        assert_f32s_close(
            &le_bytes_to_f32s(&gate_output_bytes),
            &[expected_gate],
            1e-5,
        );
        assert_f32s_close(
            &le_bytes_to_f32s(&beta_output_bytes),
            &[expected_beta],
            1e-6,
        );
    }

    #[test]
    fn cpu_aq4_matvec_add_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut residual = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&[1.25, -2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_add_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            &residual,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![113.75, 28.0]);
    }

    #[test]
    fn cpu_aq4_matvec_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut gate_index = context.alloc_buffer(3).unwrap();
        let mut gate_scale = context.alloc_buffer(3).unwrap();
        let mut gate_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_index = context.alloc_buffer(3).unwrap();
        let mut up_scale = context.alloc_buffer(3).unwrap();
        let mut up_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        gate_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        up_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        gate_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        up_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        gate_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        up_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        gate_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        up_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_silu_mul_f32(
            &gate_index,
            &gate_scale,
            &gate_codebook,
            &gate_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &up_index,
            &up_scale,
            &up_codebook,
            &up_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&[1.125, 0.3], &[2.25, 0.6]);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn cpu_aq4_matvec_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut a_index = context.alloc_buffer(3).unwrap();
        let mut a_scale = context.alloc_buffer(3).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_index = context.alloc_buffer(3).unwrap();
        let mut b_scale = context.alloc_buffer(3).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        a_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        a_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        b_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        a_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        b_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0, 0.5]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.1, -0.2]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_gate_beta_f32(
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            3,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut beta_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        gate_output
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let (expected_gate, expected_beta) = expected_linear_attn_gate_beta(
            &[1.125, 0.3],
            &[2.25, 0.6],
            &[0.0, 0.5],
            &[0.1, -0.2],
            2,
            1,
        );
        assert_f32s_close(&le_bytes_to_f32s(&gate_bytes), &expected_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&beta_bytes), &expected_beta, 1e-6);
    }

    #[test]
    fn first_hip_aq4_dequant_f32_materializes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x30], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![5.0, 10.0, 0.0, 60.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![112.5, 30.0]);
    }

    #[test]
    fn first_hip_sq_fp8_matvec_batch_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload = context.alloc_buffer(6).unwrap();
        let mut scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        payload
            .copy_from_host(0, &[0x38, 0x40, 0xb8, 0x30, 0x00, 0x38], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[0.5, -1.0, 2.0, 1.0, 2.0, -0.5]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_batch_f32(
            &payload,
            &scales,
            &input,
            2,
            3,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(
            &le_bytes_to_f32s(&output_bytes),
            &[-11.0, 1.25, 12.0, 0.25],
            1e-6,
        );
    }

    #[test]
    fn first_hip_sq_fp8_matvec_pair_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_payload = context.alloc_buffer(6).unwrap();
        let mut left_scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_payload = context.alloc_buffer(6).unwrap();
        let mut right_scales = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        let payload = [0x38, 0x40, 0xb8, 0x30, 0x00, 0x38];
        left_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        right_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        left_scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        right_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_pair_f32(
            &left_payload,
            &left_scales,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            &right_payload,
            &right_scales,
            SQ_FP8_SCALE_TENSOR,
            0,
            &input,
            2,
            2,
            3,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&left_bytes), &[-11.0, 1.25], 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&right_bytes), &[-3.5, 2.25], 1e-6);
    }

    #[test]
    fn first_hip_sq_fp8_matvec_triple_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_payload = context.alloc_buffer(6).unwrap();
        let mut first_scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_payload = context.alloc_buffer(6).unwrap();
        let mut second_scales = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_payload = context.alloc_buffer(6).unwrap();
        let mut third_scales = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        let payload = [0x38, 0x40, 0xb8, 0x30, 0x00, 0x38];
        first_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        second_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        third_payload
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        first_scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 4.0, 1.0, 0.5]),
                Some(&mut stream),
            )
            .unwrap();
        second_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        third_scales
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 0.5]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sq_fp8_matvec_triple_f32(
            &first_payload,
            &first_scales,
            SQ_FP8_SCALE_ROW_BLOCK,
            2,
            &second_payload,
            &second_scales,
            SQ_FP8_SCALE_TENSOR,
            0,
            &third_payload,
            &third_scales,
            SQ_FP8_SCALE_ROW,
            0,
            &input,
            2,
            2,
            2,
            3,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut third_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&first_bytes), &[-11.0, 1.25], 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&second_bytes), &[-3.5, 2.25], 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&third_bytes), &[-7.0, 1.125], 1e-6);
    }

    #[test]
    fn first_hip_aq4_matvec_batch_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * 2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[0.5, -1.0, 2.0, 1.0, 0.0, -0.5]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_batch_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![112.5, 30.0, -25.0, -12.5]
        );
    }

    #[test]
    fn first_hip_aq4_matvec_top1_f32_writes_partial_maximum_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let partial_count = aq4_matvec_top1_partial_count(2).unwrap();
        let mut partial_values = context
            .alloc_buffer(partial_count * std::mem::size_of::<f32>())
            .unwrap();
        let mut partial_indices = context
            .alloc_buffer(partial_count * std::mem::size_of::<u32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let written = aq4_matvec_top1_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut partial_values,
            &mut partial_indices,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(written, partial_count);

        let mut value_bytes = vec![0_u8; partial_count * std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; partial_count * std::mem::size_of::<u32>()];
        partial_values
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        partial_indices
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![112.5]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![0]);
    }

    #[test]
    fn first_hip_aq4_matvec_pair_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_index = context.alloc_buffer(2).unwrap();
        let mut left_scale = context.alloc_buffer(2).unwrap();
        let mut left_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut right_index = context.alloc_buffer(1).unwrap();
        let mut right_scale = context.alloc_buffer(1).unwrap();
        let mut right_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        left_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        left_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        right_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        right_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        left_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        right_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        left_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        right_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_pair_f32(
            &left_index,
            &left_scale,
            &left_codebook,
            &left_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &right_index,
            &right_scale,
            &right_codebook,
            &right_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            2,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_output_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&left_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&right_output_bytes), vec![28.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_triple_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_index = context.alloc_buffer(2).unwrap();
        let mut first_scale = context.alloc_buffer(2).unwrap();
        let mut first_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut second_index = context.alloc_buffer(1).unwrap();
        let mut second_scale = context.alloc_buffer(1).unwrap();
        let mut second_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_index = context.alloc_buffer(1).unwrap();
        let mut third_scale = context.alloc_buffer(1).unwrap();
        let mut third_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        first_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        first_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        second_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        second_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        third_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        third_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut first_codebook,
            &mut second_codebook,
            &mut third_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut first_scale_values,
            &mut second_scale_values,
            &mut third_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_triple_f32(
            &first_index,
            &first_scale,
            &first_codebook,
            &first_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &second_index,
            &second_scale,
            &second_codebook,
            &second_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &third_index,
            &third_scale,
            &third_codebook,
            &third_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            1,
            2,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut third_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_output_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_output_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&first_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&second_output_bytes), vec![28.0]);
        assert_eq!(le_bytes_to_f32s(&third_output_bytes), vec![38.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_add_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut residual = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&[1.25, -2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_add_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            &residual,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![113.75, 28.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut gate_index = context.alloc_buffer(3).unwrap();
        let mut gate_scale = context.alloc_buffer(3).unwrap();
        let mut gate_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_index = context.alloc_buffer(3).unwrap();
        let mut up_scale = context.alloc_buffer(3).unwrap();
        let mut up_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        gate_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        up_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        gate_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        up_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        gate_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        up_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        gate_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        up_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_silu_mul_f32(
            &gate_index,
            &gate_scale,
            &gate_codebook,
            &gate_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &up_index,
            &up_scale,
            &up_codebook,
            &up_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&[1.125, 0.3], &[2.25, 0.6]);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn first_hip_aq4_matvec_gate_beta_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut a_index = context.alloc_buffer(3).unwrap();
        let mut a_scale = context.alloc_buffer(3).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_index = context.alloc_buffer(3).unwrap();
        let mut b_scale = context.alloc_buffer(3).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        a_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        a_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        b_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        a_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        b_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0, 0.5]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.1, -0.2]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_gate_beta_f32(
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            3,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut beta_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        gate_output
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let (expected_gate, expected_beta) = expected_linear_attn_gate_beta(
            &[1.125, 0.3],
            &[2.25, 0.6],
            &[0.0, 0.5],
            &[0.1, -0.2],
            2,
            1,
        );
        assert_f32s_close(&le_bytes_to_f32s(&gate_bytes), &expected_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&beta_bytes), &expected_beta, 1e-6);
    }

    #[test]
    fn first_hip_matvec_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn first_hip_rmsnorm_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0];
        let weight_values = [0.5_f32, 1.0, 1.5, -2.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rmsnorm_f32(
            &input,
            &weight,
            input_values.len(),
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_rmsnorm(&input_values, &weight_values, epsilon);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_segmented_rmsnorm_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_f32(
            &input,
            &weight,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_segmented_rmsnorm_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let gate_values = [-1.0_f32, 0.25, 1.0, -2.0, 0.5, 3.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_silu_mul_f32(
            &input,
            &weight,
            &gate,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let normed = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected = expected_silu_mul(&gate_values, &normed);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let up_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut up = context
            .alloc_buffer(up_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        up.copy_from_host(0, &f32s_to_le_bytes(&up_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        silu_mul_f32(
            &gate,
            &up,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&gate_values, &up_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_sigmoid_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32(
            &gate,
            &input,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_sigmoid_mul_f32_in_place_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input_output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input_output
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32_in_place(
            &gate,
            &mut input_output,
            gate_values.len(),
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        input_output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_add_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let lhs_values = [-1.0_f32, 0.0, 1.0, 2.0, 8.5];
        let rhs_values = [3.0_f32, -4.0, 5.0, 6.0, -0.25];
        let mut lhs = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut rhs = context
            .alloc_buffer(rhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        lhs.copy_from_host(0, &f32s_to_le_bytes(&lhs_values), Some(&mut stream))
            .unwrap();
        rhs.copy_from_host(0, &f32s_to_le_bytes(&rhs_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        add_f32(&lhs, &rhs, lhs_values.len(), &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; lhs_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = lhs_values
            .iter()
            .zip(rhs_values.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn first_hip_rope_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let heads = 2_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 3_usize;
        let rope_base = 10000.0_f32;
        let elements = sequence_len * heads * head_dim;
        let input_values = (0..elements)
            .map(|index| (index as f32 - 11.0) / 7.0)
            .collect::<Vec<_>>();
        let expected = expected_rope(
            &input_values,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rope_f32(
            &input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn first_hip_qwen35_qk_norm_rope_f32_matches_split_norm_rope_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = q_heads * head_dim;
        let k_output_elements = kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-4);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-4);
    }

    #[test]
    fn first_hip_qwen35_qk_norm_rope_batch_f32_matches_split_norm_rope_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let sequence_len = 3_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..sequence_len * q_heads * head_dim * 2)
            .map(|index| (index as f32 - 17.0) / 19.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| (index as f32 + 11.0) / -13.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = sequence_len * q_heads * head_dim;
        let k_output_elements = sequence_len * kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_batch_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            sequence_len,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let q_projected_stride = q_heads * head_dim * 2;
        let mut expected_q_gate = Vec::with_capacity(q_output_elements);
        let mut expected_q_query = Vec::with_capacity(q_output_elements);
        for token in 0..sequence_len {
            for head in 0..q_heads {
                let source_base = token * q_projected_stride + head * 2 * head_dim;
                expected_q_query
                    .extend_from_slice(&q_projected_values[source_base..source_base + head_dim]);
                expected_q_gate.extend_from_slice(
                    &q_projected_values[source_base + head_dim..source_base + 2 * head_dim],
                );
            }
        }
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            sequence_len,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            sequence_len,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-4);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-4);
    }

    #[test]
    fn first_hip_qwen35_qk_norm_rope_paged_kv_write_f32_matches_split_norm_rope_and_paged_write_when_available()
     {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let value_dim = 5_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let cache_position = 5_usize;
        let block_size = 4_usize;
        let cache_blocks = 2_usize;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let v_projected_values = (0..kv_heads * value_dim)
            .map(|index| (index as f32 - 2.0) / 7.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let block_table_values = vec![1_u32, 0_u32];
        let physical_tokens = cache_blocks * block_size;
        let q_output_elements = q_heads * head_dim;
        let k_cache_elements = physical_tokens * kv_heads * head_dim;
        let v_cache_elements = physical_tokens * kv_heads * value_dim;
        let initial_k_cache = (0..k_cache_elements)
            .map(|index| -10.0_f32 - index as f32)
            .collect::<Vec<_>>();
        let initial_v_cache = (0..v_cache_elements)
            .map(|index| 20.0_f32 + index as f32)
            .collect::<Vec<_>>();

        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_projected = context
            .alloc_buffer(v_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(k_cache_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(v_cache_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        v_projected
            .copy_from_host(0, &f32s_to_le_bytes(&v_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        k_cache
            .copy_from_host(0, &f32s_to_le_bytes(&initial_k_cache), Some(&mut stream))
            .unwrap();
        v_cache
            .copy_from_host(0, &f32s_to_le_bytes(&initial_v_cache), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_paged_kv_write_f32(
            &q_projected,
            &k_projected,
            &v_projected,
            &q_weight,
            &k_weight,
            &block_table,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            cache_blocks,
            &mut q_gate,
            &mut q_rope,
            &mut k_cache,
            &mut v_cache,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut expected_k_cache = initial_k_cache;
        let mut expected_v_cache = initial_v_cache;
        let physical_timestep = block_table_values[cache_position / block_size] as usize
            * block_size
            + (cache_position % block_size);
        for kv_head in 0..kv_heads {
            let k_src = kv_head * head_dim;
            let k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            expected_k_cache[k_dst..k_dst + head_dim]
                .copy_from_slice(&expected_k_rope[k_src..k_src + head_dim]);
            let v_src = kv_head * value_dim;
            let v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            expected_v_cache[v_dst..v_dst + value_dim]
                .copy_from_slice(&v_projected_values[v_src..v_src + value_dim]);
        }

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_cache_bytes = vec![0_u8; k_cache_elements * std::mem::size_of::<f32>()];
        let mut v_cache_bytes = vec![0_u8; v_cache_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_cache
            .copy_to_host(0, &mut k_cache_bytes, Some(&mut stream))
            .unwrap();
        v_cache
            .copy_to_host(0, &mut v_cache_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-4);
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-4);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-5);
    }

    #[test]
    fn first_hip_causal_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32(
            &q,
            &k,
            &v,
            sequence_len,
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
    fn first_hip_causal_attn_f32_flash2_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32_flash2(
            &q,
            &k,
            &v,
            sequence_len,
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
    fn first_hip_causal_attn_batch_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let batch_count = 2_usize;
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_elements_per_batch = sequence_len * q_heads * head_dim;
        let k_elements_per_batch = sequence_len * kv_heads * head_dim;
        let v_elements_per_batch = sequence_len * kv_heads * value_dim;
        let q_values = (0..batch_count * q_elements_per_batch)
            .map(|index| (index as f32 - 19.0) / 23.0)
            .collect::<Vec<_>>();
        let k_values = (0..batch_count * k_elements_per_batch)
            .map(|index| ((index * 3) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();
        let v_values = (0..batch_count * v_elements_per_batch)
            .map(|index| ((index * 5) as f32 - 11.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for batch in 0..batch_count {
            expected.extend(expected_causal_attn(
                &q_values[batch * q_elements_per_batch..(batch + 1) * q_elements_per_batch],
                &k_values[batch * k_elements_per_batch..(batch + 1) * k_elements_per_batch],
                &v_values[batch * v_elements_per_batch..(batch + 1) * v_elements_per_batch],
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            ));
        }
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_batch_f32(
            &q,
            &k,
            &v,
            batch_count,
            sequence_len,
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
    fn first_hip_causal_attn_batch_f32_flash2_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let batch_count = 2_usize;
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_elements_per_batch = sequence_len * q_heads * head_dim;
        let k_elements_per_batch = sequence_len * kv_heads * head_dim;
        let v_elements_per_batch = sequence_len * kv_heads * value_dim;
        let q_values = (0..batch_count * q_elements_per_batch)
            .map(|index| (index as f32 - 19.0) / 23.0)
            .collect::<Vec<_>>();
        let k_values = (0..batch_count * k_elements_per_batch)
            .map(|index| ((index * 3) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();
        let v_values = (0..batch_count * v_elements_per_batch)
            .map(|index| ((index * 5) as f32 - 11.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for batch in 0..batch_count {
            expected.extend(expected_causal_attn(
                &q_values[batch * q_elements_per_batch..(batch + 1) * q_elements_per_batch],
                &k_values[batch * k_elements_per_batch..(batch + 1) * k_elements_per_batch],
                &v_values[batch * v_elements_per_batch..(batch + 1) * v_elements_per_batch],
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            ));
        }
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_batch_f32_flash2(
            &q,
            &k,
            &v,
            batch_count,
            sequence_len,
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
    fn first_hip_decode_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..cache_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..cache_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_decode_attn(
            &q_values,
            &k_values,
            &v_values,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        decode_attn_f32(
            &q,
            &k,
            &v,
            cache_len,
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
    fn first_hip_cached_prefix_attn_f32_flash2_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_values,
            &v_values,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_f32_flash2(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
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
    fn first_hip_cached_prefix_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_values,
            &v_values,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_f32(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
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
    fn first_hip_cached_prefix_attn_fp8_e4m3_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
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
    fn first_hip_cached_prefix_attn_fp8_e4m3_flash2_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_values,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_flash2(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
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
    fn first_hip_cached_prefix_attn_fp8_e4m3_flash2_fp8q_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cached_prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let (q_fp8, q_scale, q_expected) = fp8_e4m3_quantize(&q_values);
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_expected,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context.alloc_buffer(q_fp8.len()).unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &q_fp8, Some(&mut stream)).unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_flash2_fp8q(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
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
    fn first_hip_cached_prefix_attn_fp8_e4m3_rocwmma_computes_expected_values_when_available() {
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
        let cached_prefix_len = 3_usize;
        let new_tokens = 2_usize;
        let total_context = cached_prefix_len + new_tokens;
        let q_heads = 32_usize;
        let kv_heads = 2_usize;
        let head_dim = 32_usize;
        let value_dim = 32_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..new_tokens * q_heads * head_dim)
            .map(|index| (index as f32 - 97.0) / 211.0)
            .collect::<Vec<_>>();
        let k_values = (0..total_context * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 113.0) / 257.0)
            .collect::<Vec<_>>();
        let v_values = (0..total_context * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 131.0) / 293.0)
            .collect::<Vec<_>>();
        let (q_fp8, q_scale, q_expected) = fp8_e4m3_quantize(&q_values);
        let (k_fp8, k_scale, k_expected) = fp8_e4m3_quantize(&k_values);
        let (v_fp8, v_scale, v_expected) = fp8_e4m3_quantize(&v_values);
        let expected = expected_cached_prefix_attn(
            &q_expected,
            &k_expected,
            &v_expected,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context.alloc_buffer(q_fp8.len()).unwrap();
        let mut k = context.alloc_buffer(k_fp8.len()).unwrap();
        let mut v = context.alloc_buffer(v_fp8.len()).unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &q_fp8, Some(&mut stream)).unwrap();
        k.copy_from_host(0, &k_fp8, Some(&mut stream)).unwrap();
        v.copy_from_host(0, &v_fp8, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        cached_prefix_attn_fp8_e4m3_rocwmma(
            &q,
            &k,
            &v,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
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
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 5e-4);
    }

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
}
