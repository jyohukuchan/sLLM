    static AQ4_EXPERIMENTAL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct ExperimentalEnvGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl ExperimentalEnvGuard {
        fn new(name: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(name).ok();
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Self { name, previous }
        }
    }

    impl Drop for ExperimentalEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.previous.as_deref() {
                    Some(value) => std::env::set_var(self.name, value),
                    None => std::env::remove_var(self.name),
                }
            }
        }
    }

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
    fn aq4_batch_dispatch_classifier_is_conservative_on_cpu_and_ragged_shapes() {
        const LDS_ENV: &str = "ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM";
        const REGISTER_ENV: &str = "ULLM_EXPERIMENTAL_HIP_AQ4_REGISTER_BM";
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let gfx1201_devices: Vec<u32> = (1..device_count().unwrap())
            .filter(|&device_index| {
                device_info(device_index)
                    .map(|info| info.gcn_arch_name == "gfx1201")
                    .unwrap_or(false)
            })
            .collect();

        {
            let _lds = ExperimentalEnvGuard::new(LDS_ENV, None);
            let _register = ExperimentalEnvGuard::new(REGISTER_ENV, None);
            assert_eq!(
                aq4_matvec_batch_dispatch_kind_for_shape(0, 16, 32, 128, 8),
                Aq4MatvecBatchDispatchKind::Legacy
            );
            for &device_index in &gfx1201_devices {
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 8),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
            }
        }

        {
            let _lds = ExperimentalEnvGuard::new(LDS_ENV, Some("1"));
            let _register = ExperimentalEnvGuard::new(REGISTER_ENV, None);
            for &device_index in &gfx1201_devices {
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 8),
                    Aq4MatvecBatchDispatchKind::TiledLdsBm8
                );
                assert!(aq4_matvec_batch_dispatch_tiled_for_shape(
                    device_index,
                    16,
                    32,
                    128,
                    8
                ));
            }
        }

        for (value, expected) in [
            ("4", Aq4MatvecBatchDispatchKind::RegisterBm4),
            ("8", Aq4MatvecBatchDispatchKind::RegisterBm8),
        ] {
            let _lds = ExperimentalEnvGuard::new(LDS_ENV, Some("1"));
            let _register = ExperimentalEnvGuard::new(REGISTER_ENV, Some(value));
            for &device_index in &gfx1201_devices {
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 8),
                    expected
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 2),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 3),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 4),
                    if value == "4" {
                        Aq4MatvecBatchDispatchKind::RegisterBm4
                    } else {
                        Aq4MatvecBatchDispatchKind::Legacy
                    }
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 8, 32, 128, 8),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 31, 128, 8),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 127, 8),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
            }
        }

        {
            let _lds = ExperimentalEnvGuard::new(LDS_ENV, Some("1"));
            let _register = ExperimentalEnvGuard::new(REGISTER_ENV, Some("invalid"));
            for &device_index in &gfx1201_devices {
                assert_eq!(
                    aq4_matvec_batch_dispatch_kind_for_shape(device_index, 16, 32, 128, 8),
                    Aq4MatvecBatchDispatchKind::Legacy
                );
            }
        }
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
    fn cpu_buffer_zero_clears_only_requested_range() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(32).unwrap();
        buffer
            .copy_from_host(0, &[0xa5_u8; 32], Some(&mut stream))
            .unwrap();
        buffer.zero(7, 19, Some(&mut stream)).unwrap();
        buffer.zero(32, 0, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = [0_u8; 32];
        buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(&output[..7], &[0xa5_u8; 7]);
        assert_eq!(&output[7..26], &[0_u8; 19]);
        assert_eq!(&output[26..], &[0xa5_u8; 6]);
    }

    #[test]
    fn buffer_zero_rejects_out_of_bounds_range() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut buffer = context.alloc_buffer(8).unwrap();
        let err = buffer.zero(7, 2, None).unwrap_err();
        assert!(err.contains("out of bounds"), "{err}");
    }

    #[test]
    fn cpu_buffer_copies_between_runtime_buffers() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut src = context.alloc_buffer(64).unwrap();
        let mut dst = context.alloc_buffer(64).unwrap();
        let input: Vec<u8> = (0..64).map(|value| (value * 13 + 5) as u8).collect();
        src.copy_from_host(0, &input, Some(&mut stream)).unwrap();
        dst.copy_from_host(0, &[0xff_u8; 64], Some(&mut stream))
            .unwrap();

        dst.copy_from_buffer(8, &src, 17, 23, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; 64];
        dst.copy_to_host(0, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(&output[..8], &[0xff_u8; 8]);
        assert_eq!(&output[8..31], &input[17..40]);
        assert_eq!(&output[31..], &[0xff_u8; 33]);
    }

    #[test]
    fn cpu_buffer_copy_rejects_out_of_bounds_source_range() {
        let mut context = RuntimeContext::create(0).unwrap();
        let src = context.alloc_buffer(8).unwrap();
        let mut dst = context.alloc_buffer(8).unwrap();
        let err = dst.copy_from_buffer(0, &src, 7, 2, None).unwrap_err();
        assert!(err.contains("out of bounds"));
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
    fn cpu_sq_fp8_matvec_block2d_f32_shares_scales_across_adjacent_rows() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload = context.alloc_buffer(9).unwrap();
        let mut scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        payload
            .copy_from_host(0, &[0x38; 9], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 3.0, 5.0, 7.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0, 1.0, 1.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let path = sq_fp8_matvec_block2d_f32(
            &payload,
            &scales,
            &input,
            3,
            3,
            2,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(path, SqFp8ExecutionPath::CpuReference);
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 3 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![7.0, 7.0, 17.0]);
    }

    #[test]
    fn cpu_sq_fp8_matvec_block2d_batch_f32_reports_reference_path() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload = context.alloc_buffer(9).unwrap();
        let mut scales = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();

        payload
            .copy_from_host(0, &[0x38; 9], Some(&mut stream))
            .unwrap();
        scales
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[2.0, 3.0, 5.0, 7.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        let path = sq_fp8_matvec_block2d_batch_f32(
            &payload,
            &scales,
            &input,
            3,
            3,
            2,
            2,
            2,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(path, SqFp8ExecutionPath::CpuReference);
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 6 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            le_bytes_to_f32s(&output_bytes),
            vec![7.0, 7.0, 17.0, 5.0, 5.0, 12.0]
        );
    }

    #[test]
    fn sq_fp8_matvec_block2d_f32_rejects_invalid_shapes_and_buffers() {
        let mut context = RuntimeContext::create(0).unwrap();
        let payload = context.alloc_buffer(9).unwrap();
        let scales = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let zero_block_error = sq_fp8_matvec_block2d_f32(
            &payload,
            &scales,
            &input,
            3,
            3,
            0,
            2,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(zero_block_error.contains("block rows and cols must be greater than zero"));

        let scale_buffer_error = sq_fp8_matvec_block2d_f32(
            &payload,
            &scales,
            &input,
            3,
            3,
            2,
            2,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(scale_buffer_error.contains("out of bounds"));

        let matrix_overflow_error = sq_fp8_matvec_block2d_f32(
            &payload,
            &scales,
            &input,
            usize::MAX,
            2,
            2,
            2,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(matrix_overflow_error.contains("matrix element count overflows"));

        let batch_overflow_error = sq_fp8_matvec_block2d_batch_f32(
            &payload,
            &scales,
            &input,
            3,
            3,
            2,
            2,
            usize::MAX,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(batch_overflow_error.contains("input element count overflows"));
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
    fn paged_decode_attn_split_workspace_bytes_checks_boundaries_and_overflow() {
        assert_eq!(paged_decode_attn_split_workspace_bytes(1, 1, 1, 1).unwrap(), 12);
        assert_eq!(
            paged_decode_attn_split_workspace_bytes(4, 256, 513, 256).unwrap(),
            4 * 3 * 258 * std::mem::size_of::<f32>()
        );
        for args in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            assert!(paged_decode_attn_split_workspace_bytes(args.0, args.1, args.2, args.3)
                .is_err());
        }
        assert!(paged_decode_attn_split_workspace_bytes(usize::MAX, 1, 2, 1).is_err());
        assert!(paged_decode_attn_split_workspace_bytes(1, usize::MAX, 1, 1).is_err());
    }

    #[test]
    fn cpu_paged_decode_attn_split_f32_plain_and_gated_match_expected() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 5_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let source_tile = 2_usize;
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
        let workspace_bytes = paged_decode_attn_split_workspace_bytes(
            q_heads,
            value_dim,
            cache_len,
            source_tile,
        )
        .unwrap();

        let mut q = context.alloc_buffer(q_values.len() * 4).unwrap();
        let mut gate = context.alloc_buffer(gate_values.len() * 4).unwrap();
        let mut k_cache = context.alloc_buffer(k_cache_values.len() * 4).unwrap();
        let mut v_cache = context.alloc_buffer(v_cache_values.len() * 4).unwrap();
        let mut block_table = context.alloc_buffer(block_table_values.len() * 4).unwrap();
        let mut workspace = context.alloc_buffer(workspace_bytes).unwrap();
        let mut plain_output = context.alloc_buffer(expected_plain.len() * 4).unwrap();
        let mut gated_output = context.alloc_buffer(expected_gated.len() * 4).unwrap();
        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream)).unwrap();
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream)).unwrap();
        k_cache.copy_from_host(0, &f32s_to_le_bytes(&k_cache_values), Some(&mut stream)).unwrap();
        v_cache.copy_from_host(0, &f32s_to_le_bytes(&v_cache_values), Some(&mut stream)).unwrap();
        block_table.copy_from_host(0, &u32s_to_le_bytes(&block_table_values), Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        paged_decode_attn_split_f32(&q, &k_cache, &v_cache, &block_table, cache_len, block_size, cache_blocks, q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile, &mut workspace, &mut plain_output, Some(&mut stream)).unwrap();
        paged_decode_attn_split_sigmoid_gate_f32(&q, &gate, &k_cache, &v_cache, &block_table, cache_len, block_size, cache_blocks, q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile, &mut workspace, &mut gated_output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut plain_bytes = vec![0_u8; expected_plain.len() * 4];
        let mut gated_bytes = vec![0_u8; expected_gated.len() * 4];
        plain_output.copy_to_host(0, &mut plain_bytes, Some(&mut stream)).unwrap();
        gated_output.copy_to_host(0, &mut gated_bytes, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&plain_bytes), &expected_plain, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&gated_bytes), &expected_gated, 1e-5);
    }

    #[test]
    fn cpu_paged_decode_attn_split_f32_rejects_invalid_workspace_and_source_tile() {
        let mut cpu = RuntimeContext::create(0).unwrap();
        let q = cpu.alloc_buffer(4).unwrap();
        let k_cache = cpu.alloc_buffer(4).unwrap();
        let v_cache = cpu.alloc_buffer(4).unwrap();
        let mut block_table = cpu.alloc_buffer(4).unwrap();
        let mut short_workspace = cpu.alloc_buffer(11).unwrap();
        let mut output = cpu.alloc_buffer(4).unwrap();
        let error = paged_decode_attn_split_f32(&q, &k_cache, &v_cache, &block_table, 1, 1, 1, 1, 1, 1, 1, 1.0, 1, &mut short_workspace, &mut output, None).unwrap_err();
        assert!(error.contains("out of bounds") || error.contains("workspace"), "{error}");
        let mut workspace = cpu.alloc_buffer(12).unwrap();
        let error = paged_decode_attn_split_f32(&q, &k_cache, &v_cache, &block_table, 1, 1, 1, 1, 1, 1, 1, 1.0, 0, &mut workspace, &mut output, None).unwrap_err();
        assert!(error.contains("source_tile"), "{error}");
        block_table.copy_from_host(0, &u32s_to_le_bytes(&[u32::MAX]), None).unwrap();
        let error = paged_decode_attn_split_f32(&q, &k_cache, &v_cache, &block_table, 1, 1, 1, 1, 1, 1, 1, 1.0, 1, &mut workspace, &mut output, None).unwrap_err();
        assert!(error.contains("block table"), "{error}");
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
    fn cpu_aq4_matvec_batch_invalid_scale_index_fails_before_output_mutation() {
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
        index.copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream)).unwrap();
        // The last group points beyond scale_count and must be rejected before any output write.
        scale.copy_from_host(0, &[0_u8, 1, 2], Some(&mut stream)).unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&(0..16).map(|value| value as f32).collect::<Vec<_>>()),
                Some(&mut stream),
            )
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
        let sentinel = f32s_to_le_bytes(&[13.0, -7.0, 3.5, 99.0]);
        output.copy_from_host(0, &sentinel, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let error = aq4_matvec_batch_f32(
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
        .unwrap_err();
        assert!(error.contains("scale index"), "{error}");
        let mut output_bytes = vec![0_u8; sentinel.len()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output_bytes, sentinel);
    }

    #[test]
    fn cpu_aq4_register_bm8_batch_rejects_without_fallback_or_output_mutation() {
        let rows = 32_usize;
        let cols = 128_usize;
        let batch_count = 8_usize;
        let elements = rows * cols;
        let groups = elements / 16;
        let scale_count = 7_usize;
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(elements / 2).unwrap();
        let mut scale = context.alloc_buffer(groups).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(scale_count * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(batch_count * cols * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
            .unwrap();
        index.zero(0, elements / 2, Some(&mut stream)).unwrap();
        scale.zero(0, groups, Some(&mut stream)).unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&(0..16).map(|value| value as f32).collect::<Vec<_>>()),
                Some(&mut stream),
            )
            .unwrap();
        scale_values
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .zero(
                0,
                batch_count * cols * std::mem::size_of::<f32>(),
                Some(&mut stream),
            )
            .unwrap();
        let sentinel = vec![0x5a_u8; batch_count * rows * std::mem::size_of::<f32>()];
        output.copy_from_host(0, &sentinel, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        for (group_size, test_rows, test_cols, test_batch) in [
            (8_usize, rows, cols, batch_count),
            (16, rows - 1, cols, batch_count),
            (16, rows, cols - 1, batch_count),
            (16, rows, cols, batch_count - 1),
        ] {
            let error = aq4_matvec_batch_register_bm8_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                scale_count,
                group_size,
                0.75,
                0,
                test_rows,
                test_cols,
                test_batch,
                &mut output,
                Some(&mut stream),
            )
            .unwrap_err();
            assert!(
                error.contains("requires") || error.contains("at least 8"),
                "{error}"
            );
        }

        let error = aq4_matvec_batch_register_bm8_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            scale_count,
            16,
            0.75,
            0,
            rows,
            cols,
            batch_count,
            &mut output,
            Some(&mut stream),
        )
        .unwrap_err();
        assert!(error.contains("HIP gfx1201"), "{error}");

        let mut output_bytes = vec![0_u8; sentinel.len()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output_bytes, sentinel);
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
