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
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_GROUP8_RAGGED_M_DIFFERENTIAL=1"]
fn hip_aq4_wmma_group8_ragged_m_exact_buffers_match_cpu_when_enabled() {
    assert_eq!(std::env::var("ULLM_RUN_AQ4_WMMA_GROUP8_RAGGED_M_DIFFERENTIAL").as_deref(), Ok("1"));
    let device_index = (1..device_count().unwrap()).find(|&candidate| device_info(candidate).is_ok_and(|info| info.gcn_arch_name == "gfx1201")).expect("isolated gfx1201 HIP device");
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    const ROWS: usize = 16; const COLS: usize = 4096; const SCALES: usize = 7;
    let mut seed = 0x510e_527f_u32; let mut next = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed as f32 / u32::MAX as f32 };
    let indices: Vec<u8> = (0..ROWS * COLS / 2).map(|_| ((next() * 16.0) as u8 & 15) | (((next() * 16.0) as u8 & 15) << 4)).collect(); let scales: Vec<u8> = (0..ROWS * COLS / 8).map(|_| ((next() * SCALES as f32) as u8).min(6)).collect(); let codebook: Vec<f32> = (0..16).map(|_| next() - 0.5).collect(); let values: Vec<f32> = (0..SCALES).map(|_| 0.5 + next() * 0.5).collect(); let row_scales: Vec<f32> = (0..ROWS).map(|_| 0.75 + next() * 0.5).collect();
    for &m in &[127_usize, 113, 65, 8, 1, 128] {
        let input: Vec<f32> = (0..m * COLS).map(|_| next() * 2.0 - 1.0).collect(); assert_eq!(input.len(), input.capacity());
        let mut cpu = RuntimeContext::create(0).unwrap(); let mut cs = cpu.create_stream().unwrap(); let mut ci = cpu.alloc_buffer(indices.len()).unwrap(); let mut cg = cpu.alloc_buffer(scales.len()).unwrap(); let mut cc = cpu.alloc_buffer(64).unwrap(); let mut cv = cpu.alloc_buffer(SCALES * 4).unwrap(); let mut cx = cpu.alloc_buffer(input.len() * 4).unwrap(); let mut cr = cpu.alloc_buffer(ROWS * 4).unwrap(); let mut co = cpu.alloc_buffer(m * ROWS * 4).unwrap();
        ci.copy_from_host(0, &indices, Some(&mut cs)).unwrap(); cg.copy_from_host(0, &scales, Some(&mut cs)).unwrap(); cc.copy_from_host(0, &f32s_to_le_bytes(&codebook), Some(&mut cs)).unwrap(); cv.copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut cs)).unwrap(); cx.copy_from_host(0, &f32s_to_le_bytes(&input), Some(&mut cs)).unwrap(); cr.copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cs)).unwrap(); aq4_matvec_batch_f32(&ci, &cg, &cc, &cv, &cx, Some(&cr), SCALES, 8, 0.75, ROWS, ROWS, COLS, m, &mut co, Some(&mut cs)).unwrap(); cs.synchronize().unwrap(); let mut expected = vec![0; m * ROWS * 4]; co.copy_to_host(0, &mut expected, Some(&mut cs)).unwrap();
        let mut hip = RuntimeContext::create(device_index).unwrap(); let mut hs = hip.create_stream().unwrap(); let mut hi = hip.alloc_buffer(indices.len()).unwrap(); let mut hg = hip.alloc_buffer(scales.len()).unwrap(); let mut hc = hip.alloc_buffer(64).unwrap(); let mut hv = hip.alloc_buffer(SCALES * 4).unwrap(); let mut hx = hip.alloc_buffer(input.len() * 4).unwrap(); let mut hr = hip.alloc_buffer(ROWS * 4).unwrap(); let mut ho = hip.alloc_buffer(m * ROWS * 4).unwrap(); assert_eq!(hx.size().unwrap(), input.len() * 4); assert_eq!(ho.size().unwrap(), m * ROWS * 4); hi.copy_from_host(0, &indices, Some(&mut hs)).unwrap(); hg.copy_from_host(0, &scales, Some(&mut hs)).unwrap(); hc.copy_from_host(0, &f32s_to_le_bytes(&codebook), Some(&mut hs)).unwrap(); hv.copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut hs)).unwrap(); hx.copy_from_host(0, &f32s_to_le_bytes(&input), Some(&mut hs)).unwrap(); hr.copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hs)).unwrap(); aq4_matvec_batch_wmma_group8_ragged_m_prototype_f32(&hi, &hg, &hc, &hv, &hx, Some(&hr), SCALES, 8, 0.75, ROWS, ROWS, COLS, m, &mut ho, Some(&mut hs)).unwrap(); hs.synchronize().unwrap(); let mut actual = vec![0; m * ROWS * 4]; ho.copy_to_host(0, &mut actual, Some(&mut hs)).unwrap(); for (a, e) in le_bytes_to_f32s(&actual).iter().zip(le_bytes_to_f32s(&expected)) { assert!((a - e).abs() <= 0.05 + 0.01 * e.abs()); }
        if m == 128 { let mut production = hip.alloc_buffer(m * ROWS * 4).unwrap(); aq4_matvec_batch_wmma_group8_prototype_f32(&hi, &hg, &hc, &hv, &hx, Some(&hr), SCALES, 8, 0.75, ROWS, ROWS, COLS, m, &mut production, Some(&mut hs)).unwrap(); hs.synchronize().unwrap(); let mut bytes = vec![0; m * ROWS * 4]; production.copy_to_host(0, &mut bytes, Some(&mut hs)).unwrap(); assert_f32s_close(&le_bytes_to_f32s(&actual), &le_bytes_to_f32s(&bytes), 1e-5); }
    }
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
    fn first_hip_sq_fp8_matvec_block2d_paths_use_native_kernels_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
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
        assert_eq!(path, SqFp8ExecutionPath::HipKernel);
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 3 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &[7.0, 7.0, 17.0], 1e-6);

        let batch_path = sq_fp8_matvec_block2d_batch_f32(
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
        assert_eq!(batch_path, SqFp8ExecutionPath::HipKernel);
        stream.synchronize().unwrap();

        let mut batch_output_bytes = vec![0_u8; 6 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut batch_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(
            &le_bytes_to_f32s(&batch_output_bytes),
            &[7.0, 7.0, 17.0, 5.0, 5.0, 12.0],
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
    fn first_hip_aq4_register_and_forced_bm8_match_cpu_when_available() {
        let selected_device = (1..device_count().unwrap()).find(|&device_index| {
            device_info(device_index)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        });
        let Some(device_index) = selected_device else {
            return;
        };
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut seed = 0x9e37_79b9_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        for register_bm in [4_usize, 8_usize] {
            let _lds = ExperimentalEnvGuard::new("ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM", None);
            let _register = ExperimentalEnvGuard::new(
                "ULLM_EXPERIMENTAL_HIP_AQ4_REGISTER_BM",
                if register_bm == 4 { Some("4") } else { None },
            );
            for &batch_count in &[4_usize, 8, 16, 32, 64, 127, 128] {
                for &rows in &[32_usize, 64] {
                    for &cols in &[128_usize, 256] {
                        let expected_dispatch = if register_bm == 8 || batch_count < register_bm {
                            Aq4MatvecBatchDispatchKind::Legacy
                        } else {
                            Aq4MatvecBatchDispatchKind::RegisterBm4
                        };
                        assert_eq!(
                            aq4_matvec_batch_dispatch_kind_for_shape(
                                device_index,
                                16,
                                rows,
                                cols,
                                batch_count
                            ),
                            expected_dispatch
                        );
            let elements = rows * cols;
            let index_bytes = (elements + 1) / 2;
            let groups = elements / 16;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16)
                .map(|_| next_unit() * 1.0 - 0.5)
                .collect();
            let scale_values: Vec<f32> = (0..scale_count)
                .map(|_| 0.5 + next_unit() * 0.5)
                .collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            let use_row_scales = (batch_count + rows + cols) % 2 == 0;
            let row_scales: Vec<f32> = (0..rows)
                .map(|_| 0.75 + next_unit() * 0.5)
                .collect();
            let row_scale_count = if use_row_scales { rows } else { 0 };
            let tensor_scale = 0.75_f32;

            let mut cpu_context = RuntimeContext::create(0).unwrap();
            let mut cpu_stream = cpu_context.create_stream().unwrap();
            let mut cpu_index = cpu_context.alloc_buffer(index_bytes).unwrap();
            let mut cpu_scale = cpu_context.alloc_buffer(groups).unwrap();
            let mut cpu_codebook = cpu_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_scale_values = cpu_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_input = cpu_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_row_scale = if use_row_scales {
                Some(cpu_context.alloc_buffer(rows * std::mem::size_of::<f32>()).unwrap())
            } else {
                None
            };
            let mut cpu_output = cpu_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            cpu_index.copy_from_host(0, &indices, Some(&mut cpu_stream)).unwrap();
            cpu_scale
                .copy_from_host(0, &scale_indices, Some(&mut cpu_stream))
                .unwrap();
            cpu_codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut cpu_stream))
                .unwrap();
            cpu_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut cpu_stream))
                .unwrap();
            cpu_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut cpu_stream))
                .unwrap();
            if let Some(row_scale) = cpu_row_scale.as_mut() {
                row_scale
                    .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cpu_stream))
                    .unwrap();
            }
            cpu_stream.synchronize().unwrap();
            aq4_matvec_batch_f32(
                &cpu_index,
                &cpu_scale,
                &cpu_codebook,
                &cpu_scale_values,
                &cpu_input,
                cpu_row_scale.as_ref(),
                scale_count,
                16,
                tensor_scale,
                row_scale_count,
                rows,
                cols,
                batch_count,
                &mut cpu_output,
                Some(&mut cpu_stream),
            )
            .unwrap();
            cpu_stream.synchronize().unwrap();
            let mut cpu_output_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
            cpu_output
                .copy_to_host(0, &mut cpu_output_bytes, Some(&mut cpu_stream))
                .unwrap();
            cpu_stream.synchronize().unwrap();
            let expected = le_bytes_to_f32s(&cpu_output_bytes);

            let mut hip_context = RuntimeContext::create(device_index).unwrap();
            let mut hip_stream = hip_context.create_stream().unwrap();
            let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
            let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
            let mut hip_codebook = hip_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_scale_values = hip_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_input = hip_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_row_scale = if use_row_scales {
                Some(hip_context.alloc_buffer(rows * std::mem::size_of::<f32>()).unwrap())
            } else {
                None
            };
            let mut hip_output = hip_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            hip_index.copy_from_host(0, &indices, Some(&mut hip_stream)).unwrap();
            hip_scale
                .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                .unwrap();
            hip_codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut hip_stream))
                .unwrap();
            hip_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                .unwrap();
            hip_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                .unwrap();
            if let Some(row_scale) = hip_row_scale.as_mut() {
                row_scale
                    .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                    .unwrap();
            }
            hip_stream.synchronize().unwrap();
            if register_bm == 8 && batch_count >= 8 {
                aq4_matvec_batch_register_bm8_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    hip_row_scale.as_ref(),
                    scale_count,
                    16,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            } else {
                aq4_matvec_batch_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    hip_row_scale.as_ref(),
                    scale_count,
                    16,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            let mut hip_output_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
            hip_output
                .copy_to_host(0, &mut hip_output_bytes, Some(&mut hip_stream))
                .unwrap();
            hip_stream.synchronize().unwrap();
            assert_f32s_close(
                &le_bytes_to_f32s(&hip_output_bytes),
                &expected,
                1e-4,
            );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_REGISTER_BM8_GROUP8_DIFFERENTIAL=1"]
    fn hip_aq4_register_bm8_group8_production_shapes_match_cpu_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_REGISTER_BM8_GROUP8_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_REGISTER_BM8_GROUP8_DIFFERENTIAL=1 before running this GPU differential test"
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

        let mut seed = 0x6a09_e667_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        for &(rows, batch_count) in &[
            (1024_usize, 8_usize),
            (1024, 64),
            (1024, 128),
            (4096, 8),
            (4096, 64),
            (4096, 128),
        ] {
            let cols = 4096_usize;
            let elements = rows * cols;
            let index_bytes = elements / 2;
            let groups = elements / 8;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            let use_row_scales = batch_count != 64;
            let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            let row_scale_count = if use_row_scales { rows } else { 0 };
            let tensor_scale = 0.75_f32;

            let mut cpu_context = RuntimeContext::create(0).unwrap();
            let mut cpu_stream = cpu_context.create_stream().unwrap();
            let mut cpu_index = cpu_context.alloc_buffer(index_bytes).unwrap();
            let mut cpu_scale = cpu_context.alloc_buffer(groups).unwrap();
            let mut cpu_codebook = cpu_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_scale_values = cpu_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_input = cpu_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut cpu_row_scale = if use_row_scales {
                Some(
                    cpu_context
                        .alloc_buffer(rows * std::mem::size_of::<f32>())
                        .unwrap(),
                )
            } else {
                None
            };
            let mut cpu_output = cpu_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            cpu_index
                .copy_from_host(0, &indices, Some(&mut cpu_stream))
                .unwrap();
            cpu_scale
                .copy_from_host(0, &scale_indices, Some(&mut cpu_stream))
                .unwrap();
            cpu_codebook
                .copy_from_host(
                    0,
                    &f32s_to_le_bytes(&codebook_values),
                    Some(&mut cpu_stream),
                )
                .unwrap();
            cpu_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut cpu_stream))
                .unwrap();
            cpu_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut cpu_stream))
                .unwrap();
            if let Some(row_scale) = cpu_row_scale.as_mut() {
                row_scale
                    .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cpu_stream))
                    .unwrap();
            }
            cpu_stream.synchronize().unwrap();
            aq4_matvec_batch_f32(
                &cpu_index,
                &cpu_scale,
                &cpu_codebook,
                &cpu_scale_values,
                &cpu_input,
                cpu_row_scale.as_ref(),
                scale_count,
                8,
                tensor_scale,
                row_scale_count,
                rows,
                cols,
                batch_count,
                &mut cpu_output,
                Some(&mut cpu_stream),
            )
            .unwrap();
            cpu_stream.synchronize().unwrap();
            let mut expected_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
            cpu_output
                .copy_to_host(0, &mut expected_bytes, Some(&mut cpu_stream))
                .unwrap();
            cpu_stream.synchronize().unwrap();
            let expected = le_bytes_to_f32s(&expected_bytes);

            let mut hip_context = RuntimeContext::create(device_index).unwrap();
            let mut hip_stream = hip_context.create_stream().unwrap();
            let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
            let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
            let mut hip_codebook = hip_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_scale_values = hip_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_input = hip_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_row_scale = if use_row_scales {
                Some(
                    hip_context
                        .alloc_buffer(rows * std::mem::size_of::<f32>())
                        .unwrap(),
                )
            } else {
                None
            };
            let mut hip_output = hip_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            hip_index
                .copy_from_host(0, &indices, Some(&mut hip_stream))
                .unwrap();
            hip_scale
                .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                .unwrap();
            hip_codebook
                .copy_from_host(
                    0,
                    &f32s_to_le_bytes(&codebook_values),
                    Some(&mut hip_stream),
                )
                .unwrap();
            hip_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                .unwrap();
            hip_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                .unwrap();
            if let Some(row_scale) = hip_row_scale.as_mut() {
                row_scale
                    .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                    .unwrap();
            }
            hip_stream.synchronize().unwrap();
            aq4_matvec_batch_register_bm8_group8_f32(
                &hip_index,
                &hip_scale,
                &hip_codebook,
                &hip_scale_values,
                &hip_input,
                hip_row_scale.as_ref(),
                scale_count,
                8,
                tensor_scale,
                row_scale_count,
                rows,
                cols,
                batch_count,
                &mut hip_output,
                Some(&mut hip_stream),
            )
            .unwrap();
            hip_stream.synchronize().unwrap();
            let mut actual_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
            hip_output
                .copy_to_host(0, &mut actual_bytes, Some(&mut hip_stream))
                .unwrap();
            hip_stream.synchronize().unwrap();
            assert_f32s_close(&le_bytes_to_f32s(&actual_bytes), &expected, 1e-3);
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_PROTOTYPE_DIFFERENTIAL=1"]
    fn hip_aq4_wmma_prototype_m128_group16_model_shapes_match_cpu_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_WMMA_PROTOTYPE_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_WMMA_PROTOTYPE_DIFFERENTIAL=1 before running this GPU differential test"
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

        let mut seed = 0x510e_527f_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        // This stable direct ABI is the one selected by HipAq4GemmWmmaF32. It compiles the
        // production double-buffered HIPRTC source, so this test covers the actual served kernel.
        for &(family, rows, cols, use_row_scales) in &[
            ("attn_q + linear_attn_qkv", 8_192_usize, 4_096, true),
            ("linear_attn_z", 4_096, 4_096, false),
            ("linear_attn_a + linear_attn_b", 32, 4_096, true),
            ("mlp_gate + mlp_up", 12_288, 4_096, true),
            ("mlp_down", 4_096, 12_288, false),
        ] {
            let batch_count = 128_usize;
            assert_eq!(rows % 16, 0, "{family} must fill WMMA row tiles");
            assert_eq!(cols % 32, 0, "{family} must fill Wide-K pairs");
            if rows == 32 {
                assert_eq!(rows / 16, 2, "{family} must launch exactly two row tiles");
            }
            let elements = rows * cols;
            let index_bytes = elements / 2;
            let groups = elements / 16;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            // Exercise both post-accumulation row-scale cases while covering every group16
            // production projection dimension. The false cases use the ABI's null-pointer path.
            let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            let row_scale_count = if use_row_scales { rows } else { 0 };
            let tensor_scale = 0.75_f32;

            let expected = {
                let mut cpu_context = RuntimeContext::create(0).unwrap();
                let mut cpu_stream = cpu_context.create_stream().unwrap();
                let mut cpu_index = cpu_context.alloc_buffer(index_bytes).unwrap();
                let mut cpu_scale = cpu_context.alloc_buffer(groups).unwrap();
                let mut cpu_codebook = cpu_context
                    .alloc_buffer(16 * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_scale_values = cpu_context
                    .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_input = cpu_context
                    .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_row_scale = if use_row_scales {
                    Some(
                        cpu_context
                            .alloc_buffer(rows * std::mem::size_of::<f32>())
                            .unwrap(),
                    )
                } else {
                    None
                };
                let mut cpu_output = cpu_context
                    .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                    .unwrap();
                cpu_index
                    .copy_from_host(0, &indices, Some(&mut cpu_stream))
                    .unwrap();
                cpu_scale
                    .copy_from_host(0, &scale_indices, Some(&mut cpu_stream))
                    .unwrap();
                cpu_codebook
                    .copy_from_host(
                        0,
                        &f32s_to_le_bytes(&codebook_values),
                        Some(&mut cpu_stream),
                    )
                    .unwrap();
                cpu_scale_values
                    .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut cpu_stream))
                    .unwrap();
                cpu_input
                    .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut cpu_stream))
                    .unwrap();
                if let Some(row_scale) = cpu_row_scale.as_mut() {
                    row_scale
                        .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cpu_stream))
                        .unwrap();
                }
                cpu_stream.synchronize().unwrap();
                aq4_matvec_batch_f32(
                    &cpu_index,
                    &cpu_scale,
                    &cpu_codebook,
                    &cpu_scale_values,
                    &cpu_input,
                    cpu_row_scale.as_ref(),
                    scale_count,
                    16,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut cpu_output,
                    Some(&mut cpu_stream),
                )
                .unwrap();
                cpu_stream.synchronize().unwrap();
                let mut expected_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
                cpu_output
                    .copy_to_host(0, &mut expected_bytes, Some(&mut cpu_stream))
                    .unwrap();
                cpu_stream.synchronize().unwrap();
                le_bytes_to_f32s(&expected_bytes)
            };

            let actual = {
                let mut hip_context = RuntimeContext::create(device_index).unwrap();
                let mut hip_stream = hip_context.create_stream().unwrap();
                let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
                let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
                let mut hip_codebook = hip_context
                    .alloc_buffer(16 * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_scale_values = hip_context
                    .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_input = hip_context
                    .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_row_scale = if use_row_scales {
                    Some(
                        hip_context
                            .alloc_buffer(rows * std::mem::size_of::<f32>())
                            .unwrap(),
                    )
                } else {
                    None
                };
                let mut hip_output = hip_context
                    .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                    .unwrap();
                hip_index
                    .copy_from_host(0, &indices, Some(&mut hip_stream))
                    .unwrap();
                hip_scale
                    .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                    .unwrap();
                hip_codebook
                    .copy_from_host(
                        0,
                        &f32s_to_le_bytes(&codebook_values),
                        Some(&mut hip_stream),
                    )
                    .unwrap();
                hip_scale_values
                    .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                    .unwrap();
                hip_input
                    .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                    .unwrap();
                if let Some(row_scale) = hip_row_scale.as_mut() {
                    row_scale
                        .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                        .unwrap();
                }
                hip_stream.synchronize().unwrap();
                aq4_matvec_batch_wmma_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    hip_row_scale.as_ref(),
                    scale_count,
                    16,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
                hip_stream.synchronize().unwrap();
                let mut actual_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
                hip_output
                    .copy_to_host(0, &mut actual_bytes, Some(&mut hip_stream))
                    .unwrap();
                hip_stream.synchronize().unwrap();
                le_bytes_to_f32s(&actual_bytes)
            };

            // Both decoded weights and activations are rounded to FP16 before WMMA, while the
            // accumulator is FP32. The bounded differential inputs make 0.05 absolute + 1%
            // relative error a deliberate FP16 staging allowance, not a general production
            // fidelity criterion; it remains far below a layout, transpose, or scale-factor bug.
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "family={family} rows={rows} cols={cols} index={index}: actual={actual} expected={expected} tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_PROTOTYPE_TIMING=1"]
    fn hip_aq4_wmma_prototype_m128_group16_model_shapes_timing_vs_register_bm8_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_WMMA_PROTOTYPE_TIMING").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_WMMA_PROTOTYPE_TIMING=1 before running this GPU timing test"
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

        const WARMUP_ITERATIONS: usize = 3;
        const TIMED_ITERATIONS: usize = 20;
        let mut seed = 0x1f83_d9ab_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        for &(family, rows, cols) in &[
            ("attn_q + linear_attn_qkv", 8_192_usize, 4_096),
            ("linear_attn_z", 4_096, 4_096),
            ("linear_attn_a + linear_attn_b", 32, 4_096),
            ("mlp_gate + mlp_up", 12_288, 4_096),
            ("mlp_down", 4_096, 12_288),
        ] {
            let batch_count = 128_usize;
            assert_eq!(rows % 16, 0, "{family} must fill WMMA row tiles");
            assert_eq!(cols % 32, 0, "{family} must fill Wide-K pairs");
            if rows == 32 {
                assert_eq!(rows / 16, 2, "{family} must launch exactly two row tiles");
            }
            let elements = rows * cols;
            let index_bytes = elements / 2;
            let groups = elements / 16;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            let tensor_scale = 0.75_f32;

            let mut hip_context = RuntimeContext::create(device_index).unwrap();
            let mut hip_stream = hip_context.create_stream().unwrap();
            let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
            let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
            let mut hip_codebook = hip_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_scale_values = hip_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_input = hip_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_row_scale = hip_context
                .alloc_buffer(rows * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_output = hip_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            hip_index
                .copy_from_host(0, &indices, Some(&mut hip_stream))
                .unwrap();
            hip_scale
                .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                .unwrap();
            hip_codebook
                .copy_from_host(
                    0,
                    &f32s_to_le_bytes(&codebook_values),
                    Some(&mut hip_stream),
                )
                .unwrap();
            hip_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                .unwrap();
            hip_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                .unwrap();
            hip_row_scale
                .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                .unwrap();
            hip_stream.synchronize().unwrap();

            // Compile/load both modules and reach steady-state allocations before wall timing.
            for _ in 0..WARMUP_ITERATIONS {
                aq4_matvec_batch_register_bm8_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    16,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            for _ in 0..WARMUP_ITERATIONS {
                aq4_matvec_batch_wmma_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    16,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();

            let register_started = std::time::Instant::now();
            for _ in 0..TIMED_ITERATIONS {
                aq4_matvec_batch_register_bm8_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    16,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            let register_elapsed = register_started.elapsed();

            let wmma_started = std::time::Instant::now();
            for _ in 0..TIMED_ITERATIONS {
                aq4_matvec_batch_wmma_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    16,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            let wmma_elapsed = wmma_started.elapsed();

            let register_ms = register_elapsed.as_secs_f64() * 1_000.0 / TIMED_ITERATIONS as f64;
            let wmma_ms = wmma_elapsed.as_secs_f64() * 1_000.0 / TIMED_ITERATIONS as f64;
            let flops_per_launch = 2.0_f64 * rows as f64 * cols as f64 * batch_count as f64;
            let register_tflops = flops_per_launch / (register_ms / 1_000.0) / 1.0e12;
            let wmma_tflops = flops_per_launch / (wmma_ms / 1_000.0) / 1.0e12;
            assert!(register_ms.is_finite() && register_ms > 0.0);
            assert!(wmma_ms.is_finite() && wmma_ms > 0.0);
            eprintln!(
                "AQ4 WMMA prototype timing family={family} rows={rows} cols={cols} M={batch_count}: register-bm8={register_ms:.3} ms ({register_tflops:.2} TFLOPS), wmma={wmma_ms:.3} ms ({wmma_tflops:.2} TFLOPS), speedup={:.3}x",
                register_ms / wmma_ms
            );
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_RAGGED_M_DIFFERENTIAL=1"]
    fn hip_aq4_wmma_ragged_m_group16_exact_buffers_match_cpu_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_WMMA_RAGGED_M_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_WMMA_RAGGED_M_DIFFERENTIAL=1 before running this GPU differential test"
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

        // cols=4096 exercises the full Wide-K ping-pong loop.  Input and output buffers below
        // are allocated exactly for each logical M, not the kernel's physical M=128 capacity.
        const ROWS: usize = 16;
        const COLS: usize = 4_096;
        const SCALE_COUNT: usize = 7;
        const TENSOR_SCALE: f32 = 0.75;
        let mut seed = 0x6a09_e667_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        let elements = ROWS * COLS;
        let index_bytes = elements / 2;
        let groups = elements / 16;
        let mut indices = vec![0_u8; index_bytes];
        for byte in &mut indices {
            *byte = ((next_unit() * 16.0) as u8 & 0x0f)
                | (((next_unit() * 16.0) as u8 & 0x0f) << 4);
        }
        let scale_indices: Vec<u8> = (0..groups)
            .map(|_| ((next_unit() * SCALE_COUNT as f32) as u8).min(SCALE_COUNT as u8 - 1))
            .collect();
        let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
        let scale_values: Vec<f32> = (0..SCALE_COUNT)
            .map(|_| 0.5 + next_unit() * 0.5)
            .collect();
        let row_scales: Vec<f32> = (0..ROWS).map(|_| 0.75 + next_unit() * 0.5).collect();

        for &m_actual in &[127_usize, 113, 65, 8, 1, 128] {
            let mut input_values = Vec::with_capacity(m_actual * COLS);
            for _ in 0..m_actual * COLS {
                input_values.push(next_unit() * 2.0 - 1.0);
            }
            // This makes accidental host-side padding impossible; device buffers use precisely
            // these byte counts too. GPU guard-page instrumentation is intentionally left to the
            // isolated hardware validation environment.
            assert_eq!(input_values.len(), input_values.capacity());
            let expected = {
                let mut cpu_context = RuntimeContext::create(0).unwrap();
                let mut cpu_stream = cpu_context.create_stream().unwrap();
                let mut cpu_index = cpu_context.alloc_buffer(index_bytes).unwrap();
                let mut cpu_scale = cpu_context.alloc_buffer(groups).unwrap();
                let mut cpu_codebook = cpu_context.alloc_buffer(16 * std::mem::size_of::<f32>()).unwrap();
                let mut cpu_scale_values = cpu_context
                    .alloc_buffer(SCALE_COUNT * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_input = cpu_context
                    .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_row_scales = cpu_context.alloc_buffer(ROWS * std::mem::size_of::<f32>()).unwrap();
                let mut cpu_output = cpu_context
                    .alloc_buffer(m_actual * ROWS * std::mem::size_of::<f32>())
                    .unwrap();
                cpu_index.copy_from_host(0, &indices, Some(&mut cpu_stream)).unwrap();
                cpu_scale.copy_from_host(0, &scale_indices, Some(&mut cpu_stream)).unwrap();
                cpu_codebook.copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut cpu_stream)).unwrap();
                cpu_scale_values.copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut cpu_stream)).unwrap();
                cpu_input.copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut cpu_stream)).unwrap();
                cpu_row_scales.copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cpu_stream)).unwrap();
                cpu_stream.synchronize().unwrap();
                aq4_matvec_batch_f32(
                    &cpu_index, &cpu_scale, &cpu_codebook, &cpu_scale_values, &cpu_input,
                    Some(&cpu_row_scales), SCALE_COUNT, 16, TENSOR_SCALE, ROWS, ROWS, COLS,
                    m_actual, &mut cpu_output, Some(&mut cpu_stream),
                ).unwrap();
                cpu_stream.synchronize().unwrap();
                let mut bytes = vec![0_u8; m_actual * ROWS * std::mem::size_of::<f32>()];
                cpu_output.copy_to_host(0, &mut bytes, Some(&mut cpu_stream)).unwrap();
                cpu_stream.synchronize().unwrap();
                le_bytes_to_f32s(&bytes)
            };

            let mut hip_context = RuntimeContext::create(device_index).unwrap();
            let mut hip_stream = hip_context.create_stream().unwrap();
            let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
            let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
            let mut hip_codebook = hip_context.alloc_buffer(16 * std::mem::size_of::<f32>()).unwrap();
            let mut hip_scale_values = hip_context.alloc_buffer(SCALE_COUNT * std::mem::size_of::<f32>()).unwrap();
            let mut hip_input = hip_context.alloc_buffer(input_values.len() * std::mem::size_of::<f32>()).unwrap();
            let mut hip_row_scales = hip_context.alloc_buffer(ROWS * std::mem::size_of::<f32>()).unwrap();
            let mut hip_output = hip_context.alloc_buffer(m_actual * ROWS * std::mem::size_of::<f32>()).unwrap();
            assert_eq!(hip_input.size().unwrap(), input_values.len() * std::mem::size_of::<f32>());
            assert_eq!(hip_output.size().unwrap(), m_actual * ROWS * std::mem::size_of::<f32>());
            hip_index.copy_from_host(0, &indices, Some(&mut hip_stream)).unwrap();
            hip_scale.copy_from_host(0, &scale_indices, Some(&mut hip_stream)).unwrap();
            hip_codebook.copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut hip_stream)).unwrap();
            hip_scale_values.copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream)).unwrap();
            hip_input.copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream)).unwrap();
            hip_row_scales.copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream)).unwrap();
            hip_stream.synchronize().unwrap();
            aq4_matvec_batch_wmma_ragged_m_prototype_f32(
                &hip_index, &hip_scale, &hip_codebook, &hip_scale_values, &hip_input,
                Some(&hip_row_scales), SCALE_COUNT, 16, TENSOR_SCALE, ROWS, ROWS, COLS,
                m_actual, &mut hip_output, Some(&mut hip_stream),
            ).unwrap();
            hip_stream.synchronize().unwrap();
            let mut actual_bytes = vec![0_u8; m_actual * ROWS * std::mem::size_of::<f32>()];
            hip_output.copy_to_host(0, &mut actual_bytes, Some(&mut hip_stream)).unwrap();
            hip_stream.synchronize().unwrap();
            let actual = le_bytes_to_f32s(&actual_bytes);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
                assert!((actual - expected).abs() <= tolerance,
                    "ragged M={m_actual} index={index}: actual={actual} expected={expected} tolerance={tolerance}");
            }

            if m_actual == 128 {
                let mut production_output = hip_context.alloc_buffer(128 * ROWS * std::mem::size_of::<f32>()).unwrap();
                aq4_matvec_batch_wmma_prototype_f32(
                    &hip_index, &hip_scale, &hip_codebook, &hip_scale_values, &hip_input,
                    Some(&hip_row_scales), SCALE_COUNT, 16, TENSOR_SCALE, ROWS, ROWS, COLS,
                    128, &mut production_output, Some(&mut hip_stream),
                ).unwrap();
                hip_stream.synchronize().unwrap();
                let mut production_bytes = vec![0_u8; 128 * ROWS * std::mem::size_of::<f32>()];
                production_output.copy_to_host(0, &mut production_bytes, Some(&mut hip_stream)).unwrap();
                hip_stream.synchronize().unwrap();
                assert_f32s_close(&actual, &le_bytes_to_f32s(&production_bytes), 1e-5);
            }
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_RAGGED_M_TIMING=1"]
    fn hip_aq4_wmma_ragged_m127_group16_timing_vs_register_bm8_when_enabled() {
        assert_eq!(std::env::var("ULLM_RUN_AQ4_WMMA_RAGGED_M_TIMING").as_deref(), Ok("1"));
        let device_index = (1..device_count().unwrap())
            .find(|&candidate| device_info(candidate).map(|info| info.gcn_arch_name == "gfx1201").unwrap_or(false))
            .expect("isolated gfx1201 HIP device");
        let _lock = AQ4_EXPERIMENTAL_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        const M: usize = 127;
        const WARMUP: usize = 3;
        const ITERATIONS: usize = 20;
        let mut seed = 0xbb67_ae85_u32;
        let mut next_unit = || { seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (seed as f32) / (u32::MAX as f32) };
        for &(family, rows, cols) in &[("attn_q + linear_attn_qkv", 8_192_usize, 4_096_usize), ("mlp_gate + mlp_up", 12_288, 4_096), ("mlp_down", 4_096, 12_288)] {
            let mut context = RuntimeContext::create(device_index).unwrap();
            let mut stream = context.create_stream().unwrap();
            let elements = rows * cols;
            let mut index = context.alloc_buffer(elements / 2).unwrap();
            let mut scale = context.alloc_buffer(elements / 16).unwrap();
            let mut codebook = context.alloc_buffer(16 * std::mem::size_of::<f32>()).unwrap();
            let mut scale_values = context.alloc_buffer(7 * std::mem::size_of::<f32>()).unwrap();
            let mut input = context.alloc_buffer(M * cols * std::mem::size_of::<f32>()).unwrap();
            let mut row_scales = context.alloc_buffer(rows * std::mem::size_of::<f32>()).unwrap();
            let mut output = context.alloc_buffer(M * rows * std::mem::size_of::<f32>()).unwrap();
            let indices: Vec<u8> = (0..elements / 2).map(|_| ((next_unit() * 16.0) as u8 & 0x0f) | (((next_unit() * 16.0) as u8 & 0x0f) << 4)).collect();
            let scales: Vec<u8> = (0..elements / 16).map(|_| ((next_unit() * 7.0) as u8).min(6)).collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values_host: Vec<f32> = (0..7).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..M * cols).map(|_| next_unit() * 2.0 - 1.0).collect();
            let row_scale_values: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            index.copy_from_host(0, &indices, Some(&mut stream)).unwrap(); scale.copy_from_host(0, &scales, Some(&mut stream)).unwrap();
            codebook.copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream)).unwrap(); scale_values.copy_from_host(0, &f32s_to_le_bytes(&scale_values_host), Some(&mut stream)).unwrap();
            input.copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream)).unwrap(); row_scales.copy_from_host(0, &f32s_to_le_bytes(&row_scale_values), Some(&mut stream)).unwrap(); stream.synchronize().unwrap();
            for _ in 0..WARMUP { aq4_matvec_batch_register_bm8_f32(&index, &scale, &codebook, &scale_values, &input, Some(&row_scales), 7, 16, 0.75, rows, rows, cols, M, &mut output, Some(&mut stream)).unwrap(); }
            stream.synchronize().unwrap();
            for _ in 0..WARMUP { aq4_matvec_batch_wmma_ragged_m_prototype_f32(&index, &scale, &codebook, &scale_values, &input, Some(&row_scales), 7, 16, 0.75, rows, rows, cols, M, &mut output, Some(&mut stream)).unwrap(); }
            stream.synchronize().unwrap();
            let register_started = std::time::Instant::now();
            for _ in 0..ITERATIONS { aq4_matvec_batch_register_bm8_f32(&index, &scale, &codebook, &scale_values, &input, Some(&row_scales), 7, 16, 0.75, rows, rows, cols, M, &mut output, Some(&mut stream)).unwrap(); }
            stream.synchronize().unwrap();
            let register_ms = register_started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
            let ragged_started = std::time::Instant::now();
            for _ in 0..ITERATIONS { aq4_matvec_batch_wmma_ragged_m_prototype_f32(&index, &scale, &codebook, &scale_values, &input, Some(&row_scales), 7, 16, 0.75, rows, rows, cols, M, &mut output, Some(&mut stream)).unwrap(); }
            stream.synchronize().unwrap();
            let ragged_ms = ragged_started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
            assert!(register_ms.is_finite() && register_ms > 0.0 && ragged_ms.is_finite() && ragged_ms > 0.0);
            eprintln!("AQ4 WMMA ragged-M timing family={family} rows={rows} cols={cols} M={M}: register-bm8={register_ms:.3} ms, ragged-wmma={ragged_ms:.3} ms, speedup={:.3}x", register_ms / ragged_ms);
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_GROUP8_DIFFERENTIAL=1"]
    fn hip_aq4_wmma_group8_m128_target_shapes_match_cpu_when_enabled() {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_WMMA_GROUP8_DIFFERENTIAL").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_WMMA_GROUP8_DIFFERENTIAL=1 before running this GPU differential test"
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

        let mut seed = 0x9b05_688c_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        // The production direct ABI uses a separate module from both group16 WMMA and the group8
        // register kernel. These tuples exercise the served dimensions and both optional
        // row-scale ABI states.
        for &(family, rows, cols, use_row_scales) in &[
            ("attn_o + linear_attn_out", 4_096_usize, 4_096, true),
            ("attn_k + attn_v", 1_024, 4_096, false),
        ] {
            let batch_count = 128_usize;
            assert_eq!(rows % 16, 0, "{family} must fill WMMA row tiles");
            assert_eq!(cols % 32, 0, "{family} must fill Wide-K groups");
            assert_eq!(32 / 8, 4, "{family} must decode four group8 scales per Wide-K load");
            let elements = rows * cols;
            let index_bytes = elements / 2;
            let groups = elements / 8;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            let row_scale_count = if use_row_scales { rows } else { 0 };
            let tensor_scale = 0.75_f32;

            let expected = {
                let mut cpu_context = RuntimeContext::create(0).unwrap();
                let mut cpu_stream = cpu_context.create_stream().unwrap();
                let mut cpu_index = cpu_context.alloc_buffer(index_bytes).unwrap();
                let mut cpu_scale = cpu_context.alloc_buffer(groups).unwrap();
                let mut cpu_codebook = cpu_context
                    .alloc_buffer(16 * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_scale_values = cpu_context
                    .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_input = cpu_context
                    .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                    .unwrap();
                let mut cpu_row_scale = if use_row_scales {
                    Some(
                        cpu_context
                            .alloc_buffer(rows * std::mem::size_of::<f32>())
                            .unwrap(),
                    )
                } else {
                    None
                };
                let mut cpu_output = cpu_context
                    .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                    .unwrap();
                cpu_index
                    .copy_from_host(0, &indices, Some(&mut cpu_stream))
                    .unwrap();
                cpu_scale
                    .copy_from_host(0, &scale_indices, Some(&mut cpu_stream))
                    .unwrap();
                cpu_codebook
                    .copy_from_host(
                        0,
                        &f32s_to_le_bytes(&codebook_values),
                        Some(&mut cpu_stream),
                    )
                    .unwrap();
                cpu_scale_values
                    .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut cpu_stream))
                    .unwrap();
                cpu_input
                    .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut cpu_stream))
                    .unwrap();
                if let Some(row_scale) = cpu_row_scale.as_mut() {
                    row_scale
                        .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut cpu_stream))
                        .unwrap();
                }
                cpu_stream.synchronize().unwrap();
                aq4_matvec_batch_f32(
                    &cpu_index,
                    &cpu_scale,
                    &cpu_codebook,
                    &cpu_scale_values,
                    &cpu_input,
                    cpu_row_scale.as_ref(),
                    scale_count,
                    8,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut cpu_output,
                    Some(&mut cpu_stream),
                )
                .unwrap();
                cpu_stream.synchronize().unwrap();
                let mut expected_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
                cpu_output
                    .copy_to_host(0, &mut expected_bytes, Some(&mut cpu_stream))
                    .unwrap();
                cpu_stream.synchronize().unwrap();
                le_bytes_to_f32s(&expected_bytes)
            };

            let actual = {
                let mut hip_context = RuntimeContext::create(device_index).unwrap();
                let mut hip_stream = hip_context.create_stream().unwrap();
                let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
                let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
                let mut hip_codebook = hip_context
                    .alloc_buffer(16 * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_scale_values = hip_context
                    .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_input = hip_context
                    .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                    .unwrap();
                let mut hip_row_scale = if use_row_scales {
                    Some(
                        hip_context
                            .alloc_buffer(rows * std::mem::size_of::<f32>())
                            .unwrap(),
                    )
                } else {
                    None
                };
                let mut hip_output = hip_context
                    .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                    .unwrap();
                hip_index
                    .copy_from_host(0, &indices, Some(&mut hip_stream))
                    .unwrap();
                hip_scale
                    .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                    .unwrap();
                hip_codebook
                    .copy_from_host(
                        0,
                        &f32s_to_le_bytes(&codebook_values),
                        Some(&mut hip_stream),
                    )
                    .unwrap();
                hip_scale_values
                    .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                    .unwrap();
                hip_input
                    .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                    .unwrap();
                if let Some(row_scale) = hip_row_scale.as_mut() {
                    row_scale
                        .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                        .unwrap();
                }
                hip_stream.synchronize().unwrap();
                aq4_matvec_batch_wmma_group8_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    hip_row_scale.as_ref(),
                    scale_count,
                    8,
                    tensor_scale,
                    row_scale_count,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
                hip_stream.synchronize().unwrap();
                let mut actual_bytes = vec![0_u8; batch_count * rows * std::mem::size_of::<f32>()];
                hip_output
                    .copy_to_host(0, &mut actual_bytes, Some(&mut hip_stream))
                    .unwrap();
                hip_stream.synchronize().unwrap();
                le_bytes_to_f32s(&actual_bytes)
            };

            // The CPU AQ4 reference remains FP32. This kernel stages both decoded weights and
            // activations through FP16 before FP32 WMMA accumulation, so use the established
            // bounded 0.05 absolute + 1% relative allowance. It is deliberately far below the
            // error expected from a group-scale, nibble-order, transpose, or factor-placement bug.
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "family={family} rows={rows} cols={cols} index={index}: actual={actual} expected={expected} tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WMMA_GROUP8_RAGGED_M_TIMING=1"]
    fn hip_aq4_wmma_group8_ragged_m127_target_shapes_timing_vs_register_bm8_group8_when_enabled()
    {
        assert_eq!(
            std::env::var("ULLM_RUN_AQ4_WMMA_GROUP8_RAGGED_M_TIMING").as_deref(),
            Ok("1"),
            "set ULLM_RUN_AQ4_WMMA_GROUP8_RAGGED_M_TIMING=1 before running this GPU timing test"
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

        const WARMUP_ITERATIONS: usize = 3;
        const TIMED_ITERATIONS: usize = 20;
        let mut seed = 0xd807_aa98_u32;
        let mut next_unit = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32) / (u32::MAX as f32)
        };
        for &(family, rows, cols) in &[
            ("attn_o + linear_attn_out", 4_096_usize, 4_096),
            ("attn_k + attn_v", 1_024, 4_096),
        ] {
            let batch_count = 127_usize;
            assert_eq!(rows % 16, 0, "{family} must fill WMMA row tiles");
            assert_eq!(cols % 32, 0, "{family} must fill Wide-K groups");
            assert_eq!(32 / 8, 4, "{family} must decode four group8 scales per Wide-K load");
            let elements = rows * cols;
            let index_bytes = elements / 2;
            let groups = elements / 8;
            let scale_count = 7_usize;
            let mut indices = vec![0_u8; index_bytes];
            for byte in &mut indices {
                let low = (next_unit() * 16.0) as u8 & 0x0f;
                let high = (next_unit() * 16.0) as u8 & 0x0f;
                *byte = low | (high << 4);
            }
            let scale_indices: Vec<u8> = (0..groups)
                .map(|_| ((next_unit() * scale_count as f32) as u8).min(scale_count as u8 - 1))
                .collect();
            let codebook_values: Vec<f32> = (0..16).map(|_| next_unit() - 0.5).collect();
            let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next_unit() * 0.5).collect();
            let input_values: Vec<f32> = (0..batch_count * cols)
                .map(|_| next_unit() * 2.0 - 1.0)
                .collect();
            let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next_unit() * 0.5).collect();
            let tensor_scale = 0.75_f32;

            let mut hip_context = RuntimeContext::create(device_index).unwrap();
            let mut hip_stream = hip_context.create_stream().unwrap();
            let mut hip_index = hip_context.alloc_buffer(index_bytes).unwrap();
            let mut hip_scale = hip_context.alloc_buffer(groups).unwrap();
            let mut hip_codebook = hip_context
                .alloc_buffer(16 * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_scale_values = hip_context
                .alloc_buffer(scale_count * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_input = hip_context
                .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_row_scale = hip_context
                .alloc_buffer(rows * std::mem::size_of::<f32>())
                .unwrap();
            let mut hip_output = hip_context
                .alloc_buffer(batch_count * rows * std::mem::size_of::<f32>())
                .unwrap();
            hip_index
                .copy_from_host(0, &indices, Some(&mut hip_stream))
                .unwrap();
            hip_scale
                .copy_from_host(0, &scale_indices, Some(&mut hip_stream))
                .unwrap();
            hip_codebook
                .copy_from_host(
                    0,
                    &f32s_to_le_bytes(&codebook_values),
                    Some(&mut hip_stream),
                )
                .unwrap();
            hip_scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut hip_stream))
                .unwrap();
            hip_input
                .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut hip_stream))
                .unwrap();
            hip_row_scale
                .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut hip_stream))
                .unwrap();
            hip_stream.synchronize().unwrap();

            // Compile/load both independent modules and reach steady-state allocations before
            // wall timing. The scalar side is the current production group8 register kernel.
            for _ in 0..WARMUP_ITERATIONS {
                aq4_matvec_batch_register_bm8_group8_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    8,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            for _ in 0..WARMUP_ITERATIONS {
                aq4_matvec_batch_wmma_group8_ragged_m_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    8,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();

            let register_started = std::time::Instant::now();
            for _ in 0..TIMED_ITERATIONS {
                aq4_matvec_batch_register_bm8_group8_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    8,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            let register_elapsed = register_started.elapsed();

            let wmma_started = std::time::Instant::now();
            for _ in 0..TIMED_ITERATIONS {
                aq4_matvec_batch_wmma_group8_ragged_m_prototype_f32(
                    &hip_index,
                    &hip_scale,
                    &hip_codebook,
                    &hip_scale_values,
                    &hip_input,
                    Some(&hip_row_scale),
                    scale_count,
                    8,
                    tensor_scale,
                    rows,
                    rows,
                    cols,
                    batch_count,
                    &mut hip_output,
                    Some(&mut hip_stream),
                )
                .unwrap();
            }
            hip_stream.synchronize().unwrap();
            let wmma_elapsed = wmma_started.elapsed();

            let register_ms = register_elapsed.as_secs_f64() * 1_000.0 / TIMED_ITERATIONS as f64;
            let wmma_ms = wmma_elapsed.as_secs_f64() * 1_000.0 / TIMED_ITERATIONS as f64;
            let flops_per_launch = 2.0_f64 * rows as f64 * cols as f64 * batch_count as f64;
            let register_tflops = flops_per_launch / (register_ms / 1_000.0) / 1.0e12;
            let wmma_tflops = flops_per_launch / (wmma_ms / 1_000.0) / 1.0e12;
            assert!(register_ms.is_finite() && register_ms > 0.0);
            assert!(wmma_ms.is_finite() && wmma_ms > 0.0);
            eprintln!(
                "AQ4 group8 WMMA timing family={family} rows={rows} cols={cols} M={batch_count}: register-bm8-group8={register_ms:.3} ms ({register_tflops:.2} TFLOPS), wmma-group8={wmma_ms:.3} ms ({wmma_tflops:.2} TFLOPS), speedup={:.3}x",
                register_ms / wmma_ms
            );
        }
    }

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
    fn hip_gemma_proportional_rope_f32_full_geometry_preserves_partial_boundaries() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let heads = 8_usize;
        let head_dim = 512_usize;
        let rotary_dim = 128_usize;
        let input_values = (0..heads * head_dim)
            .map(|index| index as f32 * 0.03125 - 51.0)
            .collect::<Vec<_>>();
        let expected = expected_gemma_proportional_rope(
            &input_values,
            1,
            heads,
            head_dim,
            rotary_dim,
            17,
            1_000_000.0,
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
        gemma_proportional_rope_f32(
            &input,
            1,
            heads,
            head_dim,
            rotary_dim,
            17,
            1_000_000.0,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let actual = le_bytes_to_f32s(&output_bytes);
        // HIPRTC's powf/cosf differs slightly from the serial Rust reference;
        // the exact pass-through boundary checks below remain bit-exact.
        assert_f32s_close(&actual, &expected, 1e-4);
        for head in 0..heads {
            let base = head * head_dim;
            assert_eq!(
                &actual[base + rotary_dim / 2..base + head_dim / 2],
                &input_values[base + rotary_dim / 2..base + head_dim / 2],
            );
            assert_eq!(
                &actual[base + head_dim / 2 + rotary_dim / 2..base + head_dim],
                &input_values[base + head_dim / 2 + rotary_dim / 2..base + head_dim],
            );
        }
    }
