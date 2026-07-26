fn aq4_matvec_add_wide_load_gpu_device() -> u32 {
    (1..device_count().unwrap())
        .find(|&device| {
            device_info(device)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device")
}

#[allow(clippy::too_many_arguments)]
fn aq4_matvec_add_run(
    device: u32,
    shuffle: bool,
    indices: &[u8],
    scale_indices: &[u8],
    codebook: &[f32],
    scale_values: &[f32],
    input: &[f32],
    residual: &[f32],
    row_scales: Option<&[f32]>,
    group_size: usize,
    tensor_scale: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut index = context.alloc_buffer(indices.len()).unwrap();
    let mut scale = context.alloc_buffer(scale_indices.len()).unwrap();
    let mut codebook_buffer = context.alloc_buffer(codebook.len() * 4).unwrap();
    let mut scale_values_buffer = context.alloc_buffer(scale_values.len() * 4).unwrap();
    let mut input_buffer = context.alloc_buffer(input.len() * 4).unwrap();
    let mut residual_buffer = context.alloc_buffer(residual.len() * 4).unwrap();
    let mut row_scale_buffer = row_scales.map(|values| context.alloc_buffer(values.len() * 4).unwrap());
    let mut output = context.alloc_buffer(rows * 4).unwrap();
    index.copy_from_host(0, indices, Some(&mut stream)).unwrap();
    scale.copy_from_host(0, scale_indices, Some(&mut stream)).unwrap();
    codebook_buffer
        .copy_from_host(0, &f32s_to_le_bytes(codebook), Some(&mut stream))
        .unwrap();
    scale_values_buffer
        .copy_from_host(0, &f32s_to_le_bytes(scale_values), Some(&mut stream))
        .unwrap();
    input_buffer
        .copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream))
        .unwrap();
    residual_buffer
        .copy_from_host(0, &f32s_to_le_bytes(residual), Some(&mut stream))
        .unwrap();
    if let (Some(buffer), Some(values)) = (row_scale_buffer.as_mut(), row_scales) {
        buffer
            .copy_from_host(0, &f32s_to_le_bytes(values), Some(&mut stream))
            .unwrap();
    }
    let row_scale_count = row_scales.map_or(0, <[f32]>::len);
    if shuffle {
        aq4_matvec_add_shuffle_prototype_f32(
            &index,
            &scale,
            &codebook_buffer,
            &scale_values_buffer,
            &input_buffer,
            &residual_buffer,
            row_scale_buffer.as_ref(),
            scale_values.len(),
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
    } else {
        aq4_matvec_add_f32(
            &index,
            &scale,
            &codebook_buffer,
            &scale_values_buffer,
            &input_buffer,
            &residual_buffer,
            row_scale_buffer.as_ref(),
            scale_values.len(),
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut output_bytes = vec![0; rows * 4];
    output
        .copy_to_host(0, &mut output_bytes, Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();
    le_bytes_to_f32s(&output_bytes)
}

fn aq4_matvec_add_assert_matches_cpu(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_ADD_PRODUCTION_DIFFERENTIAL=1"]
fn hip_aq4_matvec_add_production_model_shapes_match_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_ADD_PRODUCTION_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_AQ4_MATVEC_ADD_PRODUCTION_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let hip_device = aq4_matvec_add_wide_load_gpu_device();
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _require_production_kernel =
        ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL", Some("1"));
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_ADD_RPB", Some("8"));
    let mut state = 0x510e_527f_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state as f32 / u32::MAX as f32
    };
    for &(family, rows, cols, group_size, has_row_scales) in &[
        ("self-attn o + linear-attn out residual", 4_096_usize, 4_096, 8, true),
        ("self-attn + linear-attn MLP down residual", 4_096, 12_288, 16, false),
    ] {
        let index_bytes = rows * cols / 2;
        let groups = rows * cols / group_size;
        let scale_count = 7_usize;
        let indices: Vec<u8> = (0..index_bytes)
            .map(|_| ((next() * 16.0) as u8 & 15) | (((next() * 16.0) as u8 & 15) << 4))
            .collect();
        let scale_indices: Vec<u8> = (0..groups)
            .map(|_| ((next() * scale_count as f32) as u8).min(scale_count as u8 - 1))
            .collect();
        let codebook: Vec<f32> = (0..16).map(|_| next() - 0.5).collect();
        let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next() * 0.5).collect();
        let input: Vec<f32> = (0..cols).map(|_| next() * 2.0 - 1.0).collect();
        let residual: Vec<f32> = (0..rows).map(|_| next() * 2.0 - 1.0).collect();
        let row_scales = has_row_scales.then(|| {
            (0..rows)
                .map(|_| 0.75 + next() * 0.5)
                .collect::<Vec<f32>>()
        });
        let expected = aq4_matvec_add_run(
            0,
            false,
            &indices,
            &scale_indices,
            &codebook,
            &scale_values,
            &input,
            &residual,
            row_scales.as_deref(),
            group_size,
            0.75,
            rows,
            cols,
        );
        let actual = aq4_matvec_add_run(
            hip_device,
            false,
            &indices,
            &scale_indices,
            &codebook,
            &scale_values,
            &input,
            &residual,
            row_scales.as_deref(),
            group_size,
            0.75,
            rows,
            cols,
        );
        let shuffle_baseline = aq4_matvec_add_run(
            hip_device,
            true,
            &indices,
            &scale_indices,
            &codebook,
            &scale_values,
            &input,
            &residual,
            row_scales.as_deref(),
            group_size,
            0.75,
            rows,
            cols,
        );
        aq4_matvec_add_assert_matches_cpu(&actual, &expected, family);
        aq4_matvec_add_assert_matches_cpu(
            &actual,
            &shuffle_baseline,
            &format!("{family} group-specialized versus shuffle baseline"),
        );
        eprintln!(
            "AQ4 matvec-add production differential family={family} rows={rows} cols={cols} group{group_size}: ok"
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_DIFFERENTIAL=1"]
fn hip_aq4_matvec_add_shuffle_prototype_matches_cpu_for_production_shapes_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let hip_device = aq4_matvec_add_wide_load_gpu_device();
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_ADD_RPB", Some("8"));
    let mut state = 0x9b05_688c_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state as f32 / u32::MAX as f32
    };
    for &(family, rows, cols, group_size, has_row_scales) in &[
        ("self-attn o + linear-attn out residual", 4_096_usize, 4_096, 8, true),
        ("self-attn + linear-attn MLP down residual", 4_096, 12_288, 16, false),
    ] {
        let index_bytes = rows * cols / 2;
        let groups = rows * cols / group_size;
        let scale_count = 7_usize;
        let indices: Vec<u8> = (0..index_bytes)
            .map(|_| ((next() * 16.0) as u8 & 15) | (((next() * 16.0) as u8 & 15) << 4))
            .collect();
        let scale_indices: Vec<u8> = (0..groups)
            .map(|_| ((next() * scale_count as f32) as u8).min(scale_count as u8 - 1))
            .collect();
        let codebook: Vec<f32> = (0..16).map(|_| next() - 0.5).collect();
        let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next() * 0.5).collect();
        let input: Vec<f32> = (0..cols).map(|_| next() * 2.0 - 1.0).collect();
        let residual: Vec<f32> = (0..rows).map(|_| next() * 2.0 - 1.0).collect();
        let row_scales = has_row_scales.then(|| {
            (0..rows)
                .map(|_| 0.75 + next() * 0.5)
                .collect::<Vec<f32>>()
        });
        let expected = aq4_matvec_add_run(
            0,
            false,
            &indices,
            &scale_indices,
            &codebook,
            &scale_values,
            &input,
            &residual,
            row_scales.as_deref(),
            group_size,
            0.75,
            rows,
            cols,
        );
        let actual = aq4_matvec_add_run(
            hip_device,
            true,
            &indices,
            &scale_indices,
            &codebook,
            &scale_values,
            &input,
            &residual,
            row_scales.as_deref(),
            group_size,
            0.75,
            rows,
            cols,
        );
        aq4_matvec_add_assert_matches_cpu(&actual, &expected, family);
        eprintln!(
            "AQ4 matvec-add shuffle differential family={family} rows={rows} cols={cols} group{group_size}: ok"
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_ADD_WIDE_LOAD_TIMING=1"]
fn hip_aq4_matvec_add_wide_load_prototype_timing_vs_production_for_production_shapes_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_ADD_WIDE_LOAD_TIMING").as_deref(),
        Ok("1")
    );
    let hip_device = aq4_matvec_add_wide_load_gpu_device();
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for &(family, rows, cols, group_size, has_row_scales) in &[
        ("self-attn o + linear-attn out residual", 4_096_usize, 4_096, 8, true),
        ("self-attn + linear-attn MLP down residual", 4_096, 12_288, 16, false),
    ] {
        let mut context = RuntimeContext::create(hip_device).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(rows * cols / 2).unwrap();
        let mut scale = context.alloc_buffer(rows * cols / group_size).unwrap();
        let mut codebook = context.alloc_buffer(16 * 4).unwrap();
        let mut scale_values = context.alloc_buffer(4).unwrap();
        let mut input = context.alloc_buffer(cols * 4).unwrap();
        let mut residual = context.alloc_buffer(rows * 4).unwrap();
        let mut row_scales = has_row_scales.then(|| context.alloc_buffer(rows * 4).unwrap());
        let mut output = context.alloc_buffer(rows * 4).unwrap();
        index
            .copy_from_host(0, &vec![0x87; rows * cols / 2], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &vec![0; rows * cols / group_size], Some(&mut stream))
            .unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&(0..16).map(|value| value as f32 / 16.0 - 0.5).collect::<Vec<_>>()),
                Some(&mut stream),
            )
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.75]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&vec![0.125; cols]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&vec![-0.25; rows]), Some(&mut stream))
            .unwrap();
        if let Some(row_scale) = row_scales.as_mut() {
            row_scale
                .copy_from_host(0, &f32s_to_le_bytes(&vec![1.25; rows]), Some(&mut stream))
                .unwrap();
        }
        let row_scale_count = usize::from(has_row_scales) * rows;
        for _ in 0..3 {
            aq4_matvec_add_f32(&index, &scale, &codebook, &scale_values, &input, &residual, row_scales.as_ref(), 1, group_size, 0.75, row_scale_count, rows, cols, &mut output, Some(&mut stream)).unwrap();
            aq4_matvec_add_wide_load_prototype_f32(&index, &scale, &codebook, &scale_values, &input, &residual, row_scales.as_ref(), 1, group_size, 0.75, row_scale_count, rows, cols, &mut output, Some(&mut stream)).unwrap();
        }
        stream.synchronize().unwrap();
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_add_f32(&index, &scale, &codebook, &scale_values, &input, &residual, row_scales.as_ref(), 1, group_size, 0.75, row_scale_count, rows, cols, &mut output, Some(&mut stream)).unwrap();
        }
        stream.synchronize().unwrap();
        let production_ms = started.elapsed().as_secs_f64() * 50.0;
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_add_wide_load_prototype_f32(&index, &scale, &codebook, &scale_values, &input, &residual, row_scales.as_ref(), 1, group_size, 0.75, row_scale_count, rows, cols, &mut output, Some(&mut stream)).unwrap();
        }
        stream.synchronize().unwrap();
        let wide_ms = started.elapsed().as_secs_f64() * 50.0;
        assert!(production_ms.is_finite() && production_ms > 0.0);
        assert!(wide_ms.is_finite() && wide_ms > 0.0);
        eprintln!(
            "AQ4 matvec-add wide-load timing family={family} rows={rows} cols={cols} group{group_size}: production={production_ms:.3} ms, wide={wide_ms:.3} ms, speedup={:.3}x",
            production_ms / wide_ms
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_TIMING=1"]
fn hip_aq4_matvec_add_shuffle_prototype_timing_vs_production_for_production_shapes_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_TIMING").as_deref(),
        Ok("1")
    );
    let hip_device = aq4_matvec_add_wide_load_gpu_device();
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_ADD_RPB", Some("8"));
    for &(family, rows, cols, group_size, has_row_scales) in &[
        ("self-attn o + linear-attn out residual", 4_096_usize, 4_096, 8, true),
        ("self-attn + linear-attn MLP down residual", 4_096, 12_288, 16, false),
    ] {
        let mut context = RuntimeContext::create(hip_device).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(rows * cols / 2).unwrap();
        let mut scale = context.alloc_buffer(rows * cols / group_size).unwrap();
        let mut codebook = context.alloc_buffer(16 * 4).unwrap();
        let mut scale_values = context.alloc_buffer(4).unwrap();
        let mut input = context.alloc_buffer(cols * 4).unwrap();
        let mut residual = context.alloc_buffer(rows * 4).unwrap();
        let mut row_scales = has_row_scales.then(|| context.alloc_buffer(rows * 4).unwrap());
        let mut output = context.alloc_buffer(rows * 4).unwrap();
        index
            .copy_from_host(0, &vec![0x87; rows * cols / 2], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &vec![0; rows * cols / group_size], Some(&mut stream))
            .unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(
                    &(0..16)
                        .map(|value| value as f32 / 16.0 - 0.5)
                        .collect::<Vec<_>>(),
                ),
                Some(&mut stream),
            )
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.75]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&vec![0.125; cols]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&vec![-0.25; rows]), Some(&mut stream))
            .unwrap();
        if let Some(row_scale) = row_scales.as_mut() {
            row_scale
                .copy_from_host(0, &f32s_to_le_bytes(&vec![1.25; rows]), Some(&mut stream))
                .unwrap();
        }
        let row_scale_count = usize::from(has_row_scales) * rows;
        for _ in 0..3 {
            aq4_matvec_add_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                &residual,
                row_scales.as_ref(),
                1,
                group_size,
                0.75,
                row_scale_count,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
            aq4_matvec_add_shuffle_prototype_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                &residual,
                row_scales.as_ref(),
                1,
                group_size,
                0.75,
                row_scale_count,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_add_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                &residual,
                row_scales.as_ref(),
                1,
                group_size,
                0.75,
                row_scale_count,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let production_ms = started.elapsed().as_secs_f64() * 50.0;
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_add_shuffle_prototype_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                &residual,
                row_scales.as_ref(),
                1,
                group_size,
                0.75,
                row_scale_count,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let shuffle_ms = started.elapsed().as_secs_f64() * 50.0;
        assert!(production_ms.is_finite() && production_ms > 0.0);
        assert!(shuffle_ms.is_finite() && shuffle_ms > 0.0);
        eprintln!(
            "AQ4 matvec-add shuffle timing family={family} rows={rows} cols={cols} group{group_size}: production={production_ms:.3} ms, shuffle={shuffle_ms:.3} ms, speedup={:.3}x",
            production_ms / shuffle_ms
        );
    }
}
