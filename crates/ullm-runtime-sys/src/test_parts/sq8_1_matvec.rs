fn sq8_1_fixture_payload() -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    const ROWS: usize = 2;
    const STRIDE: usize = 48;
    let mut payload = vec![0_u8; ROWS * STRIDE];
    for col in 0..32 {
        payload[col] = 1_i8 as u8;
        payload[STRIDE + col] = (-1_i8) as u8;
    }
    payload[32] = 2_i8 as u8;
    payload[STRIDE + 32] = (-3_i8) as u8;
    // F16 LE: row0 [1.0, 0.5], row1 [0.5, 1.0].
    let scales = vec![0x00, 0x3c, 0x00, 0x38, 0x00, 0x38, 0x00, 0x3c];
    let mut input = (0..31)
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    input.push(127.0);
    input.push(127.0);
    let first_sum = input[..32].iter().sum::<f32>();
    let expected = vec![first_sum + 127.0, -0.5 * first_sum - 381.0];
    (payload, scales, input, expected)
}

#[test]
fn cpu_sq8_1_w8a16_default_and_explicit_w8a8_match_k32_reference() {
    const ROWS: usize = 2;
    const COLS: usize = 33;
    let (payload_bytes, scale_bytes, input_values, expected) = sq8_1_fixture_payload();
    let stride = sq8_1_payload_row_stride(COLS).unwrap();
    assert_eq!(stride, 48);

    let mut context = RuntimeContext::create(0).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut payload = context.alloc_buffer(payload_bytes.len()).unwrap();
    let mut scales = context.alloc_buffer(scale_bytes.len()).unwrap();
    let mut input = context
        .alloc_buffer(COLS * std::mem::size_of::<f32>())
        .unwrap();
    let mut w8a16_output = context
        .alloc_buffer(ROWS * std::mem::size_of::<f32>())
        .unwrap();
    let mut w8a8_output = context
        .alloc_buffer(ROWS * std::mem::size_of::<f32>())
        .unwrap();
    payload
        .copy_from_host(0, &payload_bytes, Some(&mut stream))
        .unwrap();
    scales
        .copy_from_host(0, &scale_bytes, Some(&mut stream))
        .unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();

    sq8_1_matvec_w8a16_f32(
        &payload,
        &scales,
        &input,
        ROWS,
        COLS,
        stride,
        &mut w8a16_output,
        Some(&mut stream),
    )
    .unwrap();
    // W8A8 has a separate explicit entry point; it is never selected by
    // the W8A16 call above.
    sq8_1_matvec_w8a8_explicit_f32(
        &payload,
        &scales,
        &input,
        ROWS,
        COLS,
        stride,
        &mut w8a8_output,
        Some(&mut stream),
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut w8a16_bytes = vec![0_u8; ROWS * std::mem::size_of::<f32>()];
    let mut w8a8_bytes = vec![0_u8; ROWS * std::mem::size_of::<f32>()];
    w8a16_output
        .copy_to_host(0, &mut w8a16_bytes, Some(&mut stream))
        .unwrap();
    w8a8_output
        .copy_to_host(0, &mut w8a8_bytes, Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(le_bytes_to_f32s(&w8a16_bytes), expected);
    assert_eq!(le_bytes_to_f32s(&w8a8_bytes), expected);
}

#[test]
fn cpu_sq8_1_rejects_nonzero_physical_tail_and_noncanonical_stride() {
    const ROWS: usize = 2;
    const COLS: usize = 33;
    let (mut payload_bytes, scale_bytes, input_values, _) = sq8_1_fixture_payload();
    payload_bytes[33] = 1;
    let mut context = RuntimeContext::create(0).unwrap();
    let mut payload = context.alloc_buffer(payload_bytes.len()).unwrap();
    let mut scales = context.alloc_buffer(scale_bytes.len()).unwrap();
    let mut input = context
        .alloc_buffer(COLS * std::mem::size_of::<f32>())
        .unwrap();
    let mut output = context
        .alloc_buffer(ROWS * std::mem::size_of::<f32>())
        .unwrap();
    payload.copy_from_host(0, &payload_bytes, None).unwrap();
    scales.copy_from_host(0, &scale_bytes, None).unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(&input_values), None)
        .unwrap();
    let err = sq8_1_matvec_w8a16_f32(&payload, &scales, &input, ROWS, COLS, 48, &mut output, None)
        .unwrap_err();
    assert!(err.contains("malformed"), "{err}");
    assert!(
        sq8_1_matvec_w8a16_f32(&payload, &scales, &input, ROWS, COLS, 33, &mut output, None,)
            .unwrap_err()
            .contains("round_up")
    );
}

#[test]
fn cpu_sq8_1_boundary_tail_zero_and_saturation_cases_match() {
    // With every activation block max fixed at 127, ceil_fp16(max/127) is
    // exactly one.  W8A8 therefore has an exact integer oracle against the
    // W8A16 path while exercising signed endpoints, all-zero rows, and each
    // physical K=16/K=32 tail boundary.
    for cols in [1_usize, 15, 16, 17, 31, 32, 33, 65] {
        const ROWS: usize = 2;
        let stride = sq8_1_payload_row_stride(cols).unwrap();
        let groups = (cols + 31) / 32;
        let mut payload = vec![0_u8; ROWS * stride];
        let mut expected = 0_i32;
        let input = (0..cols)
            .map(|col| match col % 3 {
                0 => -127.0_f32,
                1 => 0.0_f32,
                _ => 127.0_f32,
            })
            .collect::<Vec<_>>();
        for col in 0..cols {
            let weight = match col % 4 {
                0 => -127_i8,
                1 => -1_i8,
                2 => 1_i8,
                _ => 127_i8,
            };
            payload[col] = weight as u8;
            expected += i32::from(weight) * input[col] as i32;
        }
        // Row one stays all zero, including its physical tail padding.
        let mut scales = vec![0_u8; ROWS * groups * 2];
        for group in 0..ROWS * groups {
            scales[group * 2 + 1] = 0x3c; // FP16 1.0, little endian.
        }

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut payload_buffer = context.alloc_buffer(payload.len()).unwrap();
        let mut scale_buffer = context.alloc_buffer(scales.len()).unwrap();
        let mut input_buffer = context
            .alloc_buffer(cols * std::mem::size_of::<f32>())
            .unwrap();
        let mut w8a16_output = context
            .alloc_buffer(ROWS * std::mem::size_of::<f32>())
            .unwrap();
        let mut w8a8_output = context
            .alloc_buffer(ROWS * std::mem::size_of::<f32>())
            .unwrap();
        payload_buffer
            .copy_from_host(0, &payload, Some(&mut stream))
            .unwrap();
        scale_buffer
            .copy_from_host(0, &scales, Some(&mut stream))
            .unwrap();
        input_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&input), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        sq8_1_matvec_w8a16_f32(
            &payload_buffer,
            &scale_buffer,
            &input_buffer,
            ROWS,
            cols,
            stride,
            &mut w8a16_output,
            Some(&mut stream),
        )
        .unwrap();
        sq8_1_matvec_w8a8_explicit_f32(
            &payload_buffer,
            &scale_buffer,
            &input_buffer,
            ROWS,
            cols,
            stride,
            &mut w8a8_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        for output in [&mut w8a16_output, &mut w8a8_output] {
            let mut bytes = vec![0_u8; ROWS * std::mem::size_of::<f32>()];
            output.copy_to_host(0, &mut bytes, Some(&mut stream)).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(le_bytes_to_f32s(&bytes), vec![expected as f32, 0.0], "cols={cols}");
        }
    }
}
