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
