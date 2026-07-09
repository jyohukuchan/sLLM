// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

const SQ8_CK_IMPLEMENTATION_UNAVAILABLE: c_int = 0;
const SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X128: c_int = 1;
const SQ8_CK_MEM_V1_KPADDING_TILE_16X128X256: c_int = 2;
const SQ8_CK_MEM_V1_DEFAULT_TILE_16X256X128: c_int = 3;
const SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X256: c_int = 4;

unsafe extern "C" {
    fn ullm_runtime_sq8_ck_quantize_activation_f32(
        input_buffer: *const RawRuntimeBuffer,
        m: usize,
        k: usize,
        quantized_buffer: *mut RawRuntimeBuffer,
        scale_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq8_ck_projection_f32(
        quantized_activation_buffer: *const RawRuntimeBuffer,
        activation_scale_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        weight_scale_buffer: *const RawRuntimeBuffer,
        m: usize,
        n: usize,
        k: usize,
        workspace_buffer: *mut RawRuntimeBuffer,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
        implementation: *mut c_int,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8CkImplementation {
    MemV1DefaultTile16x128x128,
    MemV1KPaddingTile16x128x256,
    MemV1DefaultTile16x256x128,
    MemV1DefaultTile16x128x256,
}

impl Sq8CkImplementation {
    fn from_raw(raw: c_int) -> Result<Self, String> {
        match raw {
            SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X128 => Ok(Self::MemV1DefaultTile16x128x128),
            SQ8_CK_MEM_V1_KPADDING_TILE_16X128X256 => Ok(Self::MemV1KPaddingTile16x128x256),
            SQ8_CK_MEM_V1_DEFAULT_TILE_16X256X128 => Ok(Self::MemV1DefaultTile16x256x128),
            SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X256 => Ok(Self::MemV1DefaultTile16x128x256),
            SQ8_CK_IMPLEMENTATION_UNAVAILABLE => Err(
                "SQ8 CK runtime returned an unavailable implementation after success".to_string(),
            ),
            _ => Err(format!(
                "SQ8 CK runtime returned unknown implementation id {raw}"
            )),
        }
    }
}

#[derive(Debug)]
pub struct Sq8CkQuantizedActivation {
    m: usize,
    k: usize,
    quantized: RuntimeBuffer,
    scales: RuntimeBuffer,
}

impl Sq8CkQuantizedActivation {
    /// Allocates the OCP E4M3 payload and F32 row-by-K128 scale buffers.
    pub fn allocate(context: &mut RuntimeContext, m: usize, k: usize) -> Result<Self, String> {
        let (quantized_bytes, scale_bytes) = sq8_ck_activation_buffer_bytes(m, k)?;
        Ok(Self {
            m,
            k,
            quantized: context.alloc_buffer(quantized_bytes)?,
            scales: context.alloc_buffer(scale_bytes)?,
        })
    }

    /// Enqueues exact dynamic quantization without synchronizing the stream.
    pub fn quantize_f32(
        &mut self,
        input: &RuntimeBuffer,
        stream: Option<&mut RuntimeStream>,
    ) -> Result<(), String> {
        sq8_ck_quantize_activation_f32(
            input,
            self.m,
            self.k,
            &mut self.quantized,
            &mut self.scales,
            stream,
        )
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the read-only OCP E4M3 payload buffer for projection or auditing.
    pub fn quantized_buffer(&self) -> &RuntimeBuffer {
        &self.quantized
    }

    /// Returns the read-only F32 scale buffer for projection or auditing.
    pub fn scale_buffer(&self) -> &RuntimeBuffer {
        &self.scales
    }

    pub fn quantized_bytes(&self) -> usize {
        self.m * self.k
    }

    pub fn scale_bytes(&self) -> usize {
        self.m * (self.k / 128) * std::mem::size_of::<f32>()
    }
}

pub fn sq8_ck_activation_buffer_bytes(m: usize, k: usize) -> Result<(usize, usize), String> {
    if !sq8_ck_m_is_measured(m) || k == 0 || !k.is_multiple_of(128) {
        return Err(
            "SQ8 CK activation requires M in {1,2,4,8,16,32,128} and K divisible by 128"
                .to_string(),
        );
    }
    let quantized_bytes = m
        .checked_mul(k)
        .ok_or_else(|| "SQ8 CK activation element count overflows".to_string())?;
    let scale_elements = m
        .checked_mul(k / 128)
        .ok_or_else(|| "SQ8 CK activation scale count overflows".to_string())?;
    let scale_bytes = scale_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ8 CK activation scale byte size overflows".to_string())?;
    Ok((quantized_bytes, scale_bytes))
}

pub fn sq8_ck_projection_buffer_bytes(m: usize, n: usize) -> Result<(usize, usize), String> {
    let elements = m
        .checked_mul(n)
        .ok_or_else(|| "SQ8 CK projection output element count overflows".to_string())?;
    let workspace_bytes = elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "SQ8 CK projection workspace byte size overflows".to_string())?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ8 CK projection output byte size overflows".to_string())?;
    Ok((workspace_bytes, output_bytes))
}

pub fn sq8_ck_quantize_activation_f32(
    input: &RuntimeBuffer,
    m: usize,
    k: usize,
    quantized: &mut RuntimeBuffer,
    scales: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq8_ck_quantize_activation_f32(
            input.raw.as_ptr(),
            m,
            k,
            quantized.raw.as_ptr(),
            scales.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sq8_ck_projection_buffers_f32(
    quantized_activation: &RuntimeBuffer,
    activation_scales: &RuntimeBuffer,
    weight: &RuntimeBuffer,
    weight_scales: &RuntimeBuffer,
    m: usize,
    n: usize,
    k: usize,
    workspace: &mut RuntimeBuffer,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<Sq8CkImplementation, String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let mut implementation = SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
    status_to_result(unsafe {
        ullm_runtime_sq8_ck_projection_f32(
            quantized_activation.raw.as_ptr(),
            activation_scales.raw.as_ptr(),
            weight.raw.as_ptr(),
            weight_scales.raw.as_ptr(),
            m,
            n,
            k,
            workspace.raw.as_ptr(),
            output.raw.as_ptr(),
            stream,
            &mut implementation,
        )
    })?;
    Sq8CkImplementation::from_raw(implementation)
}

pub fn sq8_ck_projection_f32(
    activation: &Sq8CkQuantizedActivation,
    weight: &RuntimeBuffer,
    weight_scales: &RuntimeBuffer,
    n: usize,
    workspace: &mut RuntimeBuffer,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<Sq8CkImplementation, String> {
    sq8_ck_projection_buffers_f32(
        activation.quantized_buffer(),
        activation.scale_buffer(),
        weight,
        weight_scales,
        activation.m(),
        n,
        activation.k(),
        workspace,
        output,
        stream,
    )
}

fn sq8_ck_m_is_measured(m: usize) -> bool {
    matches!(m, 1 | 2 | 4 | 8 | 16 | 32 | 128)
}

#[cfg(test)]
mod sq8_ck_tests {
    use super::*;

    #[test]
    fn typed_activation_retains_shape_and_exact_buffer_sizes() {
        let mut context = RuntimeContext::create(0).unwrap();
        let activation = Sq8CkQuantizedActivation::allocate(&mut context, 2, 256).unwrap();
        assert_eq!(activation.m(), 2);
        assert_eq!(activation.k(), 256);
        assert_eq!(activation.quantized_bytes(), 512);
        assert_eq!(activation.scale_bytes(), 16);
        assert_eq!(activation.quantized_buffer().size().unwrap(), 512);
        assert_eq!(activation.scale_buffer().size().unwrap(), 16);
    }

    #[cfg(not(feature = "rocm-ck-gfx1201"))]
    #[test]
    fn feature_off_quantization_stub_returns_explicit_error() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(128 * std::mem::size_of::<f32>())
            .unwrap();
        let mut activation = Sq8CkQuantizedActivation::allocate(&mut context, 1, 128).unwrap();
        let error = activation.quantize_f32(&input, None).unwrap_err();
        assert!(error.contains("requires Cargo feature rocm-ck-gfx1201"));
    }

    #[cfg(not(feature = "rocm-ck-gfx1201"))]
    #[test]
    fn feature_off_projection_stub_returns_explicit_error() {
        let mut context = RuntimeContext::create(0).unwrap();
        let activation = Sq8CkQuantizedActivation::allocate(&mut context, 1, 5120).unwrap();
        let weight = context.alloc_buffer(1).unwrap();
        let weight_scales = context.alloc_buffer(1).unwrap();
        let mut workspace = context.alloc_buffer(1).unwrap();
        let mut output = context.alloc_buffer(1).unwrap();
        let error = sq8_ck_projection_f32(
            &activation,
            &weight,
            &weight_scales,
            5120,
            &mut workspace,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(error.contains("requires Cargo feature rocm-ck-gfx1201"));
    }

    #[cfg(feature = "rocm-ck-gfx1201")]
    #[test]
    fn gfx1201_quantization_is_byte_and_scale_bit_exact() {
        let hip_index = (1..device_count().unwrap())
            .find(|index| device_info(*index).is_ok_and(|info| info.backend == "hip"))
            .expect("one isolated HIP device");
        let mut context = RuntimeContext::create(hip_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = (0..128)
            .map(|index| ((index as i32 - 63) as f32) * 0.03125)
            .collect::<Vec<_>>();
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let input_bytes = test_f32_bytes(&input_values);
        input
            .copy_from_host(0, &input_bytes, Some(&mut stream))
            .unwrap();
        let mut activation = Sq8CkQuantizedActivation::allocate(&mut context, 1, 128).unwrap();
        activation.quantize_f32(&input, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut actual_quantized = vec![0_u8; activation.quantized_bytes()];
        let mut actual_scale = vec![0_u8; activation.scale_bytes()];
        activation
            .quantized_buffer()
            .copy_to_host(0, &mut actual_quantized, Some(&mut stream))
            .unwrap();
        activation
            .scale_buffer()
            .copy_to_host(0, &mut actual_scale, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let maximum = input_values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let expected_scale = (maximum / 448.0).max(f32::from_bits(1));
        let expected_quantized = input_values
            .iter()
            .map(|value| test_encode_ocp_e4m3_rne(*value, expected_scale))
            .collect::<Vec<_>>();
        assert_eq!(actual_scale, expected_scale.to_le_bytes());
        assert_eq!(actual_quantized, expected_quantized);
    }

    #[cfg(feature = "rocm-ck-gfx1201")]
    #[test]
    fn gfx1201_projection_launches_measured_instance_and_converts_bf16() {
        const M: usize = 1;
        const N: usize = 5120;
        const K: usize = 5120;

        let hip_index = (1..device_count().unwrap())
            .find(|index| device_info(*index).is_ok_and(|info| info.backend == "hip"))
            .expect("one isolated HIP device");
        let mut context = RuntimeContext::create(hip_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut input = context
            .alloc_buffer(M * K * std::mem::size_of::<f32>())
            .unwrap();
        let input_host = vec![0_u8; M * K * std::mem::size_of::<f32>()];
        input
            .copy_from_host(0, &input_host, Some(&mut stream))
            .unwrap();
        let mut activation = Sq8CkQuantizedActivation::allocate(&mut context, M, K).unwrap();
        activation.quantize_f32(&input, Some(&mut stream)).unwrap();

        let mut weight = context.alloc_buffer(N * K).unwrap();
        let weight_host = vec![0_u8; N * K];
        weight
            .copy_from_host(0, &weight_host, Some(&mut stream))
            .unwrap();
        let weight_scale_values = vec![1.0_f32; (N / 128) * (K / 128)];
        let weight_scale_bytes = test_f32_bytes(&weight_scale_values);
        let mut weight_scales = context
            .alloc_buffer(weight_scale_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        weight_scales
            .copy_from_host(0, &weight_scale_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        drop(input_host);
        drop(weight_host);
        drop(weight_scale_bytes);
        let (workspace_bytes, output_bytes) = sq8_ck_projection_buffer_bytes(M, N).unwrap();
        let mut workspace = context.alloc_buffer(workspace_bytes).unwrap();
        let mut output = context.alloc_buffer(output_bytes).unwrap();

        let implementation = sq8_ck_projection_f32(
            &activation,
            &weight,
            &weight_scales,
            N,
            &mut workspace,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(
            implementation,
            Sq8CkImplementation::MemV1DefaultTile16x128x128
        );
        stream.synchronize().unwrap();
        let mut actual = vec![0_u8; output_bytes];
        output
            .copy_to_host(0, &mut actual, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert!(actual.iter().all(|byte| *byte == 0));
    }

    #[cfg(feature = "rocm-ck-gfx1201")]
    fn test_f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[cfg(feature = "rocm-ck-gfx1201")]
    fn test_positive_ocp_e4m3(code: u8) -> f32 {
        let exponent = (code >> 3) & 15;
        let mantissa = code & 7;
        if exponent == 0 {
            f32::from(mantissa) * 0.001953125
        } else {
            (1.0 + f32::from(mantissa) * 0.125) * 2.0_f32.powi(i32::from(exponent) - 7)
        }
    }

    #[cfg(feature = "rocm-ck-gfx1201")]
    fn test_encode_ocp_e4m3_rne(value: f32, scale: f32) -> u8 {
        let sign = ((value.to_bits() >> 24) & 0x80) as u8;
        let magnitude = (value / scale).abs();
        if magnitude == 0.0 {
            return sign;
        }
        if magnitude >= 448.0 {
            return sign | 0x7e;
        }
        let mut upper = 0_u8;
        while test_positive_ocp_e4m3(upper) < magnitude {
            upper += 1;
        }
        let upper_value = test_positive_ocp_e4m3(upper);
        if upper_value == magnitude {
            return sign | upper;
        }
        let lower = upper - 1;
        let lower_distance = f64::from(magnitude) - f64::from(test_positive_ocp_e4m3(lower));
        let upper_distance = f64::from(upper_value) - f64::from(magnitude);
        let encoded = if lower_distance < upper_distance {
            lower
        } else if upper_distance < lower_distance {
            upper
        } else if lower.is_multiple_of(2) {
            lower
        } else {
            upper
        };
        sign | encoded
    }
}
