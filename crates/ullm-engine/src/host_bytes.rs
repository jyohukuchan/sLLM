// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

pub fn decode_f32_le_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

pub fn encode_f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn encode_u32_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_f32_to_bytes_uses_little_endian_layout() {
        let values = [1.0_f32, -2.5_f32, f32::from_bits(0x7fc0_1234)];

        let encoded = encode_f32_to_bytes(&values);

        let mut expected = Vec::new();
        for value in values {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(encoded, expected);
        assert_eq!(
            decode_f32_le_values(&encoded)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn encode_u32_to_bytes_uses_little_endian_layout() {
        let values = [0_u32, 1_u32, 0x1234_5678_u32, u32::MAX];

        let encoded = encode_u32_to_bytes(&values);

        let mut expected = Vec::new();
        for value in values {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(encoded, expected);
    }
}
