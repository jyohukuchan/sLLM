// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

pub fn scale_values(scale_format: &str) -> Result<Vec<f32>, String> {
    if scale_format == "e8m0" {
        return Ok((0..255).map(|code| 2.0_f32.powi(code - 127)).collect());
    }
    if let Some((exp_bits, mant_bits)) = parse_unsigned_em_scale_format(scale_format) {
        return Ok(decode_unsigned_em(exp_bits, mant_bits));
    }
    Err(format!("unknown aq scale format: {scale_format}"))
}

fn parse_unsigned_em_scale_format(scale_format: &str) -> Option<(u32, u32)> {
    let rest = scale_format
        .strip_prefix("ue")
        .or_else(|| scale_format.strip_prefix('e'))?;
    let (exp, mant) = rest.split_once('m')?;
    if exp.is_empty() || mant.is_empty() {
        return None;
    }
    let exp_bits = exp.parse::<u32>().ok()?;
    let mant_bits = mant.parse::<u32>().ok()?;
    if exp_bits == 0 || exp_bits > 8 || mant_bits > 8 {
        return None;
    }
    Some((exp_bits, mant_bits))
}

fn decode_unsigned_em(exp_bits: u32, mant_bits: u32) -> Vec<f32> {
    let mut values = Vec::new();
    let bias = (1_i32 << (exp_bits - 1)) - 1;
    let max_exp = (1_u32 << exp_bits) - 1;
    let mant_count = 1_u32 << mant_bits;
    for exp in 0..max_exp {
        for mant in 0..mant_count {
            if exp == 0 {
                if mant == 0 {
                    continue;
                }
                values.push((mant as f32 / mant_count as f32) * 2.0_f32.powi(1 - bias));
            } else {
                values.push(
                    (1.0 + mant as f32 / mant_count as f32) * 2.0_f32.powi(exp as i32 - bias),
                );
            }
        }
    }
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| left == right);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_scale_table_matches_expected_shape() {
        let values = scale_values("e4m3").unwrap();
        assert_eq!(values.len(), 119);
        assert!(values.windows(2).all(|window| window[0] < window[1]));
        assert!(values[0] > 0.0);
    }

    #[test]
    fn unsigned_prefix_is_accepted_for_scale_tables() {
        assert_eq!(
            scale_values("ue4m3").unwrap(),
            scale_values("e4m3").unwrap()
        );
    }

    #[test]
    fn e8m0_scale_table_has_255_entries() {
        let values = scale_values("e8m0").unwrap();
        assert_eq!(values.len(), 255);
        assert_eq!(values[127], 1.0);
    }
}
