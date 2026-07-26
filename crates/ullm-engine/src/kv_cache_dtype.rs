// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Runtime-selectable storage contracts for paged K/V caches.
//!
//! Attention inputs and outputs remain F32.  This module describes only the
//! persistent K/V payload and, for FP8, its independently-addressable scale
//! metadata.  Keeping the layout arithmetic here makes allocation, runtime
//! ABI validation, and benchmark reporting use the same byte contract.

use std::fmt;
use std::str::FromStr;

/// Optional uniform cache selector.  Per-plane selectors take precedence.
pub const KV_CACHE_DTYPE_ENV: &str = "ULLM_KV_CACHE_DTYPE";
/// Optional persistent-key cache selector.
pub const KV_CACHE_TYPE_K_ENV: &str = "ULLM_KV_CACHE_TYPE_K";
/// Optional persistent-value cache selector.
pub const KV_CACHE_TYPE_V_ENV: &str = "ULLM_KV_CACHE_TYPE_V";

/// Storage representation of one persistent K or V value.
///
/// `Fp8E4M3Fn` uses OCP FP8 E4M3FN payload bytes and one IEEE FP16 scale per
/// `(physical_token, kv_head, plane)`.  It deliberately does not alias any
/// SQ8_0 artifact format: this is a dynamically-written cache layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvCacheDtype {
    #[default]
    F32,
    F16,
    Fp8E4M3Fn,
}

impl KvCacheDtype {
    pub const FFI_F32: u32 = 0;
    pub const FFI_F16: u32 = 1;
    pub const FFI_FP8_E4M3FN: u32 = 2;

    pub const FP8_SCALE_BYTES: usize = std::mem::size_of::<u16>();

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Fp8E4M3Fn => "fp8_e4m3fn",
        }
    }

    pub const fn ffi_code(self) -> u32 {
        match self {
            Self::F32 => Self::FFI_F32,
            Self::F16 => Self::FFI_F16,
            Self::Fp8E4M3Fn => Self::FFI_FP8_E4M3FN,
        }
    }

    pub const fn payload_bytes_per_value(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::F16 => std::mem::size_of::<u16>(),
            Self::Fp8E4M3Fn => std::mem::size_of::<u8>(),
        }
    }

    pub const fn needs_fp8_scale(self) -> bool {
        matches!(self, Self::Fp8E4M3Fn)
    }
}

/// Decodes the IEEE-754 binary16 bit pattern used for FP16 K/V payloads and
/// FP8 scale metadata.  Keeping this bit-level helper in the layout module
/// lets diagnostic readback use exactly the same on-device storage contract
/// without introducing an additional numeric dependency into the engine.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    match exponent {
        0 if mantissa == 0 => sign * 0.0,
        0 => sign * (mantissa as f32) * 2.0_f32.powi(-24),
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + (mantissa as f32) / 1024.0) * 2.0_f32.powi(exponent as i32 - 15),
    }
}

/// Decodes one OCP FP8 E4M3FN payload byte.
///
/// `0x7f`/`0xff` are NaN and are never emitted by the cache writer for finite
/// source values.  The separate FP16 scale is intentionally not applied here.
pub fn fp8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32) * 0.001953125
    } else {
        (1.0 + (mantissa as f32) * 0.125) * 2.0_f32.powi(exponent as i32 - 7)
    };
    sign * magnitude
}

impl fmt::Display for KvCacheDtype {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for KvCacheDtype {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "f16" | "fp16" | "float16" => Ok(Self::F16),
            "fp8" | "fp8_e4m3fn" | "e4m3fn" | "e4m3" => Ok(Self::Fp8E4M3Fn),
            "q8_0" | "q8" => Err(
                "Q8_0 is not an implemented KV-cache dtype; choose f32, f16, or fp8_e4m3fn"
                    .to_string(),
            ),
            _ => Err(format!(
                "unknown KV-cache dtype {value:?}; choose f32, f16, or fp8_e4m3fn"
            )),
        }
    }
}

/// Separate K/V storage selection, analogous to llama.cpp's cache-type-k/v.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KvCacheDtypes {
    pub key: KvCacheDtype,
    pub value: KvCacheDtype,
}

impl KvCacheDtypes {
    pub const fn uniform(dtype: KvCacheDtype) -> Self {
        Self {
            key: dtype,
            value: dtype,
        }
    }

    fn parse_selector(name: &str, value: &str) -> Result<KvCacheDtype, String> {
        value
            .parse()
            .map_err(|error: String| format!("{name}={value:?}: {error}"))
    }

    fn from_optional_values(
        uniform: Option<&str>,
        key: Option<&str>,
        value: Option<&str>,
    ) -> Result<Self, String> {
        let mut dtypes = match uniform {
            Some(value) => Self::uniform(Self::parse_selector(KV_CACHE_DTYPE_ENV, value)?),
            None => Self::default(),
        };
        if let Some(value) = key {
            dtypes.key = Self::parse_selector(KV_CACHE_TYPE_K_ENV, value)?;
        }
        if let Some(value) = value {
            dtypes.value = Self::parse_selector(KV_CACHE_TYPE_V_ENV, value)?;
        }
        Ok(dtypes)
    }

    /// Reads the opt-in process environment without changing the F32 default.
    ///
    /// `ULLM_KV_CACHE_DTYPE` sets both planes.  `ULLM_KV_CACHE_TYPE_K` and
    /// `ULLM_KV_CACHE_TYPE_V` then override only their respective plane.
    pub fn from_env() -> Result<Self, String> {
        fn read_optional_env(name: &str) -> Result<Option<String>, String> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(error) => Err(format!("failed to read {name}: {error}")),
            }
        }

        let uniform = read_optional_env(KV_CACHE_DTYPE_ENV)?;
        let key = read_optional_env(KV_CACHE_TYPE_K_ENV)?;
        let value = read_optional_env(KV_CACHE_TYPE_V_ENV)?;
        Self::from_optional_values(uniform.as_deref(), key.as_deref(), value.as_deref())
    }
}

/// Exact allocation accounting for one paged K/V cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheLayout {
    pub dtypes: KvCacheDtypes,
    pub physical_tokens: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub k_payload_bytes: usize,
    pub v_payload_bytes: usize,
    pub k_scale_bytes: usize,
    pub v_scale_bytes: usize,
}

impl KvCacheLayout {
    pub fn new(
        dtypes: KvCacheDtypes,
        physical_tokens: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Result<Self, String> {
        if physical_tokens == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
            return Err("KV-cache layout dimensions must be greater than zero".to_string());
        }
        let per_plane_scale_values = physical_tokens
            .checked_mul(kv_heads)
            .ok_or_else(|| "KV-cache scale count overflows".to_string())?;
        let k_values = per_plane_scale_values
            .checked_mul(head_dim)
            .ok_or_else(|| "KV-cache K payload value count overflows".to_string())?;
        let v_values = per_plane_scale_values
            .checked_mul(value_dim)
            .ok_or_else(|| "KV-cache V payload value count overflows".to_string())?;
        let k_payload_bytes = k_values
            .checked_mul(dtypes.key.payload_bytes_per_value())
            .ok_or_else(|| "KV-cache K payload byte count overflows".to_string())?;
        let v_payload_bytes = v_values
            .checked_mul(dtypes.value.payload_bytes_per_value())
            .ok_or_else(|| "KV-cache V payload byte count overflows".to_string())?;
        let scale_bytes = per_plane_scale_values
            .checked_mul(KvCacheDtype::FP8_SCALE_BYTES)
            .ok_or_else(|| "KV-cache FP8 scale byte count overflows".to_string())?;
        Ok(Self {
            dtypes,
            physical_tokens,
            kv_heads,
            head_dim,
            value_dim,
            k_payload_bytes,
            v_payload_bytes,
            k_scale_bytes: dtypes
                .key
                .needs_fp8_scale()
                .then_some(scale_bytes)
                .unwrap_or(0),
            v_scale_bytes: dtypes
                .value
                .needs_fp8_scale()
                .then_some(scale_bytes)
                .unwrap_or(0),
        })
    }

    pub fn total_bytes(self) -> Result<usize, String> {
        self.k_payload_bytes
            .checked_add(self.v_payload_bytes)
            .and_then(|value| value.checked_add(self.k_scale_bytes))
            .and_then(|value| value.checked_add(self.v_scale_bytes))
            .ok_or_else(|| "KV-cache total byte count overflows".to_string())
    }

    pub const fn k_scale_values(self) -> usize {
        if self.dtypes.key.needs_fp8_scale() {
            self.physical_tokens * self.kv_heads
        } else {
            0
        }
    }

    pub const fn v_scale_values(self) -> usize {
        if self.dtypes.value.needs_fp8_scale() {
            self.physical_tokens * self.kv_heads
        } else {
            0
        }
    }

    pub fn effective_k_bytes_per_value(self) -> f64 {
        self.k_payload_bytes.saturating_add(self.k_scale_bytes) as f64
            / (self.physical_tokens * self.kv_heads * self.head_dim) as f64
    }

    pub fn effective_v_bytes_per_value(self) -> f64 {
        self.v_payload_bytes.saturating_add(self.v_scale_bytes) as f64
            / (self.physical_tokens * self.kv_heads * self.value_dim) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_default_and_parser_keep_q8_0_out_of_scope() {
        assert_eq!(
            KvCacheDtypes::default(),
            KvCacheDtypes::uniform(KvCacheDtype::F32)
        );
        assert_eq!("f16".parse(), Ok(KvCacheDtype::F16));
        assert_eq!("fp8_e4m3fn".parse(), Ok(KvCacheDtype::Fp8E4M3Fn));
        assert!(
            "Q8_0"
                .parse::<KvCacheDtype>()
                .unwrap_err()
                .contains("not an implemented")
        );
    }

    #[test]
    fn kv_selectors_allow_kv_override_without_process_environment_mutation() {
        assert_eq!(
            KvCacheDtypes::from_optional_values(Some("f16"), None, None).unwrap(),
            KvCacheDtypes::uniform(KvCacheDtype::F16)
        );
        assert_eq!(
            KvCacheDtypes::from_optional_values(
                Some("f16"),
                Some("fp8_e4m3fn"),
                Some("f32"),
            )
            .unwrap(),
            KvCacheDtypes {
                key: KvCacheDtype::Fp8E4M3Fn,
                value: KvCacheDtype::F32,
            }
        );
        assert!(KvCacheDtypes::from_optional_values(None, Some("Q8_0"), None)
            .unwrap_err()
            .contains(KV_CACHE_TYPE_K_ENV));
    }

    #[test]
    fn qwen35_block16_layout_accounts_for_fp8_scales() {
        let f32 = KvCacheLayout::new(
            KvCacheDtypes::uniform(KvCacheDtype::F32),
            256 * 16,
            4,
            256,
            256,
        )
        .unwrap();
        let f16 = KvCacheLayout::new(
            KvCacheDtypes::uniform(KvCacheDtype::F16),
            256 * 16,
            4,
            256,
            256,
        )
        .unwrap();
        let fp8 = KvCacheLayout::new(
            KvCacheDtypes::uniform(KvCacheDtype::Fp8E4M3Fn),
            256 * 16,
            4,
            256,
            256,
        )
        .unwrap();

        assert_eq!(f32.total_bytes().unwrap(), 32 * 1024 * 1024);
        assert_eq!(f16.total_bytes().unwrap(), 16 * 1024 * 1024);
        assert_eq!(fp8.k_scale_bytes, 32 * 1024);
        assert_eq!(fp8.v_scale_bytes, 32 * 1024);
        assert_eq!(fp8.total_bytes().unwrap(), 8 * 1024 * 1024 + 64 * 1024);
        assert_eq!(fp8.effective_k_bytes_per_value(), 1.0078125);
        assert_eq!(fp8.effective_v_bytes_per_value(), 1.0078125);
    }

    #[test]
    fn mixed_kv_layout_is_explicit() {
        let layout = KvCacheLayout::new(
            KvCacheDtypes {
                key: KvCacheDtype::F16,
                value: KvCacheDtype::Fp8E4M3Fn,
            },
            16,
            4,
            256,
            256,
        )
        .unwrap();
        assert_eq!(layout.k_scale_bytes, 0);
        assert_eq!(layout.v_scale_values(), 64);
        assert_eq!(layout.effective_k_bytes_per_value(), 2.0);
        assert_eq!(layout.effective_v_bytes_per_value(), 1.0078125);
    }
}
