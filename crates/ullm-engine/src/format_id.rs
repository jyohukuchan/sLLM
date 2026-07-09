// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

pub const FORMAT_AQ4_0: &str = "AQ4_0";
pub const FORMAT_SQ8_0: &str = "SQ8_0";

pub fn canonical_format_id(value: &str) -> Option<&'static str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower == "aq4"
        || lower == "aq4_0"
        || lower == "aq4-prototype-current-runtime"
        || lower.starts_with("aq4_")
    {
        return Some(FORMAT_AQ4_0);
    }
    if lower == "sq" || lower == "sq8_0" || lower == "sq-format-v0.1" || lower.starts_with("sq-fp8")
    {
        return Some(FORMAT_SQ8_0);
    }
    None
}

pub fn is_sq8_0_alias(value: &str) -> bool {
    canonical_format_id(value) == Some(FORMAT_SQ8_0)
}

pub fn is_aq4_0_alias(value: &str) -> bool {
    canonical_format_id(value) == Some(FORMAT_AQ4_0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_public_and_legacy_aq4_ids() {
        assert_eq!(canonical_format_id("AQ4_0"), Some(FORMAT_AQ4_0));
        assert_eq!(canonical_format_id("aq4"), Some(FORMAT_AQ4_0));
        assert_eq!(
            canonical_format_id("aq4-prototype-current-runtime"),
            Some(FORMAT_AQ4_0)
        );
        assert_eq!(
            canonical_format_id("aq4_e4m3_g16_ts_flloyd16"),
            Some(FORMAT_AQ4_0)
        );
    }

    #[test]
    fn canonicalizes_public_and_legacy_sq8_ids() {
        assert_eq!(canonical_format_id("SQ8_0"), Some(FORMAT_SQ8_0));
        assert_eq!(canonical_format_id("sq"), Some(FORMAT_SQ8_0));
        assert_eq!(canonical_format_id("sq-format-v0.1"), Some(FORMAT_SQ8_0));
        assert_eq!(
            canonical_format_id("sq-fp8-w8a16-r9700-v0"),
            Some(FORMAT_SQ8_0)
        );
        assert_eq!(
            canonical_format_id("sq-fp8-w8a16-r9700-v0-qkv-layer23-k16"),
            Some(FORMAT_SQ8_0)
        );
    }

    #[test]
    fn rejects_unknown_format_ids() {
        assert_eq!(canonical_format_id(""), None);
        assert_eq!(canonical_format_id("bf16"), None);
    }
}
