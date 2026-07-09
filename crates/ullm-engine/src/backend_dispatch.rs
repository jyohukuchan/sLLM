// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendRequest<'a> {
    pub operation: &'a str,
    pub phase: &'a str,
    pub format_id: Option<&'a str>,
    pub model_arch: Option<&'a str>,
    pub gpu_arch: Option<&'a str>,
    pub gpu_name: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendImplementation<'a> {
    pub id: &'a str,
    pub operation: &'a str,
    pub phase: &'a str,
    pub format_id: Option<&'a str>,
    pub model_arch: Option<&'a str>,
    pub gpu_arch: Option<&'a str>,
    pub gpu_name: Option<&'a str>,
    pub priority: i32,
}

pub fn select_backend<'a>(
    request: &BackendRequest<'_>,
    implementations: &'a [BackendImplementation<'a>],
) -> Option<&'a BackendImplementation<'a>> {
    implementations
        .iter()
        .filter(|implementation| implementation.matches(request))
        .max_by_key(|implementation| (implementation.specificity(), implementation.priority))
}

impl BackendImplementation<'_> {
    fn matches(&self, request: &BackendRequest<'_>) -> bool {
        self.operation == request.operation
            && self.phase == request.phase
            && optional_match(self.format_id, request.format_id)
            && optional_match(self.model_arch, request.model_arch)
            && optional_match(self.gpu_arch, request.gpu_arch)
            && optional_match(self.gpu_name, request.gpu_name)
    }

    fn specificity(&self) -> i32 {
        [
            self.format_id,
            self.model_arch,
            self.gpu_arch,
            self.gpu_name,
        ]
        .into_iter()
        .filter(|value| value.is_some())
        .count() as i32
    }
}

fn optional_match(expected: Option<&str>, actual: Option<&str>) -> bool {
    match expected {
        Some(expected) => actual == Some(expected),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_gpu_decode_overrides_arch_decode() {
        let implementations = [
            BackendImplementation {
                id: "decode_Ampere",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "decode_A100",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: Some("A100_80GB"),
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "decode",
            format_id: None,
            model_arch: Some("Qwen3"),
            gpu_arch: Some("Ampere"),
            gpu_name: Some("A100_80GB"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "decode_A100");
    }

    #[test]
    fn arch_prefill_is_used_when_gpu_specific_prefill_is_absent() {
        let implementations = [
            BackendImplementation {
                id: "prefill_default",
                operation: "attention",
                phase: "prefill",
                format_id: None,
                model_arch: None,
                gpu_arch: None,
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "prefill_Ampere",
                operation: "attention",
                phase: "prefill",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("Ampere"),
                gpu_name: None,
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "prefill",
            format_id: None,
            model_arch: Some("Qwen3"),
            gpu_arch: Some("Ampere"),
            gpu_name: Some("A100_80GB"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "prefill_Ampere");
    }

    #[test]
    fn format_specific_implementation_beats_generic_arch_match() {
        let implementations = [
            BackendImplementation {
                id: "decode_RDNA4_generic",
                operation: "attention",
                phase: "decode",
                format_id: None,
                model_arch: None,
                gpu_arch: Some("RDNA4"),
                gpu_name: None,
                priority: 0,
            },
            BackendImplementation {
                id: "decode_RDNA4_SQ8_0",
                operation: "attention",
                phase: "decode",
                format_id: Some("SQ8_0"),
                model_arch: None,
                gpu_arch: Some("RDNA4"),
                gpu_name: None,
                priority: 0,
            },
        ];
        let request = BackendRequest {
            operation: "attention",
            phase: "decode",
            format_id: Some("SQ8_0"),
            model_arch: Some("Qwen3"),
            gpu_arch: Some("RDNA4"),
            gpu_name: Some("R9700"),
        };
        let selected = select_backend(&request, &implementations).unwrap();
        assert_eq!(selected.id, "decode_RDNA4_SQ8_0");
    }
}
