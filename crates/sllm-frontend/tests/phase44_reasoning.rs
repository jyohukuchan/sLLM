use sllm_core::SamplingParametersV1;
use sllm_frontend::{
    GenerationConfigV1, MAX_REASONING_TOKENS_V1, ReasoningControllerV1, ReasoningErrorV1,
    ReasoningModeV1, ReasoningPolicyV1, ThinkingModeV1,
};

#[test]
fn legacy_generation_config_has_no_reasoning_controller() {
    let config = GenerationConfigV1::new(17, SamplingParametersV1::greedy(), Vec::new())
        .expect("legacy config");
    assert!(config.reasoning().is_none());
}

#[test]
fn reasoning_budget_accepts_boundaries_and_rejects_adjacent_values() {
    assert!(ReasoningPolicyV1::enabled(Some(1), [17]).is_ok());
    assert!(ReasoningPolicyV1::enabled(Some(2_047), [17]).is_ok());
    assert!(ReasoningPolicyV1::enabled(Some(MAX_REASONING_TOKENS_V1), [17]).is_ok());
    assert!(ReasoningPolicyV1::enabled(Some(0), [17]).is_err());
    assert!(ReasoningPolicyV1::enabled(Some(MAX_REASONING_TOKENS_V1 + 1), [17]).is_err());
}

#[test]
fn max_output_admission_counts_a_multi_token_close() {
    let policy = ReasoningPolicyV1::enabled(Some(3), [17, 18, 19]).unwrap();
    assert!(policy.validate_max_new_tokens(5).is_err());
    assert!(policy.validate_max_new_tokens(6).is_ok());

    let config = GenerationConfigV1::new(5, SamplingParametersV1::greedy(), Vec::new()).unwrap();
    assert!(config.with_reasoning(policy.clone()).is_err());
    let config = GenerationConfigV1::new(6, SamplingParametersV1::greedy(), Vec::new()).unwrap();
    assert!(config.with_reasoning(policy).is_ok());
}

#[test]
fn early_close_and_forced_close_keep_marker_tokens_hidden() {
    let policy = ReasoningPolicyV1::enabled(Some(2), [17, 18]).unwrap();
    let mut controller = ReasoningControllerV1::new(policy);
    assert!(!controller.observe(17).unwrap().visible());
    assert!(controller.observe(18).unwrap().entered_answer());
    assert!(controller.observe(99).unwrap().visible());
    assert_eq!(controller.reasoning_tokens(), 0);
    assert_eq!(controller.generated_tokens(), 3);

    let policy = ReasoningPolicyV1::enabled(Some(2), [17, 18]).unwrap();
    let mut controller = ReasoningControllerV1::new(policy);
    assert!(!controller.observe(1).unwrap().visible());
    assert!(!controller.observe(2).unwrap().visible());
    assert_eq!(controller.expected_forced_token(), Some(17));
    let mask = controller
        .apply_mask(Some(&[true; 32]), Some(32))
        .unwrap()
        .unwrap();
    assert_eq!(mask.iter().filter(|value| **value).count(), 1);
    assert!(mask[17]);
    assert!(!controller.observe(17).unwrap().visible());
    assert!(controller.observe(18).unwrap().forced());
    assert!(controller.observe(42).unwrap().visible());
    assert_eq!(controller.reasoning_tokens(), 2);
    assert_eq!(controller.visible_tokens(), 1);
}

#[test]
fn grammar_intersection_and_forced_token_mismatch_fail_closed() {
    let policy = ReasoningPolicyV1::enabled(Some(1), [17]).unwrap();
    let mut controller = ReasoningControllerV1::new(policy);
    controller.observe(1).unwrap();
    let mut grammar_mask = [true; 32];
    grammar_mask[17] = false;
    assert_eq!(
        controller.apply_mask(Some(&grammar_mask), Some(32)),
        Err(ReasoningErrorV1::CandidateMaskEmpty)
    );
    assert_eq!(
        controller.apply_mask(None, Some(8)),
        Err(ReasoningErrorV1::TokenOutsideVocabulary)
    );

    let policy = ReasoningPolicyV1::enabled(Some(1), [17]).unwrap();
    let mut controller = ReasoningControllerV1::new(policy);
    controller.observe(1).unwrap();
    assert_eq!(
        controller.observe(16),
        Err(ReasoningErrorV1::ForcedTokenMismatch)
    );
}

#[test]
fn thinking_mode_mapping_and_template_default_are_explicit() {
    let policy = ReasoningPolicyV1::from_thinking(ThinkingModeV1::Enabled, Some(1), [17]).unwrap();
    assert_eq!(policy.mode(), ReasoningModeV1::Enabled);
    let default =
        ReasoningPolicyV1::from_thinking(ThinkingModeV1::TemplateDefault, Some(1), [17]).unwrap();
    assert!(!default.is_enabled());
}
