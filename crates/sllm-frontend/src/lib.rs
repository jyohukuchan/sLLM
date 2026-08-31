#![forbid(unsafe_code)]

use core::fmt;
use std::borrow::Borrow;

mod chat;
mod gemma4_mtp_generation;
mod generation;
mod inference;
mod ministral3;
mod reasoning;
mod template;
mod tokenizer;
mod tool_protocol;
mod vision;

pub use chat::{
    ChatFieldV1, ChatMessageV1, ChatRenderError, ChatRenderOptionsV1, ChatTemplateRenderResultV1,
    ChatTemplateRendererErrorV1, ChatTemplateRendererV1, GEMMA4_MOE_CHAT_TEMPLATE_FILENAME,
    GEMMA4_MOE_CHAT_TEMPLATE_SHA256, GEMMA4_MOE_CHAT_TEMPLATE_SIZE_BYTES,
    Gemma4MoeChatTemplateErrorV1, Gemma4MoeChatTemplateV1, GenericChatTemplateConfigV1,
    QWEN35_CHAT_MAX_OUTPUT_BYTES, QWEN35_CHAT_RENDERER_VERSION, QWEN35_CHAT_TEMPLATE_FILENAME,
    QWEN35_CHAT_TEMPLATE_SHA256, QWEN35_CHAT_TEMPLATE_SIZE_BYTES, Qwen35ChatMessageV1,
    Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, ThinkingModeV1, UntrustedChatMessageV1,
    UntrustedChatRequestV1, UntrustedChatValueV1,
};
pub use gemma4_mtp_generation::{GEMMA4_MTP_TARGET_HIDDEN_WIDTH, Gemma4MtpGenerationExecutorV1};
pub use generation::{
    FinishReasonV1, GenerationCancellationV1, GenerationChoiceResultV1, GenerationChoicesResultV1,
    GenerationConfigV1, GenerationExecutorV1, GenerationInputV1, GenerationOutputSinkV1,
    GenerationResultV1, GenerationServiceError, GenerationServiceV1, GenerationStepV1,
    GenerationTextFrontendV1, GenericGenerationInputV1, MAX_GENERATION_CHOICES_V1,
    MAX_STOP_STRING_BYTES_V1, MAX_STOP_STRINGS_V1, PreparedGenerationInputV1,
    QwenMtpGenerationExecutorV1, SpeculativeGenerationAdapterV1, SpeculativeGenerationExecutorV1,
    TokenUsageV1, derive_choice_seed_v1, gemma4_generation_stop_policy,
    gemma4_moe_generation_stop_policy,
};
pub use inference::{
    ApplyTemplateResultV1, FIM_TEMPLATE_VERSION_V1, FimTemplateErrorV1, FimTemplateV1,
    GenericTemplateApplyInputV1, GenericTemplateInputKindV1, GenericTemplateInputV1,
    GenericTemplateMessagesInputV1, GenericTemplateMessagesV1, InputTokenCountInputV1,
    MAX_TOKENIZER_UTILITY_INPUT_BYTES_V1, TOKENIZER_UTILITY_VERSION_V1, TemplateIdentityV1,
    TokenPieceV1, TokenizeOptionsV1, TokenizeResultV1, TokenizerUtilityErrorV1,
    TokenizerUtilityServiceV1,
};
pub use ministral3::{
    MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1, MINISTRAL3_CHAT_TEMPLATE_FILENAME,
    MINISTRAL3_CHAT_TEMPLATE_SHA256, MINISTRAL3_CHAT_TEMPLATE_SIZE_BYTES,
    MINISTRAL3_DEFAULT_SYSTEM_PROMPT, MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256,
    MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES, MINISTRAL3_FRONTEND_VERSION_V1,
    MINISTRAL3_HISTORY_FIXTURE_RENDERED_SHA256, MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_SHA256,
    MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_V1, MINISTRAL3_MAX_MESSAGES_V1,
    MINISTRAL3_SYSTEM_USER_FIXTURE_RENDERED_SHA256,
    MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_SHA256, MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_V1,
    MINISTRAL3_TOKENIZER_FILENAME, MINISTRAL3_TOKENIZER_SHA256, MINISTRAL3_TOKENIZER_SIZE_BYTES,
    Ministral3ChatRendererV1, Ministral3ChatTemplateV1, Ministral3FrontendErrorV1,
    Ministral3RenderOptionsV1, Ministral3TextFrontendV1, Ministral3TokenizerV1,
    ministral3_generation_stop_policy,
};
pub use reasoning::{
    MAX_REASONING_CLOSE_TOKENS_V1, MAX_REASONING_TOKENS_V1, ReasoningControllerV1,
    ReasoningErrorV1, ReasoningModeV1, ReasoningObservationV1, ReasoningPolicyV1,
};
pub use template::{
    GENERIC_TEMPLATE_MAX_FUEL_V1, GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1,
    GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1, GENERIC_TEMPLATE_MAX_KWARGS_V1,
    GENERIC_TEMPLATE_MAX_MESSAGES_V1, GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
    GENERIC_TEMPLATE_MAX_RECURSION_V1, GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1,
    GENERIC_TEMPLATE_PROFILE_VERSION_V1, GENERIC_TEMPLATE_REVIEWED_GEMMA4_PROFILE_VERSION_V1,
    GenericTemplateContextV1, GenericTemplateErrorV1, GenericTemplateIdentityV1,
    GenericTemplateProviderV1, GenericTemplateRenderResultV1, GenericTemplateRendererV1,
    GenericTemplateSourceV1,
};
pub use tokenizer::{
    DecodeModeV1, EosIdentitySnapshotV1, EosIdentityV1, MAX_TOKEN_PIECE_BYTES_V1,
    SpecialTokenSnapshotV1, TokenByteEntryV1, TokenByteTableV1, TokenIdContextV1, TokenIdsV1,
    TokenPieceClassV1, TokenizerError, TokenizerFrontendV1, TokenizerSnapshotV1,
};
pub use tool_protocol::{
    CanonicalGenerationEnvelopeV1, CanonicalToolCallV1, MAX_QWEN_TOOL_PROMPT_BYTES_V1,
    MAX_TOOL_ARGUMENT_BYTES_V1, MAX_TOOL_CALL_ID_BYTES_V1, MAX_TOOL_CALLS_V1,
    MAX_TOOL_DEFINITIONS_V1, MAX_TOOL_DESCRIPTION_BYTES_V1, MAX_TOOL_GENERATION_ENVELOPE_BYTES_V1,
    MAX_TOOL_HISTORY_ITEMS_V1, MAX_TOOL_NAME_BYTES_V1, MAX_TOOL_REASONING_BYTES_V1,
    MAX_TOOL_RESULT_BYTES_V1, MAX_TOOL_SCHEMA_BYTES_V1, MAX_TOOL_SCHEMA_DEPTH_V1,
    QWEN_TOOL_PROTOCOL_CLOSE_V1, QWEN_TOOL_PROTOCOL_OPEN_V1, TOOL_PROTOCOL_VERSION_V1,
    ToolCallPolicyV1, ToolCallV1, ToolChoiceV1, ToolDefinitionV1, ToolMessageRoleV1,
    ToolProtocolError, ToolProtocolItemV1, ToolProtocolV1, ToolResultV1,
};
pub use vision::{
    BoundedImageBytesV1, DecodedRgbImageV1, MAX_TOTAL_VISUAL_TOKENS_V1, ProcessedVisionInputV1,
    Qwen35VisionProcessorV1, VisionErrorV1, VisionImageFormatV1, VisionPatchPositionV1,
};

pub use sllm_core::{
    BudgetBoundary, GenerationStopPolicyV1, MaxNewTokensZero, PromptEvaluation, StopEvaluation,
    StopTokenHandling,
};

pub const GENERATION_STOP_POLICY_VERSION: u8 = 1;
pub const GENERATION_STOP_REASON_VERSION: u8 = 1;

// Tokenizer encode/decode and the fixed Qwen3.5 renderer are part of this
// frontend boundary. Keep an explicit type reference to the pinned tokenizer
// dependency.
#[allow(dead_code)]
const TOKENIZERS_DEPENDENCY_MARKER: core::marker::PhantomData<tokenizers::Tokenizer> =
    core::marker::PhantomData;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReasonKindV1 {
    StopToken,
    MaxNewTokens,
}

impl StopReasonKindV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StopToken => "stop_token",
            Self::MaxNewTokens => "max_new_tokens",
        }
    }

    pub const fn reason_token(self) -> &'static str {
        self.as_str()
    }
}

impl fmt::Display for StopReasonKindV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StopReasonV1 {
    version: u8,
    reason_version: u8,
    kind: StopReasonKindV1,
    token_id: Option<u32>,
}

impl StopReasonV1 {
    pub const fn stop_token(token_id: u32) -> Self {
        Self {
            version: GENERATION_STOP_POLICY_VERSION,
            reason_version: GENERATION_STOP_REASON_VERSION,
            kind: StopReasonKindV1::StopToken,
            token_id: Some(token_id),
        }
    }

    pub const fn max_new_tokens() -> Self {
        Self {
            version: GENERATION_STOP_POLICY_VERSION,
            reason_version: GENERATION_STOP_REASON_VERSION,
            kind: StopReasonKindV1::MaxNewTokens,
            token_id: None,
        }
    }

    pub const fn reason_token(self) -> &'static str {
        self.kind.reason_token()
    }

    pub const fn version(self) -> u8 {
        self.version
    }

    pub const fn reason_version(self) -> u8 {
        self.reason_version
    }

    pub const fn kind(self) -> StopReasonKindV1 {
        self.kind
    }

    pub const fn token_id(self) -> Option<u32> {
        self.token_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedTokenDecisionV1 {
    generated_token_id: u32,
    visible_token_id: Option<u32>,
    decode_input_token_id: Option<u32>,
    stop_reason: Option<StopReasonV1>,
}

impl GeneratedTokenDecisionV1 {
    const fn continue_token(token_id: u32) -> Self {
        Self {
            generated_token_id: token_id,
            visible_token_id: Some(token_id),
            decode_input_token_id: Some(token_id),
            stop_reason: None,
        }
    }

    const fn max_new_tokens(token_id: u32) -> Self {
        Self {
            generated_token_id: token_id,
            visible_token_id: Some(token_id),
            decode_input_token_id: None,
            stop_reason: Some(StopReasonV1::max_new_tokens()),
        }
    }

    const fn stop_token(token_id: u32) -> Self {
        Self {
            generated_token_id: token_id,
            visible_token_id: None,
            decode_input_token_id: None,
            stop_reason: Some(StopReasonV1::stop_token(token_id)),
        }
    }

    pub const fn generated_token_id(self) -> u32 {
        self.generated_token_id
    }

    pub const fn visible_token_id(self) -> Option<u32> {
        self.visible_token_id
    }

    pub const fn decode_input_token_id(self) -> Option<u32> {
        self.decode_input_token_id
    }

    pub const fn stop_reason(self) -> Option<StopReasonV1> {
        self.stop_reason
    }

    pub const fn is_terminal(self) -> bool {
        self.stop_reason.is_some()
    }

    pub const fn is_visible(self) -> bool {
        self.visible_token_id.is_some()
    }

    pub const fn should_feed_decode(self) -> bool {
        self.decode_input_token_id.is_some()
    }

    pub const fn should_continue(self) -> bool {
        !self.is_terminal()
    }

    pub const fn reason_token(self) -> Option<&'static str> {
        match self.stop_reason {
            Some(reason) => Some(reason.reason_token()),
            None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopPolicyError {
    InvalidPolicyVersion {
        actual: u8,
    },
    InvalidReasonVersion {
        actual: u8,
    },
    InvalidEvaluation,
    InvalidPromptEvaluation,
    InvalidStopTokenVisibility,
    InvalidStopTokenDecodeInput,
    InvalidBudgetBoundary,
    InvalidMaxNewTokensZero,
    EmptyStopTokenIds,
    DuplicateStopTokenId {
        token_id: u32,
    },
    AlreadyStopped,
    ArgmaxAfterZeroBudget,
    InvalidGeneratedCount,
    GeneratedCountExceedsBudget {
        generated_count_after_token: u64,
        max_new_tokens: u32,
    },
    CountOverflow,
}

impl fmt::Display for StopPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicyVersion { actual } => {
                write!(
                    formatter,
                    "unsupported generation stop policy version {actual}"
                )
            }
            Self::InvalidReasonVersion { actual } => {
                write!(
                    formatter,
                    "unsupported generation stop reason version {actual}"
                )
            }
            Self::InvalidEvaluation => formatter.write_str("invalid stop evaluation mode"),
            Self::InvalidPromptEvaluation => formatter.write_str("invalid prompt evaluation mode"),
            Self::InvalidStopTokenVisibility => {
                formatter.write_str("stop tokens must not be visible")
            }
            Self::InvalidStopTokenDecodeInput => {
                formatter.write_str("stop tokens must not feed a subsequent decode")
            }
            Self::InvalidBudgetBoundary => formatter.write_str("invalid budget boundary mode"),
            Self::InvalidMaxNewTokensZero => {
                formatter.write_str("invalid max_new_tokens=0 handling mode")
            }
            Self::EmptyStopTokenIds => formatter.write_str("stop token IDs must be nonempty"),
            Self::DuplicateStopTokenId { token_id } => {
                write!(formatter, "stop token ID {token_id} is duplicated")
            }
            Self::AlreadyStopped => formatter.write_str("generation has already stopped"),
            Self::ArgmaxAfterZeroBudget => {
                formatter.write_str("argmax must not run when max_new_tokens is zero")
            }
            Self::InvalidGeneratedCount => {
                formatter.write_str("generated token count must be nonzero")
            }
            Self::GeneratedCountExceedsBudget {
                generated_count_after_token,
                max_new_tokens,
            } => write!(
                formatter,
                "generated token count {generated_count_after_token} exceeds max_new_tokens {max_new_tokens}"
            ),
            Self::CountOverflow => formatter.write_str("generated token count overflowed"),
        }
    }
}

impl std::error::Error for StopPolicyError {}

pub fn validate_generation_stop_policy(
    policy: &GenerationStopPolicyV1,
) -> Result<(), StopPolicyError> {
    if policy.version != GENERATION_STOP_POLICY_VERSION {
        return Err(StopPolicyError::InvalidPolicyVersion {
            actual: policy.version,
        });
    }
    if policy.reason_version != GENERATION_STOP_REASON_VERSION {
        return Err(StopPolicyError::InvalidReasonVersion {
            actual: policy.reason_version,
        });
    }
    if !matches!(policy.evaluation, StopEvaluation::NewlyGeneratedAfterArgmax) {
        return Err(StopPolicyError::InvalidEvaluation);
    }
    if !matches!(policy.prompt_evaluation, PromptEvaluation::NeverStop) {
        return Err(StopPolicyError::InvalidPromptEvaluation);
    }
    if policy.stop_token.visible_output {
        return Err(StopPolicyError::InvalidStopTokenVisibility);
    }
    if policy.stop_token.subsequent_decode_input {
        return Err(StopPolicyError::InvalidStopTokenDecodeInput);
    }
    if !matches!(policy.budget_boundary, BudgetBoundary::StopTokenWins) {
        return Err(StopPolicyError::InvalidBudgetBoundary);
    }
    if !matches!(
        policy.max_new_tokens_zero,
        MaxNewTokensZero::MaxNewTokensBeforeDecode
    ) {
        return Err(StopPolicyError::InvalidMaxNewTokensZero);
    }
    if policy.stop_token_ids.is_empty() {
        return Err(StopPolicyError::EmptyStopTokenIds);
    }
    for (index, token_id) in policy.stop_token_ids.iter().enumerate() {
        if policy.stop_token_ids[..index].contains(token_id) {
            return Err(StopPolicyError::DuplicateStopTokenId {
                token_id: *token_id,
            });
        }
    }
    Ok(())
}

pub fn stop_before_decode(max_new_tokens: u32) -> Option<StopReasonV1> {
    (max_new_tokens == 0).then(StopReasonV1::max_new_tokens)
}

fn decide_after_argmax_count(
    policy: &GenerationStopPolicyV1,
    token_id: u32,
    generated_count_after_token: u64,
    max_new_tokens: u32,
) -> Result<GeneratedTokenDecisionV1, StopPolicyError> {
    if max_new_tokens == 0 {
        return Err(StopPolicyError::ArgmaxAfterZeroBudget);
    }
    if generated_count_after_token == 0 {
        return Err(StopPolicyError::InvalidGeneratedCount);
    }
    if generated_count_after_token > u64::from(max_new_tokens) {
        return Err(StopPolicyError::GeneratedCountExceedsBudget {
            generated_count_after_token,
            max_new_tokens,
        });
    }

    if policy.stop_token_ids.contains(&token_id) {
        return Ok(GeneratedTokenDecisionV1::stop_token(token_id));
    }

    let at_budget = generated_count_after_token >= u64::from(max_new_tokens);
    Ok(if at_budget {
        GeneratedTokenDecisionV1::max_new_tokens(token_id)
    } else {
        GeneratedTokenDecisionV1::continue_token(token_id)
    })
}

pub fn decide_after_argmax(
    policy: &GenerationStopPolicyV1,
    token_id: u32,
    generated_count_after_token: u32,
    max_new_tokens: u32,
) -> Result<GeneratedTokenDecisionV1, StopPolicyError> {
    validate_generation_stop_policy(policy)?;
    decide_after_argmax_count(
        policy,
        token_id,
        u64::from(generated_count_after_token),
        max_new_tokens,
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationReportV1 {
    version: u8,
    reason_version: u8,
    input_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    visible_token_ids: Vec<u32>,
    decode_input_token_ids: Vec<u32>,
    stop_reason: Option<StopReasonV1>,
}

impl GenerationReportV1 {
    fn new(input_token_ids: &[u32]) -> Self {
        Self {
            version: GENERATION_STOP_POLICY_VERSION,
            reason_version: GENERATION_STOP_REASON_VERSION,
            input_token_ids: input_token_ids.to_vec(),
            generated_token_ids: Vec::new(),
            visible_token_ids: Vec::new(),
            decode_input_token_ids: Vec::new(),
            stop_reason: None,
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stop_reason.is_some()
    }

    pub const fn version(&self) -> u8 {
        self.version
    }

    pub const fn reason_version(&self) -> u8 {
        self.reason_version
    }

    pub fn input_token_ids(&self) -> &[u32] {
        &self.input_token_ids
    }

    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    pub fn visible_token_ids(&self) -> &[u32] {
        &self.visible_token_ids
    }

    pub fn decode_input_token_ids(&self) -> &[u32] {
        &self.decode_input_token_ids
    }

    pub const fn stop_reason(&self) -> Option<StopReasonV1> {
        self.stop_reason
    }

    pub fn reason_token(&self) -> Option<&'static str> {
        self.stop_reason.map(StopReasonV1::reason_token)
    }

    pub fn stop_token_id(&self) -> Option<u32> {
        self.stop_reason.and_then(StopReasonV1::token_id)
    }
}

#[derive(Clone, Debug)]
pub struct GenerationStopControllerV1 {
    policy: GenerationStopPolicyV1,
    max_new_tokens: u32,
    report: GenerationReportV1,
}

impl GenerationStopControllerV1 {
    pub fn new<P>(policy: P, max_new_tokens: u32) -> Result<Self, StopPolicyError>
    where
        P: Borrow<GenerationStopPolicyV1>,
    {
        Self::new_with_input_token_ids(policy, max_new_tokens, &[])
    }

    pub fn new_with_input_token_ids<P>(
        policy: P,
        max_new_tokens: u32,
        input_token_ids: &[u32],
    ) -> Result<Self, StopPolicyError>
    where
        P: Borrow<GenerationStopPolicyV1>,
    {
        validate_generation_stop_policy(policy.borrow())?;
        let mut report = GenerationReportV1::new(input_token_ids);
        if let Some(reason) = stop_before_decode(max_new_tokens) {
            report.stop_reason = Some(reason);
        }
        Ok(Self {
            policy: policy.borrow().clone(),
            max_new_tokens,
            report,
        })
    }

    pub fn observe_generated(
        &mut self,
        token_id: u32,
    ) -> Result<GeneratedTokenDecisionV1, StopPolicyError> {
        if self.is_stopped() {
            return Err(StopPolicyError::AlreadyStopped);
        }

        // Retain every generated argmax before exposing the decision. Stop IDs
        // are deliberately absent only from the visible and decode-input lists.
        self.report.generated_token_ids.push(token_id);
        let generated_count_after_token = match u64::try_from(self.report.generated_token_ids.len())
        {
            Ok(count) => count,
            Err(_) => {
                self.report.generated_token_ids.pop();
                return Err(StopPolicyError::CountOverflow);
            }
        };
        let decision = match decide_after_argmax_count(
            &self.policy,
            token_id,
            generated_count_after_token,
            self.max_new_tokens,
        ) {
            Ok(decision) => decision,
            Err(error) => {
                self.report.generated_token_ids.pop();
                return Err(error);
            }
        };

        if let Some(visible_token_id) = decision.visible_token_id() {
            self.report.visible_token_ids.push(visible_token_id);
        }
        if let Some(decode_input_token_id) = decision.decode_input_token_id() {
            self.report
                .decode_input_token_ids
                .push(decode_input_token_id);
        }
        if let Some(reason) = decision.stop_reason() {
            self.report.stop_reason = Some(reason);
        }
        Ok(decision)
    }

    pub fn observe_generated_token(
        &mut self,
        token_id: u32,
    ) -> Result<GeneratedTokenDecisionV1, StopPolicyError> {
        self.observe_generated(token_id)
    }

    pub fn policy(&self) -> &GenerationStopPolicyV1 {
        &self.policy
    }

    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    pub fn is_stopped(&self) -> bool {
        self.report.is_stopped()
    }

    pub fn report(&self) -> &GenerationReportV1 {
        &self.report
    }

    pub fn into_report(self) -> GenerationReportV1 {
        self.report
    }
}

pub type GeneratedTokenDecision = GeneratedTokenDecisionV1;
pub type GenerationReport = GenerationReportV1;
pub type GenerationStopController = GenerationStopControllerV1;
pub type StopControllerV1 = GenerationStopControllerV1;
pub type StopReason = StopReasonV1;
pub type StopReasonKind = StopReasonKindV1;

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{
        BudgetBoundary, MaxNewTokensZero, PromptEvaluation, StopEvaluation, StopTokenHandling,
    };

    fn valid_policy() -> GenerationStopPolicyV1 {
        GenerationStopPolicyV1 {
            version: GENERATION_STOP_POLICY_VERSION,
            stop_token_ids: vec![248_046, 248_044],
            evaluation: StopEvaluation::NewlyGeneratedAfterArgmax,
            prompt_evaluation: PromptEvaluation::NeverStop,
            stop_token: StopTokenHandling {
                visible_output: false,
                subsequent_decode_input: false,
            },
            budget_boundary: BudgetBoundary::StopTokenWins,
            max_new_tokens_zero: MaxNewTokensZero::MaxNewTokensBeforeDecode,
            reason_version: GENERATION_STOP_REASON_VERSION,
        }
    }

    #[test]
    fn pinned_dependency_is_available_without_frontend_behavior() {
        let _ = super::TOKENIZERS_DEPENDENCY_MARKER;
    }

    #[test]
    fn max_zero_stops_before_any_generated_token_observation() {
        let mut controller = GenerationStopControllerV1::new(valid_policy(), 0).unwrap();
        assert!(controller.is_stopped());
        assert!(controller.report().generated_token_ids().is_empty());
        assert!(controller.report().visible_token_ids().is_empty());
        assert_eq!(controller.report().reason_token(), Some("max_new_tokens"));
        assert_eq!(controller.report().reason_version(), 1);
        assert_eq!(
            controller.observe_generated(248_046),
            Err(StopPolicyError::AlreadyStopped)
        );
    }

    #[test]
    fn prompt_ids_are_reported_but_never_stop_evaluated() {
        let mut controller = GenerationStopControllerV1::new_with_input_token_ids(
            valid_policy(),
            2,
            &[248_046, 248_044],
        )
        .unwrap();
        assert_eq!(controller.report().input_token_ids(), [248_046, 248_044]);
        let decision = controller.observe_generated(7).unwrap();
        assert_eq!(decision.visible_token_id(), Some(7));
        assert_eq!(controller.report().generated_token_ids(), [7]);
    }

    #[test]
    fn both_qwen_stop_ids_stop_immediately_and_are_hidden() {
        for stop_id in [248_046, 248_044] {
            let mut controller = GenerationStopControllerV1::new(valid_policy(), 17).unwrap();
            let decision = controller.observe_generated(stop_id).unwrap();
            assert_eq!(decision.generated_token_id(), stop_id);
            assert_eq!(decision.visible_token_id(), None);
            assert!(!decision.should_feed_decode());
            assert_eq!(decision.reason_token(), Some("stop_token"));
            assert_eq!(controller.report().generated_token_ids(), [stop_id]);
            assert!(controller.report().visible_token_ids().is_empty());
            assert!(controller.report().decode_input_token_ids().is_empty());
            assert_eq!(controller.report().stop_token_id(), Some(stop_id));
        }
    }

    #[test]
    fn stop_token_wins_at_the_budget_boundary() {
        let mut controller = GenerationStopControllerV1::new(valid_policy(), 1).unwrap();
        let decision = controller.observe_generated(248_046).unwrap();
        assert_eq!(decision.reason_token(), Some("stop_token"));
        assert!(controller.report().visible_token_ids().is_empty());
    }

    #[test]
    fn non_stop_token_at_budget_is_visible_but_not_fed() {
        let mut controller = GenerationStopControllerV1::new(valid_policy(), 3).unwrap();
        assert!(
            controller
                .observe_generated(3)
                .unwrap()
                .should_feed_decode()
        );
        assert!(
            controller
                .observe_generated(17)
                .unwrap()
                .should_feed_decode()
        );
        let decision = controller.observe_generated(5).unwrap();
        assert_eq!(decision.visible_token_id(), Some(5));
        assert!(!decision.should_feed_decode());
        assert_eq!(decision.reason_token(), Some("max_new_tokens"));
        assert_eq!(controller.report().generated_token_ids(), [3, 17, 5]);
        assert_eq!(controller.report().visible_token_ids(), [3, 17, 5]);
        assert_eq!(controller.report().decode_input_token_ids(), [3, 17]);
    }

    #[test]
    fn non_aligned_three_and_seventeen_budgets_are_exact() {
        for budget in [3, 17] {
            let mut controller = GenerationStopControllerV1::new(valid_policy(), budget).unwrap();
            for index in 0..budget {
                let decision = controller.observe_generated(index + 1).unwrap();
                assert_eq!(decision.generated_token_id(), index + 1);
                assert_eq!(decision.should_feed_decode(), index + 1 < budget);
            }
            assert!(controller.is_stopped());
            assert_eq!(
                controller.report().generated_token_ids().len(),
                budget as usize
            );
            assert_eq!(
                controller.report().visible_token_ids().len(),
                budget as usize
            );
            assert_eq!(
                controller.report().decode_input_token_ids().len(),
                (budget - 1) as usize
            );
        }
    }

    #[test]
    fn observe_after_stop_fails_closed_without_mutating_report() {
        let mut controller = GenerationStopControllerV1::new(valid_policy(), 1).unwrap();
        controller.observe_generated(9).unwrap();
        let before = controller.report().clone();
        assert_eq!(
            controller.observe_generated(10),
            Err(StopPolicyError::AlreadyStopped)
        );
        assert_eq!(controller.report(), &before);
    }

    #[test]
    fn invalid_policy_fields_and_ids_are_rejected() {
        let mut policy = valid_policy();
        policy.version = 2;
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::InvalidPolicyVersion { actual: 2 })
        );

        let mut policy = valid_policy();
        policy.reason_version = 2;
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::InvalidReasonVersion { actual: 2 })
        );

        let mut policy = valid_policy();
        policy.stop_token.visible_output = true;
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::InvalidStopTokenVisibility)
        );

        let mut policy = valid_policy();
        policy.stop_token.subsequent_decode_input = true;
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::InvalidStopTokenDecodeInput)
        );

        let mut policy = valid_policy();
        policy.stop_token_ids.clear();
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::EmptyStopTokenIds)
        );

        let mut policy = valid_policy();
        policy.stop_token_ids = vec![u32::MAX, u32::MAX];
        assert_eq!(
            validate_generation_stop_policy(&policy),
            Err(StopPolicyError::DuplicateStopTokenId { token_id: u32::MAX })
        );
        let mut policy = valid_policy();
        policy.stop_token_ids = vec![u32::MAX];
        assert!(validate_generation_stop_policy(&policy).is_ok());
    }

    #[test]
    fn public_reason_tokens_and_standalone_decision_are_typed() {
        assert_eq!(StopReasonKindV1::StopToken.as_str(), "stop_token");
        assert_eq!(
            StopReasonKindV1::MaxNewTokens.reason_token(),
            "max_new_tokens"
        );
        assert_eq!(
            stop_before_decode(0).unwrap().reason_token(),
            "max_new_tokens"
        );
        assert!(stop_before_decode(1).is_none());

        let policy = valid_policy();
        assert!(GenerationStopControllerV1::new(&policy, 1).is_ok());

        let decision = decide_after_argmax(&valid_policy(), 12, 1, 2).unwrap();
        assert_eq!(decision.generated_token_id(), 12);
        assert_eq!(decision.visible_token_id(), Some(12));
        assert_eq!(decision.decode_input_token_id(), Some(12));
        assert!(decision.should_feed_decode());
        assert_eq!(
            decide_after_argmax(&valid_policy(), 12, 0, 2),
            Err(StopPolicyError::InvalidGeneratedCount)
        );
        assert_eq!(
            decide_after_argmax(&valid_policy(), 12, 1, 0),
            Err(StopPolicyError::ArgmaxAfterZeroBudget)
        );
    }

    #[test]
    fn standalone_decision_rejects_over_budget_before_stop_evaluation() {
        let policy = valid_policy();
        for token_id in [248_046, 248_044, 7] {
            let error = decide_after_argmax(&policy, token_id, 2, 1).unwrap_err();
            assert_eq!(
                error,
                StopPolicyError::GeneratedCountExceedsBudget {
                    generated_count_after_token: 2,
                    max_new_tokens: 1,
                }
            );
            assert!(error.to_string().contains("exceeds max_new_tokens"));
        }
    }

    #[test]
    fn standalone_boundaries_are_checked_without_a_large_loop() {
        let policy = valid_policy();
        for max_new_tokens in [3, 17, 255, 256, 257] {
            let ordinary = decide_after_argmax(&policy, 7, max_new_tokens, max_new_tokens).unwrap();
            assert_eq!(
                ordinary.stop_reason().unwrap().kind(),
                StopReasonKindV1::MaxNewTokens
            );
            assert_eq!(ordinary.visible_token_id(), Some(7));
            assert!(!ordinary.should_feed_decode());

            let over_budget =
                decide_after_argmax(&policy, 7, max_new_tokens.saturating_add(1), max_new_tokens);
            assert_eq!(
                over_budget,
                Err(StopPolicyError::GeneratedCountExceedsBudget {
                    generated_count_after_token: u64::from(max_new_tokens) + 1,
                    max_new_tokens,
                })
            );
        }

        let u32_max = decide_after_argmax(&policy, 7, u32::MAX, u32::MAX).unwrap();
        assert_eq!(
            u32_max.stop_reason().unwrap().kind(),
            StopReasonKindV1::MaxNewTokens
        );
        assert_eq!(u32_max.generated_token_id(), 7);
    }

    #[test]
    fn typed_reason_invariants_and_read_only_accessors_are_consistent() {
        let stop = StopReasonV1::stop_token(u32::MAX);
        assert_eq!(stop.version(), GENERATION_STOP_POLICY_VERSION);
        assert_eq!(stop.reason_version(), GENERATION_STOP_REASON_VERSION);
        assert_eq!(stop.kind(), StopReasonKindV1::StopToken);
        assert_eq!(stop.token_id(), Some(u32::MAX));

        let budget = StopReasonV1::max_new_tokens();
        assert_eq!(budget.version(), GENERATION_STOP_POLICY_VERSION);
        assert_eq!(budget.reason_version(), GENERATION_STOP_REASON_VERSION);
        assert_eq!(budget.kind(), StopReasonKindV1::MaxNewTokens);
        assert_eq!(budget.token_id(), None);

        let mut controller = GenerationStopControllerV1::new(valid_policy(), 1).unwrap();
        let decision = controller.observe_generated(9).unwrap();
        let reason = decision.stop_reason().unwrap();
        assert_eq!(reason.kind(), StopReasonKindV1::MaxNewTokens);
        assert_eq!(reason.token_id(), None);
        assert_eq!(decision.visible_token_id(), Some(9));
        assert_eq!(decision.decode_input_token_id(), None);
        assert!(!decision.should_feed_decode());

        let report = controller.report();
        assert_eq!(report.version(), GENERATION_STOP_POLICY_VERSION);
        assert_eq!(report.reason_version(), GENERATION_STOP_REASON_VERSION);
        assert_eq!(report.generated_token_ids(), report.visible_token_ids());
        assert_eq!(report.generated_token_ids().len(), 1);
        assert!(report.decode_input_token_ids().is_empty());
        assert_eq!(report.stop_reason(), Some(reason));
    }

    #[test]
    fn controller_cannot_reach_over_budget_and_error_does_not_mutate_report() {
        let mut controller = GenerationStopControllerV1::new(valid_policy(), 1).unwrap();
        let decision = controller.observe_generated(3).unwrap();
        assert_eq!(
            decision.stop_reason().unwrap().kind(),
            StopReasonKindV1::MaxNewTokens
        );
        let report_before = controller.report().clone();

        assert_eq!(
            controller.observe_generated(17),
            Err(StopPolicyError::AlreadyStopped)
        );
        assert_eq!(controller.report(), &report_before);
        assert_eq!(controller.report().generated_token_ids(), [3]);
    }
}
