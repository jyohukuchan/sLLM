//! Transport-independent render/tokenize/prefill/decode/sampling service.

use core::fmt;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sllm_core::{
    CompiledGrammar, DeviceTokenSelectorRequestV1, DraftProposalV1, DraftProviderV1,
    Gemma4ExecutionRequest, Gemma4ModelLock, GrammarError, MAX_SPECULATIVE_DRAFT_WIDTH_V1,
    QwenExecutionRequest, SamplerChainConfigV1, SamplerChainV1, SamplingError,
    SamplingParametersV1, SamplingRandomSource, SamplingSelectionV1, SpeculativeAccountingV1,
    SpeculativeError, TokenTrie, verify_target_selected,
};

use crate::reasoning::{ReasoningControllerV1, ReasoningErrorV1, ReasoningPolicyV1};
use crate::{
    ChatMessageV1, ChatRenderOptionsV1, ChatTemplateRendererV1, DecodeModeV1,
    GenerationStopPolicyV1, GenericTemplateIdentityV1, GenericTemplateInputV1,
    GenericTemplateProviderV1, Qwen35ChatTemplateV1, TokenByteTableV1, TokenIdsV1,
    TokenizerFrontendV1, TokenizerUtilityErrorV1, validate_generation_stop_policy,
};

pub const MAX_STOP_STRINGS_V1: usize = 4;
pub const MAX_STOP_STRING_BYTES_V1: usize = 1_048_576;
pub const MAX_GENERATION_CHOICES_V1: usize = 8;

pub fn gemma4_generation_stop_policy(
    lock: &Gemma4ModelLock,
) -> Result<GenerationStopPolicyV1, GenerationServiceError> {
    let stop_token_ids = lock
        .model
        .tokenizer_contract
        .stop_token_ids
        .iter()
        .map(|&token| u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let policy = GenerationStopPolicyV1 {
        version: 1,
        stop_token_ids,
        evaluation: crate::StopEvaluation::NewlyGeneratedAfterArgmax,
        prompt_evaluation: crate::PromptEvaluation::NeverStop,
        stop_token: crate::StopTokenHandling {
            visible_output: false,
            subsequent_decode_input: false,
        },
        budget_boundary: crate::BudgetBoundary::StopTokenWins,
        max_new_tokens_zero: crate::MaxNewTokensZero::MaxNewTokensBeforeDecode,
        reason_version: 1,
    };
    validate_generation_stop_policy(&policy)
        .map_err(|_| GenerationServiceError::InvalidStopPolicy)?;
    Ok(policy)
}

pub fn gemma4_moe_generation_stop_policy() -> Result<GenerationStopPolicyV1, GenerationServiceError>
{
    let policy = sllm_core::gemma4_moe_generation_stop_policy();
    validate_generation_stop_policy(&policy)
        .map_err(|_| GenerationServiceError::InvalidStopPolicy)?;
    Ok(policy)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationInputV1 {
    Prompt(String),
    Messages {
        messages: Vec<ChatMessageV1>,
        options: ChatRenderOptionsV1,
    },
    /// Explicit continuation input.  The assistant prefix is part of the
    /// model context, but is not emitted as generated output.
    PromptWithAssistantPrefill {
        prompt: String,
        assistant_prefill: String,
    },
    /// Explicit chat continuation input.  This is intentionally separate
    /// from `Messages` so the legacy rendering/token sequence remains exact.
    MessagesWithAssistantPrefill {
        messages: Vec<ChatMessageV1>,
        options: ChatRenderOptionsV1,
        assistant_prefill: String,
    },
    /// A capability-gated generic template input.  The provider renders this
    /// before tokenization; raw and Gemma text variants remain rejected by the
    /// generic utility contract.
    GenericTemplate(Box<GenericGenerationInputV1>),
}

/// A generic template request whose bounded render is completed before model
/// execution.  The rendered bytes and complete source/profile/kwargs/render
/// identity are retained so cache and checkpoint adapters can bind exact
/// prompt provenance without re-rendering on a GPU path.
#[derive(Clone, Debug)]
pub struct GenericGenerationInputV1 {
    provider: GenericTemplateProviderV1,
    input: GenericTemplateInputV1,
    rendered: String,
    identity: GenericTemplateIdentityV1,
}

impl GenericGenerationInputV1 {
    pub fn new(
        provider: GenericTemplateProviderV1,
        input: GenericTemplateInputV1,
    ) -> Result<Self, GenerationServiceError> {
        let render: Result<_, TokenizerUtilityErrorV1> = match &input {
            GenericTemplateInputV1::Json(context) => provider
                .render_context(context)
                .map_err(TokenizerUtilityErrorV1::GenericTemplate),
            GenericTemplateInputV1::Messages(messages) => provider
                .render_context(messages.context())
                .map_err(TokenizerUtilityErrorV1::GenericTemplate),
            GenericTemplateInputV1::RawText(_) => {
                Err(TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput {
                    kind: crate::GenericTemplateInputKindV1::RawText,
                })
            }
            GenericTemplateInputV1::GemmaRawText(_) => {
                Err(TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput {
                    kind: crate::GenericTemplateInputKindV1::GemmaRawText,
                })
            }
        };
        let render = render.map_err(generic_template_error)?;
        if render.rendered().is_empty() {
            return Err(GenerationServiceError::GenericTemplate(
                TokenizerUtilityErrorV1::InvalidTemplateResult,
            ));
        }
        Ok(Self {
            provider,
            input,
            rendered: render.rendered().to_owned(),
            identity: render.identity().clone(),
        })
    }

    pub fn provider(&self) -> &GenericTemplateProviderV1 {
        &self.provider
    }

    pub fn input(&self) -> &GenericTemplateInputV1 {
        &self.input
    }

    pub fn rendered(&self) -> &str {
        &self.rendered
    }

    pub fn rendered_bytes(&self) -> &[u8] {
        self.rendered.as_bytes()
    }

    pub fn identity(&self) -> &GenericTemplateIdentityV1 {
        &self.identity
    }
}

impl PartialEq for GenericGenerationInputV1 {
    fn eq(&self, other: &Self) -> bool {
        self.provider.digest() == other.provider.digest()
            && self.input == other.input
            && self.rendered == other.rendered
            && self.identity == other.identity
    }
}

impl Eq for GenericGenerationInputV1 {}

fn generic_template_error(error: TokenizerUtilityErrorV1) -> GenerationServiceError {
    GenerationServiceError::GenericTemplate(error)
}

/// Tokenized generation input with an explicit assistant continuation
/// boundary.  `token_ids()` is the complete context passed to the executor;
/// `assistant_prefill_token_ids()` is used only for grammar state priming and
/// accounting/visibility semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedGenerationInputV1 {
    assistant_prefill_start: usize,
    token_ids: Vec<u32>,
    generic_template_identity: Option<GenericTemplateIdentityV1>,
}

impl PreparedGenerationInputV1 {
    pub fn from_token_ids(
        base_token_ids: Vec<u32>,
        assistant_prefill_token_ids: Vec<u32>,
    ) -> Result<Self, GenerationServiceError> {
        if base_token_ids.is_empty() {
            return Err(GenerationServiceError::EmptyPromptTokens);
        }
        let combined_len = base_token_ids
            .len()
            .checked_add(assistant_prefill_token_ids.len())
            .ok_or(GenerationServiceError::CountOverflow)?;
        let mut token_ids = Vec::with_capacity(combined_len);
        token_ids.extend_from_slice(&base_token_ids);
        token_ids.extend_from_slice(&assistant_prefill_token_ids);
        Ok(Self {
            assistant_prefill_start: base_token_ids.len(),
            token_ids,
            generic_template_identity: None,
        })
    }

    fn with_generic_template_identity(mut self, identity: GenericTemplateIdentityV1) -> Self {
        self.generic_template_identity = Some(identity);
        self
    }

    pub fn base_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.assistant_prefill_start]
    }

    pub fn assistant_prefill_token_ids(&self) -> &[u32] {
        &self.token_ids[self.assistant_prefill_start..]
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn generic_template_identity(&self) -> Option<&GenericTemplateIdentityV1> {
        self.generic_template_identity.as_ref()
    }
}

#[derive(Clone, Debug)]
pub struct GenerationConfigV1 {
    max_new_tokens: u32,
    sampling: SamplingParametersV1,
    stop_strings: Vec<String>,
    sampler_chain: Option<SamplerChainConfigV1>,
    grammar: Option<CompiledGrammar>,
    ignore_stop_tokens: bool,
    device_selector_seed: Option<u64>,
    reasoning: Option<ReasoningPolicyV1>,
}

impl PartialEq for GenerationConfigV1 {
    fn eq(&self, other: &Self) -> bool {
        self.max_new_tokens == other.max_new_tokens
            && self.sampling == other.sampling
            && self.stop_strings == other.stop_strings
            && self.sampler_chain == other.sampler_chain
            && self.ignore_stop_tokens == other.ignore_stop_tokens
            && self.device_selector_seed == other.device_selector_seed
            && self.reasoning == other.reasoning
            // Compiled grammars intentionally do not expose structural
            // equality; state-count equality keeps this compatibility trait
            // useful for error assertions without comparing private NFAs.
            && self.grammar.as_ref().map(CompiledGrammar::state_count)
                == other.grammar.as_ref().map(CompiledGrammar::state_count)
    }
}

impl GenerationConfigV1 {
    pub fn new(
        max_new_tokens: u32,
        sampling: SamplingParametersV1,
        stop_strings: Vec<String>,
    ) -> Result<Self, GenerationServiceError> {
        if max_new_tokens == 0 {
            return Err(GenerationServiceError::InvalidMaxNewTokens);
        }
        if stop_strings.len() > MAX_STOP_STRINGS_V1 {
            return Err(GenerationServiceError::TooManyStopStrings);
        }
        let mut total_bytes = 0_usize;
        for (index, stop) in stop_strings.iter().enumerate() {
            if stop.is_empty() {
                return Err(GenerationServiceError::EmptyStopString { index });
            }
            total_bytes = total_bytes
                .checked_add(stop.len())
                .ok_or(GenerationServiceError::StopStringsTooLarge)?;
            if total_bytes > MAX_STOP_STRING_BYTES_V1 {
                return Err(GenerationServiceError::StopStringsTooLarge);
            }
            if stop_strings[..index].contains(stop) {
                return Err(GenerationServiceError::DuplicateStopString { index });
            }
        }
        Ok(Self {
            max_new_tokens,
            sampling,
            stop_strings,
            sampler_chain: None,
            grammar: None,
            ignore_stop_tokens: false,
            device_selector_seed: None,
            reasoning: None,
        })
    }

    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    pub const fn sampling(&self) -> SamplingParametersV1 {
        self.sampling
    }

    pub fn stop_strings(&self) -> &[String] {
        &self.stop_strings
    }

    /// Returns the explicit sampler-chain configuration, when advanced
    /// sampling is enabled. `None` retains the exact legacy profile-v1 path.
    pub fn sampler_chain(&self) -> Option<&SamplerChainConfigV1> {
        self.sampler_chain.as_ref()
    }

    /// Installs a validated, backend-neutral sampler chain while preserving
    /// the legacy constructor and call sites.
    pub fn with_sampler_chain(
        mut self,
        chain: SamplerChainConfigV1,
    ) -> Result<Self, GenerationServiceError> {
        chain.validate()?;
        if chain.parameters != self.sampling {
            return Err(GenerationServiceError::SamplingConfigurationMismatch);
        }
        self.sampler_chain = Some(chain);
        Ok(self)
    }

    pub fn with_grammar(mut self, grammar: CompiledGrammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    pub fn grammar(&self) -> Option<&CompiledGrammar> {
        self.grammar.as_ref()
    }

    /// Masks every stop token in the sampler candidate set. This is useful for
    /// OpenAI-compatible `ignore_eos` behavior and is deliberately config
    /// scoped so it cannot mutate the model's reviewed stop policy.
    pub fn with_ignore_stop_tokens(mut self, enabled: bool) -> Self {
        self.ignore_stop_tokens = enabled;
        self
    }

    pub const fn ignore_stop_tokens(&self) -> bool {
        self.ignore_stop_tokens
    }

    /// Pins the request-local random stream used by an eligible prepared GPU
    /// selector. The host sampler must be initialized from the same seed so a
    /// capability decision cannot silently change request determinism.
    pub fn with_device_selector_seed(mut self, seed: u64) -> Self {
        self.device_selector_seed = Some(seed);
        self
    }

    pub const fn device_selector_seed(&self) -> Option<u64> {
        self.device_selector_seed
    }

    /// Installs the additive reasoning policy.  Callers that do not install a
    /// policy retain the exact legacy generation path.
    pub fn with_reasoning(
        mut self,
        policy: ReasoningPolicyV1,
    ) -> Result<Self, GenerationServiceError> {
        policy.validate_max_new_tokens(self.max_new_tokens)?;
        self.reasoning = Some(policy);
        Ok(self)
    }

    pub fn reasoning(&self) -> Option<&ReasoningPolicyV1> {
        self.reasoning.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReasonV1 {
    Stop,
    Length,
}

impl FinishReasonV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsageV1 {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    reasoning_tokens: u64,
}

impl TokenUsageV1 {
    pub const fn prompt_tokens(self) -> u64 {
        self.prompt_tokens
    }

    pub const fn completion_tokens(self) -> u64 {
        self.completion_tokens
    }

    pub const fn total_tokens(self) -> u64 {
        self.total_tokens
    }

    pub const fn reasoning_tokens(self) -> u64 {
        self.reasoning_tokens
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationResultV1 {
    input_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    visible_token_ids: Vec<u32>,
    decode_input_token_ids: Vec<u32>,
    output_text: String,
    finish_reason: FinishReasonV1,
    stop_token_id: Option<u32>,
    matched_stop: Option<String>,
    usage: TokenUsageV1,
    decode_steps: u32,
    selections: Vec<SamplingSelectionV1>,
    reasoning_token_ids: Vec<u32>,
}

impl GenerationResultV1 {
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

    pub fn output_text(&self) -> &str {
        &self.output_text
    }

    pub const fn finish_reason(&self) -> FinishReasonV1 {
        self.finish_reason
    }

    pub const fn stop_token_id(&self) -> Option<u32> {
        self.stop_token_id
    }

    pub fn matched_stop(&self) -> Option<&str> {
        self.matched_stop.as_deref()
    }

    pub const fn usage(&self) -> TokenUsageV1 {
        self.usage
    }

    pub const fn decode_steps(&self) -> u32 {
        self.decode_steps
    }

    /// Sampling metadata in generated-token order. The vector includes a
    /// stop token when one was selected, matching `generated_token_ids`.
    pub fn selections(&self) -> &[SamplingSelectionV1] {
        &self.selections
    }

    /// Generated token history consumed while the reasoning controller was
    /// active. Closing-marker tokens are included because they are real model
    /// selections, while the visible token history excludes them.
    pub fn reasoning_token_ids(&self) -> &[u32] {
        &self.reasoning_token_ids
    }

    pub const fn reasoning_tokens(&self) -> u64 {
        self.usage.reasoning_tokens()
    }
}

/// One independently generated choice. The index is stable across buffered
/// and streaming transports and is never inferred from completion order.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationChoiceResultV1 {
    index: u32,
    result: GenerationResultV1,
}

impl GenerationChoiceResultV1 {
    pub const fn index(&self) -> u32 {
        self.index
    }

    pub const fn result(&self) -> &GenerationResultV1 {
        &self.result
    }
}

/// Bounded multi-choice result with OpenAI-compatible aggregate accounting:
/// the shared prompt is counted once and generated tokens are summed.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationChoicesResultV1 {
    choices: Vec<GenerationChoiceResultV1>,
    usage: TokenUsageV1,
}

impl GenerationChoicesResultV1 {
    pub fn new(results: Vec<GenerationResultV1>) -> Result<Self, GenerationServiceError> {
        if results.is_empty() || results.len() > MAX_GENERATION_CHOICES_V1 {
            return Err(GenerationServiceError::InvalidChoiceCount);
        }
        let prompt = results[0].input_token_ids();
        if results
            .iter()
            .skip(1)
            .any(|result| result.input_token_ids() != prompt)
        {
            return Err(GenerationServiceError::InconsistentChoicePrompt);
        }
        let prompt_tokens = results[0].usage().prompt_tokens();
        let completion_tokens = results.iter().try_fold(0_u64, |total, result| {
            total
                .checked_add(result.usage().completion_tokens())
                .ok_or(GenerationServiceError::CountOverflow)
        })?;
        let reasoning_tokens = results.iter().try_fold(0_u64, |total, result| {
            total
                .checked_add(result.usage().reasoning_tokens())
                .ok_or(GenerationServiceError::CountOverflow)
        })?;
        let total_tokens = prompt_tokens
            .checked_add(completion_tokens)
            .ok_or(GenerationServiceError::CountOverflow)?;
        let choices = results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                Ok(GenerationChoiceResultV1 {
                    index: u32::try_from(index)
                        .map_err(|_| GenerationServiceError::CountOverflow)?,
                    result,
                })
            })
            .collect::<Result<Vec<_>, GenerationServiceError>>()?;
        Ok(Self {
            choices,
            usage: TokenUsageV1 {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens,
            },
        })
    }

    pub fn choices(&self) -> &[GenerationChoiceResultV1] {
        &self.choices
    }

    pub const fn usage(&self) -> TokenUsageV1 {
        self.usage
    }
}

/// Derives a deterministic per-choice stream while preserving the exact
/// legacy stream for choice zero. Unseeded requests remain unseeded so each
/// choice obtains entropy from its independently created random source.
pub const fn derive_choice_seed_v1(seed: Option<u64>, choice_index: u32) -> Option<u64> {
    let Some(seed) = seed else {
        return None;
    };
    if choice_index == 0 {
        return Some(seed);
    }
    let mut value = seed ^ (choice_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    Some(value ^ (value >> 31))
}

#[derive(Clone, Debug)]
pub struct GenerationCancellationV1 {
    cancelled: Arc<AtomicBool>,
}

impl Default for GenerationCancellationV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationCancellationV1 {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationStepV1 {
    device_argmax: u32,
    last_logits: Option<Vec<f32>>,
    device_selection: Option<SamplingSelectionV1>,
}

impl GenerationStepV1 {
    pub fn new(device_argmax: u32, last_logits: Option<Vec<f32>>) -> Self {
        Self {
            device_argmax,
            last_logits,
            device_selection: None,
        }
    }

    pub fn from_device_selection(selection: SamplingSelectionV1) -> Self {
        Self {
            device_argmax: selection.token_id,
            last_logits: None,
            device_selection: Some(selection),
        }
    }

    pub const fn device_argmax(&self) -> u32 {
        self.device_argmax
    }

    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    pub fn device_selection(&self) -> Option<&SamplingSelectionV1> {
        self.device_selection.as_ref()
    }
}

pub trait GenerationExecutorV1 {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError>;

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError>;

    /// Optional pre-decode admission hook. Context-window adapters use this
    /// to transactionally rebuild a compact owner before the token is
    /// submitted; legacy executors retain the no-op default.
    fn before_decode(&mut self, _token_id: u32) -> Result<(), GenerationServiceError> {
        Ok(())
    }

    fn supports_device_selector(&self) -> bool {
        false
    }

    fn prefill_with_device_selector(
        &mut self,
        _input_token_ids: &[u32],
        _selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        Err(GenerationServiceError::DeviceSelectorUnsupported)
    }

    fn decode_with_device_selector(
        &mut self,
        _token_id: u32,
        _selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        Err(GenerationServiceError::DeviceSelectorUnsupported)
    }

    /// Finalize request-local work which was staged ahead of the sequential
    /// generation loop. Ordinary executors have nothing to do; speculative
    /// adapters use this hook to discard unconsumed target rows on a clean
    /// stop or length boundary.
    fn finish(&mut self) -> Result<(), GenerationServiceError> {
        Ok(())
    }

    fn cancel(&mut self);
}

/// Internal exact-speculation seam. It returns only canonical target steps in
/// visible order; proposals and rejected rows never cross this boundary.
pub trait SpeculativeGenerationExecutorV1: GenerationExecutorV1 {
    fn speculative_decode_greedy(
        &mut self,
        pending_token: u32,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError>;

    /// Returns the provider owned by the target adapter, when it has one.
    /// Keeping this seam on the executor lets the legacy Qwen MTP owner use
    /// the same provider/verification loop as externally supplied providers
    /// without changing its public constructor or call sites.
    fn draft_provider(&mut self) -> Option<&mut dyn DraftProviderV1> {
        None
    }

    fn has_draft_provider(&self) -> bool {
        false
    }

    /// Maximum proposal width accepted by this target graph.  Targets which
    /// do not expose a provider retain the legacy width for compatibility.
    fn speculative_draft_width(&self) -> usize {
        MAX_SPECULATIVE_DRAFT_WIDTH_V1
    }

    /// Verifies one provider proposal and returns only canonical target steps
    /// in publication order.  The default preserves the old exact-greedy
    /// executor contract; model adapters override it when they can consume a
    /// proposal from a model-neutral provider.
    fn speculative_decode_with_proposal(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        let _ = proposal;
        self.speculative_decode_greedy(pending_token)
    }

    /// Publish exactly the input rows consumed by the sequential frontend from
    /// the most recently staged speculative block. Implementations which do
    /// not stage target state retain the no-op compatibility behavior.
    fn finalize_speculative_decode(
        &mut self,
        _committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingSpeculativeBlockV1 {
    total_input_rows: usize,
    consumed_input_rows: usize,
}

/// Adapts an exact speculative executor to the unchanged one-token generation
/// loop. Sampled requests request logits and therefore use the canonical
/// target-only decode path without consuming proposal RNG or changing public
/// sampling state.
pub struct SpeculativeGenerationAdapterV1<E> {
    inner: E,
    queued: VecDeque<(u32, GenerationStepV1)>,
    provider: Option<Box<dyn DraftProviderV1>>,
    committed_target_tokens: Vec<u32>,
    accounting: SpeculativeAccountingV1,
    max_draft_width: usize,
    pending_block: Option<PendingSpeculativeBlockV1>,
}

impl<E: SpeculativeGenerationExecutorV1> SpeculativeGenerationAdapterV1<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            queued: VecDeque::new(),
            provider: None,
            committed_target_tokens: Vec::new(),
            accounting: SpeculativeAccountingV1::default(),
            max_draft_width: MAX_SPECULATIVE_DRAFT_WIDTH_V1,
            pending_block: None,
        }
    }

    /// Creates an adapter with an explicit model-neutral draft provider.
    /// The provider owns its own state/RNG; only target steps cross the
    /// executor boundary.
    pub fn with_provider<P: DraftProviderV1 + 'static>(inner: E, provider: P) -> Self {
        Self::with_provider_and_draft_width(inner, provider, MAX_SPECULATIVE_DRAFT_WIDTH_V1)
            .expect("the default speculative width is valid")
    }

    pub fn with_provider_and_draft_width<P: DraftProviderV1 + 'static>(
        inner: E,
        provider: P,
        max_draft_width: usize,
    ) -> Result<Self, GenerationServiceError> {
        if !(1..=MAX_SPECULATIVE_DRAFT_WIDTH_V1).contains(&max_draft_width) {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::DraftWidthExceeded.to_string(),
            ));
        }
        let mut adapter = Self::new(inner);
        adapter.provider = Some(Box::new(provider));
        adapter.max_draft_width = max_draft_width;
        Ok(adapter)
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    pub fn into_inner(self) -> E {
        self.inner
    }

    pub const fn accounting(&self) -> SpeculativeAccountingV1 {
        self.accounting
    }

    pub fn provider(&self) -> Option<&dyn DraftProviderV1> {
        self.provider.as_deref()
    }

    fn finalize_pending_block(&mut self) -> Result<(), GenerationServiceError> {
        let Some(pending) = self.pending_block.take() else {
            return Ok(());
        };
        if pending.consumed_input_rows == 0
            || pending.consumed_input_rows > pending.total_input_rows
        {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::InvalidDecision.to_string(),
            ));
        }
        self.inner
            .finalize_speculative_decode(pending.consumed_input_rows)?;
        self.queued.clear();
        Ok(())
    }

    fn stage_steps(
        &mut self,
        steps: Vec<GenerationStepV1>,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let total_input_rows = steps.len();
        let mut steps = steps.into_iter();
        let first = steps.next().ok_or_else(|| {
            GenerationServiceError::Execution(
                "speculative executor returned no canonical target step".to_owned(),
            )
        })?;
        let mut expected_input = first.device_argmax();
        for step in steps {
            let next_expected = step.device_argmax();
            self.queued.push_back((expected_input, step));
            expected_input = next_expected;
        }
        self.pending_block = Some(PendingSpeculativeBlockV1 {
            total_input_rows,
            // The token supplied to the speculative decode call is the first
            // input row and was already selected by the sequential frontend.
            consumed_input_rows: 1,
        });
        if total_input_rows == 1 {
            self.finalize_pending_block()?;
        }
        Ok(first)
    }

    fn proposal(
        &mut self,
        pending_token: u32,
    ) -> Result<Option<DraftProposalV1>, GenerationServiceError> {
        self.committed_target_tokens.push(pending_token);
        let target_width = self.inner.speculative_draft_width();
        if !(1..=MAX_SPECULATIVE_DRAFT_WIDTH_V1).contains(&target_width) {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::DraftWidthExceeded.to_string(),
            ));
        }
        let max_width = self.max_draft_width.min(target_width);
        if let Some(provider) = self.provider.as_mut() {
            return provider
                .propose(&self.committed_target_tokens, max_width)
                .map_err(GenerationServiceError::from);
        }
        self.inner
            .draft_provider()
            .map(|provider| {
                provider
                    .propose(&self.committed_target_tokens, max_width)
                    .map_err(GenerationServiceError::from)
            })
            .transpose()
            .map(|proposal| proposal.flatten())
    }

    fn record_proposal(
        &mut self,
        proposal: &DraftProposalV1,
        steps: &[GenerationStepV1],
    ) -> Result<(), GenerationServiceError> {
        if steps.is_empty() || steps.len() > proposal.token_ids().len() + 1 {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::InvalidDecision.to_string(),
            ));
        }
        let target_tokens = steps
            .iter()
            .map(GenerationStepV1::device_argmax)
            .collect::<Vec<_>>();
        let accepted = proposal
            .token_ids()
            .iter()
            .zip(&target_tokens)
            .take_while(|(draft, target)| draft == target)
            .count();
        let expected_steps = if accepted == proposal.token_ids().len() {
            proposal.token_ids().len() + 1
        } else {
            accepted + 1
        };
        if steps.len() != expected_steps {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::InvalidDecision.to_string(),
            ));
        }
        // A rejecting target commits only the accepted prefix and its
        // replacement row.  verify_target_selected intentionally stops at
        // that first mismatch, so padding the unavailable tail is safe.
        let mut verification_tokens = target_tokens.clone();
        if verification_tokens.len() < proposal.token_ids().len() + 1 {
            let filler = *verification_tokens.last().ok_or_else(|| {
                GenerationServiceError::Speculative(SpeculativeError::InvalidDecision.to_string())
            })?;
            verification_tokens.resize(proposal.token_ids().len() + 1, filler);
        }
        let decision = verify_target_selected(proposal.token_ids(), &verification_tokens)
            .map_err(GenerationServiceError::from)?;
        self.accounting
            .record(proposal, &decision)
            .map_err(GenerationServiceError::from)
    }
}

impl<E: SpeculativeGenerationExecutorV1> GenerationExecutorV1
    for SpeculativeGenerationAdapterV1<E>
{
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        self.finalize_pending_block()?;
        self.queued.clear();
        self.committed_target_tokens = input_token_ids.to_vec();
        self.accounting = SpeculativeAccountingV1::default();
        if let Some(provider) = self.provider.as_mut() {
            provider.reset().map_err(GenerationServiceError::from)?;
        } else if let Some(provider) = self.inner.draft_provider() {
            provider.reset().map_err(GenerationServiceError::from)?;
        }
        self.inner.prefill(input_token_ids, include_last_logits)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if include_last_logits {
            if !self.queued.is_empty() {
                return Err(GenerationServiceError::Execution(
                    "sampling mode changed while speculative rows were queued".to_owned(),
                ));
            }
            return self.inner.decode(token_id, true);
        }
        if let Some((expected_input, step)) = self.queued.pop_front() {
            if token_id != expected_input {
                return Err(GenerationServiceError::Execution(
                    "generation loop input differs from the accepted speculative token".to_owned(),
                ));
            }
            self.committed_target_tokens.push(token_id);
            let pending = self.pending_block.as_mut().ok_or_else(|| {
                GenerationServiceError::Speculative(SpeculativeError::InvalidDecision.to_string())
            })?;
            pending.consumed_input_rows = pending
                .consumed_input_rows
                .checked_add(1)
                .ok_or(GenerationServiceError::CountOverflow)?;
            if pending.consumed_input_rows == pending.total_input_rows {
                self.finalize_pending_block()?;
            }
            return Ok(step);
        }
        // Preserve the original executor-only contract for adapters that do
        // not expose a model-neutral provider. This is retained for API
        // compatibility with custom exact-greedy executors.
        if self.provider.is_none() && !self.inner.has_draft_provider() {
            let steps = self.inner.speculative_decode_greedy(token_id)?;
            return self.stage_steps(steps);
        }
        let Some(proposal) = self.proposal(token_id)? else {
            return self.inner.decode(token_id, false);
        };
        let steps = self
            .inner
            .speculative_decode_with_proposal(token_id, &proposal)?;
        self.record_proposal(&proposal, &steps)?;
        self.stage_steps(steps)
    }

    fn finish(&mut self) -> Result<(), GenerationServiceError> {
        self.finalize_pending_block()?;
        self.inner.finish()
    }

    fn cancel(&mut self) {
        let _ = self.finalize_pending_block();
        self.queued.clear();
        self.inner.cancel();
    }
}

/// Qwen3.5 target+MTP owner for exact greedy speculation.
///
/// `draft_width` is the number of MTP proposal tokens in one speculative
/// block. The target verify block contains one additional row for the pending
/// token, so its required row capacity is `draft_width + 1`.
struct PendingQwenSpeculativeBlockV1 {
    hidden_rows_bf16: Vec<u16>,
    target_accepted_draft_tokens: usize,
    target_input_rows: usize,
    proposed_draft_tokens: usize,
}

pub struct QwenMtpGenerationExecutorV1 {
    target: QwenExecutionRequest,
    mtp: QwenExecutionRequest,
    last_target_hidden_bf16: Vec<u16>,
    draft_width: usize,
    proposal_blocks: u64,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    committed_target_rows: u64,
    pending_speculative_block: Option<PendingQwenSpeculativeBlockV1>,
}

impl QwenMtpGenerationExecutorV1 {
    const HIDDEN_WIDTH: usize = 2_560;
    /// Maximum number of MTP proposal tokens in one generation block.
    ///
    /// This keeps the public generation transaction aligned with the largest
    /// width supported by the serial-equivalent Qwen target graph path.
    pub const MAX_DRAFT_WIDTH: usize = 8;

    pub fn new(target: QwenExecutionRequest, mtp: QwenExecutionRequest) -> Self {
        Self {
            target,
            mtp,
            last_target_hidden_bf16: Vec::new(),
            draft_width: 2,
            proposal_blocks: 0,
            proposed_draft_tokens: 0,
            accepted_draft_tokens: 0,
            committed_target_rows: 0,
            pending_speculative_block: None,
        }
    }

    pub fn new_with_draft_width(
        target: QwenExecutionRequest,
        mtp: QwenExecutionRequest,
        draft_width: usize,
    ) -> Result<Self, GenerationServiceError> {
        Self::validate_draft_width(draft_width)?;
        Ok(Self {
            target,
            mtp,
            last_target_hidden_bf16: Vec::new(),
            draft_width,
            proposal_blocks: 0,
            proposed_draft_tokens: 0,
            accepted_draft_tokens: 0,
            committed_target_rows: 0,
            pending_speculative_block: None,
        })
    }

    fn validate_draft_width(draft_width: usize) -> Result<(), GenerationServiceError> {
        if Self::target_block_rows_for_draft_width(draft_width).is_none() {
            return Err(GenerationServiceError::Execution(
                "MTP generation draft width must be in 1..=8".to_owned(),
            ));
        }
        Ok(())
    }

    /// Returns the target verify row count for a draft-token width.
    ///
    /// The first row consumes the pending token and each remaining row
    /// verifies one proposed draft token.
    pub const fn target_block_rows_for_draft_width(draft_width: usize) -> Option<usize> {
        if draft_width == 0 || draft_width > Self::MAX_DRAFT_WIDTH {
            None
        } else {
            Some(draft_width + 1)
        }
    }

    pub const fn draft_width(&self) -> usize {
        self.draft_width
    }

    pub fn target(&self) -> &QwenExecutionRequest {
        &self.target
    }

    pub fn mtp(&self) -> &QwenExecutionRequest {
        &self.mtp
    }

    pub const fn proposal_blocks(&self) -> u64 {
        self.proposal_blocks
    }

    pub const fn proposed_draft_tokens(&self) -> u64 {
        self.proposed_draft_tokens
    }

    pub const fn accepted_draft_tokens(&self) -> u64 {
        self.accepted_draft_tokens
    }

    pub const fn committed_target_rows(&self) -> u64 {
        self.committed_target_rows
    }

    fn step_from_output(
        output: &sllm_core::QwenExecutionOutput,
        row: usize,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = *output
            .token_ids()
            .get(row)
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        Ok(GenerationStepV1::new(
            u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            None,
        ))
    }
}

impl GenerationExecutorV1 for QwenMtpGenerationExecutorV1 {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        _: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.pending_speculative_block.is_some() {
            return Err(GenerationServiceError::Execution(
                "speculative target block must be finalized before prefill".to_owned(),
            ));
        }
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .target
            .prefill_with_mtp_state(&input)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let hidden = output.hidden_states_bf16().ok_or_else(|| {
            GenerationServiceError::Execution("target prefill omitted MTP hidden rows".to_owned())
        })?;
        if hidden.len() != input.len() * Self::HIDDEN_WIDTH {
            return Err(GenerationServiceError::Execution(
                "target prefill MTP hidden row count differs".to_owned(),
            ));
        }
        let zero = vec![0_u16; Self::HIDDEN_WIDTH];
        self.mtp
            .prefill_mtp(input[0], &zero)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        for index in 1..input.len() {
            self.mtp
                .decode_mtp(
                    input[index],
                    &hidden[(index - 1) * Self::HIDDEN_WIDTH..index * Self::HIDDEN_WIDTH],
                )
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        self.last_target_hidden_bf16 = hidden[(input.len() - 1) * Self::HIDDEN_WIDTH..].to_vec();
        let final_row = output
            .token_ids()
            .len()
            .checked_sub(1)
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        Self::step_from_output(&output, final_row)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = if include_last_logits {
            self.target.decode_with_last_logits(token)
        } else {
            self.target.decode(token)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let argmax = Self::step_from_output(&output, 0)?;
        Ok(GenerationStepV1::new(
            argmax.device_argmax(),
            output.last_logits().map(<[f32]>::to_vec),
        ))
    }

    fn cancel(&mut self) {
        self.target.cancel();
        self.mtp.cancel();
    }
}

impl SpeculativeGenerationExecutorV1 for QwenMtpGenerationExecutorV1 {
    fn draft_provider(&mut self) -> Option<&mut dyn DraftProviderV1> {
        Some(self)
    }

    fn has_draft_provider(&self) -> bool {
        true
    }

    fn speculative_draft_width(&self) -> usize {
        self.draft_width
    }

    fn speculative_decode_greedy(
        &mut self,
        pending_token: u32,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        let proposal = self
            .propose_mtp_draft(pending_token, self.draft_width)
            .map_err(GenerationServiceError::from)?
            .ok_or_else(|| {
                GenerationServiceError::Execution("MTP provider returned no draft".to_owned())
            })?;
        self.speculative_decode_with_proposal(pending_token, &proposal)
    }

    fn speculative_decode_with_proposal(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        self.verify_mtp_draft(pending_token, proposal)
    }

    fn finalize_speculative_decode(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        self.finalize_mtp_draft(committed_input_rows)
    }
}

impl DraftProviderV1 for QwenMtpGenerationExecutorV1 {
    fn kind(&self) -> sllm_core::DraftProviderKindV1 {
        sllm_core::DraftProviderKindV1::QwenMtp
    }

    fn propose(
        &mut self,
        committed_target_tokens: &[u32],
        max_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        if committed_target_tokens.len() > sllm_core::MAX_SPECULATIVE_HISTORY_TOKENS_V1 {
            return Err(SpeculativeError::HistoryLimitExceeded);
        }
        if !(1..=MAX_SPECULATIVE_DRAFT_WIDTH_V1).contains(&max_width) {
            return Err(if max_width == 0 {
                SpeculativeError::ZeroDraftWidth
            } else {
                SpeculativeError::DraftWidthExceeded
            });
        }
        let pending_token = committed_target_tokens
            .last()
            .copied()
            .ok_or(SpeculativeError::HistoryLimitExceeded)?;
        self.propose_mtp_draft(pending_token, max_width)
    }
}

impl QwenMtpGenerationExecutorV1 {
    fn propose_mtp_draft(
        &mut self,
        pending_token: u32,
        requested_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        let width = requested_width.min(self.draft_width);
        if width == 0 {
            return Err(SpeculativeError::ZeroDraftWidth);
        }
        if self.last_target_hidden_bf16.len() != Self::HIDDEN_WIDTH {
            return Err(SpeculativeError::HistoryLimitExceeded);
        }
        let pending = i32::try_from(pending_token)
            .map_err(|_| SpeculativeError::TokenOutOfVocabulary(pending_token))?;
        let mut drafts = Vec::with_capacity(width);
        let mut proposal_token = pending;
        let mut proposal_hidden = self.last_target_hidden_bf16.clone();
        for _ in 0..width {
            let proposal = self
                .mtp
                .decode_mtp(proposal_token, &proposal_hidden)
                .map_err(|_| SpeculativeError::InvalidDecision)?;
            proposal_token = *proposal
                .token_ids()
                .first()
                .ok_or(SpeculativeError::ZeroDraftWidth)?;
            proposal_hidden = proposal
                .hidden_states_bf16()
                .ok_or(SpeculativeError::InvalidDecision)?
                .to_vec();
            drafts.push(
                u32::try_from(proposal_token)
                    .map_err(|_| SpeculativeError::TokenOutOfVocabulary(proposal_token as u32))?,
            );
        }
        Ok(Some(DraftProposalV1::new(self.kind(), drafts)?))
    }

    fn verify_mtp_draft(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        if self.pending_speculative_block.is_some() {
            return Err(GenerationServiceError::Execution(
                "previous speculative target block is still pending".to_owned(),
            ));
        }
        let drafts = proposal.token_ids();
        if drafts.is_empty() || drafts.len() > self.draft_width {
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::DraftWidthExceeded.to_string(),
            ));
        }
        let pending =
            i32::try_from(pending_token).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let mut block_inputs = Vec::with_capacity(self.draft_width + 1);
        block_inputs.push(pending);
        block_inputs.extend(
            drafts
                .iter()
                .copied()
                .map(|token| {
                    i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let block = self
            .target
            .decode_block_with_mtp_state(&block_inputs)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let hidden = block.hidden_states_bf16().ok_or_else(|| {
            GenerationServiceError::Execution("target verify omitted hidden rows".to_owned())
        })?;
        let draft_width = drafts.len();
        if block.token_ids().len() != draft_width + 1
            || hidden.len() != (draft_width + 1) * Self::HIDDEN_WIDTH
        {
            return Err(GenerationServiceError::Execution(
                "target verify row count differs from draft width".to_owned(),
            ));
        }
        let mut accepted = 0_usize;
        while accepted < draft_width {
            let draft = drafts[accepted];
            let target = u32::try_from(block.token_ids()[accepted])
                .map_err(|_| GenerationServiceError::TokenIdOverflow)?;
            if draft != target {
                break;
            }
            accepted += 1;
        }
        let committed_rows = if accepted == draft_width {
            draft_width + 1
        } else {
            accepted + 1
        };
        let steps = (0..committed_rows)
            .map(|row| Self::step_from_output(&block, row))
            .collect::<Result<Vec<_>, _>>()?;
        if proposal.provider() == sllm_core::DraftProviderKindV1::QwenMtp {
            // The MTP provider already executed one transition per proposed
            // token. Retain only rows accepted by the target, adding the final
            // all-accept row which was not needed to produce a proposal.
            if accepted != draft_width {
                for _ in 0..draft_width.saturating_sub(committed_rows) {
                    self.mtp
                        .rewind_last_decode_transition()
                        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                }
            } else {
                let previous_hidden_start = (draft_width - 1) * Self::HIDDEN_WIDTH;
                self.mtp
                    .decode_mtp(
                        i32::try_from(drafts[draft_width - 1])
                            .map_err(|_| GenerationServiceError::TokenIdOverflow)?,
                        &hidden[previous_hidden_start..previous_hidden_start + Self::HIDDEN_WIDTH],
                    )
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            }
        } else {
            // An external or n-gram provider does not touch MTP state. Advance
            // the request-local MTP owner along the target-accepted inputs so
            // later blocks retain the same hidden-state alignment as Qwen MTP.
            for (row, &block_input) in block_inputs.iter().take(committed_rows).enumerate() {
                let hidden_before = if row == 0 {
                    self.last_target_hidden_bf16.as_slice()
                } else {
                    let start = (row - 1) * Self::HIDDEN_WIDTH;
                    &hidden[start..start + Self::HIDDEN_WIDTH]
                };
                self.mtp
                    .decode_mtp(block_input, hidden_before)
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            }
        }
        self.pending_speculative_block = Some(PendingQwenSpeculativeBlockV1 {
            hidden_rows_bf16: hidden.to_vec(),
            target_accepted_draft_tokens: accepted,
            target_input_rows: committed_rows,
            proposed_draft_tokens: draft_width,
        });
        Ok(steps)
    }

    fn finalize_mtp_draft(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        let pending = self.pending_speculative_block.take().ok_or_else(|| {
            GenerationServiceError::Execution(
                "no speculative target block is pending finalization".to_owned(),
            )
        })?;
        if committed_input_rows == 0 || committed_input_rows > pending.target_input_rows {
            self.pending_speculative_block = Some(pending);
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::InvalidDecision.to_string(),
            ));
        }

        let proposal_blocks = self
            .proposal_blocks
            .checked_add(1)
            .ok_or(GenerationServiceError::CountOverflow)?;
        let proposed_draft_tokens = self
            .proposed_draft_tokens
            .checked_add(
                u64::try_from(pending.proposed_draft_tokens)
                    .map_err(|_| GenerationServiceError::CountOverflow)?,
            )
            .ok_or(GenerationServiceError::CountOverflow)?;
        let committed_accepted = pending
            .target_accepted_draft_tokens
            .min(committed_input_rows);
        let accepted_draft_tokens = self
            .accepted_draft_tokens
            .checked_add(
                u64::try_from(committed_accepted)
                    .map_err(|_| GenerationServiceError::CountOverflow)?,
            )
            .ok_or(GenerationServiceError::CountOverflow)?;
        let committed_target_rows = self
            .committed_target_rows
            .checked_add(
                u64::try_from(committed_input_rows)
                    .map_err(|_| GenerationServiceError::CountOverflow)?,
            )
            .ok_or(GenerationServiceError::CountOverflow)?;

        for _ in committed_input_rows..pending.target_input_rows {
            self.mtp
                .rewind_last_decode_transition()
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        self.target
            .resolve_decode_block(committed_input_rows)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let hidden_start = (committed_input_rows - 1) * Self::HIDDEN_WIDTH;
        self.last_target_hidden_bf16 =
            pending.hidden_rows_bf16[hidden_start..hidden_start + Self::HIDDEN_WIDTH].to_vec();
        self.proposal_blocks = proposal_blocks;
        self.proposed_draft_tokens = proposed_draft_tokens;
        self.accepted_draft_tokens = accepted_draft_tokens;
        self.committed_target_rows = committed_target_rows;
        Ok(())
    }
}

/// Minimal text frontend seam used by the transport-independent service.
/// Production uses the verified tokenizer; tests can model byte-fallback
/// boundaries without loading a model asset.
pub trait GenerationTextFrontendV1 {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError>;

    /// Encodes an explicit assistant continuation without injecting prompt
    /// special tokens.  Lightweight test frontends inherit the generation
    /// encoder because they do not model special-token insertion.
    fn encode_assistant_prefill(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.encode_generation(text)
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError>;

    /// Returns the decoder-aware raw token pieces needed by constrained
    /// generation. Lightweight frontends may keep the default fail-closed
    /// implementation; production tokenizer frontends override it.
    fn token_byte_table(&self) -> Result<&TokenByteTableV1, GenerationServiceError> {
        Err(GenerationServiceError::TokenBytesUnsupported)
    }
}

/// Receives text that is safe to expose to a transport. Implementations must
/// apply their own bounded backpressure and return promptly after cancellation.
pub trait GenerationOutputSinkV1 {
    fn publish(&mut self, delta: &str) -> Result<(), GenerationServiceError>;

    /// Receives hidden reasoning deltas.  The default intentionally drops
    /// them so callers that only consume visible answer text cannot
    /// accidentally expose private reasoning content.  A protocol adapter
    /// may opt in when it has a dedicated reasoning transport channel.
    fn publish_reasoning(&mut self, _delta: &str) -> Result<(), GenerationServiceError> {
        Ok(())
    }
}

struct IgnoreGenerationOutput;

impl GenerationOutputSinkV1 for IgnoreGenerationOutput {
    fn publish(&mut self, _: &str) -> Result<(), GenerationServiceError> {
        Ok(())
    }
}

impl GenerationTextFrontendV1 for TokenizerFrontendV1 {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.encode(text)
            .map(|ids| ids.as_slice().to_vec())
            .map_err(|_| GenerationServiceError::Tokenize)
    }

    fn encode_assistant_prefill(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.encode_without_special_tokens(text)
            .map(|ids| ids.as_slice().to_vec())
            .map_err(|_| GenerationServiceError::Tokenize)
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        self.decode(
            &TokenIdsV1::from_slice(token_ids),
            DecodeModeV1::PreserveSpecialTokens,
        )
        .map_err(|_| GenerationServiceError::Decode)
    }

    fn token_byte_table(&self) -> Result<&TokenByteTableV1, GenerationServiceError> {
        Ok(TokenizerFrontendV1::token_byte_table(self))
    }
}

impl GenerationExecutorV1 for QwenExecutionRequest {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = if include_last_logits {
            self.prefill_with_last_logits(&input)
        } else {
            self.prefill(&input)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        Ok(GenerationStepV1::new(
            u32::try_from(argmax).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            output.last_logits().map(<[f32]>::to_vec),
        ))
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = if include_last_logits {
            self.decode_with_last_logits(token)
        } else {
            self.decode(token)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        if output.token_ids().len() != 1 {
            return Err(GenerationServiceError::MissingDeviceArgmax);
        }
        Ok(GenerationStepV1::new(
            u32::try_from(output.token_ids()[0])
                .map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            output.last_logits().map(<[f32]>::to_vec),
        ))
    }

    fn supports_device_selector(&self) -> bool {
        true
    }

    fn prefill_with_device_selector(
        &mut self,
        input_token_ids: &[u32],
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = QwenExecutionRequest::prefill_with_device_selector(self, &input, selector)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let selection = output
            .selection()
            .cloned()
            .ok_or(GenerationServiceError::MissingDeviceSelection)?;
        Ok(GenerationStepV1::from_device_selection(selection))
    }

    fn decode_with_device_selector(
        &mut self,
        token_id: u32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = QwenExecutionRequest::decode_with_device_selector(self, token, selector)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let selection = output
            .selection()
            .cloned()
            .ok_or(GenerationServiceError::MissingDeviceSelection)?;
        Ok(GenerationStepV1::from_device_selection(selection))
    }

    fn cancel(&mut self) {
        QwenExecutionRequest::cancel(self);
    }
}

impl GenerationExecutorV1 for Gemma4ExecutionRequest {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = if include_last_logits {
            Gemma4ExecutionRequest::prefill_with_last_logits(self, &input)
        } else {
            Gemma4ExecutionRequest::prefill(self, &input)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        Ok(GenerationStepV1::new(
            u32::try_from(argmax).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            output.last_logits().map(<[f32]>::to_vec),
        ))
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = if include_last_logits {
            Gemma4ExecutionRequest::decode_with_last_logits(self, token)
        } else {
            Gemma4ExecutionRequest::decode(self, token)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        if output.token_ids().len() != 1 {
            return Err(GenerationServiceError::MissingDeviceArgmax);
        }
        Ok(GenerationStepV1::new(
            u32::try_from(output.token_ids()[0])
                .map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            output.last_logits().map(<[f32]>::to_vec),
        ))
    }

    fn supports_device_selector(&self) -> bool {
        true
    }

    fn prefill_with_device_selector(
        &mut self,
        input_token_ids: &[u32],
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = Gemma4ExecutionRequest::prefill_with_device_selector(self, &input, selector)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let selection = output
            .selection()
            .cloned()
            .ok_or(GenerationServiceError::MissingDeviceSelection)?;
        Ok(GenerationStepV1::from_device_selection(selection))
    }

    fn decode_with_device_selector(
        &mut self,
        token_id: u32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = Gemma4ExecutionRequest::decode_with_device_selector(self, token, selector)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let selection = output
            .selection()
            .cloned()
            .ok_or(GenerationServiceError::MissingDeviceSelection)?;
        Ok(GenerationStepV1::from_device_selection(selection))
    }

    fn cancel(&mut self) {
        Gemma4ExecutionRequest::cancel(self);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationServiceError {
    InvalidMaxNewTokens,
    TooManyStopStrings,
    EmptyStopString { index: usize },
    DuplicateStopString { index: usize },
    StopStringsTooLarge,
    MissingRenderer,
    Render,
    Tokenize,
    EmptyPromptTokens,
    InvalidAssistantPrefill,
    Decode,
    NonPrefixDecode,
    TokenIdOverflow,
    CountOverflow,
    MissingDeviceArgmax,
    MissingDeviceSelection,
    DeviceSelectorUnsupported,
    InvalidStopPolicy,
    InvalidChoiceCount,
    InconsistentChoicePrompt,
    SamplingConfigurationMismatch,
    TokenBytesUnsupported,
    GenericTemplate(TokenizerUtilityErrorV1),
    Speculative(String),
    Reasoning(ReasoningErrorV1),
    Grammar(GrammarError),
    Sampling(SamplingError),
    Execution(String),
    Output(String),
    Cancelled,
}

impl fmt::Display for GenerationServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaxNewTokens => {
                formatter.write_str("max_new_tokens must be greater than zero")
            }
            Self::TooManyStopStrings => {
                formatter.write_str("at most four stop strings are supported")
            }
            Self::EmptyStopString { index } => write!(formatter, "stop string {index} is empty"),
            Self::DuplicateStopString { index } => {
                write!(formatter, "stop string {index} is duplicated")
            }
            Self::StopStringsTooLarge => {
                formatter.write_str("stop strings exceed the bounded byte limit")
            }
            Self::MissingRenderer => formatter.write_str("chat messages require a renderer"),
            Self::Render => formatter.write_str("generation input could not be rendered"),
            Self::Tokenize => formatter.write_str("generation input could not be tokenized"),
            Self::EmptyPromptTokens => {
                formatter.write_str("generation input produced no token IDs")
            }
            Self::InvalidAssistantPrefill => formatter.write_str(
                "assistant prefill must be a nonempty proper suffix of the generation input",
            ),
            Self::Decode => formatter.write_str("generated token IDs could not be decoded"),
            Self::NonPrefixDecode => formatter
                .write_str("incremental tokenizer output changed an already decoded prefix"),
            Self::TokenIdOverflow => {
                formatter.write_str("token ID does not fit the execution contract")
            }
            Self::CountOverflow => formatter.write_str("generation token accounting overflowed"),
            Self::MissingDeviceArgmax => {
                formatter.write_str("execution published no device Argmax token")
            }
            Self::MissingDeviceSelection => {
                formatter.write_str("execution published no prepared device selection")
            }
            Self::DeviceSelectorUnsupported => {
                formatter.write_str("execution does not support the prepared device selector")
            }
            Self::InvalidStopPolicy => {
                formatter.write_str("generation stop-token policy is invalid")
            }
            Self::InvalidChoiceCount => {
                formatter.write_str("generation choice count must be between one and eight")
            }
            Self::InconsistentChoicePrompt => {
                formatter.write_str("generation choices must share the exact prompt token IDs")
            }
            Self::SamplingConfigurationMismatch => formatter.write_str(
                "sampler-chain parameters must match the generation sampling parameters",
            ),
            Self::GenericTemplate(error) => error.fmt(formatter),
            Self::Reasoning(error) => error.fmt(formatter),
            Self::TokenBytesUnsupported => {
                formatter.write_str("generation frontend does not expose raw token bytes")
            }
            Self::Speculative(reason) => {
                write!(formatter, "speculative generation failed: {reason}")
            }
            Self::Grammar(error) => error.fmt(formatter),
            Self::Sampling(error) => error.fmt(formatter),
            Self::Execution(reason) => write!(formatter, "generation execution failed: {reason}"),
            Self::Output(reason) => write!(formatter, "generation output failed: {reason}"),
            Self::Cancelled => formatter.write_str("generation was cancelled"),
        }
    }
}

impl std::error::Error for GenerationServiceError {}

impl From<SamplingError> for GenerationServiceError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<GrammarError> for GenerationServiceError {
    fn from(error: GrammarError) -> Self {
        Self::Grammar(error)
    }
}

impl From<SpeculativeError> for GenerationServiceError {
    fn from(error: SpeculativeError) -> Self {
        Self::Speculative(error.to_string())
    }
}

impl From<ReasoningErrorV1> for GenerationServiceError {
    fn from(error: ReasoningErrorV1) -> Self {
        Self::Reasoning(error)
    }
}

pub struct GenerationServiceV1<'a> {
    tokenizer: &'a dyn GenerationTextFrontendV1,
    renderer: Option<ChatTemplateRendererV1<'a>>,
    stop_policy: &'a GenerationStopPolicyV1,
}

impl<'a> GenerationServiceV1<'a> {
    pub fn new(
        tokenizer: &'a dyn GenerationTextFrontendV1,
        renderer: Option<&'a Qwen35ChatTemplateV1>,
        stop_policy: &'a GenerationStopPolicyV1,
    ) -> Result<Self, GenerationServiceError> {
        Self::new_with_chat_renderer(
            tokenizer,
            renderer.map(ChatTemplateRendererV1::qwen35),
            stop_policy,
        )
    }

    /// Constructs the shared generation service with a model-selected chat
    /// renderer. Existing Qwen callers keep using [`Self::new`]; reviewed
    /// non-Qwen models can select the same message preparation path with a
    /// bounded [`ChatTemplateRendererV1::Generic`] provider.
    pub fn new_with_chat_renderer(
        tokenizer: &'a dyn GenerationTextFrontendV1,
        renderer: Option<ChatTemplateRendererV1<'a>>,
        stop_policy: &'a GenerationStopPolicyV1,
    ) -> Result<Self, GenerationServiceError> {
        validate_generation_stop_policy(stop_policy)
            .map_err(|_| GenerationServiceError::InvalidStopPolicy)?;
        Ok(Self {
            tokenizer,
            renderer,
            stop_policy,
        })
    }

    pub fn generate(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input: &GenerationInputV1,
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        let mut sink = IgnoreGenerationOutput;
        self.generate_with_sink(executor, input, config, cancellation, random, &mut sink)
    }

    pub fn generate_with_sink(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input: &GenerationInputV1,
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        let prepared = self.prepare_input_plan(input)?;
        self.generate_prepared_with_sink(executor, &prepared, config, cancellation, random, sink)
    }

    /// Runs the same renderer/tokenizer path as [`Self::generate`] while
    /// allowing a model owner to size its request graph before execution.
    pub fn prepare_input(
        &self,
        input: &GenerationInputV1,
    ) -> Result<Vec<u32>, GenerationServiceError> {
        Ok(self.prepare_input_plan(input)?.token_ids().to_vec())
    }

    /// Prepares a transport-independent input while retaining the explicit
    /// assistant-prefix boundary needed by grammar, usage, and visibility
    /// semantics.  Existing `prepare_input` callers receive the same token
    /// IDs as before for the legacy variants.
    pub fn prepare_input_plan(
        &self,
        input: &GenerationInputV1,
    ) -> Result<PreparedGenerationInputV1, GenerationServiceError> {
        let (rendered, assistant_prefill, generic_identity) = match input {
            GenerationInputV1::Prompt(prompt) => (prompt.clone(), None, None),
            GenerationInputV1::Messages { messages, options } => {
                let rendered = self
                    .renderer
                    .as_ref()
                    .ok_or(GenerationServiceError::MissingRenderer)?
                    .render(messages, *options)
                    .map_err(|_| GenerationServiceError::Render)?;
                (
                    rendered.rendered().to_owned(),
                    None,
                    rendered.generic_identity().cloned(),
                )
            }
            GenerationInputV1::PromptWithAssistantPrefill {
                prompt,
                assistant_prefill,
            } => (prompt.clone(), Some(assistant_prefill.as_str()), None),
            GenerationInputV1::MessagesWithAssistantPrefill {
                messages,
                options,
                assistant_prefill,
            } => {
                let rendered = self
                    .renderer
                    .as_ref()
                    .ok_or(GenerationServiceError::MissingRenderer)?
                    .render_with_assistant_prefill(messages, *options)
                    .map_err(|_| GenerationServiceError::Render)?;
                (
                    rendered.rendered().to_owned(),
                    Some(assistant_prefill.as_str()),
                    rendered.generic_identity().cloned(),
                )
            }
            GenerationInputV1::GenericTemplate(generic) => (
                generic.rendered().to_owned(),
                None,
                Some(generic.identity().clone()),
            ),
        };
        let base_token_ids = self.tokenizer.encode_generation(&rendered)?;
        let assistant_prefill_token_ids = assistant_prefill
            .map(|prefill| self.tokenizer.encode_assistant_prefill(prefill))
            .transpose()?
            .unwrap_or_default();
        let prepared =
            PreparedGenerationInputV1::from_token_ids(base_token_ids, assistant_prefill_token_ids)?;
        if let Some(identity) = generic_identity {
            Ok(prepared.with_generic_template_identity(identity))
        } else {
            Ok(prepared)
        }
    }

    pub fn generate_prepared(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input: &PreparedGenerationInputV1,
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        let mut sink = IgnoreGenerationOutput;
        self.generate_prepared_with_sink(executor, input, config, cancellation, random, &mut sink)
    }

    pub fn generate_prepared_with_sink(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input: &PreparedGenerationInputV1,
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        let result = self.generate_tokens_inner(
            executor,
            input.token_ids(),
            input.assistant_prefill_token_ids(),
            config,
            cancellation,
            random,
            sink,
        );
        if result.is_err() {
            executor.cancel();
        }
        result
    }

    pub fn generate_tokens(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input_token_ids: &[u32],
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        let mut sink = IgnoreGenerationOutput;
        self.generate_tokens_with_sink(
            executor,
            input_token_ids,
            config,
            cancellation,
            random,
            &mut sink,
        )
    }

    pub fn generate_tokens_with_sink(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input_token_ids: &[u32],
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        if input_token_ids.is_empty() {
            return Err(GenerationServiceError::EmptyPromptTokens);
        }
        let result = self.generate_tokens_inner(
            executor,
            input_token_ids,
            &[],
            config,
            cancellation,
            random,
            sink,
        );
        if result.is_err() {
            executor.cancel();
        }
        result
    }

    /// Generate from a complete prompt while preserving the suffix that is an
    /// assistant continuation. The suffix is already in model context and is
    /// never republished, but it primes stop and grammar state before the
    /// first generated token.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_tokens_with_assistant_prefill_sink(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input_token_ids: &[u32],
        assistant_prefill_token_ids: &[u32],
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        if input_token_ids.is_empty()
            || assistant_prefill_token_ids.len() >= input_token_ids.len()
            || !input_token_ids.ends_with(assistant_prefill_token_ids)
        {
            return Err(GenerationServiceError::InvalidAssistantPrefill);
        }
        let result = self.generate_tokens_inner(
            executor,
            input_token_ids,
            assistant_prefill_token_ids,
            config,
            cancellation,
            random,
            sink,
        );
        if result.is_err() {
            executor.cancel();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_tokens_inner(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input_token_ids: &[u32],
        assistant_prefill_token_ids: &[u32],
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        check_cancelled(cancellation)?;
        if let Some(policy) = config.reasoning() {
            policy.validate_max_new_tokens(config.max_new_tokens())?;
        }
        let mut reasoning_controller = config.reasoning().cloned().map(ReasoningControllerV1::new);
        let sampler_config = config
            .sampler_chain
            .clone()
            .unwrap_or_else(|| SamplerChainConfigV1::legacy(config.sampling));
        let include_logits = sampler_config.requires_logits()
            || config.grammar.is_some()
            || config.ignore_stop_tokens
            // A bounded reasoning policy may need to intersect a singleton
            // forced-close mask with host logits.  Request logits up front so
            // the controller never falls back to host-side token insertion.
            || reasoning_controller
                .as_ref()
                .is_some_and(ReasoningControllerV1::is_enabled);
        let mut sampler = SamplerChainV1::new(sampler_config, input_token_ids)?;
        let (mut grammar_state, grammar_trie) = if let Some(grammar) = config.grammar.as_ref() {
            let table = self.tokenizer.token_byte_table()?;
            let pieces = table
                .as_slice()
                .iter()
                .map(|entry| {
                    if entry.is_grammar_eligible() {
                        entry.bytes().map(|bytes| bytes.to_vec())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let trie = TokenTrie::new_optional(pieces)?;
            (Some(grammar.initial_state()), Some(trie))
        } else {
            (None, None)
        };
        // An explicit assistant continuation is already part of the prompt
        // context.  Prime the grammar with its exact raw token bytes before
        // constructing the first candidate mask; invalid prefixes therefore
        // fail closed before any executor work is submitted.
        if !assistant_prefill_token_ids.is_empty() {
            if let Some(state) = grammar_state.as_mut() {
                let table = self.tokenizer.token_byte_table()?;
                for &token in assistant_prefill_token_ids {
                    let entry = table
                        .entry(token)
                        .ok_or(GenerationServiceError::TokenIdOverflow)?;
                    let bytes = entry
                        .bytes()
                        .ok_or(GenerationServiceError::TokenBytesUnsupported)?;
                    state.accept(bytes)?;
                }
            }
        }
        let use_device_selector = config.device_selector_seed.is_some()
            && sampler.supports_device_selector()
            && executor.supports_device_selector();
        let selector_vocab_size = if use_device_selector {
            Some(self.tokenizer.token_byte_table()?.len())
        } else {
            None
        };
        let mut device_mask = if use_device_selector {
            let base_mask = build_valid_token_mask(
                grammar_state.as_ref(),
                grammar_trie.as_ref(),
                config.ignore_stop_tokens,
                &self.stop_policy.stop_token_ids,
                selector_vocab_size,
            )?;
            reasoning_controller
                .as_ref()
                .map(|controller| controller.apply_mask(base_mask.as_deref(), selector_vocab_size))
                .transpose()?
                .flatten()
                .or(base_mask)
        } else {
            None
        };
        let mut step = if use_device_selector {
            let selector = sampler.prepare_device_selector(
                selector_vocab_size.expect("device selector vocabulary was resolved"),
                device_mask.as_deref(),
                config
                    .device_selector_seed
                    .expect("device selector seed was resolved"),
                0,
            )?;
            executor.prefill_with_device_selector(input_token_ids, &selector)?
        } else {
            executor.prefill(input_token_ids, include_logits)?
        };
        let mut matcher = IncrementalStopMatcher::new(config.stop_strings.clone());
        let mut generated = Vec::new();
        let mut selections = Vec::new();
        let mut normal_tokens = Vec::<u32>::new();
        let mut visible_token_ids = Vec::<u32>::new();
        let mut reasoning_token_ids = Vec::<u32>::new();
        let mut decoded_snapshots = Vec::<String>::new();
        let mut decode_inputs = Vec::new();
        let mut decode_context_tokens = assistant_prefill_token_ids.to_vec();
        let assistant_decoded = if decode_context_tokens.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode_generation(&decode_context_tokens)?
        };
        let assistant_stable_end = assistant_decoded.trim_end_matches('\u{fffd}').len();
        let mut decoded = assistant_decoded[..assistant_stable_end].to_owned();
        let mut unstable_utf8_tail = assistant_decoded[assistant_stable_end..].to_owned();
        // Prime only the suffix which can begin a stop sequence. Complete stop
        // strings wholly inside prompt state are not generation stops, and no
        // bytes released by priming are ever sent to the visible sink.
        matcher.prime(&decoded);
        let mut generated_decoded = String::new();
        let mut output_text = String::new();
        let mut finish_reason = None;
        let mut stop_token_id = None;
        let mut matched_stop = None;
        let mut decode_steps = 0_u32;

        for index in 0..config.max_new_tokens {
            check_cancelled(cancellation)?;
            let base_mask = if use_device_selector {
                device_mask.take()
            } else {
                build_valid_token_mask(
                    grammar_state.as_ref(),
                    grammar_trie.as_ref(),
                    config.ignore_stop_tokens,
                    &self.stop_policy.stop_token_ids,
                    step.last_logits.as_ref().map(Vec::len),
                )?
            };
            let valid_mask = reasoning_controller
                .as_ref()
                .map(|controller| {
                    controller.apply_mask(
                        base_mask.as_deref(),
                        if use_device_selector {
                            selector_vocab_size
                        } else {
                            step.last_logits.as_ref().map(Vec::len)
                        },
                    )
                })
                .transpose()?
                .flatten()
                .or(base_mask);
            let selection = if use_device_selector {
                step.device_selection
                    .clone()
                    .ok_or(GenerationServiceError::MissingDeviceSelection)?
            } else {
                sampler.select_with_mask(
                    step.device_argmax,
                    step.last_logits.as_deref(),
                    valid_mask.as_deref(),
                    random,
                )?
            };
            let token = selection.token_id;
            if valid_mask
                .as_ref()
                .is_some_and(|mask| !mask.get(token as usize).copied().unwrap_or(false))
            {
                return Err(GenerationServiceError::Reasoning(
                    ReasoningErrorV1::ForcedTokenMismatch,
                ));
            }
            let is_stop_token = self.stop_policy.stop_token_ids.contains(&token);
            let reasoning_observation = reasoning_controller
                .as_mut()
                .filter(|_| !is_stop_token)
                .map(|controller| controller.observe(token))
                .transpose()?;
            let token_visible = reasoning_observation
                .as_ref()
                .is_none_or(|observation| observation.visible())
                && !is_stop_token;
            if !is_stop_token {
                if let Some(observation) = reasoning_observation {
                    if !observation.visible() {
                        reasoning_token_ids.push(token);
                    } else {
                        visible_token_ids.push(token);
                    }
                    if observation.entered_answer() {
                        // Any stop prefix held while reasoning is hidden must
                        // not join the visible answer across the closing marker.
                        matcher = IncrementalStopMatcher::new(config.stop_strings.clone());
                        generated_decoded.clear();
                        decoded_snapshots.clear();
                    }
                } else {
                    visible_token_ids.push(token);
                }
            }
            if !is_stop_token {
                if let Some(state) = grammar_state.as_mut() {
                    let table = self.tokenizer.token_byte_table()?;
                    let entry = table
                        .entry(token)
                        .ok_or(GenerationServiceError::TokenIdOverflow)?;
                    let bytes = entry
                        .bytes()
                        .ok_or(GenerationServiceError::TokenBytesUnsupported)?;
                    state.accept(bytes)?;
                }
            }
            sampler.accept(token)?;
            generated.push(token);
            selections.push(selection);

            if self.stop_policy.stop_token_ids.contains(&token) {
                finish_reason = Some(FinishReasonV1::Stop);
                stop_token_id = Some(token);
                if token_visible && !normal_tokens.is_empty() && !unstable_utf8_tail.is_empty() {
                    let tail = matcher.push(&unstable_utf8_tail);
                    publish_visible(&mut output_text, &tail.visible, sink)?;
                }
                publish_visible(&mut output_text, &matcher.finish(), sink)?;
                break;
            }

            decode_context_tokens.push(token);
            let candidate_ids = &decode_context_tokens;
            let candidate = self.tokenizer.decode_generation(candidate_ids)?;
            // Hugging Face byte-fallback decoding can temporarily end in one
            // or more replacement characters and repair that suffix after a
            // later token completes the UTF-8 sequence. Never publish or feed
            // that unstable suffix to the stop matcher early.
            let stable_end = candidate.trim_end_matches('\u{fffd}').len();
            let stable_candidate = &candidate[..stable_end];
            let delta = stable_candidate
                .strip_prefix(&decoded)
                .ok_or(GenerationServiceError::NonPrefixDecode)?;
            let match_result = if token_visible {
                matcher.push(delta)
            } else {
                StopMatch {
                    visible: String::new(),
                    matched: None,
                }
            };
            if token_visible {
                publish_visible(&mut output_text, &match_result.visible, sink)?;
            } else if !delta.is_empty() {
                // Reasoning content is excluded from `output_text` and
                // visible token accounting.  Only a protocol adapter that
                // explicitly implements the reasoning channel receives it.
                sink.publish_reasoning(delta)?;
            }
            if token_visible {
                generated_decoded.push_str(delta);
            }
            decoded = stable_candidate.to_owned();
            unstable_utf8_tail = candidate[stable_end..].to_owned();
            normal_tokens.push(token);
            if token_visible {
                decoded_snapshots.push(format!("{generated_decoded}{unstable_utf8_tail}"));
            }
            if let Some(stop) = match_result.matched {
                finish_reason = Some(FinishReasonV1::Stop);
                matched_stop = Some(stop);
                break;
            }

            if index + 1 == config.max_new_tokens {
                let reasoning_output_open = reasoning_controller
                    .as_ref()
                    .is_some_and(ReasoningControllerV1::is_reasoning);
                if !reasoning_output_open && !unstable_utf8_tail.is_empty() {
                    let tail = matcher.push(&unstable_utf8_tail);
                    publish_visible(&mut output_text, &tail.visible, sink)?;
                    if let Some(stop) = tail.matched {
                        finish_reason = Some(FinishReasonV1::Stop);
                        matched_stop = Some(stop);
                        break;
                    }
                }
                publish_visible(&mut output_text, &matcher.finish(), sink)?;
                finish_reason = Some(FinishReasonV1::Length);
                break;
            }
            decode_inputs.push(token);
            check_cancelled(cancellation)?;
            step = if use_device_selector {
                executor.before_decode(token)?;
                let base_mask = build_valid_token_mask(
                    grammar_state.as_ref(),
                    grammar_trie.as_ref(),
                    config.ignore_stop_tokens,
                    &self.stop_policy.stop_token_ids,
                    selector_vocab_size,
                )?;
                device_mask = reasoning_controller
                    .as_ref()
                    .map(|controller| {
                        controller.apply_mask(base_mask.as_deref(), selector_vocab_size)
                    })
                    .transpose()?
                    .flatten()
                    .or(base_mask);
                let selector = sampler.prepare_device_selector(
                    selector_vocab_size.expect("device selector vocabulary was resolved"),
                    device_mask.as_deref(),
                    config
                        .device_selector_seed
                        .expect("device selector seed was resolved"),
                    u64::from(index) + 1,
                )?;
                executor.decode_with_device_selector(token, &selector)?
            } else {
                executor.before_decode(token)?;
                executor.decode(token, include_logits)?
            };
            decode_steps = decode_steps
                .checked_add(1)
                .ok_or(GenerationServiceError::CountOverflow)?;
        }

        executor.finish()?;
        let finish_reason = finish_reason.ok_or(GenerationServiceError::CountOverflow)?;
        let visible_count = decoded_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                !snapshot.is_empty() && output_text.starts_with(snapshot.as_str())
            })
            .map(|(index, _)| index + 1)
            .max()
            .unwrap_or(0);
        let visible_token_ids = if reasoning_controller.is_some() {
            visible_token_ids[..visible_count.min(visible_token_ids.len())].to_vec()
        } else {
            normal_tokens[..visible_count].to_vec()
        };
        let prompt_tokens = u64::try_from(input_token_ids.len())
            .map_err(|_| GenerationServiceError::CountOverflow)?;
        let completion_tokens =
            u64::try_from(generated.len()).map_err(|_| GenerationServiceError::CountOverflow)?;
        let total_tokens = prompt_tokens
            .checked_add(completion_tokens)
            .ok_or(GenerationServiceError::CountOverflow)?;
        let reasoning_tokens = reasoning_controller
            .as_ref()
            .map(ReasoningControllerV1::reasoning_tokens)
            .unwrap_or(0);
        Ok(GenerationResultV1 {
            input_token_ids: input_token_ids.to_vec(),
            generated_token_ids: generated,
            visible_token_ids,
            decode_input_token_ids: decode_inputs,
            output_text,
            finish_reason,
            stop_token_id,
            matched_stop,
            usage: TokenUsageV1 {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens: u64::from(reasoning_tokens),
            },
            decode_steps,
            selections,
            reasoning_token_ids,
        })
    }
}

fn build_valid_token_mask(
    grammar_state: Option<&sllm_core::GrammarState>,
    grammar_trie: Option<&TokenTrie>,
    ignore_stop_tokens: bool,
    stop_token_ids: &[u32],
    vocab_size_hint: Option<usize>,
) -> Result<Option<Vec<bool>>, GenerationServiceError> {
    let mut mask = if let (Some(state), Some(trie)) = (grammar_state, grammar_trie) {
        let mut mask = match state.valid_token_mask_with_trie(trie) {
            Ok(mask) => mask,
            Err(GrammarError::AllTokensMasked)
                if state.is_finished() && state.is_utf8_boundary() =>
            {
                vec![false; trie.token_count()]
            }
            Err(error) => return Err(error.into()),
        };
        // Special stop rows are deliberately absent from the grammar trie.
        // They become eligible only after a complete UTF-8 accept state.
        if state.is_finished() && state.is_utf8_boundary() {
            for &stop in stop_token_ids {
                if let Ok(index) = usize::try_from(stop) {
                    if let Some(value) = mask.get_mut(index) {
                        *value = true;
                    }
                }
            }
        }
        Some(mask)
    } else if ignore_stop_tokens {
        let vocab_size = vocab_size_hint.ok_or(SamplingError::MissingLogits)?;
        Some(vec![true; vocab_size])
    } else {
        None
    };

    if ignore_stop_tokens {
        if let Some(mask) = mask.as_mut() {
            for &stop in stop_token_ids {
                if let Ok(index) = usize::try_from(stop) {
                    if let Some(value) = mask.get_mut(index) {
                        *value = false;
                    }
                }
            }
        }
    }
    if mask
        .as_ref()
        .is_some_and(|values| !values.iter().any(|value| *value))
    {
        return Err(GrammarError::AllTokensMasked.into());
    }
    Ok(mask)
}

fn publish_visible(
    output: &mut String,
    delta: &str,
    sink: &mut impl GenerationOutputSinkV1,
) -> Result<(), GenerationServiceError> {
    if delta.is_empty() {
        return Ok(());
    }
    sink.publish(delta)?;
    output.push_str(delta);
    Ok(())
}

fn check_cancelled(cancellation: &GenerationCancellationV1) -> Result<(), GenerationServiceError> {
    if cancellation.is_cancelled() {
        Err(GenerationServiceError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct StopMatch {
    visible: String,
    matched: Option<String>,
}

#[derive(Debug)]
struct IncrementalStopMatcher {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl IncrementalStopMatcher {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            stopped: false,
        }
    }

    fn push(&mut self, delta: &str) -> StopMatch {
        debug_assert!(!self.stopped);
        self.pending.push_str(delta);
        let matched = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop).map(|start| (start, stop)))
            .min_by(|(left_start, left), (right_start, right)| {
                left_start
                    .cmp(right_start)
                    .then_with(|| left.len().cmp(&right.len()))
            })
            .map(|(start, stop)| (start, stop.clone()));
        if let Some((start, stop)) = matched {
            let visible = self.pending[..start].to_owned();
            self.pending.clear();
            self.stopped = true;
            return StopMatch {
                visible,
                matched: Some(stop),
            };
        }

        let hold = self
            .pending
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(self.pending.len()))
            .filter(|&index| {
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(&self.pending[index..]))
            })
            .map(|index| self.pending.len() - index)
            .max()
            .unwrap_or(0);
        let safe = self.pending.len() - hold;
        let visible = self.pending[..safe].to_owned();
        self.pending.drain(..safe);
        StopMatch {
            visible,
            matched: None,
        }
    }

    fn prime(&mut self, context: &str) {
        debug_assert!(!self.stopped);
        let mut best = None;
        for (start, _) in context
            .char_indices()
            .chain(std::iter::once((context.len(), '\0')))
        {
            let suffix = &context[start..];
            if self
                .stops
                .iter()
                .any(|stop| suffix.len() < stop.len() && stop.starts_with(suffix))
                && best.is_none_or(|current: usize| suffix.len() > current)
            {
                best = Some(suffix.len());
            }
        }
        self.pending = best
            .map(|length| context[context.len() - length..].to_owned())
            .unwrap_or_default();
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{
        BudgetBoundary, MaxNewTokensZero, PromptEvaluation, StopEvaluation, StopTokenHandling,
    };
    use std::collections::VecDeque;

    #[test]
    fn gemma_stop_policy_is_derived_from_the_reviewed_lock() {
        let lock = sllm_core::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-bf16.json"
        ))
        .unwrap();
        let policy = gemma4_generation_stop_policy(&lock).unwrap();
        assert_eq!(policy.stop_token_ids, [1]);
        assert_eq!(
            policy.evaluation,
            sllm_core::StopEvaluation::NewlyGeneratedAfterArgmax
        );
        assert!(!policy.stop_token.visible_output);
        assert!(!policy.stop_token.subsequent_decode_input);
    }

    #[test]
    fn gemma4_moe_stop_policy_is_the_reviewed_eos_turn_and_tool_boundary() {
        let policy = gemma4_moe_generation_stop_policy().unwrap();
        assert_eq!(policy.stop_token_ids, [1, 106, 50]);
        assert_eq!(
            policy.evaluation,
            sllm_core::StopEvaluation::NewlyGeneratedAfterArgmax
        );
        assert_eq!(
            policy.prompt_evaluation,
            sllm_core::PromptEvaluation::NeverStop
        );
        assert!(!policy.stop_token.visible_output);
        assert!(!policy.stop_token.subsequent_decode_input);
    }

    struct FixedRandom(f64);

    impl SamplingRandomSource for FixedRandom {
        fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
            Ok(self.0)
        }
    }

    #[derive(Default)]
    struct TinyExecutor {
        steps: VecDeque<GenerationStepV1>,
        include_logits: Vec<bool>,
        prefill_inputs: Vec<Vec<u32>>,
        decode_inputs: Vec<u32>,
        cancel_count: u32,
    }

    impl TinyExecutor {
        fn argmax(tokens: impl IntoIterator<Item = u32>) -> Self {
            Self {
                steps: tokens
                    .into_iter()
                    .map(|token| GenerationStepV1::new(token, None))
                    .collect(),
                ..Self::default()
            }
        }

        fn next(
            &mut self,
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.include_logits.push(include_last_logits);
            self.steps.pop_front().ok_or_else(|| {
                GenerationServiceError::Execution("tiny executor exhausted".to_owned())
            })
        }
    }

    impl GenerationExecutorV1 for TinyExecutor {
        fn prefill(
            &mut self,
            input_token_ids: &[u32],
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.prefill_inputs.push(input_token_ids.to_vec());
            self.next(include_last_logits)
        }

        fn decode(
            &mut self,
            token_id: u32,
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.decode_inputs.push(token_id);
            self.next(include_last_logits)
        }

        fn cancel(&mut self) {
            self.cancel_count += 1;
        }
    }

    struct TinyDeviceSelectorExecutor {
        selections: VecDeque<SamplingSelectionV1>,
        requests: Vec<(u64, u64, usize, usize)>,
        prefill_inputs: Vec<Vec<u32>>,
        decode_inputs: Vec<u32>,
        cancel_count: u32,
    }

    impl TinyDeviceSelectorExecutor {
        fn select(&mut self, request: &DeviceTokenSelectorRequestV1) -> GenerationStepV1 {
            self.requests.push((
                request.seed(),
                request.counter(),
                request.additive_logits().len(),
                request
                    .valid_mask()
                    .iter()
                    .filter(|&&value| value != 0)
                    .count(),
            ));
            GenerationStepV1::from_device_selection(
                self.selections
                    .pop_front()
                    .expect("device selection fixture exhausted"),
            )
        }
    }

    impl GenerationExecutorV1 for TinyDeviceSelectorExecutor {
        fn prefill(
            &mut self,
            _: &[u32],
            _: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            panic!("host prefill must not be used after the device route is selected")
        }

        fn decode(&mut self, _: u32, _: bool) -> Result<GenerationStepV1, GenerationServiceError> {
            panic!("host decode must not be used after the device route is selected")
        }

        fn supports_device_selector(&self) -> bool {
            true
        }

        fn prefill_with_device_selector(
            &mut self,
            input_token_ids: &[u32],
            selector: &DeviceTokenSelectorRequestV1,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.prefill_inputs.push(input_token_ids.to_vec());
            Ok(self.select(selector))
        }

        fn decode_with_device_selector(
            &mut self,
            token_id: u32,
            selector: &DeviceTokenSelectorRequestV1,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.decode_inputs.push(token_id);
            Ok(self.select(selector))
        }

        fn cancel(&mut self) {
            self.cancel_count += 1;
        }
    }

    struct TinySpeculativeExecutor {
        prefill_step: Option<GenerationStepV1>,
        target_only_steps: VecDeque<GenerationStepV1>,
        batches: VecDeque<Vec<GenerationStepV1>>,
        speculative_inputs: Vec<u32>,
        target_only_inputs: Vec<u32>,
        finalized_rows: Vec<usize>,
        cancelled: bool,
    }

    struct TinyProviderExecutor {
        prefill_step: Option<GenerationStepV1>,
        batches: VecDeque<Vec<GenerationStepV1>>,
        proposals: Vec<Vec<u32>>,
        target_only_inputs: Vec<u32>,
        finalized_rows: Vec<usize>,
        cancelled: bool,
    }

    impl GenerationExecutorV1 for TinyProviderExecutor {
        fn prefill(
            &mut self,
            _: &[u32],
            _: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.prefill_step.take().ok_or_else(|| {
                GenerationServiceError::Execution("missing provider prefill".to_owned())
            })
        }

        fn decode(
            &mut self,
            token_id: u32,
            _: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.target_only_inputs.push(token_id);
            Err(GenerationServiceError::Execution(
                "provider path unexpectedly used target-only decode".to_owned(),
            ))
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    impl SpeculativeGenerationExecutorV1 for TinyProviderExecutor {
        fn speculative_decode_greedy(
            &mut self,
            _: u32,
        ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
            Err(GenerationServiceError::Execution(
                "legacy provider path unexpectedly used".to_owned(),
            ))
        }

        fn speculative_decode_with_proposal(
            &mut self,
            _: u32,
            proposal: &DraftProposalV1,
        ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
            self.proposals.push(proposal.token_ids().to_vec());
            self.batches.pop_front().ok_or_else(|| {
                GenerationServiceError::Execution("missing provider target batch".to_owned())
            })
        }

        fn finalize_speculative_decode(
            &mut self,
            committed_input_rows: usize,
        ) -> Result<(), GenerationServiceError> {
            self.finalized_rows.push(committed_input_rows);
            Ok(())
        }
    }

    impl GenerationExecutorV1 for TinySpeculativeExecutor {
        fn prefill(
            &mut self,
            _: &[u32],
            _: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.prefill_step.take().ok_or_else(|| {
                GenerationServiceError::Execution("missing speculative prefill".to_owned())
            })
        }

        fn decode(
            &mut self,
            token_id: u32,
            _: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            self.target_only_inputs.push(token_id);
            self.target_only_steps.pop_front().ok_or_else(|| {
                GenerationServiceError::Execution("missing target-only step".to_owned())
            })
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    impl SpeculativeGenerationExecutorV1 for TinySpeculativeExecutor {
        fn speculative_decode_greedy(
            &mut self,
            pending_token: u32,
        ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
            self.speculative_inputs.push(pending_token);
            self.batches.pop_front().ok_or_else(|| {
                GenerationServiceError::Execution("missing speculative batch".to_owned())
            })
        }

        fn finalize_speculative_decode(
            &mut self,
            committed_input_rows: usize,
        ) -> Result<(), GenerationServiceError> {
            self.finalized_rows.push(committed_input_rows);
            Ok(())
        }
    }

    struct PieceFrontend;

    impl GenerationTextFrontendV1 for PieceFrontend {
        fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
            if text.is_empty() {
                Err(GenerationServiceError::Tokenize)
            } else {
                Ok(vec![1, 3, 17])
            }
        }

        fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
            let mut output = String::new();
            for (index, token) in token_ids.iter().enumerate() {
                output.push_str(match token {
                    5 => "A",
                    6 => "B",
                    7 => "C",
                    10 => "abc",
                    // Models a byte-fallback prefix that becomes valid UTF-8
                    // only after a later token; no replacement bytes leak.
                    11 if token_ids.get(index + 1) == Some(&12) => "",
                    11 => "�",
                    12 => "終わりtail",
                    13 => "ab",
                    14 => "c",
                    15 => "終",
                    16 => "わりtail",
                    _ => "?",
                });
            }
            Ok(output)
        }
    }

    struct GrammarPieceFrontend {
        table: TokenByteTableV1,
    }

    impl GrammarPieceFrontend {
        fn new() -> Self {
            Self {
                table: TokenByteTableV1::from_tokenizer_json(
                    include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json"),
                    16,
                )
                .expect("grammar fixture table"),
            }
        }
    }

    impl GenerationTextFrontendV1 for GrammarPieceFrontend {
        fn encode_generation(&self, _: &str) -> Result<Vec<u32>, GenerationServiceError> {
            Ok(vec![1])
        }

        fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
            Ok(token_ids
                .iter()
                .map(|token| match token {
                    1 => "hello",
                    2 => "world",
                    _ => "?",
                })
                .collect())
        }

        fn token_byte_table(&self) -> Result<&TokenByteTableV1, GenerationServiceError> {
            Ok(&self.table)
        }
    }

    fn policy() -> GenerationStopPolicyV1 {
        GenerationStopPolicyV1 {
            version: 1,
            stop_token_ids: vec![99],
            evaluation: StopEvaluation::NewlyGeneratedAfterArgmax,
            prompt_evaluation: PromptEvaluation::NeverStop,
            stop_token: StopTokenHandling {
                visible_output: false,
                subsequent_decode_input: false,
            },
            budget_boundary: BudgetBoundary::StopTokenWins,
            max_new_tokens_zero: MaxNewTokensZero::MaxNewTokensBeforeDecode,
            reason_version: 1,
        }
    }

    #[test]
    fn qwen_mtp_draft_width_accepts_supported_boundaries_and_maps_to_target_rows() {
        for draft_width in [1, 2, 3, 4, 7, 8] {
            QwenMtpGenerationExecutorV1::validate_draft_width(draft_width)
                .expect("supported draft width");
            assert_eq!(
                QwenMtpGenerationExecutorV1::target_block_rows_for_draft_width(draft_width),
                Some(draft_width + 1)
            );
        }
        assert_eq!(QwenMtpGenerationExecutorV1::MAX_DRAFT_WIDTH, 8);
    }

    #[test]
    fn qwen_mtp_draft_width_rejects_values_above_supported_graph_limit() {
        for draft_width in [9, usize::MAX] {
            let error = QwenMtpGenerationExecutorV1::validate_draft_width(draft_width)
                .expect_err("draft width exceeds supported graph limit");
            assert_eq!(
                QwenMtpGenerationExecutorV1::target_block_rows_for_draft_width(draft_width),
                None
            );
            assert_eq!(
                error,
                GenerationServiceError::Execution(
                    "MTP generation draft width must be in 1..=8".to_owned()
                )
            );
        }
    }

    #[test]
    fn eligible_sampling_uses_one_fixed_selected_only_device_route() {
        let frontend = GrammarPieceFrontend::new();
        let mut stop_policy = policy();
        stop_policy.stop_token_ids = vec![9];
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(
            2,
            SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap(),
            vec![],
        )
        .unwrap()
        .with_device_selector_seed(7);
        let selection = |token_id| SamplingSelectionV1 {
            token_id,
            logprob: -0.25,
            top_logprobs: Vec::new(),
        };
        let mut executor = TinyDeviceSelectorExecutor {
            selections: VecDeque::from([selection(1), selection(9)]),
            requests: Vec::new(),
            prefill_inputs: Vec::new(),
            decode_inputs: Vec::new(),
            cancel_count: 0,
        };
        let prepared = PreparedGenerationInputV1::from_token_ids(vec![1], vec![3, 17]).unwrap();

        let result = service
            .generate_prepared(
                &mut executor,
                &prepared,
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.99),
            )
            .unwrap();

        assert_eq!(result.generated_token_ids(), [1, 9]);
        assert_eq!(result.stop_token_id(), Some(9));
        assert_eq!(executor.prefill_inputs, [vec![1, 3, 17]]);
        assert_eq!(executor.decode_inputs, [1]);
        assert_eq!(executor.requests, [(7, 0, 16, 16), (7, 1, 16, 16)]);
        assert_eq!(executor.cancel_count, 0);
    }

    #[test]
    fn stop_matcher_holds_cross_token_prefix_and_hides_stop() {
        let mut matcher = IncrementalStopMatcher::new(vec!["終わり".to_owned(), "stop".to_owned()]);
        let first = matcher.push("abc終");
        assert_eq!(first.visible, "abc");
        assert_eq!(first.matched, None);
        let second = matcher.push("わりtail");
        assert_eq!(second.visible, "");
        assert_eq!(second.matched.as_deref(), Some("終わり"));
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn stop_matcher_flushes_unmatched_partial_at_length() {
        let mut matcher = IncrementalStopMatcher::new(vec!["stop".to_owned()]);
        assert_eq!(matcher.push("hello st").visible, "hello ");
        assert_eq!(matcher.finish(), "st");
    }

    #[test]
    fn greedy_service_preserves_argmax_sequence_and_reports_usage() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(3, SamplingParametersV1::greedy(), vec![]).unwrap();
        let mut executor = TinyExecutor::argmax([5, 6, 7]);
        let result = service
            .generate_tokens(
                &mut executor,
                &[1, 3, 17],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(f64::NAN),
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [5, 6, 7]);
        assert_eq!(result.decode_input_token_ids(), [5, 6]);
        assert_eq!(result.output_text(), "ABC");
        assert_eq!(result.finish_reason(), FinishReasonV1::Length);
        assert_eq!(result.usage().prompt_tokens(), 3);
        assert_eq!(result.usage().completion_tokens(), 3);
        assert_eq!(result.usage().total_tokens(), 6);
        assert_eq!(executor.include_logits, [false, false, false]);
        assert_eq!(executor.cancel_count, 0);
        assert_eq!(result.selections().len(), 3);
        assert_eq!(result.selections()[0].token_id, 5);
    }

    #[test]
    fn assistant_prefill_is_combined_for_execution_but_hidden_from_output() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let prepared = PreparedGenerationInputV1::from_token_ids(vec![1, 3], vec![17]).unwrap();
        let config =
            GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec!["?".to_owned()])
                .unwrap();
        let mut executor = TinyExecutor::argmax([5]);
        let mut sink = RecordingSink {
            deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            fail_after: None,
        };
        let result = service
            .generate_prepared_with_sink(
                &mut executor,
                &prepared,
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
                &mut sink,
            )
            .unwrap();

        assert_eq!(prepared.token_ids(), [1, 3, 17]);
        assert_eq!(executor.prefill_inputs, [vec![1, 3, 17]]);
        assert_eq!(result.input_token_ids(), [1, 3, 17]);
        assert_eq!(result.generated_token_ids(), [5]);
        assert_eq!(result.output_text(), "A");
        assert_eq!(result.matched_stop(), None);
        assert_eq!(sink.deltas, ["A"]);
        assert_eq!(result.usage().prompt_tokens(), 3);
        assert_eq!(result.usage().completion_tokens(), 1);
    }

    #[test]
    fn assistant_prefill_primes_ascii_and_unicode_stop_prefixes_without_republishing_them() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();

        for (prefill, generated, stop) in [
            (vec![13], 14, "abc"),
            (vec![15], 16, "終わり"),
            // Token 11 is an unstable byte-fallback prefix. Token 12 repairs
            // it into the complete Unicode stop without exposing U+FFFD.
            (vec![11], 12, "終わり"),
        ] {
            let prepared = PreparedGenerationInputV1::from_token_ids(vec![1], prefill).unwrap();
            let config =
                GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![stop.to_owned()])
                    .unwrap();
            let mut executor = TinyExecutor::argmax([generated]);
            let result = service
                .generate_prepared(
                    &mut executor,
                    &prepared,
                    &config,
                    &GenerationCancellationV1::new(),
                    &mut FixedRandom(0.0),
                )
                .unwrap();
            assert_eq!(result.output_text(), "");
            assert_eq!(result.matched_stop(), Some(stop));
            assert_eq!(result.finish_reason(), FinishReasonV1::Stop);
            assert!(!result.output_text().contains('\u{fffd}'));
            assert_eq!(result.usage().prompt_tokens(), 2);
            assert_eq!(result.usage().completion_tokens(), 1);
        }
    }

    #[test]
    fn assistant_prefill_primes_grammar_before_executor_and_rejects_invalid_prefix() {
        let frontend = GrammarPieceFrontend::new();
        let mut stop_policy = policy();
        stop_policy.stop_token_ids = vec![9];
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![])
            .unwrap()
            .with_grammar(CompiledGrammar::compile("root ::= \"hello\"\n").unwrap());

        let valid = PreparedGenerationInputV1::from_token_ids(vec![1], vec![1]).unwrap();
        let mut stop_logits = vec![0.0_f32; 16];
        stop_logits[9] = 20.0;
        let mut executor = TinyExecutor {
            steps: [GenerationStepV1::new(9, Some(stop_logits))].into(),
            ..TinyExecutor::default()
        };
        let mut sink = RecordingSink {
            deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            fail_after: None,
        };
        let result = service
            .generate_prepared_with_sink(
                &mut executor,
                &valid,
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
                &mut sink,
            )
            .unwrap();
        assert_eq!(executor.prefill_inputs, [vec![1, 1]]);
        assert_eq!(result.generated_token_ids(), [9]);
        assert_eq!(result.output_text(), "");
        assert!(sink.deltas.is_empty());
        assert_eq!(result.usage().prompt_tokens(), 2);

        let invalid = PreparedGenerationInputV1::from_token_ids(vec![1], vec![2]).unwrap();
        let mut invalid_executor = TinyExecutor::argmax([9]);
        assert!(matches!(
            service.generate_prepared(
                &mut invalid_executor,
                &invalid,
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            ),
            Err(GenerationServiceError::Grammar(_))
        ));
        assert!(invalid_executor.prefill_inputs.is_empty());
        assert_eq!(invalid_executor.cancel_count, 1);
    }

    #[test]
    fn grammar_masks_tokens_and_retains_selection_metadata() {
        let frontend = GrammarPieceFrontend::new();
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let grammar = CompiledGrammar::compile("root ::= \"hello\"\n").unwrap();
        let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![])
            .unwrap()
            .with_grammar(grammar);
        let mut logits = vec![0.0_f32; 16];
        logits[1] = 1.0;
        logits[2] = 4.0;
        let mut executor = TinyExecutor {
            steps: [GenerationStepV1::new(2, Some(logits))].into(),
            ..TinyExecutor::default()
        };
        let result = service
            .generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [1]);
        assert_eq!(result.output_text(), "hello");
        assert_eq!(result.selections()[0].token_id, 1);
        assert!(executor.include_logits.iter().all(|value| *value));
    }

    #[test]
    fn grammar_terminal_allows_stop_token_only_at_accept_boundary() {
        let frontend = GrammarPieceFrontend::new();
        let mut stop_policy = policy();
        stop_policy.stop_token_ids = vec![9];
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(2, SamplingParametersV1::greedy(), vec![])
            .unwrap()
            .with_grammar(CompiledGrammar::compile("root ::= \"hello\"\n").unwrap());
        let mut first_logits = vec![0.0_f32; 16];
        first_logits[1] = 1.0;
        first_logits[2] = 4.0;
        let mut stop_logits = vec![0.0_f32; 16];
        stop_logits[9] = 20.0;
        let mut executor = TinyExecutor {
            steps: VecDeque::from([
                GenerationStepV1::new(2, Some(first_logits)),
                GenerationStepV1::new(9, Some(stop_logits)),
            ]),
            ..TinyExecutor::default()
        };
        let result = service
            .generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [1, 9]);
        assert_eq!(result.stop_token_id(), Some(9));
        assert_eq!(result.finish_reason(), FinishReasonV1::Stop);
    }

    #[test]
    fn grammar_frontend_without_raw_bytes_fails_closed() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![])
            .unwrap()
            .with_grammar(CompiledGrammar::compile("root ::= \"A\"\n").unwrap());
        let mut executor = TinyExecutor::argmax([5]);
        assert_eq!(
            service.generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            ),
            Err(GenerationServiceError::TokenBytesUnsupported)
        );
        assert_eq!(executor.cancel_count, 1);
    }

    #[test]
    fn ignore_stop_tokens_masks_stop_argmax_in_configured_request() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![])
            .unwrap()
            .with_ignore_stop_tokens(true);
        let mut logits = vec![0.0_f32; 100];
        logits[5] = 10.0;
        logits[99] = 20.0;
        let mut executor = TinyExecutor {
            steps: [GenerationStepV1::new(99, Some(logits))].into(),
            ..TinyExecutor::default()
        };
        let result = service
            .generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [5]);
    }

    #[test]
    fn speculative_adapter_feeds_only_accepted_target_steps_to_the_normal_loop() {
        let inner = TinySpeculativeExecutor {
            prefill_step: Some(GenerationStepV1::new(5, None)),
            target_only_steps: VecDeque::new(),
            batches: VecDeque::from([vec![
                GenerationStepV1::new(6, None),
                GenerationStepV1::new(7, None),
            ]]),
            speculative_inputs: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter = SpeculativeGenerationAdapterV1::new(inner);
        assert_eq!(adapter.prefill(&[1], false).unwrap().device_argmax(), 5);
        assert_eq!(adapter.decode(5, false).unwrap().device_argmax(), 6);
        assert_eq!(adapter.decode(6, false).unwrap().device_argmax(), 7);
        assert_eq!(adapter.inner().speculative_inputs, [5]);
        assert!(adapter.inner().target_only_inputs.is_empty());
        assert_eq!(adapter.inner().finalized_rows, [2]);
    }

    #[test]
    fn speculative_tail_is_finalized_to_only_consumed_rows_at_length_and_stop() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();

        for (stops, expected_reason, expected_text) in [
            (Vec::new(), FinishReasonV1::Length, "AB"),
            (vec!["B".to_owned()], FinishReasonV1::Stop, "A"),
        ] {
            let inner = TinySpeculativeExecutor {
                prefill_step: Some(GenerationStepV1::new(5, None)),
                target_only_steps: VecDeque::new(),
                batches: VecDeque::from([vec![
                    GenerationStepV1::new(6, None),
                    GenerationStepV1::new(7, None),
                    GenerationStepV1::new(5, None),
                ]]),
                speculative_inputs: Vec::new(),
                target_only_inputs: Vec::new(),
                finalized_rows: Vec::new(),
                cancelled: false,
            };
            let mut adapter = SpeculativeGenerationAdapterV1::new(inner);
            let config = GenerationConfigV1::new(2, SamplingParametersV1::greedy(), stops).unwrap();
            let result = service
                .generate_tokens(
                    &mut adapter,
                    &[1],
                    &config,
                    &GenerationCancellationV1::new(),
                    &mut FixedRandom(f64::NAN),
                )
                .unwrap();
            assert_eq!(result.generated_token_ids(), [5, 6]);
            assert_eq!(result.output_text(), expected_text);
            assert_eq!(result.finish_reason(), expected_reason);
            assert_eq!(result.usage().completion_tokens(), 2);
            assert_eq!(adapter.inner().finalized_rows, [1]);
            assert!(adapter.queued.is_empty());
        }
    }

    #[test]
    fn speculative_cancel_finalizes_consumed_rows_before_invalidating_owner() {
        let inner = TinySpeculativeExecutor {
            prefill_step: Some(GenerationStepV1::new(5, None)),
            target_only_steps: VecDeque::new(),
            batches: VecDeque::from([vec![
                GenerationStepV1::new(6, None),
                GenerationStepV1::new(7, None),
            ]]),
            speculative_inputs: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter = SpeculativeGenerationAdapterV1::new(inner);
        adapter.prefill(&[1], false).unwrap();
        adapter.decode(5, false).unwrap();
        adapter.cancel();
        assert_eq!(adapter.inner().finalized_rows, [1]);
        assert!(adapter.inner().cancelled);
        assert!(adapter.queued.is_empty());
    }

    #[test]
    fn speculative_adapter_uses_target_only_path_when_logits_are_requested() {
        let inner = TinySpeculativeExecutor {
            prefill_step: Some(GenerationStepV1::new(5, Some(vec![0.0, 1.0]))),
            target_only_steps: VecDeque::from([GenerationStepV1::new(6, Some(vec![1.0, 0.0]))]),
            batches: VecDeque::new(),
            speculative_inputs: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter = SpeculativeGenerationAdapterV1::new(inner);
        adapter.prefill(&[1], true).unwrap();
        adapter.decode(1, true).unwrap();
        assert!(adapter.inner().speculative_inputs.is_empty());
        assert_eq!(adapter.inner().target_only_inputs, [1]);
    }

    #[test]
    fn model_neutral_ngram_provider_shares_acceptance_and_accounting_loop() {
        let provider = sllm_core::NgramDraftProviderV1::new(1, 1).unwrap();
        let inner = TinyProviderExecutor {
            prefill_step: Some(GenerationStepV1::new(1, None)),
            batches: VecDeque::from([vec![
                GenerationStepV1::new(2, None),
                GenerationStepV1::new(9, None),
            ]]),
            proposals: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter =
            SpeculativeGenerationAdapterV1::with_provider_and_draft_width(inner, provider, 1)
                .unwrap();
        assert_eq!(
            adapter.prefill(&[1, 2, 1], false).unwrap().device_argmax(),
            1
        );
        assert_eq!(adapter.decode(1, false).unwrap().device_argmax(), 2);
        assert_eq!(adapter.decode(2, false).unwrap().device_argmax(), 9);
        assert_eq!(adapter.inner().proposals, [vec![2]]);
        assert_eq!(adapter.accounting().proposal_blocks(), 1);
        assert_eq!(adapter.accounting().proposed_tokens(), 1);
        assert_eq!(adapter.accounting().accepted_tokens(), 1);
        assert_eq!(adapter.accounting().rejected_tokens(), 0);
        assert_eq!(adapter.accounting().emitted_target_tokens(), 2);
        assert!(adapter.inner().target_only_inputs.is_empty());
        assert_eq!(adapter.inner().finalized_rows, [2]);
    }

    #[test]
    fn model_neutral_partial_accept_finalizes_target_replacement_rows_exactly() {
        let provider = sllm_core::NgramDraftProviderV1::new(1, 1).unwrap();
        let inner = TinyProviderExecutor {
            prefill_step: Some(GenerationStepV1::new(1, None)),
            // History [1,2,3,1] proposes [2,3]. The target accepts 2 and
            // replaces 3 with 9, so exactly two input rows may be published.
            batches: VecDeque::from([vec![
                GenerationStepV1::new(2, None),
                GenerationStepV1::new(9, None),
            ]]),
            proposals: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter =
            SpeculativeGenerationAdapterV1::with_provider_and_draft_width(inner, provider, 2)
                .unwrap();
        adapter.prefill(&[1, 2, 3, 1], false).unwrap();
        assert_eq!(adapter.decode(1, false).unwrap().device_argmax(), 2);
        assert_eq!(adapter.decode(2, false).unwrap().device_argmax(), 9);
        assert_eq!(adapter.inner().proposals, [vec![2, 3]]);
        assert_eq!(adapter.inner().finalized_rows, [2]);
        assert_eq!(adapter.accounting().accepted_tokens(), 1);
        assert_eq!(adapter.accounting().rejected_tokens(), 1);
        assert_eq!(adapter.accounting().emitted_target_tokens(), 2);
    }

    #[test]
    fn model_neutral_external_provider_rejection_publishes_only_target_replacement() {
        #[derive(Default)]
        struct DraftModel {
            reset_prefixes: Vec<Vec<u32>>,
        }

        impl sllm_core::ExternalDraftModelV1 for DraftModel {
            fn model_fingerprint(&self) -> &str {
                "draft-model"
            }

            fn tokenizer_fingerprint(&self) -> &str {
                "tok"
            }

            fn vocabulary_size(&self) -> u32 {
                32
            }

            fn reset_to_prefix(
                &mut self,
                committed_target_tokens: &[u32],
            ) -> Result<(), SpeculativeError> {
                self.reset_prefixes.push(committed_target_tokens.to_vec());
                Ok(())
            }

            fn propose_next(&mut self, _: Option<u32>) -> Result<u32, SpeculativeError> {
                Ok(7)
            }
        }

        let compatibility =
            sllm_core::ExternalDraftCompatibilityV1::new("tok", "tok", 32, 32).unwrap();
        let provider =
            sllm_core::ExternalDraftProviderV1::new(DraftModel::default(), compatibility).unwrap();
        let inner = TinyProviderExecutor {
            prefill_step: Some(GenerationStepV1::new(1, None)),
            batches: VecDeque::from([vec![GenerationStepV1::new(8, None)]]),
            proposals: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter =
            SpeculativeGenerationAdapterV1::with_provider_and_draft_width(inner, provider, 1)
                .unwrap();
        adapter.prefill(&[4], false).unwrap();
        assert_eq!(adapter.decode(1, false).unwrap().device_argmax(), 8);
        assert_eq!(adapter.inner().proposals, [vec![7]]);
        assert_eq!(adapter.accounting().accepted_tokens(), 0);
        assert_eq!(adapter.accounting().rejected_tokens(), 1);
        assert_eq!(adapter.accounting().emitted_target_tokens(), 1);
        assert_eq!(adapter.inner().finalized_rows, [1]);
    }

    #[test]
    fn model_neutral_provider_width_is_fail_closed_at_both_boundaries() {
        for width in [0, 9, usize::MAX] {
            let inner = TinyProviderExecutor {
                prefill_step: Some(GenerationStepV1::new(1, None)),
                batches: VecDeque::new(),
                proposals: Vec::new(),
                target_only_inputs: Vec::new(),
                finalized_rows: Vec::new(),
                cancelled: false,
            };
            let provider = sllm_core::NgramDraftProviderV1::new(1, 1).unwrap();
            assert!(
                SpeculativeGenerationAdapterV1::with_provider_and_draft_width(
                    inner, provider, width,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn model_neutral_all_accept_requires_the_target_bonus_row() {
        let provider = sllm_core::NgramDraftProviderV1::new(1, 1).unwrap();
        let inner = TinyProviderExecutor {
            prefill_step: Some(GenerationStepV1::new(1, None)),
            // History [1,2,1] proposes 2. Returning only that accepted row is
            // invalid because an all-accept target block must include bonus.
            batches: VecDeque::from([vec![GenerationStepV1::new(2, None)]]),
            proposals: Vec::new(),
            target_only_inputs: Vec::new(),
            finalized_rows: Vec::new(),
            cancelled: false,
        };
        let mut adapter =
            SpeculativeGenerationAdapterV1::with_provider_and_draft_width(inner, provider, 1)
                .unwrap();
        adapter.prefill(&[1, 2, 1], false).unwrap();
        assert!(matches!(
            adapter.decode(1, false),
            Err(GenerationServiceError::Speculative(_))
        ));
        assert_eq!(adapter.accounting(), SpeculativeAccountingV1::default());
    }

    #[test]
    fn string_stop_crosses_token_and_utf8_fallback_boundaries_without_leaking() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config =
            GenerationConfigV1::new(7, SamplingParametersV1::greedy(), vec!["終わり".to_owned()])
                .unwrap();
        let mut executor = TinyExecutor::argmax([10, 11, 12]);
        let result = service
            .generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [10, 11, 12]);
        assert_eq!(result.output_text(), "abc");
        assert_eq!(result.matched_stop(), Some("終わり"));
        assert_eq!(result.finish_reason(), FinishReasonV1::Stop);
        assert!(!result.output_text().contains("終わり"));
        assert_eq!(result.visible_token_ids(), [10]);
        assert_eq!(executor.decode_inputs, [10, 11]);
    }

    #[test]
    fn incomplete_utf8_replacement_is_flushed_only_at_length() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(2, SamplingParametersV1::greedy(), vec![]).unwrap();
        let mut executor = TinyExecutor::argmax([10, 11]);
        let result = service
            .generate_tokens(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
            )
            .unwrap();
        assert_eq!(result.output_text(), "abc�");
        assert_eq!(result.visible_token_ids(), [10, 11]);
        assert_eq!(result.finish_reason(), FinishReasonV1::Length);
    }

    #[test]
    fn cancellation_and_sampling_error_invalidate_only_request_owner() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), vec![]).unwrap();
        let cancellation = GenerationCancellationV1::new();
        cancellation.cancel();
        let mut cancelled = TinyExecutor::argmax([5]);
        assert_eq!(
            service.generate_tokens(
                &mut cancelled,
                &[1],
                &config,
                &cancellation,
                &mut FixedRandom(0.0)
            ),
            Err(GenerationServiceError::Cancelled),
        );
        assert_eq!(cancelled.cancel_count, 1);
        assert!(cancelled.include_logits.is_empty());

        let sampled = SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap();
        let sampled_config = GenerationConfigV1::new(1, sampled, vec![]).unwrap();
        let mut broken = TinyExecutor {
            steps: [GenerationStepV1::new(0, Some(vec![f32::NAN, 0.0]))].into(),
            ..TinyExecutor::default()
        };
        assert!(matches!(
            service.generate_tokens(
                &mut broken,
                &[1],
                &sampled_config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0)
            ),
            Err(GenerationServiceError::Sampling(SamplingError::NanLogit {
                token_id: 0
            }))
        ));
        assert_eq!(broken.cancel_count, 1);
    }

    struct RecordingSink {
        deltas: Vec<String>,
        reasoning_deltas: Vec<String>,
        fail_after: Option<usize>,
    }

    impl GenerationOutputSinkV1 for RecordingSink {
        fn publish(&mut self, delta: &str) -> Result<(), GenerationServiceError> {
            if self.fail_after == Some(self.deltas.len()) {
                return Err(GenerationServiceError::Output(
                    "consumer disconnected".to_owned(),
                ));
            }
            self.deltas.push(delta.to_owned());
            Ok(())
        }

        fn publish_reasoning(&mut self, delta: &str) -> Result<(), GenerationServiceError> {
            self.reasoning_deltas.push(delta.to_owned());
            Ok(())
        }
    }

    #[test]
    fn output_sink_receives_only_visible_deltas_and_failure_cancels_request() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(3, SamplingParametersV1::greedy(), vec![]).unwrap();
        let mut executor = TinyExecutor::argmax([5, 6, 7]);
        let mut sink = RecordingSink {
            deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            fail_after: None,
        };
        let result = service
            .generate_tokens_with_sink(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
                &mut sink,
            )
            .unwrap();
        assert_eq!(sink.deltas, ["A", "B", "C"]);
        assert_eq!(sink.deltas.concat(), result.output_text());

        let mut executor = TinyExecutor::argmax([5, 6, 7]);
        let mut sink = RecordingSink {
            deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            fail_after: Some(1),
        };
        assert_eq!(
            service.generate_tokens_with_sink(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
                &mut sink,
            ),
            Err(GenerationServiceError::Output(
                "consumer disconnected".to_owned()
            ))
        );
        assert_eq!(sink.deltas, ["A"]);
        assert_eq!(executor.cancel_count, 1);
    }

    #[test]
    fn reasoning_stop_boundary_trims_visible_token_history_after_marker() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let reasoning = ReasoningPolicyV1::enabled(Some(1), [13, 14]).unwrap();
        let config =
            GenerationConfigV1::new(4, SamplingParametersV1::greedy(), vec!["bc".to_owned()])
                .unwrap()
                .with_reasoning(reasoning)
                .unwrap();
        let logits = |token: usize| {
            let mut values = vec![0.0_f32; 32];
            values[token] = 10.0;
            values
        };
        let mut executor = TinyExecutor {
            steps: VecDeque::from([
                GenerationStepV1::new(5, Some(logits(5))),
                GenerationStepV1::new(13, Some(logits(13))),
                GenerationStepV1::new(14, Some(logits(14))),
                GenerationStepV1::new(10, Some(logits(10))),
            ]),
            ..TinyExecutor::default()
        };
        let mut sink = RecordingSink {
            deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            fail_after: None,
        };
        let result = service
            .generate_tokens_with_sink(
                &mut executor,
                &[1],
                &config,
                &GenerationCancellationV1::new(),
                &mut FixedRandom(0.0),
                &mut sink,
            )
            .unwrap();
        assert_eq!(result.generated_token_ids(), [5, 13, 14, 10]);
        assert_eq!(result.reasoning_token_ids(), [5, 13, 14]);
        assert_eq!(result.reasoning_tokens(), 1);
        assert_eq!(result.output_text(), "a");
        assert!(result.visible_token_ids().is_empty());
        assert_eq!(result.matched_stop(), Some("bc"));
        assert_eq!(sink.deltas, ["a"]);
        assert_eq!(sink.reasoning_deltas, ["A", "ab", "c"]);
    }

    #[test]
    fn config_rejects_stop_boundaries_and_duplicates() {
        let sampling = SamplingParametersV1::greedy();
        assert_eq!(
            GenerationConfigV1::new(0, sampling, vec![]),
            Err(GenerationServiceError::InvalidMaxNewTokens)
        );
        assert!(matches!(
            GenerationConfigV1::new(1, sampling, vec![String::new()]),
            Err(GenerationServiceError::EmptyStopString { index: 0 })
        ));
        assert!(matches!(
            GenerationConfigV1::new(1, sampling, vec!["x".into(), "x".into()]),
            Err(GenerationServiceError::DuplicateStopString { index: 1 })
        ));
        assert_eq!(
            GenerationConfigV1::new(
                1,
                sampling,
                vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()]
            ),
            Err(GenerationServiceError::TooManyStopStrings)
        );
    }

    #[test]
    fn choice_seed_zero_preserves_legacy_and_other_choices_are_stable() {
        assert_eq!(derive_choice_seed_v1(None, 7), None);
        assert_eq!(derive_choice_seed_v1(Some(42), 0), Some(42));
        let first = derive_choice_seed_v1(Some(42), 1).unwrap();
        assert_eq!(derive_choice_seed_v1(Some(42), 1), Some(first));
        assert_ne!(first, 42);
        assert_ne!(derive_choice_seed_v1(Some(42), 2), Some(first));
    }

    #[test]
    fn multi_choice_accounting_counts_shared_prompt_once() {
        let frontend = PieceFrontend;
        let stop_policy = policy();
        let service = GenerationServiceV1::new(&frontend, None, &stop_policy).unwrap();
        let config = GenerationConfigV1::new(2, SamplingParametersV1::greedy(), vec![]).unwrap();
        let generate = |tokens| {
            let mut executor = TinyExecutor::argmax(tokens);
            service
                .generate_tokens(
                    &mut executor,
                    &[1, 2, 3],
                    &config,
                    &GenerationCancellationV1::new(),
                    &mut FixedRandom(0.0),
                )
                .unwrap()
        };
        let choices =
            GenerationChoicesResultV1::new(vec![generate([5, 6]), generate([7, 5])]).unwrap();
        assert_eq!(choices.choices().len(), 2);
        assert_eq!(choices.choices()[0].index(), 0);
        assert_eq!(choices.choices()[1].index(), 1);
        assert_eq!(choices.usage().prompt_tokens(), 3);
        assert_eq!(choices.usage().completion_tokens(), 4);
        assert_eq!(choices.usage().total_tokens(), 7);
    }
}
