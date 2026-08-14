//! Transport-independent render/tokenize/prefill/decode/sampling service.

use core::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sllm_core::{
    ProfileSamplerV1, QwenExecutionRequest, SamplingError, SamplingParametersV1,
    SamplingRandomSource,
};

use crate::{
    DecodeModeV1, GenerationStopPolicyV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1,
    Qwen35RenderOptionsV1, TokenIdsV1, TokenizerFrontendV1, validate_generation_stop_policy,
};

pub const MAX_STOP_STRINGS_V1: usize = 4;
pub const MAX_STOP_STRING_BYTES_V1: usize = 1_048_576;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationInputV1 {
    Prompt(String),
    Messages {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationConfigV1 {
    max_new_tokens: u32,
    sampling: SamplingParametersV1,
    stop_strings: Vec<String>,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
}

impl GenerationStepV1 {
    pub fn new(device_argmax: u32, last_logits: Option<Vec<f32>>) -> Self {
        Self {
            device_argmax,
            last_logits,
        }
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

    fn cancel(&mut self);
}

/// Minimal text frontend seam used by the transport-independent service.
/// Production uses the verified tokenizer; tests can model byte-fallback
/// boundaries without loading a model asset.
pub trait GenerationTextFrontendV1 {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError>;
    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError>;
}

/// Receives text that is safe to expose to a transport. Implementations must
/// apply their own bounded backpressure and return promptly after cancellation.
pub trait GenerationOutputSinkV1 {
    fn publish(&mut self, delta: &str) -> Result<(), GenerationServiceError>;
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

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        self.decode(
            &TokenIdsV1::from_slice(token_ids),
            DecodeModeV1::PreserveSpecialTokens,
        )
        .map_err(|_| GenerationServiceError::Decode)
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

    fn cancel(&mut self) {
        QwenExecutionRequest::cancel(self);
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
    Decode,
    NonPrefixDecode,
    TokenIdOverflow,
    CountOverflow,
    MissingDeviceArgmax,
    InvalidStopPolicy,
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
            Self::InvalidStopPolicy => {
                formatter.write_str("generation stop-token policy is invalid")
            }
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

pub struct GenerationServiceV1<'a> {
    tokenizer: &'a dyn GenerationTextFrontendV1,
    renderer: Option<&'a Qwen35ChatTemplateV1>,
    stop_policy: &'a GenerationStopPolicyV1,
}

impl<'a> GenerationServiceV1<'a> {
    pub fn new(
        tokenizer: &'a dyn GenerationTextFrontendV1,
        renderer: Option<&'a Qwen35ChatTemplateV1>,
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
        let prompt = self.prepare_input(input)?;
        self.generate_tokens_with_sink(executor, &prompt, config, cancellation, random, sink)
    }

    /// Runs the same renderer/tokenizer path as [`Self::generate`] while
    /// allowing a model owner to size its request graph before execution.
    pub fn prepare_input(
        &self,
        input: &GenerationInputV1,
    ) -> Result<Vec<u32>, GenerationServiceError> {
        let rendered = match input {
            GenerationInputV1::Prompt(prompt) => prompt.clone(),
            GenerationInputV1::Messages { messages, options } => self
                .renderer
                .ok_or(GenerationServiceError::MissingRenderer)?
                .render(messages, *options)
                .map_err(|_| GenerationServiceError::Render)?,
        };
        let prompt = self.tokenizer.encode_generation(&rendered)?;
        if prompt.is_empty() {
            return Err(GenerationServiceError::EmptyPromptTokens);
        }
        Ok(prompt)
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

    fn generate_tokens_inner(
        &self,
        executor: &mut impl GenerationExecutorV1,
        input_token_ids: &[u32],
        config: &GenerationConfigV1,
        cancellation: &GenerationCancellationV1,
        random: &mut impl SamplingRandomSource,
        sink: &mut impl GenerationOutputSinkV1,
    ) -> Result<GenerationResultV1, GenerationServiceError> {
        check_cancelled(cancellation)?;
        let include_logits = config.sampling.requires_logits();
        let mut step = executor.prefill(input_token_ids, include_logits)?;
        let mut sampler = ProfileSamplerV1::new(config.sampling, input_token_ids)?;
        let mut matcher = IncrementalStopMatcher::new(config.stop_strings.clone());
        let mut generated = Vec::new();
        let mut normal_tokens = Vec::<u32>::new();
        let mut decoded_snapshots = Vec::<String>::new();
        let mut decode_inputs = Vec::new();
        let mut decoded = String::new();
        let mut unstable_utf8_tail = String::new();
        let mut output_text = String::new();
        let mut finish_reason = None;
        let mut stop_token_id = None;
        let mut matched_stop = None;
        let mut decode_steps = 0_u32;

        for index in 0..config.max_new_tokens {
            check_cancelled(cancellation)?;
            let token = sampler.select(step.device_argmax, step.last_logits.as_deref(), random)?;
            sampler.accept(token)?;
            generated.push(token);

            if self.stop_policy.stop_token_ids.contains(&token) {
                finish_reason = Some(FinishReasonV1::Stop);
                stop_token_id = Some(token);
                if !unstable_utf8_tail.is_empty() {
                    let tail = matcher.push(&unstable_utf8_tail);
                    publish_visible(&mut output_text, &tail.visible, sink)?;
                }
                publish_visible(&mut output_text, &matcher.finish(), sink)?;
                break;
            }

            let candidate_ids = normal_tokens
                .iter()
                .copied()
                .chain(std::iter::once(token))
                .collect::<Vec<_>>();
            let candidate = self.tokenizer.decode_generation(&candidate_ids)?;
            // Hugging Face byte-fallback decoding can temporarily end in one
            // or more replacement characters and repair that suffix after a
            // later token completes the UTF-8 sequence. Never publish or feed
            // that unstable suffix to the stop matcher early.
            let stable_end = candidate.trim_end_matches('\u{fffd}').len();
            let stable_candidate = &candidate[..stable_end];
            let delta = stable_candidate
                .strip_prefix(&decoded)
                .ok_or(GenerationServiceError::NonPrefixDecode)?;
            let match_result = matcher.push(delta);
            publish_visible(&mut output_text, &match_result.visible, sink)?;
            decoded = stable_candidate.to_owned();
            unstable_utf8_tail = candidate[stable_end..].to_owned();
            normal_tokens.push(token);
            decoded_snapshots.push(candidate);
            if let Some(stop) = match_result.matched {
                finish_reason = Some(FinishReasonV1::Stop);
                matched_stop = Some(stop);
                break;
            }

            if index + 1 == config.max_new_tokens {
                if !unstable_utf8_tail.is_empty() {
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
            step = executor.decode(token, include_logits)?;
            decode_steps = decode_steps
                .checked_add(1)
                .ok_or(GenerationServiceError::CountOverflow)?;
        }

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
        let visible_token_ids = normal_tokens[..visible_count].to_vec();
        let prompt_tokens = u64::try_from(input_token_ids.len())
            .map_err(|_| GenerationServiceError::CountOverflow)?;
        let completion_tokens =
            u64::try_from(generated.len()).map_err(|_| GenerationServiceError::CountOverflow)?;
        let total_tokens = prompt_tokens
            .checked_add(completion_tokens)
            .ok_or(GenerationServiceError::CountOverflow)?;
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
            },
            decode_steps,
        })
    }
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
            _: &[u32],
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
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
                    _ => "?",
                });
            }
            Ok(output)
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
}
