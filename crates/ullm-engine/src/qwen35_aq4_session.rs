// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Load-once Qwen3.5 AQ4 inference session.
//!
//! The session owns one resident model and only resets request-owned state between requests.
//! Preparing a token is deliberately separate from publishing and committing it, so cancellation
//! and publisher failures cannot make an unobserved token part of the decode history.

use crate::inference_api::{
    CancellationToken, FinishReason, InferenceRequest, ReleaseOutcome, ReleaseSummary,
};
use crate::qwen35_aq4_model_runtime::{Qwen35Aq4ModelLoadConfig, Qwen35Aq4ModelRuntime};
use crate::worker_driver::{InferenceSession, PublishedAdvance, SessionAdvance};

pub const QWEN35_AQ4_ROTARY_DIM: usize = 64;
pub const QWEN35_AQ4_ROPE_BASE: f32 = 10_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Aq4SessionStatus {
    Ready,
    Prefilling,
    PreparedToken,
    Decoding,
    Terminal,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Aq4SessionConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub sync_each_layer_for_timing: bool,
}

impl Qwen35Aq4SessionConfig {
    pub fn greedy(max_new_tokens: usize, eos_token_ids: Vec<usize>) -> Self {
        Self {
            max_new_tokens,
            eos_token_ids,
            rotary_dim: QWEN35_AQ4_ROTARY_DIM,
            rope_base: QWEN35_AQ4_ROPE_BASE,
            sync_each_layer_for_timing: false,
        }
    }
}

/// Minimal model boundary used by the serving state machine and CPU-only contract tests.
pub trait Qwen35Aq4SessionModel {
    fn context_length(&self) -> usize;

    fn vocab_size(&self) -> usize;

    fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        position: usize,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<(), String>;

    fn top_token_from_last_layer(&mut self, label: &str) -> Result<usize, String>;

    fn reset_all_request_state_synchronized(&mut self) -> Result<(), String>;

    fn shutdown_synchronized(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Qwen35Aq4SessionModel for Qwen35Aq4ModelRuntime {
    fn context_length(&self) -> usize {
        self.geometry().context_length
    }

    fn vocab_size(&self) -> usize {
        self.geometry().vocab
    }

    fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        position: usize,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<(), String> {
        Qwen35Aq4ModelRuntime::dispatch_token(
            self,
            token_id,
            rotary_dim,
            rope_base,
            position,
            position,
            sync_each_layer_for_timing,
            label,
        )?;
        Ok(())
    }

    fn top_token_from_last_layer(&mut self, label: &str) -> Result<usize, String> {
        let logits = self.top_logits_from_last_layer(1, label)?;
        let top = logits
            .first()
            .ok_or_else(|| "Qwen3.5 AQ4 top-1 returned no token".to_string())?;
        if !top.logit.is_finite() {
            return Err("Qwen3.5 AQ4 top-1 returned a non-finite logit".to_string());
        }
        Ok(top.token_id)
    }

    fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
        Qwen35Aq4ModelRuntime::reset_all_request_state_synchronized(self)
    }

    fn shutdown_synchronized(&mut self) -> Result<(), String> {
        self.synchronize()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4PreparedToken {
    pub token_id: usize,
    pub generated_index: usize,
    pub cache_len: usize,
    pub terminal_reason: Option<FinishReason>,
    nonce: u64,
}

#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    cancel: CancellationToken,
    prompt_tokens_processed: usize,
    generated_tokens: usize,
    decode_input: Option<usize>,
    terminal_outcome: Option<ReleaseOutcome>,
}

pub struct Qwen35Aq4InferenceSession<M = Qwen35Aq4ModelRuntime> {
    model: M,
    config: Qwen35Aq4SessionConfig,
    status: Qwen35Aq4SessionStatus,
    active: Option<ActiveRequest>,
    pending: Option<Qwen35Aq4PreparedToken>,
    next_nonce: u64,
}

impl Qwen35Aq4InferenceSession<Qwen35Aq4ModelRuntime> {
    /// Loads the resident model exactly once and transfers it into the reusable session.
    pub fn load(
        model_config: Qwen35Aq4ModelLoadConfig,
        session_config: Qwen35Aq4SessionConfig,
    ) -> Result<Self, String> {
        let model = Qwen35Aq4ModelRuntime::load(model_config)?;
        if !model.has_resident_embedding() || !model.supports_device_logits() {
            return Err(
                "Qwen3.5 AQ4 session requires resident embedding, final norm, and LM head"
                    .to_string(),
            );
        }
        Self::from_model(model, session_config)
    }
}

impl<M: Qwen35Aq4SessionModel> Qwen35Aq4InferenceSession<M> {
    pub fn from_model(model: M, config: Qwen35Aq4SessionConfig) -> Result<Self, String> {
        validate_config(&model, &config)?;
        Ok(Self {
            model,
            config,
            status: Qwen35Aq4SessionStatus::Ready,
            active: None,
            pending: None,
            next_nonce: 0,
        })
    }

    pub fn status(&self) -> Qwen35Aq4SessionStatus {
        self.status
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    fn fail<T>(&mut self, message: impl Into<String>) -> Result<T, String> {
        self.status = Qwen35Aq4SessionStatus::Failed;
        Err(message.into())
    }

    fn prepare_token(
        &mut self,
        label: &str,
    ) -> Result<SessionAdvance<Qwen35Aq4PreparedToken>, String> {
        let token_id = match self.model.top_token_from_last_layer(label) {
            Ok(token_id) => token_id,
            Err(error) => return self.fail(format!("{label} top-1 failed: {error}")),
        };
        if token_id >= self.model.vocab_size() {
            return self.fail(format!(
                "{label} top-1 token {token_id} exceeds vocabulary size {}",
                self.model.vocab_size()
            ));
        }
        let (generated_index, cache_len, next_generated, max_new_tokens) =
            match self.active.as_ref() {
                Some(active) => {
                    let generated_index = active.generated_tokens;
                    let Some(cache_len) =
                        active.prompt_token_ids.len().checked_add(generated_index)
                    else {
                        return self.fail("Qwen3.5 AQ4 prepared cache length overflows");
                    };
                    let Some(next_generated) = generated_index.checked_add(1) else {
                        return self.fail("Qwen3.5 AQ4 generated token count overflows");
                    };
                    (
                        generated_index,
                        cache_len,
                        next_generated,
                        active.max_new_tokens,
                    )
                }
                None => {
                    return self.fail("Qwen3.5 AQ4 token preparation has no active request");
                }
            };
        let terminal_reason = if self.config.eos_token_ids.contains(&token_id) {
            Some(FinishReason::Stop)
        } else if next_generated == max_new_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        let Some(next_nonce) = self.next_nonce.checked_add(1) else {
            return self.fail("Qwen3.5 AQ4 prepared token nonce overflows");
        };
        let prepared = Qwen35Aq4PreparedToken {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            nonce: self.next_nonce,
        };
        self.next_nonce = next_nonce;
        self.pending = Some(prepared.clone());
        self.status = Qwen35Aq4SessionStatus::PreparedToken;
        Ok(SessionAdvance::Token {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            prepared,
        })
    }

    fn reset_with_outcome(&mut self, outcome: ReleaseOutcome) -> Result<ReleaseSummary, String> {
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 reset has no active request".to_string())?;
        let summary = ReleaseSummary {
            request_id: active.request_id.clone(),
            outcome,
            prompt_tokens: active.prompt_token_ids.len(),
            generated_tokens: active.generated_tokens,
            reset_complete: true,
        };
        if let Err(error) = self.model.reset_all_request_state_synchronized() {
            self.status = Qwen35Aq4SessionStatus::Failed;
            return Err(format!("Qwen3.5 AQ4 request reset failed: {error}"));
        }
        self.active = None;
        self.pending = None;
        self.status = Qwen35Aq4SessionStatus::Ready;
        Ok(summary)
    }
}

impl<M: Qwen35Aq4SessionModel> InferenceSession for Qwen35Aq4InferenceSession<M> {
    type Prepared = Qwen35Aq4PreparedToken;

    fn start_request(
        &mut self,
        request: InferenceRequest,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        if self.status != Qwen35Aq4SessionStatus::Ready {
            return Err(format!(
                "Qwen3.5 AQ4 start requires Ready, got {:?}",
                self.status
            ));
        }
        request
            .validate_for_worker(
                self.model.context_length(),
                self.config.max_new_tokens,
                self.model.vocab_size(),
                &self.config.eos_token_ids,
                1,
            )
            .map_err(|error| error.to_string())?;
        if request.sampling.temperature != 0.0 || request.sampling.top_p != 1.0 {
            return Err(
                "Qwen3.5 AQ4 session supports greedy sampling only (temperature=0, top_p=1, top_k=1)"
                    .to_string(),
            );
        }
        self.active = Some(ActiveRequest {
            request_id: request.request_id,
            prompt_token_ids: request.prompt_token_ids,
            max_new_tokens: request.max_new_tokens,
            cancel,
            prompt_tokens_processed: 0,
            generated_tokens: 0,
            decode_input: None,
            terminal_outcome: None,
        });
        self.pending = None;
        self.status = Qwen35Aq4SessionStatus::Prefilling;
        Ok(())
    }

    fn prepare_advance(&mut self) -> Result<SessionAdvance<Self::Prepared>, String> {
        if !matches!(
            self.status,
            Qwen35Aq4SessionStatus::Prefilling | Qwen35Aq4SessionStatus::Decoding
        ) {
            return Err(format!(
                "Qwen3.5 AQ4 prepare requires Prefilling or Decoding, got {:?}",
                self.status
            ));
        }
        let cancelled = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 prepare has no active request".to_string())?
            .cancel
            .is_cancelled();
        if cancelled {
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Ok(SessionAdvance::CancellationObserved);
        }

        if self.status == Qwen35Aq4SessionStatus::Prefilling {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 prepare has no active request".to_string())?;
            if active.prompt_tokens_processed == active.prompt_token_ids.len() {
                return self.prepare_token("Qwen3.5 AQ4 prefill");
            }
        }

        let (token_id, position, prompt_progress) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 prepare has no active request".to_string())?;
            match self.status {
                Qwen35Aq4SessionStatus::Prefilling => {
                    let position = active.prompt_tokens_processed;
                    (active.prompt_token_ids[position], position, true)
                }
                Qwen35Aq4SessionStatus::Decoding => {
                    let token_id = active.decode_input.ok_or_else(|| {
                        "Qwen3.5 AQ4 decode has no committed input token".to_string()
                    })?;
                    let position = active
                        .prompt_token_ids
                        .len()
                        .checked_add(active.generated_tokens)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or_else(|| "Qwen3.5 AQ4 decode position overflows".to_string())?;
                    (token_id, position, false)
                }
                _ => unreachable!("status checked above"),
            }
        };
        let label = if prompt_progress {
            "Qwen3.5 AQ4 prefill"
        } else {
            "Qwen3.5 AQ4 decode"
        };
        if let Err(error) = self.model.dispatch_token(
            token_id,
            self.config.rotary_dim,
            self.config.rope_base,
            position,
            self.config.sync_each_layer_for_timing,
            label,
        ) {
            return self.fail(format!("{label} token dispatch failed: {error}"));
        }
        if prompt_progress {
            let active = self.active.as_mut().expect("active request checked above");
            active.prompt_tokens_processed += 1;
            return Ok(SessionAdvance::PromptProgress {
                prompt_tokens_processed: active.prompt_tokens_processed,
                cache_len: active.prompt_tokens_processed,
                execution_width: 1,
            });
        }
        self.prepare_token(label)
    }

    fn publish_prepared<F>(
        &mut self,
        prepared: Self::Prepared,
        publish: F,
    ) -> Result<PublishedAdvance, String>
    where
        F: FnOnce(usize) -> Result<(), String>,
    {
        if self.status != Qwen35Aq4SessionStatus::PreparedToken {
            return Err(format!(
                "Qwen3.5 AQ4 publish requires PreparedToken, got {:?}",
                self.status
            ));
        }
        if self.pending.as_ref() != Some(&prepared) {
            return Err("Qwen3.5 AQ4 publication handle does not match pending token".to_string());
        }
        let cancel = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 publication has no active request".to_string())?
            .cancel
            .clone();
        let next_generated = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 publication has no active request".to_string())?;
            if active.generated_tokens != prepared.generated_index {
                return Err(
                    "Qwen3.5 AQ4 prepared token index does not match active request".to_string(),
                );
            }
            active
                .generated_tokens
                .checked_add(1)
                .ok_or_else(|| "Qwen3.5 AQ4 generated token count overflows".to_string())?
        };
        let publication = cancel.publication_guard()?;
        if cancel.is_cancelled() {
            drop(publication);
            self.pending = None;
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Ok(PublishedAdvance::CancellationObserved);
        }
        if let Err(error) = publish(prepared.token_id) {
            drop(publication);
            self.pending = None;
            // A publisher failure does not poison resident model state. The caller must abort it.
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Err(format!(
                "Qwen3.5 AQ4 token publisher failed before commit: {error}"
            ));
        }
        // The callback cannot mutate the session, so every fallible commit precondition was
        // checked before publication. No error may be introduced after the public side effect.
        let active = self
            .active
            .as_mut()
            .expect("active request validated before publication");
        active.generated_tokens = next_generated;
        active.decode_input = Some(prepared.token_id);
        if let Some(reason) = prepared.terminal_reason {
            active.terminal_outcome = Some(match reason {
                FinishReason::Stop => ReleaseOutcome::Stop,
                FinishReason::Length => ReleaseOutcome::Length,
            });
            self.status = Qwen35Aq4SessionStatus::Terminal;
        } else {
            self.status = Qwen35Aq4SessionStatus::Decoding;
        }
        self.pending = None;
        drop(publication);
        Ok(PublishedAdvance::Token {
            token_id: prepared.token_id,
            generated_index: prepared.generated_index,
            cache_len: prepared.cache_len,
            terminal_reason: prepared.terminal_reason,
        })
    }

    fn finish_and_reset(&mut self) -> Result<ReleaseSummary, String> {
        if self.status != Qwen35Aq4SessionStatus::Terminal {
            return Err(format!(
                "Qwen3.5 AQ4 finish requires Terminal, got {:?}",
                self.status
            ));
        }
        let outcome = self
            .active
            .as_ref()
            .and_then(|active| active.terminal_outcome)
            .filter(|outcome| matches!(outcome, ReleaseOutcome::Stop | ReleaseOutcome::Length))
            .ok_or_else(|| "Qwen3.5 AQ4 finish has no completed outcome".to_string())?;
        self.reset_with_outcome(outcome)
    }

    fn abort_and_reset(&mut self) -> Result<ReleaseSummary, String> {
        if matches!(
            self.status,
            Qwen35Aq4SessionStatus::Ready | Qwen35Aq4SessionStatus::Failed
        ) {
            return Err(format!(
                "Qwen3.5 AQ4 abort requires an active reusable request, got {:?}",
                self.status
            ));
        }
        self.reset_with_outcome(ReleaseOutcome::Cancelled)
    }

    fn shutdown(&mut self) -> Result<(), String> {
        if let Err(error) = self.model.shutdown_synchronized() {
            self.status = Qwen35Aq4SessionStatus::Failed;
            return Err(format!("Qwen3.5 AQ4 session shutdown sync failed: {error}"));
        }
        Ok(())
    }
}

fn validate_config<M: Qwen35Aq4SessionModel>(
    model: &M,
    config: &Qwen35Aq4SessionConfig,
) -> Result<(), String> {
    if model.context_length() == 0 || model.vocab_size() == 0 {
        return Err("Qwen3.5 AQ4 session model geometry must be nonzero".to_string());
    }
    if config.max_new_tokens == 0 || config.max_new_tokens > model.context_length() {
        return Err(format!(
            "Qwen3.5 AQ4 session max_new_tokens must be in 1..={}, got {}",
            model.context_length(),
            config.max_new_tokens
        ));
    }
    if config.eos_token_ids.is_empty() {
        return Err("Qwen3.5 AQ4 session requires at least one EOS token".to_string());
    }
    let mut eos = config.eos_token_ids.clone();
    eos.sort_unstable();
    eos.dedup();
    if eos.len() != config.eos_token_ids.len()
        || eos.iter().any(|token_id| *token_id >= model.vocab_size())
    {
        return Err(
            "Qwen3.5 AQ4 session EOS tokens must be unique and inside the vocabulary".to_string(),
        );
    }
    if config.rotary_dim == 0 || config.rotary_dim % 2 != 0 {
        return Err("Qwen3.5 AQ4 session rotary_dim must be a positive even number".to_string());
    }
    if !config.rope_base.is_finite() || config.rope_base <= 0.0 {
        return Err("Qwen3.5 AQ4 session rope_base must be positive and finite".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_api::SamplingParams;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct ScriptedModel {
        context: usize,
        vocab: usize,
        logits: VecDeque<Result<usize, String>>,
        dispatches: Vec<(usize, usize, usize, u32)>,
        resets: usize,
        fail_reset: bool,
        shutdowns: usize,
        fail_shutdown: bool,
    }

    impl Qwen35Aq4SessionModel for ScriptedModel {
        fn context_length(&self) -> usize {
            self.context
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn dispatch_token(
            &mut self,
            token_id: usize,
            rotary_dim: usize,
            rope_base: f32,
            position: usize,
            _: bool,
            _: &str,
        ) -> Result<(), String> {
            self.dispatches
                .push((token_id, position, rotary_dim, rope_base.to_bits()));
            Ok(())
        }

        fn top_token_from_last_layer(&mut self, _: &str) -> Result<usize, String> {
            self.logits
                .pop_front()
                .unwrap_or_else(|| Err("script exhausted".to_string()))
        }

        fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
            self.resets += 1;
            if self.fail_reset {
                Err("scripted reset failure".to_string())
            } else {
                Ok(())
            }
        }

        fn shutdown_synchronized(&mut self) -> Result<(), String> {
            self.shutdowns += 1;
            if self.fail_shutdown {
                Err("scripted shutdown failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    fn model(tokens: &[usize]) -> ScriptedModel {
        ScriptedModel {
            context: 16,
            vocab: 32,
            logits: tokens.iter().copied().map(Ok).collect(),
            ..ScriptedModel::default()
        }
    }

    fn session(tokens: &[usize]) -> Qwen35Aq4InferenceSession<ScriptedModel> {
        Qwen35Aq4InferenceSession::from_model(
            model(tokens),
            Qwen35Aq4SessionConfig::greedy(8, vec![2]),
        )
        .unwrap()
    }

    fn request(id: &str, prompt: &[usize], max_new_tokens: usize) -> InferenceRequest {
        InferenceRequest::new_with_eos(
            id,
            prompt.to_vec(),
            max_new_tokens,
            vec![2],
            SamplingParams::greedy_with_top_k(0, 1),
        )
    }

    fn prepared(advance: SessionAdvance<Qwen35Aq4PreparedToken>) -> Qwen35Aq4PreparedToken {
        match advance {
            SessionAdvance::Token { prepared, .. } => prepared,
            other => panic!("expected token, got {other:?}"),
        }
    }

    fn next_prepared(
        session: &mut Qwen35Aq4InferenceSession<ScriptedModel>,
    ) -> Qwen35Aq4PreparedToken {
        loop {
            match session.prepare_advance().unwrap() {
                SessionAdvance::PromptProgress { .. } => {}
                advance => return prepared(advance),
            }
        }
    }

    #[test]
    fn prompt_progresses_one_token_at_a_time_and_uses_explicit_rope_config() {
        let mut session = session(&[9]);
        session
            .start_request(request("r1", &[4, 5, 6], 2), CancellationToken::new())
            .unwrap();
        for expected in 1..=3 {
            assert_eq!(
                session.prepare_advance().unwrap(),
                SessionAdvance::PromptProgress {
                    prompt_tokens_processed: expected,
                    cache_len: expected,
                    execution_width: 1,
                }
            );
        }
        let token = next_prepared(&mut session);
        assert_eq!(token.token_id, 9);
        assert_eq!(
            session.model().dispatches,
            vec![
                (4, 0, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
                (5, 1, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
                (6, 2, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
            ]
        );
    }

    #[test]
    fn eos_commits_stop_and_reset_summary() {
        let mut session = session(&[2]);
        session
            .start_request(request("eos", &[4], 4), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        assert_eq!(token.terminal_reason, Some(FinishReason::Stop));
        let published = session.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(matches!(published, PublishedAdvance::Token { .. }));
        assert_eq!(
            session.finish_and_reset().unwrap(),
            ReleaseSummary {
                request_id: "eos".to_string(),
                outcome: ReleaseOutcome::Stop,
                prompt_tokens: 1,
                generated_tokens: 1,
                reset_complete: true,
            }
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);
    }

    #[test]
    fn max_new_tokens_commits_length() {
        let mut session = session(&[7, 8]);
        session
            .start_request(request("length", &[4], 2), CancellationToken::new())
            .unwrap();
        let first = next_prepared(&mut session);
        session.publish_prepared(first, |_| Ok(())).unwrap();
        let second = next_prepared(&mut session);
        assert_eq!(second.terminal_reason, Some(FinishReason::Length));
        session.publish_prepared(second, |_| Ok(())).unwrap();
        assert_eq!(
            session.finish_and_reset().unwrap().outcome,
            ReleaseOutcome::Length
        );
        assert_eq!(session.model().dispatches[1].0, 7);
    }

    #[test]
    fn same_resident_model_serves_two_requests_with_reset_between_them() {
        let mut session = session(&[2, 2]);
        for id in ["one", "two"] {
            session
                .start_request(request(id, &[4], 1), CancellationToken::new())
                .unwrap();
            let token = next_prepared(&mut session);
            session.publish_prepared(token, |_| Ok(())).unwrap();
            assert_eq!(session.finish_and_reset().unwrap().request_id, id);
        }
        assert_eq!(session.model().resets, 2);
        assert_eq!(session.model().dispatches.len(), 2);
    }

    #[test]
    fn cancellation_after_prepare_skips_callback_and_does_not_commit() {
        let mut session = session(&[7]);
        let cancel = CancellationToken::new();
        session
            .start_request(request("cancel", &[4], 2), cancel.clone())
            .unwrap();
        let token = next_prepared(&mut session);
        cancel.cancel();
        let mut called = false;
        assert_eq!(
            session
                .publish_prepared(token, |_| {
                    called = true;
                    Ok(())
                })
                .unwrap(),
            PublishedAdvance::CancellationObserved
        );
        assert!(!called);
        assert_eq!(session.abort_and_reset().unwrap().generated_tokens, 0);
    }

    #[test]
    fn publisher_failure_does_not_commit_and_can_be_aborted() {
        let mut session = session(&[7]);
        session
            .start_request(request("publish", &[4], 2), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        let error = session
            .publish_prepared(token, |_| Err("closed".to_string()))
            .unwrap_err();
        assert!(error.contains("before commit"));
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Terminal);
        assert_eq!(session.abort_and_reset().unwrap().generated_tokens, 0);
    }

    #[test]
    fn prepared_handle_cannot_be_committed_twice() {
        let mut session = session(&[7, 2]);
        session
            .start_request(request("twice", &[4], 2), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        session.publish_prepared(token.clone(), |_| Ok(())).unwrap();
        let error = session.publish_prepared(token, |_| Ok(())).unwrap_err();
        assert!(error.contains("requires PreparedToken"));
    }

    #[test]
    fn reset_failure_poisons_session_and_rejects_reuse() {
        let mut session = session(&[2]);
        session.model.fail_reset = true;
        session
            .start_request(request("poison", &[4], 1), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        session.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(
            session
                .finish_and_reset()
                .unwrap_err()
                .contains("reset failed")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
        assert!(
            session
                .start_request(request("reuse", &[4], 1), CancellationToken::new())
                .unwrap_err()
                .contains("got Failed")
        );
    }

    #[test]
    fn start_rejects_non_greedy_and_context_overflow() {
        let mut session = session(&[2]);
        let mut non_greedy = request("sample", &[4], 1);
        non_greedy.sampling.temperature = 0.5;
        assert!(
            session
                .start_request(non_greedy, CancellationToken::new())
                .is_err()
        );
        assert!(
            session
                .start_request(request("large", &[4; 15], 2), CancellationToken::new())
                .unwrap_err()
                .contains("exceeds context")
        );
    }

    #[test]
    fn start_rejects_prompt_vocabulary_and_eos_contract_mismatches() {
        let mut session = session(&[2]);
        assert!(
            session
                .start_request(request("vocab", &[32], 1), CancellationToken::new())
                .unwrap_err()
                .contains("exceeds vocabulary")
        );
        let wrong_eos = InferenceRequest::new_with_eos(
            "eos-contract",
            vec![4],
            1,
            vec![3],
            SamplingParams::greedy_with_top_k(0, 1),
        );
        assert!(
            session
                .start_request(wrong_eos, CancellationToken::new())
                .unwrap_err()
                .contains("eos_token_ids must be")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);
    }

    #[test]
    fn prepare_counter_failure_poisons_session_before_proposal() {
        let mut session = session(&[7]);
        session.next_nonce = u64::MAX;
        session
            .start_request(request("overflow", &[4], 2), CancellationToken::new())
            .unwrap();
        assert!(matches!(
            session.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: 1,
                ..
            }
        ));
        assert!(
            session
                .prepare_advance()
                .unwrap_err()
                .contains("nonce overflows")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
    }

    #[test]
    fn graceful_shutdown_synchronizes_model_and_reports_failure() {
        let mut session = session(&[]);
        session.shutdown().unwrap();
        assert_eq!(session.model().shutdowns, 1);
        session.model.fail_shutdown = true;
        assert!(
            session
                .shutdown()
                .unwrap_err()
                .contains("shutdown sync failed")
        );
        assert_eq!(session.model().shutdowns, 2);
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
    }
}
