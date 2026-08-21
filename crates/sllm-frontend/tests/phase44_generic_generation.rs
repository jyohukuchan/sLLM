use std::cell::Cell;

use serde_json::json;
use sha2::{Digest, Sha256};
use sllm_core::{
    BudgetBoundary, MaxNewTokensZero, PromptEvaluation, SamplingParametersV1, StopEvaluation,
};
use sllm_frontend::{
    GenerationCancellationV1, GenerationConfigV1, GenerationExecutorV1, GenerationInputV1,
    GenerationServiceError, GenerationServiceV1, GenerationStepV1, GenerationStopPolicyV1,
    GenerationTextFrontendV1, GenericGenerationInputV1, GenericTemplateErrorV1,
    GenericTemplateInputKindV1, GenericTemplateInputV1, GenericTemplateMessagesInputV1,
    GenericTemplateProviderV1, StopTokenHandling, TokenizerUtilityErrorV1,
};

fn digest(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn provider(source: &str) -> GenericTemplateProviderV1 {
    GenericTemplateProviderV1::new(source, &digest(source)).expect("valid generic provider")
}

fn messages() -> GenericTemplateMessagesInputV1 {
    GenericTemplateMessagesInputV1::new(vec![json!({
        "role": "user",
        "content": "hello"
    })])
    .expect("valid message context")
}

fn generic_input(source: &str) -> GenericGenerationInputV1 {
    GenericGenerationInputV1::new(
        provider(source),
        GenericTemplateInputV1::messages(messages()),
    )
    .expect("valid generic generation input")
}

fn stop_policy() -> GenerationStopPolicyV1 {
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

struct OracleTokenizer {
    calls: Cell<usize>,
    fail: bool,
}

impl GenerationTextFrontendV1 for OracleTokenizer {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail {
            return Err(GenerationServiceError::Tokenize);
        }
        assert!(text == "hello" || text == "legacy");
        Ok(vec![1, 2])
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        Ok(token_ids
            .iter()
            .map(|token| match token {
                7 => "ok",
                _ => "?",
            })
            .collect())
    }
}

struct CountingExecutor {
    prefill_calls: usize,
    decode_calls: usize,
    cancel_calls: usize,
}

struct FixedRandom;

impl sllm_core::SamplingRandomSource for FixedRandom {
    fn next_unit_f64(&mut self) -> Result<f64, sllm_core::SamplingError> {
        Ok(0.0)
    }
}

impl CountingExecutor {
    fn new() -> Self {
        Self {
            prefill_calls: 0,
            decode_calls: 0,
            cancel_calls: 0,
        }
    }
}

impl GenerationExecutorV1 for CountingExecutor {
    fn prefill(&mut self, _: &[u32], _: bool) -> Result<GenerationStepV1, GenerationServiceError> {
        self.prefill_calls += 1;
        Ok(GenerationStepV1::new(7, None))
    }

    fn decode(&mut self, _: u32, _: bool) -> Result<GenerationStepV1, GenerationServiceError> {
        self.decode_calls += 1;
        Ok(GenerationStepV1::new(7, None))
    }

    fn cancel(&mut self) {
        self.cancel_calls += 1;
    }
}

#[test]
fn generic_render_tokenize_prepare_and_generate_share_one_loop_and_identity() {
    let generic = generic_input("{{ messages[0].content }}");
    assert_eq!(generic.rendered_bytes(), b"hello");
    assert_eq!(
        generic.identity().source_size_bytes(),
        "{{ messages[0].content }}".len()
    );
    assert!(!generic.identity().template_digest().is_empty());
    assert!(!generic.identity().kwargs_digest().is_empty());
    assert!(!generic.identity().rendered_digest().is_empty());

    let tokenizer = OracleTokenizer {
        calls: Cell::new(0),
        fail: false,
    };
    let policy = stop_policy();
    let service = GenerationServiceV1::new(&tokenizer, None, &policy).unwrap();
    let input = GenerationInputV1::GenericTemplate(Box::new(generic.clone()));
    let prepared = service.prepare_input_plan(&input).unwrap();
    assert_eq!(prepared.token_ids(), [1, 2]);
    assert!(prepared.assistant_prefill_token_ids().is_empty());
    assert_eq!(
        prepared.generic_template_identity(),
        Some(generic.identity())
    );

    let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), Vec::new()).unwrap();
    let mut executor = CountingExecutor::new();
    let result = service
        .generate(
            &mut executor,
            &input,
            &config,
            &GenerationCancellationV1::new(),
            &mut FixedRandom,
        )
        .unwrap();
    assert_eq!(result.output_text(), "ok");
    assert_eq!(executor.prefill_calls, 1);
    assert_eq!(executor.decode_calls, 0);
    assert_eq!(tokenizer.calls.get(), 2);
}

#[test]
fn legacy_prepared_inputs_keep_identity_none() {
    let tokenizer = OracleTokenizer {
        calls: Cell::new(0),
        fail: false,
    };
    let policy = stop_policy();
    let service = GenerationServiceV1::new(&tokenizer, None, &policy).unwrap();
    let prepared = service
        .prepare_input_plan(&GenerationInputV1::Prompt("legacy".to_owned()))
        .unwrap();
    assert_eq!(prepared.generic_template_identity(), None);
    assert_eq!(prepared.token_ids(), [1, 2]);
}

#[test]
fn unsupported_raw_and_gemma_generic_inputs_fail_before_executor() {
    for (input, kind) in [
        (
            GenericTemplateInputV1::raw_text("raw"),
            GenericTemplateInputKindV1::RawText,
        ),
        (
            GenericTemplateInputV1::gemma_raw_text("gemma"),
            GenericTemplateInputKindV1::GemmaRawText,
        ),
    ] {
        let error = GenericGenerationInputV1::new(provider("fixed"), input)
            .expect_err("unsupported generic input must fail closed");
        assert!(matches!(
            error,
            GenerationServiceError::GenericTemplate(
                TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput { kind: actual }
            ) if actual == kind
        ));
    }
}

#[test]
fn render_and_tokenizer_failures_do_not_call_executor() {
    let render_error = GenericGenerationInputV1::new(
        provider("{{ missing_value }}"),
        GenericTemplateInputV1::messages(messages()),
    )
    .expect_err("undefined template variable must fail closed");
    assert!(matches!(
        render_error,
        GenerationServiceError::GenericTemplate(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::UndefinedValue
        ))
    ));

    let empty_error = GenericGenerationInputV1::new(
        provider("{% if false %}x{% endif %}"),
        GenericTemplateInputV1::messages(messages()),
    )
    .expect_err("empty rendered output must fail closed");
    assert!(matches!(
        empty_error,
        GenerationServiceError::GenericTemplate(TokenizerUtilityErrorV1::InvalidTemplateResult)
    ));

    let tokenizer = OracleTokenizer {
        calls: Cell::new(0),
        fail: true,
    };
    let policy = stop_policy();
    let service = GenerationServiceV1::new(&tokenizer, None, &policy).unwrap();
    let input = GenerationInputV1::GenericTemplate(Box::new(generic_input("fixed")));
    let config = GenerationConfigV1::new(1, SamplingParametersV1::greedy(), Vec::new()).unwrap();
    let mut executor = CountingExecutor::new();
    let error = service
        .generate(
            &mut executor,
            &input,
            &config,
            &GenerationCancellationV1::new(),
            &mut FixedRandom,
        )
        .expect_err("tokenizer failure must reject before executor");
    assert_eq!(error, GenerationServiceError::Tokenize);
    assert_eq!(executor.prefill_calls, 0);
    assert_eq!(executor.decode_calls, 0);
    assert_eq!(executor.cancel_calls, 0);
}
