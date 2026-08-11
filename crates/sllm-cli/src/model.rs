use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sllm_core::{
    Backend, ExecutionSessionRequest, ModelLock, QwenExecutionRequest, VerifiedCache,
    WeightClassification, build_qwen35_graph, build_verified_weight_load_plan, read_model_lock,
};
use sllm_frontend::{
    DecodeModeV1, GenerationReportV1, GenerationStopControllerV1, GenerationStopPolicyV1,
    Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, ThinkingModeV1, TokenIdsV1,
    TokenizerFrontendV1,
};
use sllm_hip::HipBackend;

const REPORT_SCHEMA: &str = "model-frontend-cli-report-v1";
const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOKEN_IDS: usize = 1_048_576;
const MAX_NEW_TOKENS: u32 = 4096;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Eq, PartialEq)]
enum GenerationInput {
    Prompt(String),
    Messages {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
}

#[derive(Debug, Eq, PartialEq)]
struct GenerateRequest {
    input: GenerationInput,
    max_new_tokens: u32,
    device_index: u32,
    target: String,
}

trait GreedyExecution {
    fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String>;
    fn decode_one(&mut self, token_id: i32) -> Result<i32, String>;
}

impl GreedyExecution for QwenExecutionRequest {
    fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String> {
        let output = self
            .prefill(input_token_ids)
            .map_err(|_| "Qwen prefill failed".to_owned())?;
        output
            .token_ids()
            .last()
            .copied()
            .ok_or_else(|| "Qwen prefill published no argmax token".to_owned())
    }

    fn decode_one(&mut self, token_id: i32) -> Result<i32, String> {
        let output = self
            .decode(token_id)
            .map_err(|_| "Qwen decode failed".to_owned())?;
        if output.token_ids().len() != 1 {
            return Err("Qwen decode published a non-singleton argmax".to_owned());
        }
        Ok(output.token_ids()[0])
    }
}

struct GenerationOutcome {
    report: GenerationReportV1,
    decode_steps: u32,
}

fn run_greedy_generation(
    executor: &mut impl GreedyExecution,
    policy: &GenerationStopPolicyV1,
    max_new_tokens: u32,
    input_token_ids: &[u32],
) -> Result<GenerationOutcome, String> {
    let input_i32 = input_token_ids
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|_| "generation input token does not fit I32".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut generated = executor.prefill_last(&input_i32)?;
    let mut controller = GenerationStopControllerV1::new_with_input_token_ids(
        policy,
        max_new_tokens,
        input_token_ids,
    )
    .map_err(|_| "generation stop policy could not be initialized".to_owned())?;
    let mut decode_steps = 0_u32;
    loop {
        let generated_u32 =
            u32::try_from(generated).map_err(|_| "Qwen argmax token was negative".to_owned())?;
        let decision = controller
            .observe_generated(generated_u32)
            .map_err(|_| "generated token violated the stop policy".to_owned())?;
        let Some(decode_input) = decision.decode_input_token_id() else {
            break;
        };
        generated = executor.decode_one(
            i32::try_from(decode_input).map_err(|_| "decode token does not fit I32".to_owned())?,
        )?;
        decode_steps = decode_steps
            .checked_add(1)
            .ok_or_else(|| "decode step count overflowed".to_owned())?;
    }
    Ok(GenerationOutcome {
        report: controller.into_report(),
        decode_steps,
    })
}

#[derive(Debug, Eq, PartialEq)]
enum Operation {
    Verify,
    Tokenize {
        text: String,
    },
    Render {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
    Decode {
        ids: TokenIdsV1,
        mode: DecodeModeV1,
    },
    Generate(GenerateRequest),
}

#[derive(Debug, Eq, PartialEq)]
struct Request {
    lock: PathBuf,
    cache: PathBuf,
    operation: Operation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelIdentity {
    repo_id: String,
    resolved_revision: String,
    lock_fingerprint: String,
}

trait ModelFrontendBackend {
    fn identity(&self) -> ModelIdentity;
    fn verify(&self) -> Result<Value, String>;
    fn tokenize(&self, text: &str) -> Result<Value, String>;
    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String>;
    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String>;
    fn generate(&self, request: &GenerateRequest) -> Result<Value, String>;
}

struct ProductionBackend {
    lock: ModelLock,
    cache: Arc<VerifiedCache>,
}

impl ProductionBackend {
    fn open(request: &Request) -> Result<Self, String> {
        let lock = read_model_lock(&request.lock)
            .map_err(|_| "model lock could not be read or validated".to_owned())?;
        let cache = lock
            .verify_cache(&request.cache)
            .map_err(|_| "model cache does not match the lock".to_owned())?;
        Ok(Self {
            lock,
            cache: Arc::new(cache),
        })
    }
}

impl ModelFrontendBackend for ProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: self.lock.model().repo_id.clone(),
            resolved_revision: self.lock.model().resolved_revision.clone(),
            lock_fingerprint: self.lock.fingerprint().to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        let plan = build_verified_weight_load_plan(&self.lock, &self.cache)
            .map_err(|_| "verified tensors do not form the fixed model load plan".to_owned())?;
        let loadable = plan
            .entries
            .iter()
            .filter(|entry| entry.classification != WeightClassification::KnownUnconsumed)
            .count();
        Ok(json!({
            "kind": "verify-model",
            "locked_files": self.lock.model().files.len(),
            "verified_files": self.cache.files.len(),
            "tensor_count": self.cache.tensors().count(),
            "weight_entries": plan.entries.len(),
            "loadable_entries": loadable,
            "known_unconsumed_entries": plan.entries.len() - loadable,
            "total_destination_bytes": plan.total_destination_bytes,
            "plan_digest": plan.digest_hex(),
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        let tokenizer = TokenizerFrontendV1::from_verified_cache(&self.lock, &self.cache)
            .map_err(|_| "verified tokenizer could not be constructed".to_owned())?;
        let ids = tokenizer
            .encode(text)
            .map_err(|_| "text could not be tokenized".to_owned())?;
        Ok(json!({"kind": "tokenize", "count": ids.len(), "token_ids": ids.as_slice()}))
    }

    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let renderer = Qwen35ChatTemplateV1::from_verified_cache(&self.lock, &self.cache)
            .map_err(|_| "verified chat renderer could not be constructed".to_owned())?;
        let text = renderer
            .render(messages, options)
            .map_err(|_| "chat messages could not be rendered".to_owned())?;
        Ok(json!({"kind": "render", "text": text}))
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = TokenizerFrontendV1::from_verified_cache(&self.lock, &self.cache)
            .map_err(|_| "verified tokenizer could not be constructed".to_owned())?;
        let text = tokenizer
            .decode(ids, mode)
            .map_err(|_| "token IDs could not be decoded".to_owned())?;
        Ok(json!({"kind": "decode", "text": text}))
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        let tokenizer = TokenizerFrontendV1::from_verified_cache(&self.lock, &self.cache)
            .map_err(|_| "verified tokenizer could not be constructed".to_owned())?;
        let (input_kind, rendered) = match &request.input {
            GenerationInput::Prompt(prompt) => ("prompt", prompt.clone()),
            GenerationInput::Messages { messages, options } => {
                let renderer = Qwen35ChatTemplateV1::from_verified_cache(&self.lock, &self.cache)
                    .map_err(|_| {
                    "verified chat renderer could not be constructed".to_owned()
                })?;
                let rendered = renderer
                    .render(messages, *options)
                    .map_err(|_| "chat messages could not be rendered".to_owned())?;
                ("messages", rendered)
            }
        };
        let input = tokenizer
            .encode(&rendered)
            .map_err(|_| "generation input could not be tokenized".to_owned())?;
        if input.is_empty() {
            return Err("generation input produced no token IDs".to_owned());
        }
        let input_len = u64::try_from(input.len())
            .map_err(|_| "generation input token count overflowed".to_owned())?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "generation state capacity overflowed".to_owned())?;
        let plan = build_verified_weight_load_plan(&self.lock, &self.cache)
            .map_err(|_| "verified tensors do not form the fixed model load plan".to_owned())?;
        let graph = build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
            .map_err(|_| "generation graph does not satisfy the fixed Qwen contract".to_owned())?;
        let plan_digest = plan.digest_hex();
        let model_fingerprint = self.lock.fingerprint().to_owned();

        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|_| "exact HIP execution session could not be opened".to_owned())?;

        let execution = (|| -> Result<Value, String> {
            let mut owner = QwenExecutionRequest::new(
                Arc::clone(&session),
                graph,
                plan,
                Arc::clone(&self.cache),
                COMPLETION_TIMEOUT,
            )
            .map_err(|_| "Qwen request provisioning failed".to_owned())?;
            let outcome = run_greedy_generation(
                &mut owner,
                self.lock.generation_stop_policy(),
                request.max_new_tokens,
                input.as_slice(),
            )?;
            let report = outcome.report;
            let visible = TokenIdsV1::from_slice(report.visible_token_ids());
            let text = tokenizer
                .decode(&visible, DecodeModeV1::PreserveSpecialTokens)
                .map_err(|_| "visible generated token IDs could not be decoded".to_owned())?;
            let stop = report
                .stop_reason()
                .ok_or_else(|| "generation ended without a stop reason".to_owned())?;
            Ok(json!({
                "kind": "generate",
                "input_kind": input_kind,
                "input_token_ids": report.input_token_ids(),
                "generated_token_ids": report.generated_token_ids(),
                "visible_token_ids": report.visible_token_ids(),
                "decode_input_token_ids": report.decode_input_token_ids(),
                "output_text": text,
                "stop_reason": {
                    "version": stop.version(),
                    "reason_version": stop.reason_version(),
                    "kind": stop.reason_token(),
                    "token_id": stop.token_id(),
                },
                "execution": {
                    "selected_backend": "hip",
                    "target": request.target,
                    "device_index": request.device_index,
                    "model_fingerprint": model_fingerprint,
                    "plan_digest": plan_digest,
                    "prefill_tokens": input.len(),
                    "decode_steps": outcome.decode_steps,
                    "fallback_used": false,
                },
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        let mut result = execution?;
        let object = result
            .as_object_mut()
            .ok_or_else(|| "generation result was not an object".to_owned())?;
        object.insert(
            "timing_ns".to_owned(),
            Value::from(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)),
        );
        object.insert(
            "cleanup".to_owned(),
            json!({"retryable_cleanup": 0, "durable_quarantine": 0}),
        );
        Ok(result)
    }
}

pub(crate) fn run(
    command: &str,
    arguments: impl Iterator<Item = String>,
) -> Result<String, String> {
    let request = parse(command, arguments)?;
    let backend = ProductionBackend::open(&request)?;
    execute(command, request.operation, &backend)
}

fn execute(
    command: &str,
    operation: Operation,
    backend: &impl ModelFrontendBackend,
) -> Result<String, String> {
    let result = match operation {
        Operation::Verify => backend.verify()?,
        Operation::Tokenize { text } => backend.tokenize(&text)?,
        Operation::Render { messages, options } => backend.render(&messages, options)?,
        Operation::Decode { ids, mode } => backend.decode(&ids, mode)?,
        Operation::Generate(request) => backend.generate(&request)?,
    };
    serialize_report(command, &backend.identity(), result)
}

fn serialize_report(
    command: &str,
    identity: &ModelIdentity,
    result: Value,
) -> Result<String, String> {
    let generation = command == "generate";
    serde_json::to_string(&json!({
        "schema_version": REPORT_SCHEMA,
        "command": command,
        "state": "PASS",
        "model": {
            "repo_id": identity.repo_id,
            "resolved_revision": identity.resolved_revision,
            "lock_fingerprint": identity.lock_fingerprint,
        },
        "scope": {
            "offline": true,
            "gpu_execution": generation,
            "model_execution": generation,
            "generation": generation,
        },
        "result": result,
    }))
    .map_err(|_| "model frontend report could not be serialized".to_owned())
}

fn parse(command: &str, arguments: impl Iterator<Item = String>) -> Result<Request, String> {
    let mut lock = None;
    let mut cache = None;
    let mut text = None;
    let mut token_ids = None;
    let mut messages = Vec::new();
    let mut thinking = None;
    let mut no_generation_prompt = false;
    let mut skip_special_tokens = false;
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut device_index = None;
    let mut target = None;
    let mut greedy = false;
    let mut message_bytes = 0_usize;
    let mut arguments = arguments.peekable();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--lock" => set_once(&mut lock, take_value(&mut arguments, "--lock")?, "--lock")?,
            "--cache" => set_once(
                &mut cache,
                take_value(&mut arguments, "--cache")?,
                "--cache",
            )?,
            "--text" if command == "tokenize" => {
                let value = take_value(&mut arguments, "--text")?;
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--text exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut text, value, "--text")?;
            }
            "--tokens" if command == "decode" => {
                let value = take_value(&mut arguments, "--tokens")?;
                set_once(&mut token_ids, parse_token_ids(&value)?, "--tokens")?;
            }
            "--skip-special-tokens" if command == "decode" => {
                if skip_special_tokens {
                    return Err("duplicate --skip-special-tokens".to_owned());
                }
                skip_special_tokens = true;
            }
            "--prompt" if command == "generate" => {
                let value = take_value(&mut arguments, "--prompt")?;
                if value.is_empty() {
                    return Err("--prompt must not be empty".to_owned());
                }
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--prompt exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut prompt, value, "--prompt")?;
            }
            "--max-new-tokens" if command == "generate" => {
                let value = take_value(&mut arguments, "--max-new-tokens")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--max-new-tokens must be an unsigned decimal U32".to_owned());
                }
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| "--max-new-tokens must be an unsigned decimal U32".to_owned())?;
                if parsed == 0 || parsed > MAX_NEW_TOKENS {
                    return Err(format!("--max-new-tokens must be in [1,{MAX_NEW_TOKENS}]"));
                }
                set_once(&mut max_new_tokens, parsed, "--max-new-tokens")?;
            }
            "--device-index" if command == "generate" => {
                let value = take_value(&mut arguments, "--device-index")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--device-index must be an unsigned decimal U32".to_owned());
                }
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| "--device-index must be an unsigned decimal U32".to_owned())?;
                set_once(&mut device_index, parsed, "--device-index")?;
            }
            "--target" if command == "generate" => {
                let value = take_value(&mut arguments, "--target")?;
                if value != "gfx1030" && value != "gfx1201" {
                    return Err("--target must be gfx1030 or gfx1201".to_owned());
                }
                set_once(&mut target, value, "--target")?;
            }
            "--greedy" if command == "generate" => {
                if greedy {
                    return Err("duplicate --greedy".to_owned());
                }
                greedy = true;
            }
            "--message" if command == "render" || command == "generate" => {
                let value = take_value(&mut arguments, "--message")?;
                message_bytes = message_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "--message input size overflow".to_owned())?;
                if message_bytes > MAX_TEXT_BYTES || messages.len() == 4096 {
                    return Err("render message input exceeds the bounded CLI limit".to_owned());
                }
                messages.push(parse_message(&value)?);
            }
            "--thinking" if command == "render" || command == "generate" => {
                let value = match take_value(&mut arguments, "--thinking")?.as_str() {
                    "default" => ThinkingModeV1::TemplateDefault,
                    "enabled" => ThinkingModeV1::Enabled,
                    "disabled" => ThinkingModeV1::Disabled,
                    _ => return Err("--thinking must be default, enabled, or disabled".to_owned()),
                };
                set_once(&mut thinking, value, "--thinking")?;
            }
            "--no-generation-prompt" if command == "render" => {
                if no_generation_prompt {
                    return Err("duplicate --no-generation-prompt".to_owned());
                }
                no_generation_prompt = true;
            }
            value => return Err(format!("unexpected argument `{value}` for `{command}`")),
        }
    }

    let lock = PathBuf::from(lock.ok_or_else(|| "missing required --lock PATH".to_owned())?);
    let cache = PathBuf::from(cache.ok_or_else(|| "missing required --cache PATH".to_owned())?);
    let operation = match command {
        "verify-model" => Operation::Verify,
        "tokenize" => Operation::Tokenize {
            text: text.ok_or_else(|| "missing required --text TEXT".to_owned())?,
        },
        "render" => {
            if messages.is_empty() {
                return Err("render requires at least one --message ROLE:CONTENT".to_owned());
            }
            Operation::Render {
                messages,
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: !no_generation_prompt,
                    thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                },
            }
        }
        "decode" => Operation::Decode {
            ids: token_ids.ok_or_else(|| "missing required --tokens IDS".to_owned())?,
            mode: if skip_special_tokens {
                DecodeModeV1::SkipSpecialTokens
            } else {
                DecodeModeV1::PreserveSpecialTokens
            },
        },
        "generate" => {
            if !greedy {
                return Err("generate requires explicit --greedy mode".to_owned());
            }
            let options = Qwen35RenderOptionsV1 {
                add_generation_prompt: true,
                thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
            };
            let input = match (prompt, messages.is_empty()) {
                (Some(prompt), true) => {
                    if thinking.is_some() {
                        return Err("--thinking is valid only with generate --message".to_owned());
                    }
                    GenerationInput::Prompt(prompt)
                }
                (None, false) => GenerationInput::Messages { messages, options },
                (Some(_), false) => {
                    return Err(
                        "generate accepts either --prompt or --message, not both".to_owned()
                    );
                }
                (None, true) => {
                    return Err("generate requires --prompt or at least one --message".to_owned());
                }
            };
            Operation::Generate(GenerateRequest {
                input,
                max_new_tokens: max_new_tokens
                    .ok_or_else(|| "generate requires --max-new-tokens".to_owned())?,
                device_index: device_index
                    .ok_or_else(|| "generate requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "generate requires --target".to_owned())?,
            })
        }
        _ => return Err("internal unsupported model command".to_owned()),
    };
    Ok(Request {
        lock,
        cache,
        operation,
    })
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("duplicate {flag}"));
    }
    Ok(())
}

fn take_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_token_ids(value: &str) -> Result<TokenIdsV1, String> {
    if value.is_empty() {
        return Err("--tokens must not be empty".to_owned());
    }
    let mut ids = Vec::new();
    for item in value.split(',') {
        if ids.len() == MAX_TOKEN_IDS {
            return Err("--tokens exceeds the 1048576-ID input limit".to_owned());
        }
        if item.is_empty() || item.bytes().any(|byte| !byte.is_ascii_digit()) {
            return Err("--tokens must be comma-separated unsigned decimal IDs".to_owned());
        }
        ids.push(
            item.parse::<u32>()
                .map_err(|_| "--tokens contains an ID outside u32".to_owned())?,
        );
    }
    Ok(TokenIdsV1::from_slice(&ids))
}

fn parse_message(value: &str) -> Result<Qwen35ChatMessageV1, String> {
    let (role, content) = value
        .split_once(':')
        .ok_or_else(|| "--message must use ROLE:CONTENT".to_owned())?;
    if content.len() > MAX_TEXT_BYTES {
        return Err("--message content exceeds the 16 MiB input limit".to_owned());
    }
    match role {
        "system" => Ok(Qwen35ChatMessageV1::system(content)),
        "user" => Ok(Qwen35ChatMessageV1::user(content)),
        "assistant" => Ok(Qwen35ChatMessageV1::assistant(content, None)),
        _ => Err("message role must be system, user, or assistant".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct TinyBackend;

    struct SequenceExecution {
        outputs: VecDeque<i32>,
        prefill_inputs: Vec<Vec<i32>>,
        decode_inputs: Vec<i32>,
    }

    impl SequenceExecution {
        fn new(outputs: impl IntoIterator<Item = i32>) -> Self {
            Self {
                outputs: outputs.into_iter().collect(),
                prefill_inputs: Vec::new(),
                decode_inputs: Vec::new(),
            }
        }

        fn next(&mut self) -> Result<i32, String> {
            self.outputs
                .pop_front()
                .ok_or_else(|| "fake sequence exhausted".to_owned())
        }
    }

    impl GreedyExecution for SequenceExecution {
        fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String> {
            self.prefill_inputs.push(input_token_ids.to_vec());
            self.next()
        }

        fn decode_one(&mut self, token_id: i32) -> Result<i32, String> {
            self.decode_inputs.push(token_id);
            self.next()
        }
    }

    fn qwen_stop_policy() -> GenerationStopPolicyV1 {
        sllm_core::parse_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .unwrap()
        .generation_stop_policy()
        .clone()
    }

    impl ModelFrontendBackend for TinyBackend {
        fn identity(&self) -> ModelIdentity {
            ModelIdentity {
                repo_id: "Qwen/Qwen3.5-4B".to_owned(),
                resolved_revision: "8".repeat(40),
                lock_fingerprint: format!("sha256:{}", "3".repeat(64)),
            }
        }

        fn verify(&self) -> Result<Value, String> {
            Ok(json!({
                "kind": "verify-model", "locked_files": 3, "verified_files": 3,
                "tensor_count": 17, "weight_entries": 17, "loadable_entries": 3,
                "known_unconsumed_entries": 14, "total_destination_bytes": 17,
                "plan_digest": format!("sha256:{}", "9".repeat(64)),
            }))
        }

        fn tokenize(&self, text: &str) -> Result<Value, String> {
            assert_eq!(text, "abc");
            Ok(json!({"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}))
        }

        fn render(
            &self,
            messages: &[Qwen35ChatMessageV1],
            options: Qwen35RenderOptionsV1,
        ) -> Result<Value, String> {
            assert_eq!(messages.len(), 1);
            assert_eq!(options.thinking, ThinkingModeV1::Disabled);
            Ok(json!({"kind": "render", "text": "rendered"}))
        }

        fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
            assert_eq!(ids.as_slice(), &[1, 3, 17]);
            assert_eq!(mode, DecodeModeV1::SkipSpecialTokens);
            Ok(json!({"kind": "decode", "text": "decoded"}))
        }

        fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
            assert_eq!(request.max_new_tokens, 3);
            assert_eq!(request.device_index, 0);
            assert_eq!(request.target, "gfx1030");
            assert!(matches!(request.input, GenerationInput::Prompt(ref text) if text == "abc"));
            Ok(json!({
                "kind": "generate",
                "input_token_ids": [1, 3, 17],
                "generated_token_ids": [7, 8, 9],
                "visible_token_ids": [7, 8, 9],
                "decode_input_token_ids": [7, 8],
                "output_text": "generated",
                "stop_reason": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": null},
                "execution": {"selected_backend": "hip", "target": "gfx1030", "device_index": 0, "fallback_used": false},
                "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
            }))
        }
    }

    fn parse_args(command: &str, args: &[&str]) -> Result<Request, String> {
        parse(command, args.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn all_model_entrances_parse_without_touching_hip() {
        let common = ["--lock", "lock.json", "--cache", "cache"];
        assert_eq!(
            parse_args("verify-model", &common).unwrap().operation,
            Operation::Verify
        );
        assert!(matches!(
            parse_args(
                "tokenize",
                &["--lock", "lock.json", "--cache", "cache", "--text", "abc"]
            )
            .unwrap()
            .operation,
            Operation::Tokenize { .. }
        ));
        assert!(matches!(
            parse_args(
                "render",
                &[
                    "--message",
                    "user:a:b",
                    "--thinking",
                    "disabled",
                    "--lock",
                    "lock.json",
                    "--cache",
                    "cache"
                ]
            )
            .unwrap()
            .operation,
            Operation::Render { .. }
        ));
        assert!(matches!(
            parse_args(
                "decode",
                &[
                    "--tokens",
                    "1,3,17",
                    "--skip-special-tokens",
                    "--cache",
                    "cache",
                    "--lock",
                    "lock.json"
                ]
            )
            .unwrap()
            .operation,
            Operation::Decode { .. }
        ));
        assert!(matches!(
            parse_args(
                "generate",
                &[
                    "--lock",
                    "lock.json",
                    "--cache",
                    "cache",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "3",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--greedy"
                ]
            )
            .unwrap()
            .operation,
            Operation::Generate(GenerateRequest {
                input: GenerationInput::Prompt(_),
                max_new_tokens: 3,
                device_index: 0,
                target: _
            })
        ));
        let unicode = parse_args(
            "generate",
            &[
                "--lock",
                "lock.json",
                "--cache",
                "cache",
                "--message",
                "user:雪とGPU",
                "--thinking",
                "disabled",
                "--max-new-tokens",
                "17",
                "--device-index",
                "0",
                "--target",
                "gfx1201",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            unicode.operation,
            Operation::Generate(GenerateRequest {
                input: GenerationInput::Messages { .. },
                max_new_tokens: 17,
                target,
                ..
            }) if target == "gfx1201"
        ));
    }

    #[test]
    fn greedy_controller_excludes_stop_tokens_and_stops_exactly_at_budget() {
        let policy = qwen_stop_policy();

        let mut first_stop = SequenceExecution::new([248046]);
        let outcome = run_greedy_generation(&mut first_stop, &policy, 3, &[1, 3, 17]).unwrap();
        assert_eq!(outcome.report.generated_token_ids(), &[248046]);
        assert!(outcome.report.visible_token_ids().is_empty());
        assert!(outcome.report.decode_input_token_ids().is_empty());
        assert_eq!(outcome.report.stop_token_id(), Some(248046));
        assert_eq!(outcome.decode_steps, 0);

        let mut second_stop = SequenceExecution::new([7, 248044]);
        let outcome = run_greedy_generation(&mut second_stop, &policy, 3, &[1, 3, 17]).unwrap();
        assert_eq!(outcome.report.generated_token_ids(), &[7, 248044]);
        assert_eq!(outcome.report.visible_token_ids(), &[7]);
        assert_eq!(outcome.report.decode_input_token_ids(), &[7]);
        assert_eq!(second_stop.decode_inputs, [7]);
        assert_eq!(outcome.report.stop_token_id(), Some(248044));

        for budget in [1_u32, 3, 17, 255, 256, 257] {
            let mut executor = SequenceExecution::new(std::iter::repeat_n(7, budget as usize));
            let outcome =
                run_greedy_generation(&mut executor, &policy, budget, &[1, 3, 17]).unwrap();
            assert_eq!(outcome.report.generated_token_ids().len(), budget as usize);
            assert_eq!(outcome.report.visible_token_ids().len(), budget as usize);
            assert_eq!(
                outcome.report.decode_input_token_ids().len(),
                budget.saturating_sub(1) as usize
            );
            assert_eq!(outcome.report.reason_token(), Some("max_new_tokens"));
            assert_eq!(outcome.decode_steps, budget.saturating_sub(1));
        }
    }

    #[test]
    fn greedy_controller_rejects_negative_or_exhausted_executor_output() {
        let policy = qwen_stop_policy();
        let mut negative = SequenceExecution::new([-1]);
        assert!(run_greedy_generation(&mut negative, &policy, 3, &[1]).is_err());

        let mut exhausted = SequenceExecution::new([7]);
        assert!(run_greedy_generation(&mut exhausted, &policy, 3, &[1]).is_err());
    }

    #[test]
    fn tiny_backend_executes_all_success_entrances() {
        let cases = [
            ("verify-model", vec!["--lock", "x", "--cache", "y"]),
            (
                "tokenize",
                vec!["--lock", "x", "--cache", "y", "--text", "abc"],
            ),
            (
                "render",
                vec![
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--message",
                    "user:abc",
                    "--thinking",
                    "disabled",
                ],
            ),
            (
                "decode",
                vec![
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--tokens",
                    "1,3,17",
                    "--skip-special-tokens",
                ],
            ),
            (
                "generate",
                vec![
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "3",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--greedy",
                ],
            ),
        ];
        for (command, args) in cases {
            let request = parse_args(command, &args).unwrap();
            let output = execute(command, request.operation, &TinyBackend).unwrap();
            let document: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(document["command"], command);
            assert_eq!(document["result"]["kind"], command);
            assert_eq!(document["state"], "PASS");
        }
    }

    #[test]
    fn malformed_and_cross_command_arguments_fail_closed() {
        assert!(parse_args("tokenize", &["--lock", "x", "--cache", "y"]).is_err());
        assert!(
            parse_args(
                "decode",
                &["--lock", "x", "--cache", "y", "--tokens", "1,,2"]
            )
            .is_err()
        );
        for arguments in [
            vec!["--lock", "x", "--cache", "y", "--prompt", "x"],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--message",
                "user:y",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--max-new-tokens",
                "0",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx9999",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--thinking",
                "disabled",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "",
                "--max-new-tokens",
                "+3",
                "--device-index",
                "+0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        ] {
            assert!(parse_args("generate", &arguments).is_err());
        }
        assert!(
            parse_args(
                "render",
                &["--lock", "x", "--cache", "y", "--message", "tool:x"]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "verify-model",
                &["--lock", "x", "--lock", "z", "--cache", "y"]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "decode",
                &[
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--tokens",
                    "1",
                    "--skip-special-tokens",
                    "--skip-special-tokens"
                ]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "render",
                &[
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--message",
                    "user:x",
                    "--thinking",
                    "enabled",
                    "--thinking",
                    "disabled"
                ]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "verify-model",
                &["--lock", "x", "--cache", "y", "--text", "x"]
            )
            .is_err()
        );
    }

    #[test]
    fn token_boundaries_include_non_aligned_values() {
        assert_eq!(parse_token_ids("1,3,17").unwrap().as_slice(), &[1, 3, 17]);
        assert!(parse_token_ids("").is_err());
        assert!(parse_token_ids("4294967296").is_err());
        assert!(parse_token_ids("+1").is_err());
    }

    #[test]
    fn serialized_success_uses_the_versioned_closed_envelope() {
        let lock = sllm_core::parse_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .unwrap();
        let identity = ModelIdentity {
            repo_id: lock.model().repo_id.clone(),
            resolved_revision: lock.model().resolved_revision.clone(),
            lock_fingerprint: lock.fingerprint().to_owned(),
        };
        for (command, result) in [
            (
                "tokenize",
                json!({"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}),
            ),
            ("render", json!({"kind": "render", "text": "prompt"})),
            ("decode", json!({"kind": "decode", "text": "text"})),
            (
                "generate",
                json!({"kind": "generate", "generated_token_ids": [1]}),
            ),
        ] {
            let output = serialize_report(command, &identity, result).unwrap();
            assert!(!output.contains('\n'));
            let document: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(document.as_object().unwrap().len(), 6);
            assert_eq!(document["schema_version"], REPORT_SCHEMA);
            assert_eq!(document["command"], command);
            assert_eq!(document["state"], "PASS");
            assert_eq!(document["result"]["kind"], command);
            assert_eq!(document["scope"]["gpu_execution"], command == "generate");
        }
    }
}
