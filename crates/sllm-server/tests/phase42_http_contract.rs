use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::Router;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sllm_core::{fingerprint_for_json, parse_model_lock, verify_model_cache};
use sllm_frontend::{
    ApplyTemplateResultV1, DecodeModeV1, GenerationCancellationV1, Qwen35ChatMessageV1,
    Qwen35RenderOptionsV1, TemplateIdentityV1, TokenIdsV1, TokenizeOptionsV1, TokenizeResultV1,
    TokenizerFrontendV1, TokenizerUtilityServiceV1,
};
use sllm_server::{
    BackendCompletionV1, BackendEmbeddingBatchV1, BackendEmbeddingInputV1,
    BackendEmbeddingRequestV1, BackendEmbeddingVectorV1, BackendErrorV1, BackendInfillCapabilityV1,
    BackendTokenLogprobV1, BackendTopLogprobV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    FinishReasonV1, GenerationDeltaSinkV1, ModelRegistryEntryV1, ModelRegistryV1,
    SchedulerConfigV1, SchedulerV1, ServerConfigV1, TokenUsageV1, build_router_v1,
};

const FINGERPRINT: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Phase42Backend {
    tokenizer: TokenizerFrontendV1,
    _directory: TestDirectory,
}

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn copy_cache(directory: &Path) {
    let base = repository_path("ci/fixtures/model-lock-v1/cache");
    for entry in fs::read_dir(base).expect("read fixture cache") {
        let entry = entry.expect("read fixture cache entry");
        fs::copy(entry.path(), directory.join(entry.file_name())).expect("copy fixture cache file");
    }
}

fn set_file_metadata(lock: &mut Value, path: &str, bytes: &[u8]) {
    let file = lock["model"]["files"]
        .as_array_mut()
        .expect("lock files array")
        .iter_mut()
        .find(|file| file["path"] == path)
        .expect("fixture file entry");
    file["size_bytes"] = json!(bytes.len());
    file["sha256"] = json!(format!("{:x}", Sha256::digest(bytes)));
}

fn phase42_backend() -> Phase42Backend {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = TestDirectory(std::env::temp_dir().join(format!(
        "sllm-phase42-http-{}-{sequence}",
        std::process::id()
    )));
    fs::create_dir(&directory.0).expect("create fixture directory");
    copy_cache(&directory.0);
    let tokenizer = include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json");
    let config = br#"{
  "architectures": ["FixtureModel"],
  "model_type": "fixture",
  "eos_token_id": 8,
  "text_config": {"model_type": "fixture_text", "eos_token_id": 8}
}
"#;
    let tokenizer_config = br#"{
  "eos_token": "<|im_end|>",
  "added_tokens_decoder": {
    "8": {"content": "<|endoftext|>", "special": true},
    "9": {"content": "<|im_end|>", "special": true}
  }
}
"#;
    fs::write(directory.0.join("tokenizer.json"), tokenizer).unwrap();
    fs::write(directory.0.join("config.json"), config).unwrap();
    fs::write(directory.0.join("tokenizer_config.json"), tokenizer_config).unwrap();
    let mut lock: Value = serde_json::from_slice(
        &fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json")).unwrap(),
    )
    .unwrap();
    set_file_metadata(&mut lock, "tokenizer.json", tokenizer);
    set_file_metadata(&mut lock, "config.json", config);
    set_file_metadata(&mut lock, "tokenizer_config.json", tokenizer_config);
    let contract = &mut lock["model"]["tokenizer_contract"];
    contract["files"] = json!([
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json"
    ]);
    contract["vocab_size"] = json!(12);
    contract["eos_token_id"] = json!(8);
    contract["special_token_ids"] = json!({"bos": 10, "eos": 8});
    contract["stop_identity"] = json!({
        "config_eos": {"token": "<|endoftext|>", "token_id": 8, "source_file": "config.json"},
        "tokenizer_eos": {"token": "<|im_end|>", "token_id": 9, "source_files": ["tokenizer_config.json", "tokenizer.json"]}
    });
    contract["generation_stop_policy"]["stop_token_ids"] = json!([9, 8]);
    let fingerprint = fingerprint_for_json(&serde_json::to_vec(&lock).unwrap()).unwrap();
    lock["fingerprint"] = json!(fingerprint);
    let lock = parse_model_lock(&serde_json::to_vec(&lock).unwrap()).unwrap();
    let cache = verify_model_cache(&lock, &directory.0).unwrap();
    let tokenizer = TokenizerFrontendV1::from_verified_cache(&lock, &cache).unwrap();
    Phase42Backend {
        tokenizer,
        _directory: directory,
    }
}

impl ChatGenerationBackendV1 for Phase42Backend {
    fn reviewed_chat_template_available(&self) -> bool {
        true
    }

    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        if cancellation.is_cancelled() {
            return Err(BackendErrorV1::new("cancelled"));
        }
        sink.publish("ok")?;
        if request.logprobs().is_some() {
            sink.publish_logprobs(vec![BackendTokenLogprobV1 {
                token: "ok".to_owned(),
                bytes: Some(b"ok".to_vec()),
                logprob: -0.25,
                top_logprobs: vec![BackendTopLogprobV1 {
                    token: "ok".to_owned(),
                    bytes: Some(b"ok".to_vec()),
                    logprob: -0.25,
                }],
            }])?;
        }
        let prompt_tokens = if let Some((tokens, digest)) = request.prepared_infill() {
            let expected = sllm_frontend::FimTemplateV1::new(8, 9, 10).unwrap();
            if digest != expected.digest()
                || tokens.first() != Some(&8)
                || tokens.last() != Some(&10)
                || !tokens.contains(&9)
            {
                return Err(BackendErrorV1::new(
                    "synthetic FIM input was not marker/digest bound",
                ));
            }
            u64::try_from(tokens.len()).unwrap()
        } else {
            2
        };
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(prompt_tokens, 1).unwrap(),
            matched_stop: None,
        })
    }

    fn embed(
        &self,
        request: &BackendEmbeddingRequestV1,
        _: &GenerationCancellationV1,
    ) -> Result<BackendEmbeddingBatchV1, BackendErrorV1> {
        let vectors = request
            .inputs()
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let values = if index == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                BackendEmbeddingVectorV1::new(values, self.validate_embedding_input(input).unwrap())
                    .unwrap()
            })
            .collect();
        BackendEmbeddingBatchV1::new(2, vectors)
    }

    fn embedding_dimension(&self) -> Option<u32> {
        Some(2)
    }

    fn validate_embedding_input(
        &self,
        input: &BackendEmbeddingInputV1,
    ) -> Result<u64, BackendErrorV1> {
        let count = match input {
            BackendEmbeddingInputV1::Text(text) => self
                .tokenizer
                .encode(text)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?
                .len(),
            BackendEmbeddingInputV1::TokenIds(tokens) => {
                if tokens
                    .iter()
                    .any(|id| u64::from(*id) >= self.tokenizer.snapshot().vocab_size())
                {
                    return Err(BackendErrorV1::new("unknown token ID"));
                }
                tokens.len()
            }
        };
        u64::try_from(count).map_err(|_| BackendErrorV1::new("token count overflow"))
    }

    fn tokenize_utility(
        &self,
        text: &str,
        options: TokenizeOptionsV1,
    ) -> Result<TokenizeResultV1, BackendErrorV1> {
        TokenizerUtilityServiceV1::new(&self.tokenizer, None)
            .tokenize(text, options)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn detokenize_utility(
        &self,
        token_ids: &[u32],
        mode: DecodeModeV1,
    ) -> Result<String, BackendErrorV1> {
        TokenizerUtilityServiceV1::new(&self.tokenizer, None)
            .detokenize_ids(token_ids, mode)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn apply_template_utility(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<ApplyTemplateResultV1, BackendErrorV1> {
        if messages.is_empty() || !options.add_generation_prompt {
            return Err(BackendErrorV1::new("synthetic template input is invalid"));
        }
        let rendered = "hello".to_owned();
        let token_ids = self
            .tokenizer
            .encode(&rendered)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        let identity = TemplateIdentityV1::from_verified_parts(
            "synthetic-template-v1",
            1,
            "synthetic-fixture",
            FINGERPRINT,
            1,
        )
        .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        ApplyTemplateResultV1::from_verified_parts(
            rendered,
            TokenIdsV1::from_slice(token_ids.as_slice()),
            identity,
        )
        .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn tokenize_infill_content(&self, text: &str) -> Result<Vec<u32>, BackendErrorV1> {
        self.tokenizer
            .encode_without_special_tokens(text)
            .map(|tokens| tokens.as_slice().to_vec())
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn infill_capability(&self) -> Option<BackendInfillCapabilityV1> {
        Some(
            BackendInfillCapabilityV1::new(
                sllm_frontend::FimTemplateV1::new(8, 9, 10).unwrap(),
                4096,
                "synthetic-fixture",
            )
            .unwrap(),
        )
    }
}

struct NoFimBackend;

impl ChatGenerationBackendV1 for NoFimBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        _: &GenerationCancellationV1,
        _: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        Err(BackendErrorV1::new("not used"))
    }
}

fn router() -> (Router, SchedulerV1) {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(phase42_backend());
    let entry =
        ModelRegistryEntryV1::new("phase42-test", 1_700_000_000, "sllm", FINGERPRINT, backend)
            .unwrap();
    let no_fim: Arc<dyn ChatGenerationBackendV1> = Arc::new(NoFimBackend);
    let no_fim_entry =
        ModelRegistryEntryV1::new("phase42-no-fim", 1_700_000_000, "sllm", FINGERPRINT, no_fim)
            .unwrap();
    let registry = ModelRegistryV1::new(vec![entry, no_fim_entry]).unwrap();
    let scheduler = SchedulerV1::new(SchedulerConfigV1::new(8, 8, Duration::from_secs(5)).unwrap());
    let app = build_router_v1(registry, scheduler.clone(), ServerConfigV1::default());
    (app, scheduler)
}

async fn serve(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, task)
}

fn request(path: &str, body: &[u8]) -> Vec<u8> {
    let mut value = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    value.extend_from_slice(body);
    value
}

async fn raw_http(address: SocketAddr, request: Vec<u8>) -> (u16, String) {
    tokio::task::spawn_blocking(move || {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        socket.write_all(&request).unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).unwrap();
        let split = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let headers = String::from_utf8_lossy(&response[..split]).into_owned();
        let status = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let raw_body = &response[split + 4..];
        let body = if headers
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            String::from_utf8_lossy(&decode_chunked(raw_body)).into_owned()
        } else {
            String::from_utf8_lossy(raw_body).into_owned()
        };
        (status, body)
    })
    .await
    .unwrap()
}

fn decode_chunked(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let line_end = input
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let size = usize::from_str_radix(
            std::str::from_utf8(&input[..line_end])
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            16,
        )
        .unwrap();
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
    output
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase42_completion_embeddings_rerank_and_infill_routes() {
    let (app, scheduler) = router();
    let (address, server) = serve(app).await;

    let (status, body) = raw_http(
        address,
        request(
            "/v1/completions",
            br#"{"model":"phase42-test","prompt":["a","b"],"n":2,"stream":false,"max_tokens":2}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"].as_array().unwrap().len(), 4);
    assert_eq!(
        json["choices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|choice| choice["index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    // Batched Completions counts each prompt once, while completion usage is
    // accumulated independently for all n choices.
    assert_eq!(json["usage"]["prompt_tokens"], 4);
    assert_eq!(json["usage"]["completion_tokens"], 4);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/completions",
            br#"{"model":"phase42-test","prompt":"a","logprobs":1,"max_tokens":2}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    let json: Value = serde_json::from_str(&body).unwrap();
    let logprobs = &json["choices"][0]["logprobs"];
    assert_eq!(logprobs["tokens"], json!(["ok"]));
    assert_eq!(logprobs["token_logprobs"], json!([-0.25]));
    assert_eq!(logprobs["text_offset"], json!([0]));
    assert_eq!(logprobs["top_logprobs"][0]["ok"], -0.25);
    assert!(logprobs.get("content").is_none());

    let (status, body) = raw_http(
        address,
        request(
            "/v1/completions",
            br#"{"model":"phase42-test","prompt":"a","stream":true,"max_tokens":2}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    assert!(body.contains("text_completion"));
    assert!(body.ends_with("data: [DONE]\n\n"));

    let (status, body) = raw_http(
        address,
        request(
            "/v1/embeddings",
            br#"{"model":"phase42-test","input":["q","d"],"encoding_format":"base64"}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 2);
    assert!(!json["data"][0]["embedding"].as_str().unwrap().is_empty());
    assert_eq!(json["usage"]["prompt_tokens"], 2);
    assert_eq!(json["usage"]["total_tokens"], 2);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/embeddings",
            br#"{"model":"phase42-test","input":"q","dimensions":3}"#,
        ),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body.contains("dimensions"));

    let (status, body) = raw_http(
        address,
        request(
            "/v1/rerank",
            br#"{"model":"phase42-test","query":"q","documents":["a","b"],"top_n":1,"return_documents":true}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
    assert_eq!(json["results"][0]["index"], 0);
    assert_eq!(json["usage"]["total_tokens"], 3);

    let (status, _) = raw_http(
        address,
        request(
            "/v1/infill",
            br#"{"model":"phase42-test","prefix":"fn ","suffix":"()","stream":false,"max_tokens":2}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/infill",
            br#"{"model":"phase42-no-fim","prefix":"fn ","suffix":"()","stream":false,"max_tokens":2}"#,
        ),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body.contains("unsupported_parameter"));

    let (status, body) = raw_http(
        address,
        request(
            "/v1/tokenize",
            br#"{"model":"phase42-test","text":"hello world","with_pieces":true}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["tokens"], json!([1, 2]));
    assert_eq!(json["count"], 2);
    assert_eq!(json["pieces"][0]["value"], "hello");
    assert_eq!(json["model_lock_fingerprint"], FINGERPRINT);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/detokenize",
            br#"{"model":"phase42-test","tokens":[1,2],"skip_special_tokens":false}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["text"],
        "hello world"
    );

    let (status, body) = raw_http(
        address,
        request(
            "/v1/apply-template",
            br#"{"model":"phase42-test","messages":[{"role":"user","content":"hello"}],"add_generation_prompt":true,"thinking":false}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    let json: Value = serde_json::from_str(&body).unwrap();
    assert!(json["prompt"].as_str().unwrap().contains("hello"));
    assert!(json["count"].as_u64().unwrap() > 0);
    assert_eq!(json["template"]["digest"], FINGERPRINT);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/input-tokens",
            br#"{"model":"phase42-test","text":"hello"}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["count"], 1);

    let (status, body) = raw_http(
        address,
        request(
            "/v1/input-tokens",
            br#"{"model":"phase42-test","messages":[{"role":"user","content":"hello"}],"add_generation_prompt":true,"thinking":false}"#,
        ),
    )
    .await;
    assert_eq!(status, 200);
    assert!(
        serde_json::from_str::<Value>(&body).unwrap()["count"]
            .as_u64()
            .unwrap()
            > 0
    );

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase42_rejects_unknown_fields_and_wrong_content_type() {
    let (app, scheduler) = router();
    let (address, server) = serve(app).await;
    let (status, body) = raw_http(
        address,
        request(
            "/v1/completions",
            br#"{"model":"phase42-test","prompt":"x","tools":[]}"#,
        ),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body.contains("unsupported_parameter"));
    scheduler.shutdown();
    server.abort();
}
