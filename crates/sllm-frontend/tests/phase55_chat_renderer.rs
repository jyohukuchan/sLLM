use std::cell::RefCell;

use sha2::{Digest, Sha256};
use sllm_core::{BudgetBoundary, MaxNewTokensZero, PromptEvaluation, StopEvaluation};
use sllm_frontend::{
    ChatMessageV1, ChatRenderError, ChatRenderOptionsV1, ChatTemplateRendererErrorV1,
    ChatTemplateRendererV1, GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1, GenerationInputV1,
    GenerationServiceError, GenerationServiceV1, GenerationStopPolicyV1, GenerationTextFrontendV1,
    GenericChatTemplateConfigV1, GenericTemplateErrorV1, GenericTemplateProviderV1,
    StopTokenHandling, ThinkingModeV1, UntrustedChatMessageV1, UntrustedChatRequestV1,
};

fn digest(source: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(source.as_bytes()))
}

fn provider(source: &str) -> GenericTemplateProviderV1 {
    GenericTemplateProviderV1::new(source, &digest(source)).expect("synthetic template is valid")
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

#[derive(Default)]
struct CapturingTokenizer {
    inputs: RefCell<Vec<String>>,
}

impl GenerationTextFrontendV1 for CapturingTokenizer {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.inputs.borrow_mut().push(text.to_owned());
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        token_ids
            .iter()
            .map(|&token| u8::try_from(token).map(char::from))
            .collect::<Result<String, _>>()
            .map_err(|_| GenerationServiceError::Tokenize)
    }
}

#[test]
fn generic_renderer_handles_normal_roles_special_tokens_and_literal_special_characters() {
    let source = concat!(
        "{{ bos_token }}",
        "{% for message in messages %}<{{ message.role }}>{{ message.content }}</{{ message.role }}>{% endfor %}",
        "{% if add_generation_prompt %}<assistant>{% endif %}",
        "[tools={{ tools|length }}]",
    );
    let provider = provider(source);
    let config = GenericChatTemplateConfigV1::new().with_special_token("bos_token", "<bos>");
    let renderer = ChatTemplateRendererV1::generic_with_config(&provider, config);
    let messages = vec![
        ChatMessageV1::system("rule {{ stays data }} & <tag>"),
        ChatMessageV1::user("雪 \"quoted\" 'single'\\slash\nsecond"),
        ChatMessageV1::assistant("answer & <ok>", None),
        ChatMessageV1::user("next 🌍"),
    ];
    let rendered = renderer
        .render(&messages, ChatRenderOptionsV1::default())
        .expect("ordinary text chat renders");

    assert_eq!(
        rendered.rendered(),
        concat!(
            "<bos>",
            "<system>rule {{ stays data }} & <tag></system>",
            "<user>雪 \"quoted\" 'single'\\slash\nsecond</user>",
            "<assistant>answer & <ok></assistant>",
            "<user>next 🌍</user>",
            "<assistant>[tools=0]",
        )
    );
    let identity = rendered
        .generic_identity()
        .expect("generic provider identity is retained");
    assert_eq!(identity.template_digest(), digest(source));
    assert_eq!(identity.rendered_size_bytes(), rendered.rendered().len());
}

#[test]
fn generic_renderer_rejects_empty_unknown_role_tool_metadata_and_empty_output() {
    let chat_provider = provider("{{ messages[0].content }}");
    let renderer = ChatTemplateRendererV1::generic(&chat_provider);
    assert_eq!(
        renderer.render(&[], ChatRenderOptionsV1::default()),
        Err(ChatTemplateRendererErrorV1::Chat(
            ChatRenderError::EmptyMessages
        ))
    );

    let unknown = UntrustedChatRequestV1::text(vec![UntrustedChatMessageV1::text(
        "developer",
        "not admitted",
    )]);
    assert_eq!(
        renderer.render_untrusted(unknown),
        Err(ChatTemplateRendererErrorV1::Chat(
            ChatRenderError::UnsupportedRole { index: 0 }
        ))
    );

    let mut with_tools =
        UntrustedChatRequestV1::text(vec![UntrustedChatMessageV1::text("user", "hello")]);
    with_tools.messages[0].tool_calls = Some(sllm_frontend::UntrustedChatValueV1::Array);
    assert_eq!(
        renderer.render_untrusted(with_tools),
        Err(ChatTemplateRendererErrorV1::Chat(
            ChatRenderError::UnsupportedToolInput { index: Some(0) }
        ))
    );

    let empty_provider = provider("{% if false %}x{% endif %}");
    let empty_renderer = ChatTemplateRendererV1::generic(&empty_provider);
    assert_eq!(
        empty_renderer.render(
            &[ChatMessageV1::user("hello")],
            ChatRenderOptionsV1::default(),
        ),
        Err(ChatTemplateRendererErrorV1::EmptyTemplateOutput)
    );
}

#[test]
fn generic_renderer_uses_existing_exact_output_cap_without_truncation() {
    let provider = provider("{{ messages[0].content }}");
    let renderer = ChatTemplateRendererV1::generic(&provider);
    let exact = "x".repeat(GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1);
    let rendered = renderer
        .render(
            &[ChatMessageV1::user(exact)],
            ChatRenderOptionsV1 {
                add_generation_prompt: false,
                thinking: ThinkingModeV1::Disabled,
            },
        )
        .expect("exact output cap is admitted");
    assert_eq!(
        rendered.rendered().len(),
        GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1
    );

    let oversized = "x".repeat(GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1 + 1);
    assert!(matches!(
        renderer.render(
            &[ChatMessageV1::user(oversized)],
            ChatRenderOptionsV1::default(),
        ),
        Err(ChatTemplateRendererErrorV1::GenericTemplate(
            GenericTemplateErrorV1::OutputTooLarge {
                max_bytes: GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
                ..
            }
        ))
    ));
}

#[test]
fn generation_service_uses_generic_renderer_on_the_normal_messages_path() {
    let source = "{% for message in messages %}{{ message.role }}={{ message.content }};{% endfor %}{% if add_generation_prompt %}assistant={% endif %}";
    let provider = provider(source);
    let renderer = ChatTemplateRendererV1::generic(&provider);
    let messages = vec![ChatMessageV1::user("hello")];
    let options = ChatRenderOptionsV1::default();
    let direct = renderer
        .render(&messages, options)
        .expect("direct generic render succeeds");
    let expected_identity = direct
        .generic_identity()
        .expect("generic identity exists")
        .clone();

    let tokenizer = CapturingTokenizer::default();
    let policy = stop_policy();
    let service = GenerationServiceV1::new_with_chat_renderer(&tokenizer, Some(renderer), &policy)
        .expect("service accepts model-neutral renderer");
    let prepared = service
        .prepare_input_plan(&GenerationInputV1::Messages { messages, options })
        .expect("normal messages use the generic renderer");

    assert_eq!(
        tokenizer.inputs.borrow().as_slice(),
        ["user=hello;assistant="]
    );
    assert_eq!(
        prepared.generic_template_identity(),
        Some(&expected_identity)
    );
    assert_eq!(
        prepared.token_ids(),
        "user=hello;assistant="
            .bytes()
            .map(u32::from)
            .collect::<Vec<_>>()
    );

    tokenizer.inputs.borrow_mut().clear();
    let continued = service
        .prepare_input_plan(&GenerationInputV1::MessagesWithAssistantPrefill {
            messages: vec![ChatMessageV1::user("hello")],
            options: ChatRenderOptionsV1 {
                add_generation_prompt: false,
                thinking: ThinkingModeV1::Disabled,
            },
            assistant_prefill: "prefix".to_owned(),
        })
        .expect("generic renderer opens an assistant turn before a continuation");
    assert_eq!(
        tokenizer.inputs.borrow().as_slice(),
        ["user=hello;assistant=", "prefix"]
    );
    assert_eq!(
        continued.assistant_prefill_token_ids(),
        [112, 114, 101, 102, 105, 120]
    );
}

#[test]
fn raw_prompt_remains_available_without_any_chat_renderer() {
    let tokenizer = CapturingTokenizer::default();
    let policy = stop_policy();
    let service = GenerationServiceV1::new_with_chat_renderer(&tokenizer, None, &policy)
        .expect("raw-prompt service does not require a renderer");
    let prepared = service
        .prepare_input_plan(&GenerationInputV1::Prompt("生の prompt 🌍".to_owned()))
        .expect("raw prompt remains usable");

    assert_eq!(tokenizer.inputs.borrow().as_slice(), ["生の prompt 🌍"]);
    assert_eq!(
        prepared.token_ids(),
        "生の prompt 🌍".bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(prepared.generic_template_identity(), None);
}
