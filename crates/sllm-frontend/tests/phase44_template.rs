use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sllm_frontend::{
    GENERIC_TEMPLATE_MAX_FUEL_V1, GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1,
    GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1, GENERIC_TEMPLATE_MAX_KWARGS_V1,
    GENERIC_TEMPLATE_MAX_MESSAGES_V1, GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
    GENERIC_TEMPLATE_MAX_RECURSION_V1, GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1,
    GenericTemplateContextV1, GenericTemplateErrorV1, GenericTemplateProviderV1,
};

fn digest(source: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(source.as_bytes()))
}

fn provider(source: &str) -> GenericTemplateProviderV1 {
    GenericTemplateProviderV1::new(source, &digest(source)).expect("valid template")
}

#[test]
fn canonical_roles_special_tokens_unicode_and_kwargs_render() {
    let template = r#"{% for message in messages %}{{ message.role }}:{{ message.content }}
{% endfor %}{% if add_generation_prompt %}{{ special_tokens.assistant }}{% endif %}{{ kwargs.label }}"#;
    let renderer = provider(template);
    let mut kwargs = Map::new();
    kwargs.insert("label".to_owned(), json!("終端"));
    let context = json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "こんにちは 🌍"}
        ],
        "special_tokens": {"assistant": "<|assistant|>"},
        "add_generation_prompt": true
    });
    let rendered = renderer.render_with_kwargs(context, &kwargs).unwrap();
    assert_eq!(
        rendered.rendered(),
        "system:You are helpful.\nuser:こんにちは 🌍\n<|assistant|>終端"
    );
    assert_eq!(rendered.identity().profile_version(), 1);
    assert_eq!(rendered.identity().template_digest(), digest(template));
    assert_eq!(
        rendered.identity().rendered_digest(),
        digest(rendered.rendered())
    );
    assert_eq!(rendered.identity().source_size_bytes(), template.len());
}

#[test]
fn verified_qwen_vision_special_tokens_are_admitted() {
    let renderer = provider(
        "{{ special_tokens.vision_start }}{{ special_tokens.image_pad }}{{ special_tokens.video_pad }}",
    );
    let rendered = renderer
        .render(json!({
            "special_tokens": {
                "vision_start": "<|vision_start|>",
                "vision_end": "<|vision_end|>",
                "vision_pad": "<|vision_pad|>",
                "image_pad": "<|image_pad|>",
                "video_pad": "<|video_pad|>"
            }
        }))
        .unwrap();
    assert_eq!(
        rendered.rendered(),
        "<|vision_start|><|image_pad|><|video_pad|>"
    );
}

#[test]
fn llama_compatible_conditionals_and_macros_are_supported() {
    let template = r#"{% macro role_line(message) %}{{ message['role'] }}={{ message['content'] }}{% endmacro %}{% for message in messages %}{{ role_line(message) }};{% endfor %}{% if enable_thinking %}<think>{% else %}<answer>{% endif %}"#;
    let rendered = provider(template)
        .render(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": false
        }))
        .unwrap();
    assert_eq!(rendered.rendered(), "user=hello;<answer>");
}

#[test]
fn strict_unknown_values_filters_functions_and_private_attributes_fail_closed() {
    let undefined = provider("{{ missing }}").render(json!({}));
    assert!(matches!(
        undefined,
        Err(GenericTemplateErrorV1::UndefinedValue)
    ));

    let filter = provider("{{ 'x'|not_a_filter }}").render(json!({}));
    assert!(matches!(filter, Err(GenericTemplateErrorV1::UnknownFilter)));

    let function = provider("{{ not_a_function() }}").render(json!({}));
    assert!(matches!(
        function,
        Err(GenericTemplateErrorV1::UnknownFunction)
    ));

    let private =
        GenericTemplateProviderV1::new("{{ value.__class__ }}", &digest("{{ value.__class__ }}"));
    assert!(matches!(
        private,
        Err(GenericTemplateErrorV1::UnsafeAttributeAccess)
    ));
}

#[test]
fn include_import_and_inheritance_are_rejected_before_render() {
    for (source, directive) in [
        ("{% include 'other' %}", "include"),
        ("{% import 'other' as other %}", "import"),
        ("{% from 'other' import value %}", "from"),
        ("{% extends 'base' %}", "extends"),
    ] {
        let error = GenericTemplateProviderV1::new(source, &digest(source)).unwrap_err();
        assert_eq!(
            error,
            GenericTemplateErrorV1::UnsupportedDirective { directive }
        );
    }
}

#[test]
fn source_digest_and_source_size_boundaries_are_strict() {
    let source = "x".repeat(GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1);
    let renderer = GenericTemplateProviderV1::new(&source, &digest(&source)).unwrap();
    assert_eq!(
        renderer.source_size_bytes(),
        GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1
    );

    let oversized = format!("{source}x");
    assert!(matches!(
        GenericTemplateProviderV1::new(&oversized, &digest(&oversized)),
        Err(GenericTemplateErrorV1::SourceTooLarge { .. })
    ));

    let wrong = GenericTemplateProviderV1::new(
        "x",
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    );
    assert!(matches!(
        wrong,
        Err(GenericTemplateErrorV1::DigestMismatch { .. })
    ));
    assert!(matches!(
        GenericTemplateProviderV1::new(
            "x",
            "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ),
        Err(GenericTemplateErrorV1::InvalidDigest)
    ));
}

#[test]
fn output_writer_is_bounded_without_truncating() {
    let source = "{{ 'x' * 16777217 }}";
    let error = provider(source).render(json!({})).unwrap_err();
    assert!(matches!(
        error,
        GenericTemplateErrorV1::OutputTooLarge {
            max_bytes: GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
            ..
        }
    ));
}

#[test]
fn fuel_and_recursion_limits_are_enforced() {
    let fuel_source = "{% for i in messages %}{% for j in messages %}{% if false %}x{% endif %}{% endfor %}{% endfor %}";
    let messages = (0..1024)
        .map(|_| json!({"role": "user", "content": "x"}))
        .collect::<Vec<_>>();
    let error = provider(fuel_source)
        .render(json!({"messages": messages}))
        .unwrap_err();
    assert!(matches!(error, GenericTemplateErrorV1::FuelExhausted));

    let recursion_source = "{% macro recurse(value) %}{% if value > 0 %}{{ recurse(value - 1) }}{% endif %}{% endmacro %}{{ recurse(100) }}";
    let error = provider(recursion_source).render(json!({})).unwrap_err();
    assert!(matches!(error, GenericTemplateErrorV1::RecursionLimit));
    assert_eq!(GENERIC_TEMPLATE_MAX_RECURSION_V1, 32);
    assert_eq!(GENERIC_TEMPLATE_MAX_FUEL_V1, 1_000_000);
}

#[test]
fn messages_kwargs_and_context_depth_limits_have_boundary_checks() {
    let messages = (0..GENERIC_TEMPLATE_MAX_MESSAGES_V1)
        .map(|_| json!({"role": "user", "content": "x"}))
        .collect::<Vec<_>>();
    assert!(
        provider("{{ messages|length }}")
            .render(json!({"messages": messages}))
            .is_ok()
    );
    let messages = (0..=GENERIC_TEMPLATE_MAX_MESSAGES_V1)
        .map(|_| json!({"role": "user", "content": "x"}))
        .collect::<Vec<_>>();
    assert!(matches!(
        provider("{{ messages|length }}").render(json!({"messages": messages})),
        Err(GenericTemplateErrorV1::TooManyMessages { .. })
    ));

    let mut kwargs = Map::new();
    for index in 0..GENERIC_TEMPLATE_MAX_KWARGS_V1 {
        kwargs.insert(format!("key{index}"), json!(index));
    }
    let context = GenericTemplateContextV1::with_kwargs(json!({}), &kwargs).unwrap();
    assert!(
        provider("{{ kwargs.key0 }}")
            .render_context(&context)
            .is_ok()
    );
    kwargs.insert("overflow".to_owned(), json!(true));
    assert!(matches!(
        GenericTemplateContextV1::with_kwargs(json!({}), &kwargs),
        Err(GenericTemplateErrorV1::TooManyKwargs { .. })
    ));

    let oversized = json!({"kwargs": {"value": "x".repeat(GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1)}});
    assert!(matches!(
        provider("{{ kwargs.value }}").render(oversized),
        Err(GenericTemplateErrorV1::KwargsTooLarge { .. })
    ));

    let mut nested = Value::Null;
    for _ in 0..=GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1 {
        nested = json!({"next": nested});
    }
    assert!(matches!(
        provider("{{ kwargs.next }}").render(json!({"kwargs": {"next": nested}})),
        Err(GenericTemplateErrorV1::ContextTooDeep { .. })
    ));
}

#[test]
fn non_object_context_and_non_object_kwargs_are_rejected() {
    assert!(matches!(
        provider("x").render(json!(null)),
        Err(GenericTemplateErrorV1::ContextNotObject)
    ));
    assert!(matches!(
        provider("x").render(json!({"kwargs": []})),
        Err(GenericTemplateErrorV1::KwargsNotObject)
    ));
    assert!(matches!(
        provider("x").render(json!({"prompt": "raw"})),
        Err(GenericTemplateErrorV1::UnknownContextField)
    ));
    assert!(matches!(
        provider("x").render(json!({"messages": [{"content": "missing role"}]})),
        Err(GenericTemplateErrorV1::InvalidContext)
    ));
    assert!(matches!(
        provider("x").render(json!({"special_tokens": {"private": "x"}})),
        Err(GenericTemplateErrorV1::InvalidContext)
    ));
}

#[test]
fn builtin_allowlist_rejects_unprofiled_filters_tests_and_globals() {
    assert!(matches!(
        provider("{{ value|pprint }}").render(json!({"kwargs": {"value": "x"}})),
        Err(GenericTemplateErrorV1::UnknownFilter)
    ));
    assert!(matches!(
        provider("{{ value|safe }}").render(json!({"kwargs": {"value": "x"}})),
        Err(GenericTemplateErrorV1::UnknownFilter)
    ));
    assert!(matches!(
        provider("{{ value is odd }}").render(json!({"kwargs": {"value": 3}})),
        Err(GenericTemplateErrorV1::UnknownTest)
    ));
    assert!(matches!(
        provider("{{ range(3)|length }}").render(json!({})),
        Err(GenericTemplateErrorV1::UnknownFunction)
    ));
}
