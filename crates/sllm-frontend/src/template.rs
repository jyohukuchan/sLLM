//! A bounded, data-only Jinja-compatible template provider.
//!
//! The reviewed Qwen renderer lives in [`crate::Qwen35ChatTemplateV1`] and is
//! intentionally not implemented in terms of this module.  This provider is
//! an explicit opt-in for model or request templates whose source and digest
//! have already been verified by a caller.  It never installs a loader and it
//! only accepts JSON values as template data.

use core::fmt;
use std::cell::{Cell, RefCell};
use std::io::{self, Write};
use std::rc::Rc;
use std::sync::Arc;

use minijinja::value::{Value as MiniJinjaValue, ValueKind, from_args};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const GENERIC_TEMPLATE_PROFILE_VERSION_V1: u8 = 1;
pub const GENERIC_TEMPLATE_REVIEWED_GEMMA4_PROFILE_VERSION_V1: u8 = 2;
pub const GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1: usize = 64 * 1024;
pub const GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1: usize = 16 * 1024 * 1024;
pub const GENERIC_TEMPLATE_MAX_MESSAGES_V1: usize = 1_024;
pub const GENERIC_TEMPLATE_MAX_KWARGS_V1: usize = 64;
pub const GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1: usize = 1024 * 1024;
pub const GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1: usize = 32;
pub const GENERIC_TEMPLATE_MAX_RECURSION_V1: usize = 32;
pub const GENERIC_TEMPLATE_MAX_FUEL_V1: u64 = 1_000_000;

const TEMPLATE_NAME: &str = "sllm-generic-template";

// MiniJinja's builtins are intentionally reduced to the exact Phase44
// fixture allowlist.  In particular, dynamic attribute helpers, debug
// formatting, and collection introspection are not part of this profile.
const DISALLOWED_FILTERS: &[&str] = &[
    "safe",
    "escape",
    "e",
    "capitalize",
    "title",
    "dictsort",
    "items",
    "reverse",
    "split",
    "lines",
    "d",
    "round",
    "abs",
    "int",
    "float",
    "attr",
    "min",
    "max",
    "bool",
    "string",
    "batch",
    "slice",
    "sum",
    "indent",
    "select",
    "reject",
    "selectattr",
    "rejectattr",
    "map",
    "groupby",
    "chain",
    "zip",
    "pprint",
    "format",
];

const DISALLOWED_TESTS: &[&str] = &[
    "safe",
    "escaped",
    "odd",
    "even",
    "divisibleby",
    "startingwith",
    "endingwith",
    "lower",
    "upper",
    "sameas",
    "eq",
    "equalto",
    "==",
    "ne",
    "!=",
    "lt",
    "lessthan",
    "<",
    "le",
    "<=",
    "gt",
    "greaterthan",
    ">",
    "ge",
    ">=",
    "in",
    "true",
    "false",
    "filter",
    "test",
];

const DISALLOWED_GLOBALS: &[&str] = &["range", "dict", "debug", "namespace"];

/// Errors returned before a template is admitted to rendering or when a
/// bounded render fails.  Error variants deliberately contain no source or
/// context payloads, so callers can expose them without leaking prompts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenericTemplateErrorV1 {
    EmptySource,
    SourceTooLarge { bytes: usize, max_bytes: usize },
    InvalidSourceUtf8,
    InvalidDigest,
    DigestMismatch { expected: String, actual: String },
    UnsupportedDirective { directive: &'static str },
    UnsafeAttributeAccess,
    Compile { kind: ErrorKind },
    ContextNotObject,
    ContextTooDeep { depth: usize, max_depth: usize },
    MessagesNotArray,
    TooManyMessages { count: usize, max_messages: usize },
    KwargsNotObject,
    TooManyKwargs { count: usize, max_kwargs: usize },
    KwargsTooLarge { bytes: usize, max_bytes: usize },
    UnknownContextField,
    InvalidContext,
    UndefinedValue,
    UnknownFilter,
    UnknownTest,
    UnknownFunction,
    UnknownMethod,
    FuelExhausted,
    RecursionLimit,
    OutputTooLarge { bytes: usize, max_bytes: usize },
    Render { kind: ErrorKind },
}

impl fmt::Display for GenericTemplateErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => formatter.write_str("template source is empty"),
            Self::SourceTooLarge { bytes, max_bytes } => {
                write!(
                    formatter,
                    "template source is {bytes} bytes; maximum is {max_bytes}"
                )
            }
            Self::InvalidSourceUtf8 => formatter.write_str("template source is not UTF-8"),
            Self::InvalidDigest => {
                formatter.write_str("template digest is not a lowercase sha256 digest")
            }
            Self::DigestMismatch { .. } => {
                formatter.write_str("template digest does not match source")
            }
            Self::UnsupportedDirective { directive } => {
                write!(formatter, "template directive {directive} is unsupported")
            }
            Self::UnsafeAttributeAccess => formatter.write_str("unsafe template attribute access"),
            Self::Compile { kind } => write!(formatter, "template compilation failed: {kind}"),
            Self::ContextNotObject => formatter.write_str("template context must be a JSON object"),
            Self::ContextTooDeep { depth, max_depth } => {
                write!(
                    formatter,
                    "template context depth {depth} exceeds {max_depth}"
                )
            }
            Self::MessagesNotArray => formatter.write_str("template messages must be an array"),
            Self::TooManyMessages {
                count,
                max_messages,
            } => {
                write!(
                    formatter,
                    "template has {count} messages; maximum is {max_messages}"
                )
            }
            Self::KwargsNotObject => formatter.write_str("template kwargs must be an object"),
            Self::TooManyKwargs { count, max_kwargs } => {
                write!(
                    formatter,
                    "template has {count} kwargs; maximum is {max_kwargs}"
                )
            }
            Self::KwargsTooLarge { bytes, max_bytes } => {
                write!(
                    formatter,
                    "template kwargs are {bytes} bytes; maximum is {max_bytes}"
                )
            }
            Self::UnknownContextField => {
                formatter.write_str("template context contains an unknown field")
            }
            Self::InvalidContext => formatter.write_str("template context could not be serialized"),
            Self::UndefinedValue => formatter.write_str("template referenced an undefined value"),
            Self::UnknownFilter => formatter.write_str("template referenced an unknown filter"),
            Self::UnknownTest => formatter.write_str("template referenced an unknown test"),
            Self::UnknownFunction => formatter.write_str("template referenced an unknown function"),
            Self::UnknownMethod => formatter.write_str("template referenced an unknown method"),
            Self::FuelExhausted => formatter.write_str("template fuel limit exhausted"),
            Self::RecursionLimit => formatter.write_str("template recursion limit exhausted"),
            Self::OutputTooLarge { bytes, max_bytes } => {
                write!(
                    formatter,
                    "template output is {bytes} bytes; maximum is {max_bytes}"
                )
            }
            Self::Render { kind } => write!(formatter, "template render failed: {kind}"),
        }
    }
}

impl std::error::Error for GenericTemplateErrorV1 {}

/// The identity of a verified generic template source and one render.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericTemplateIdentityV1 {
    profile_version: u8,
    template_digest: String,
    source_size_bytes: usize,
    kwargs_digest: String,
    rendered_digest: String,
    rendered_size_bytes: usize,
}

impl GenericTemplateIdentityV1 {
    pub const fn profile_version(&self) -> u8 {
        self.profile_version
    }

    pub const fn version(&self) -> u8 {
        self.profile_version()
    }

    pub fn template_digest(&self) -> &str {
        &self.template_digest
    }

    pub fn source_digest(&self) -> &str {
        self.template_digest()
    }

    pub const fn source_size_bytes(&self) -> usize {
        self.source_size_bytes
    }

    pub fn kwargs_digest(&self) -> &str {
        &self.kwargs_digest
    }

    pub fn rendered_digest(&self) -> &str {
        &self.rendered_digest
    }

    pub fn output_digest(&self) -> &str {
        self.rendered_digest()
    }

    pub const fn rendered_size_bytes(&self) -> usize {
        self.rendered_size_bytes
    }
}

/// A checked JSON-only context.  The renderer also validates a context on
/// every call, because callers may pass a plain `serde_json::Value` directly.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GenericTemplateContextV1(Value);

impl GenericTemplateContextV1 {
    pub fn new(value: Value) -> Result<Self, GenericTemplateErrorV1> {
        validate_context(&value)?;
        Ok(Self(value))
    }

    pub fn from_json(value: Value) -> Result<Self, GenericTemplateErrorV1> {
        Self::new(value)
    }

    pub fn from_serialize<S: Serialize>(value: S) -> Result<Self, GenericTemplateErrorV1> {
        let value =
            serde_json::to_value(value).map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
        Self::new(value)
    }

    pub fn with_kwargs<S: Serialize>(
        value: S,
        kwargs: &Map<String, Value>,
    ) -> Result<Self, GenericTemplateErrorV1> {
        let mut value =
            serde_json::to_value(value).map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
        let object = value
            .as_object_mut()
            .ok_or(GenericTemplateErrorV1::ContextNotObject)?;
        let kwargs = Value::Object(kwargs.clone());
        // `custom_kwargs` is the versioned wire spelling. Keep `kwargs` as a
        // local alias for llama-style templates; both remain bounded and are
        // included in the same digest.
        object.insert("custom_kwargs".to_owned(), kwargs.clone());
        object.insert("kwargs".to_owned(), kwargs);
        Self::new(value)
    }

    pub fn value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

/// A source whose UTF-8 bytes and caller-provided digest have been verified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericTemplateSourceV1 {
    source: String,
    digest: String,
}

impl GenericTemplateSourceV1 {
    pub fn new(source: &str, expected_digest: &str) -> Result<Self, GenericTemplateErrorV1> {
        let source = source.as_bytes();
        Self::from_bytes(source, expected_digest)
    }

    pub fn from_str(source: &str, expected_digest: &str) -> Result<Self, GenericTemplateErrorV1> {
        Self::new(source, expected_digest)
    }

    pub fn from_bytes(
        source: &[u8],
        expected_digest: &str,
    ) -> Result<Self, GenericTemplateErrorV1> {
        if source.is_empty() {
            return Err(GenericTemplateErrorV1::EmptySource);
        }
        if source.len() > GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1 {
            return Err(GenericTemplateErrorV1::SourceTooLarge {
                bytes: source.len(),
                max_bytes: GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1,
            });
        }
        let source =
            std::str::from_utf8(source).map_err(|_| GenericTemplateErrorV1::InvalidSourceUtf8)?;
        let actual = sha256_digest(source.as_bytes());
        validate_digest(expected_digest)?;
        if actual != expected_digest {
            return Err(GenericTemplateErrorV1::DigestMismatch {
                expected: expected_digest.to_owned(),
                actual,
            });
        }
        validate_directives(source)?;
        Ok(Self {
            source: source.to_owned(),
            digest: expected_digest.to_owned(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn size_bytes(&self) -> usize {
        self.source.len()
    }

    pub fn compile(self) -> Result<GenericTemplateProviderV1, GenericTemplateErrorV1> {
        GenericTemplateProviderV1::from_source(self)
    }
}

/// Generic, explicitly opt-in, sandboxed template provider.
#[derive(Clone)]
pub struct GenericTemplateProviderV1 {
    source: GenericTemplateSourceV1,
    environment: Environment<'static>,
    profile_version: u8,
}

#[derive(Clone, Copy)]
enum GenericTemplateBuiltinProfile {
    Restricted,
    ReviewedGemma4,
}

/// Compatibility alias for callers that use “renderer” terminology.
pub type GenericTemplateRendererV1 = GenericTemplateProviderV1;

impl fmt::Debug for GenericTemplateProviderV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenericTemplateProviderV1")
            .field("digest", &self.source.digest())
            .field("size_bytes", &self.source.size_bytes())
            .field("profile_version", &self.profile_version)
            .finish()
    }
}

impl GenericTemplateProviderV1 {
    pub fn new(source: &str, expected_digest: &str) -> Result<Self, GenericTemplateErrorV1> {
        GenericTemplateSourceV1::new(source, expected_digest)?.compile()
    }

    pub fn from_bytes(
        source: &[u8],
        expected_digest: &str,
    ) -> Result<Self, GenericTemplateErrorV1> {
        GenericTemplateSourceV1::from_bytes(source, expected_digest)?.compile()
    }

    fn from_source(source: GenericTemplateSourceV1) -> Result<Self, GenericTemplateErrorV1> {
        Self::from_source_with_profile(source, GenericTemplateBuiltinProfile::Restricted)
    }

    pub(crate) fn from_reviewed_gemma4_bytes(
        source: &[u8],
        expected_digest: &str,
    ) -> Result<Self, GenericTemplateErrorV1> {
        let source = GenericTemplateSourceV1::from_bytes(source, expected_digest)?;
        Self::from_source_with_profile(source, GenericTemplateBuiltinProfile::ReviewedGemma4)
    }

    fn from_source_with_profile(
        source: GenericTemplateSourceV1,
        profile: GenericTemplateBuiltinProfile,
    ) -> Result<Self, GenericTemplateErrorV1> {
        let mut environment = Environment::<'static>::new();
        environment.set_undefined_behavior(UndefinedBehavior::Strict);
        environment.set_recursion_limit(GENERIC_TEMPLATE_MAX_RECURSION_V1);
        environment.set_fuel(Some(GENERIC_TEMPLATE_MAX_FUEL_V1));
        for filter in DISALLOWED_FILTERS {
            if !matches!(profile, GenericTemplateBuiltinProfile::ReviewedGemma4)
                || !matches!(*filter, "dictsort" | "map" | "upper")
            {
                environment.remove_filter(filter);
            }
        }
        for test in DISALLOWED_TESTS {
            environment.remove_test(test);
        }
        for global in DISALLOWED_GLOBALS {
            if !matches!(profile, GenericTemplateBuiltinProfile::ReviewedGemma4)
                || !matches!(*global, "range" | "namespace")
            {
                environment.remove_global(global);
            }
        }
        if matches!(profile, GenericTemplateBuiltinProfile::ReviewedGemma4) {
            environment.set_unknown_method_callback(|_, value, method, args| {
                match (value.kind(), method) {
                    (ValueKind::String, "split") => {
                        let (separator, max_splits): (Option<Arc<str>>, Option<i64>) =
                            from_args(args)?;
                        minijinja::filters::split(value, separator, max_splits)
                    }
                    (ValueKind::Map, "get") => {
                        let (key, default): (&MiniJinjaValue, Option<&MiniJinjaValue>) =
                            from_args(args)?;
                        let found = value.get_item(key)?;
                        if found.is_undefined() {
                            Ok(default.cloned().unwrap_or_else(|| MiniJinjaValue::from(())))
                        } else {
                            Ok(found)
                        }
                    }
                    _ => Err(Error::from(ErrorKind::UnknownMethod)),
                }
            });
        }
        environment
            .add_template_owned(TEMPLATE_NAME.to_owned(), source.source.clone())
            .map_err(|error| GenericTemplateErrorV1::Compile { kind: error.kind() })?;
        Ok(Self {
            source,
            environment,
            profile_version: match profile {
                GenericTemplateBuiltinProfile::Restricted => GENERIC_TEMPLATE_PROFILE_VERSION_V1,
                GenericTemplateBuiltinProfile::ReviewedGemma4 => {
                    GENERIC_TEMPLATE_REVIEWED_GEMMA4_PROFILE_VERSION_V1
                }
            },
        })
    }

    pub fn source(&self) -> &GenericTemplateSourceV1 {
        &self.source
    }

    pub fn digest(&self) -> &str {
        self.source.digest()
    }

    pub fn source_size_bytes(&self) -> usize {
        self.source.size_bytes()
    }

    pub const fn profile_version(&self) -> u8 {
        self.profile_version
    }

    pub fn render<S: Serialize>(
        &self,
        context: S,
    ) -> Result<GenericTemplateRenderResultV1, GenericTemplateErrorV1> {
        let value =
            serde_json::to_value(context).map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
        self.render_value(value)
    }

    pub fn render_context(
        &self,
        context: &GenericTemplateContextV1,
    ) -> Result<GenericTemplateRenderResultV1, GenericTemplateErrorV1> {
        self.render_value(context.value().clone())
    }

    pub fn render_with_kwargs<S: Serialize>(
        &self,
        context: S,
        kwargs: &Map<String, Value>,
    ) -> Result<GenericTemplateRenderResultV1, GenericTemplateErrorV1> {
        let context = GenericTemplateContextV1::with_kwargs(context, kwargs)?;
        self.render_context(&context)
    }

    pub fn render_json(
        &self,
        context: Value,
    ) -> Result<GenericTemplateRenderResultV1, GenericTemplateErrorV1> {
        self.render_value(context)
    }

    fn render_value(
        &self,
        context: Value,
    ) -> Result<GenericTemplateRenderResultV1, GenericTemplateErrorV1> {
        validate_context(&context)?;
        let kwargs_digest = kwargs_digest(&context)?;
        let bytes = Rc::new(RefCell::new(Vec::new()));
        let exceeded = Rc::new(Cell::new(false));
        let attempted = Rc::new(Cell::new(0usize));
        let writer = BoundedWriter {
            bytes: Rc::clone(&bytes),
            exceeded: Rc::clone(&exceeded),
            attempted: Rc::clone(&attempted),
            max_bytes: GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
        };
        let template = self
            .environment
            .get_template(TEMPLATE_NAME)
            .map_err(|error| map_engine_error(error, &exceeded, &attempted))?;
        template
            .render_captured_to(context, writer)
            .map_err(|error| map_engine_error(error, &exceeded, &attempted))?;
        let output = Rc::try_unwrap(bytes)
            .ok()
            .map(RefCell::into_inner)
            .unwrap_or_default();
        let rendered_size_bytes = output.len();
        let rendered = String::from_utf8(output).map_err(|_| GenericTemplateErrorV1::Render {
            kind: ErrorKind::WriteFailure,
        })?;
        let identity = GenericTemplateIdentityV1 {
            profile_version: self.profile_version,
            template_digest: self.source.digest.clone(),
            source_size_bytes: self.source.size_bytes(),
            kwargs_digest,
            rendered_digest: sha256_digest(rendered.as_bytes()),
            rendered_size_bytes,
        };
        Ok(GenericTemplateRenderResultV1 { rendered, identity })
    }
}

/// The bounded result of one template render.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericTemplateRenderResultV1 {
    rendered: String,
    identity: GenericTemplateIdentityV1,
}

impl GenericTemplateRenderResultV1 {
    pub fn rendered(&self) -> &str {
        &self.rendered
    }

    pub fn text(&self) -> &str {
        self.rendered()
    }

    pub fn identity(&self) -> &GenericTemplateIdentityV1 {
        &self.identity
    }

    pub fn into_rendered(self) -> String {
        self.rendered
    }
}

struct BoundedWriter {
    bytes: Rc<RefCell<Vec<u8>>>,
    exceeded: Rc<Cell<bool>>,
    attempted: Rc<Cell<usize>>,
    max_bytes: usize,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let current = self.bytes.borrow().len();
        let next = current.saturating_add(bytes.len());
        if next > self.max_bytes {
            self.exceeded.set(true);
            self.attempted.set(next);
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "generic template output limit exceeded",
            ));
        }
        self.bytes.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn map_engine_error(
    error: minijinja::Error,
    exceeded: &Cell<bool>,
    attempted: &Cell<usize>,
) -> GenericTemplateErrorV1 {
    if exceeded.get() {
        return GenericTemplateErrorV1::OutputTooLarge {
            bytes: attempted.get(),
            max_bytes: GENERIC_TEMPLATE_MAX_OUTPUT_BYTES_V1,
        };
    }
    match error.kind() {
        ErrorKind::UndefinedError => GenericTemplateErrorV1::UndefinedValue,
        ErrorKind::UnknownFilter => GenericTemplateErrorV1::UnknownFilter,
        ErrorKind::UnknownTest => GenericTemplateErrorV1::UnknownTest,
        ErrorKind::UnknownFunction => GenericTemplateErrorV1::UnknownFunction,
        ErrorKind::UnknownMethod => GenericTemplateErrorV1::UnknownMethod,
        ErrorKind::OutOfFuel => GenericTemplateErrorV1::FuelExhausted,
        ErrorKind::InvalidOperation if error.to_string().contains("recursion limit exceeded") => {
            GenericTemplateErrorV1::RecursionLimit
        }
        ErrorKind::TemplateNotFound | ErrorKind::BadInclude => {
            GenericTemplateErrorV1::Render { kind: error.kind() }
        }
        _ => GenericTemplateErrorV1::Render { kind: error.kind() },
    }
}

fn validate_digest(digest: &str) -> Result<(), GenericTemplateErrorV1> {
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return Err(GenericTemplateErrorV1::InvalidDigest);
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(GenericTemplateErrorV1::InvalidDigest);
    }
    Ok(())
}

fn sha256_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        use fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_directives(source: &str) -> Result<(), GenericTemplateErrorV1> {
    let mut cursor = 0;
    while cursor < source.len() {
        let block = source[cursor..].find("{%");
        let expression = source[cursor..].find("{{");
        let (relative, is_block) = match (block, expression) {
            (None, None) => break,
            (Some(block), None) => (block, true),
            (None, Some(expression)) => (expression, false),
            (Some(block), Some(expression)) => (block.min(expression), block <= expression),
        };
        let start = cursor + relative;
        let close_marker = if is_block { "%}" } else { "}}" };
        let close = source[start + 2..]
            .find(close_marker)
            .map(|offset| (start + 2 + offset, close_marker));
        let Some((end, delimiter)) = close else {
            break;
        };
        let expression = source[start + 2..end].trim_start_matches('-').trim();
        if is_block {
            let mut words = expression.split_whitespace();
            match words.next().unwrap_or_default() {
                "include" => {
                    return Err(GenericTemplateErrorV1::UnsupportedDirective {
                        directive: "include",
                    });
                }
                "import" => {
                    return Err(GenericTemplateErrorV1::UnsupportedDirective {
                        directive: "import",
                    });
                }
                "extends" => {
                    return Err(GenericTemplateErrorV1::UnsupportedDirective {
                        directive: "extends",
                    });
                }
                "from" if words.any(|word| word == "import") => {
                    return Err(GenericTemplateErrorV1::UnsupportedDirective { directive: "from" });
                }
                _ => {}
            }
        }
        if expression
            .split_whitespace()
            .any(|word| word.starts_with("__"))
            || expression.contains(".__")
            || expression.contains("['__")
            || expression.contains("[\"__")
        {
            return Err(GenericTemplateErrorV1::UnsafeAttributeAccess);
        }
        cursor = end + delimiter.len();
    }
    Ok(())
}

fn validate_context(value: &Value) -> Result<(), GenericTemplateErrorV1> {
    if !value.is_object() {
        return Err(GenericTemplateErrorV1::ContextNotObject);
    }
    validate_value_depth(value, 0)?;
    let object = value.as_object().expect("checked object");
    for (key, value) in object {
        if !is_allowed_context_key(key) {
            return Err(GenericTemplateErrorV1::UnknownContextField);
        }
        if is_special_token_key(key) && !value.is_string() {
            return Err(GenericTemplateErrorV1::InvalidContext);
        }
    }
    if let Some(messages) = object.get("messages") {
        let Some(messages) = messages.as_array() else {
            return Err(GenericTemplateErrorV1::MessagesNotArray);
        };
        if messages.len() > GENERIC_TEMPLATE_MAX_MESSAGES_V1 {
            return Err(GenericTemplateErrorV1::TooManyMessages {
                count: messages.len(),
                max_messages: GENERIC_TEMPLATE_MAX_MESSAGES_V1,
            });
        }
        for message in messages {
            let Some(message) = message.as_object() else {
                return Err(GenericTemplateErrorV1::InvalidContext);
            };
            if message.get("role").and_then(Value::as_str).is_none()
                || !message.contains_key("content")
            {
                return Err(GenericTemplateErrorV1::InvalidContext);
            }
        }
    }
    if let Some(special_tokens) = object.get("special_tokens") {
        let Some(special_tokens) = special_tokens.as_object() else {
            return Err(GenericTemplateErrorV1::InvalidContext);
        };
        if special_tokens.len() > GENERIC_TEMPLATE_MAX_KWARGS_V1 {
            return Err(GenericTemplateErrorV1::TooManyKwargs {
                count: special_tokens.len(),
                max_kwargs: GENERIC_TEMPLATE_MAX_KWARGS_V1,
            });
        }
        for (key, value) in special_tokens {
            if !is_special_token_key(key) || !value.is_string() {
                return Err(GenericTemplateErrorV1::InvalidContext);
            }
        }
    }
    for key in ["add_generation_prompt", "enable_thinking"] {
        if let Some(value) = object.get(key) {
            if !value.is_boolean() {
                return Err(GenericTemplateErrorV1::InvalidContext);
            }
        }
    }
    if let Some(reasoning_effort) = object.get("reasoning_effort") {
        let Some(reasoning_effort) = reasoning_effort.as_str() else {
            return Err(GenericTemplateErrorV1::InvalidContext);
        };
        if reasoning_effort.is_empty()
            || reasoning_effort.len() > 32
            || !reasoning_effort.is_ascii()
        {
            return Err(GenericTemplateErrorV1::InvalidContext);
        }
    }
    for key in ["kwargs", "custom_kwargs"] {
        if let Some(kwargs) = object.get(key) {
            validate_kwargs(kwargs)?;
        }
    }
    if let (Some(kwargs), Some(custom_kwargs)) = (object.get("kwargs"), object.get("custom_kwargs"))
    {
        let mut kwargs_bytes = Vec::new();
        let mut custom_kwargs_bytes = Vec::new();
        canonical_json(kwargs, &mut kwargs_bytes)
            .map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
        canonical_json(custom_kwargs, &mut custom_kwargs_bytes)
            .map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
        if kwargs_bytes != custom_kwargs_bytes {
            return Err(GenericTemplateErrorV1::InvalidContext);
        }
    }
    Ok(())
}

fn validate_kwargs(value: &Value) -> Result<(), GenericTemplateErrorV1> {
    let Some(kwargs) = value.as_object() else {
        return Err(GenericTemplateErrorV1::KwargsNotObject);
    };
    if kwargs.len() > GENERIC_TEMPLATE_MAX_KWARGS_V1 {
        return Err(GenericTemplateErrorV1::TooManyKwargs {
            count: kwargs.len(),
            max_kwargs: GENERIC_TEMPLATE_MAX_KWARGS_V1,
        });
    }
    if kwargs.keys().any(|key| key.starts_with("__")) {
        return Err(GenericTemplateErrorV1::UnsafeAttributeAccess);
    }
    let mut bytes = Vec::new();
    canonical_json(value, &mut bytes).map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
    if bytes.len() > GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1 {
        return Err(GenericTemplateErrorV1::KwargsTooLarge {
            bytes: bytes.len(),
            max_bytes: GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1,
        });
    }
    Ok(())
}

fn is_allowed_context_key(key: &str) -> bool {
    matches!(
        key,
        "messages"
            | "tools"
            | "special_tokens"
            | "add_generation_prompt"
            | "enable_thinking"
            | "reasoning_effort"
            | "kwargs"
            | "custom_kwargs"
    ) || is_special_token_key(key)
}

fn is_special_token_key(key: &str) -> bool {
    matches!(
        key,
        "assistant"
            | "assistant_token"
            | "begin_of_text"
            | "bos"
            | "bos_token"
            | "cls_token"
            | "eom"
            | "eom_token"
            | "eos"
            | "eos_token"
            | "end_of_text"
            | "eot"
            | "eot_token"
            | "fim_middle"
            | "fim_pad"
            | "fim_prefix"
            | "fim_suffix"
            | "im_end"
            | "im_start"
            | "mask_token"
            | "pad_token"
            | "reasoning_end"
            | "reasoning_start"
            | "sep_token"
            | "start_header_id"
            | "end_header_id"
            | "system"
            | "user"
            | "unk_token"
            | "vision_start"
            | "vision_end"
            | "vision_pad"
            | "image_pad"
            | "video_pad"
    )
}

fn kwargs_digest(value: &Value) -> Result<String, GenericTemplateErrorV1> {
    let kwargs = value
        .as_object()
        .and_then(|object| object.get("custom_kwargs").or_else(|| object.get("kwargs")))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    validate_kwargs(&kwargs)?;
    let mut bytes = Vec::new();
    canonical_json(&kwargs, &mut bytes).map_err(|_| GenericTemplateErrorV1::InvalidContext)?;
    Ok(sha256_digest(&bytes))
}

fn validate_value_depth(value: &Value, depth: usize) -> Result<(), GenericTemplateErrorV1> {
    if depth > GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1 {
        return Err(GenericTemplateErrorV1::ContextTooDeep {
            depth,
            max_depth: GENERIC_TEMPLATE_MAX_KWARGS_DEPTH_V1,
        });
    }
    match value {
        Value::Array(values) => values
            .iter()
            .try_for_each(|value| validate_value_depth(value, depth + 1)),
        Value::Object(values) => values
            .values()
            .try_for_each(|value| validate_value_depth(value, depth + 1)),
        _ => Ok(()),
    }
}

fn canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_writer(output, value)
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                canonical_json(value, output)?;
            }
            output.push(b']');
            Ok(())
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys: Vec<&str> = values.keys().map(String::as_str).collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)?;
                output.push(b':');
                canonical_json(&values[key], output)?;
            }
            output.push(b'}');
            Ok(())
        }
    }
}
