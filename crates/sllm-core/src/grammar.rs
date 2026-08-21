//! Bounded byte-oriented grammars for constrained generation.
//!
//! The implementation deliberately keeps grammar matching independent from a
//! tokenizer.  A grammar consumes raw token bytes, so a token may end in the
//! middle of an UTF-8 sequence without requiring a lossy `String` decode.
//! Rules are compiled to a bounded epsilon-NFA and the runtime carries an
//! epsilon-closed set of states.  This is small enough for host and GPU
//! selector callers to turn into a valid-token bitset without making the GPU
//! responsible for parsing a grammar.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const MAX_GRAMMAR_BYTES: usize = 64 * 1024;
pub const MAX_GRAMMAR_RULES: usize = 1024;
pub const MAX_GRAMMAR_NAME_BYTES: usize = 128;
pub const MAX_GRAMMAR_ALTERNATIVES: usize = 256;
pub const MAX_GRAMMAR_NESTING: usize = 32;
pub const MAX_GRAMMAR_REPEAT: usize = 4096;
pub const MAX_GRAMMAR_STACK: usize = 128;
pub const MAX_GRAMMAR_ACTIVE_STATES: usize = 65_536;
pub const MAX_TOKEN_PIECE_BYTES: usize = 128;
pub const MAX_TOKEN_TRIE_NODES: usize = 33_554_432;
pub const MAX_JSON_ENUM: usize = 256;
pub const MAX_JSON_PROPERTIES: usize = 1024;
pub const GRAMMAR_RUNTIME_STATE_SCHEMA_V1: &str = "sllm-grammar-runtime-state-v1";
const GRAMMAR_RUNTIME_MAGIC: [u8; 8] = *b"SLLMGRM1";
const GRAMMAR_RUNTIME_VERSION: u16 = 1;
pub const MAX_GRAMMAR_RUNTIME_STATE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrammarError {
    InputTooLarge { limit: usize },
    EmptyInput,
    Parse { offset: usize, message: String },
    DuplicateRule(String),
    UnknownRule(String),
    RuleLimit { limit: usize },
    RuleNameTooLong { limit: usize },
    AlternativeLimit { limit: usize },
    NestingLimit { limit: usize },
    RepeatLimit { limit: usize },
    UnboundedRepetition,
    InvalidRange,
    InvalidEscape,
    LeftRecursion(String),
    CompileLimit { limit: usize },
    StackLimit { limit: usize },
    ActiveStateLimit { limit: usize },
    EmptyTokenPiece,
    TokenPieceTooLarge { limit: usize },
    InvalidTokenId { token_id: usize, token_count: usize },
    TokenTrieLimit { limit: usize },
    InvalidUtf8 { offset: usize },
    TokenRejected,
    AllTokensMasked,
    JsonNotObject,
    JsonTooLarge { limit: usize },
    JsonDepthLimit { limit: usize },
    JsonEnumLimit { limit: usize },
    JsonPropertyLimit { limit: usize },
    UnsupportedSchemaKeyword(String),
    InvalidSchema(String),
    RemoteReference(String),
    RecursiveReference(String),
    RuntimeStateTooLarge,
    RuntimeStateMalformed,
    RuntimeStateVersionUnsupported { version: u16 },
    RuntimeStateIdentityMismatch,
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge { limit } => write!(f, "grammar input exceeds {limit} bytes"),
            Self::EmptyInput => f.write_str("grammar input is empty"),
            Self::Parse { offset, message } => {
                write!(f, "grammar parse error at {offset}: {message}")
            }
            Self::DuplicateRule(name) => write!(f, "duplicate grammar rule {name}"),
            Self::UnknownRule(name) => write!(f, "unknown grammar rule {name}"),
            Self::RuleLimit { limit } => write!(f, "grammar rule limit {limit} exceeded"),
            Self::RuleNameTooLong { limit } => write!(f, "grammar rule name exceeds {limit} bytes"),
            Self::AlternativeLimit { limit } => {
                write!(f, "grammar alternative limit {limit} exceeded")
            }
            Self::NestingLimit { limit } => write!(f, "grammar nesting limit {limit} exceeded"),
            Self::RepeatLimit { limit } => write!(f, "grammar repetition limit {limit} exceeded"),
            Self::UnboundedRepetition => f.write_str("unbounded grammar repetition is unsupported"),
            Self::InvalidRange => f.write_str("invalid grammar byte range"),
            Self::InvalidEscape => f.write_str("invalid grammar escape"),
            Self::LeftRecursion(rule) => write!(f, "left recursion is unsupported in rule {rule}"),
            Self::CompileLimit { limit } => write!(f, "compiled grammar exceeds {limit} states"),
            Self::StackLimit { limit } => write!(f, "grammar stack limit {limit} exceeded"),
            Self::ActiveStateLimit { limit } => {
                write!(f, "grammar active-state limit {limit} exceeded")
            }
            Self::EmptyTokenPiece => f.write_str("token piece must not be empty"),
            Self::TokenPieceTooLarge { limit } => write!(f, "token piece exceeds {limit} bytes"),
            Self::InvalidTokenId {
                token_id,
                token_count,
            } => write!(
                f,
                "token ID {token_id} is outside vocabulary size {token_count}"
            ),
            Self::TokenTrieLimit { limit } => write!(f, "token trie exceeds {limit} nodes"),
            Self::InvalidUtf8 { offset } => {
                write!(f, "invalid UTF-8 prefix at byte offset {offset}")
            }
            Self::TokenRejected => f.write_str("token bytes are not accepted by the grammar"),
            Self::AllTokensMasked => f.write_str("grammar masks every token"),
            Self::JsonNotObject => f.write_str("JSON Schema must be an object"),
            Self::JsonTooLarge { limit } => {
                write!(f, "serialized JSON Schema exceeds {limit} bytes")
            }
            Self::JsonDepthLimit { limit } => write!(f, "JSON Schema nesting exceeds {limit}"),
            Self::JsonEnumLimit { limit } => write!(f, "JSON enum exceeds {limit} values"),
            Self::JsonPropertyLimit { limit } => write!(f, "JSON property count exceeds {limit}"),
            Self::UnsupportedSchemaKeyword(keyword) => {
                write!(f, "unsupported JSON Schema keyword {keyword}")
            }
            Self::InvalidSchema(message) => write!(f, "invalid JSON Schema: {message}"),
            Self::RemoteReference(reference) => write!(
                f,
                "remote JSON Schema reference is unsupported: {reference}"
            ),
            Self::RecursiveReference(reference) => write!(
                f,
                "recursive JSON Schema reference is unsupported: {reference}"
            ),
            Self::RuntimeStateTooLarge => {
                f.write_str("grammar runtime state exceeds its bounded size")
            }
            Self::RuntimeStateMalformed => f.write_str("grammar runtime state is malformed"),
            Self::RuntimeStateVersionUnsupported { version } => {
                write!(f, "unsupported grammar runtime state version {version}")
            }
            Self::RuntimeStateIdentityMismatch => {
                f.write_str("grammar runtime state identity differs from the grammar")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ByteSet([u64; 4]);

impl ByteSet {
    fn empty() -> Self {
        Self([0; 4])
    }

    fn any() -> Self {
        Self([u64::MAX; 4])
    }

    fn one(value: u8) -> Self {
        let mut set = Self::empty();
        set.insert(value);
        set
    }

    fn insert(&mut self, value: u8) {
        self.0[(value / 64) as usize] |= 1_u64 << (value % 64);
    }

    fn contains(self, value: u8) -> bool {
        (self.0[(value / 64) as usize] & (1_u64 << (value % 64))) != 0
    }
}

#[derive(Clone, Debug)]
enum Expr {
    Empty,
    Literal(Vec<u8>),
    Bytes(ByteSet),
    Ref(String),
    Sequence(Vec<Expr>),
    Alternation(Vec<Expr>),
    Repeat {
        expr: Box<Expr>,
        min: usize,
        max: usize,
    },
}

#[derive(Clone, Debug)]
struct Rule {
    name: String,
    expr: Expr,
}

#[derive(Clone, Debug)]
enum Edge {
    Epsilon(usize),
    Bytes(ByteSet, usize),
    Call {
        target: usize,
        return_state: usize,
    },
    Return,
    RepeatEnter {
        target: usize,
        id: usize,
    },
    RepeatExit {
        target: usize,
        body_start: usize,
        id: usize,
        min: usize,
        max: usize,
    },
}

#[derive(Clone, Debug, Default)]
struct NfaState {
    edges: Vec<Edge>,
    accept: bool,
}

#[derive(Clone, Debug)]
pub struct CompiledGrammar {
    states: Vec<NfaState>,
    start: usize,
    accept: usize,
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Utf8State {
    #[default]
    Complete,
    Partial {
        expected: u8,
        seen: u8,
    },
}

impl Utf8State {
    fn push(&mut self, bytes: &[u8], offset: usize) -> Result<(), GrammarError> {
        for (index, &byte) in bytes.iter().enumerate() {
            match self {
                Self::Complete => {
                    if byte <= 0x7f {
                        continue;
                    }
                    let expected = match byte {
                        0xc2..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf4 => 4,
                        _ => {
                            return Err(GrammarError::InvalidUtf8 {
                                offset: offset + index,
                            });
                        }
                    };
                    *self = Self::Partial { expected, seen: 1 };
                }
                Self::Partial { expected, seen } => {
                    if !(0x80..=0xbf).contains(&byte) {
                        return Err(GrammarError::InvalidUtf8 {
                            offset: offset + index,
                        });
                    }
                    *seen += 1;
                    if *seen == *expected {
                        *self = Self::Complete;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct GrammarState {
    grammar: CompiledGrammar,
    active: Vec<ActivePath>,
    utf8: Utf8State,
    accepted_bytes: usize,
}

impl fmt::Debug for GrammarState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrammarState")
            .field("active_state_count", &self.active.len())
            .field("utf8_boundary", &self.is_utf8_boundary())
            .field("accepted_bytes", &self.accepted_bytes)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ActivePath {
    state: usize,
    repeats: BTreeMap<usize, usize>,
    returns: Vec<usize>,
}

impl GrammarState {
    pub fn reset(&mut self) {
        self.active.clear();
        epsilon_closure(
            &self.grammar,
            [ActivePath {
                state: self.grammar.start,
                repeats: BTreeMap::new(),
                returns: Vec::new(),
            }],
            &mut self.active,
        );
        self.utf8 = Utf8State::Complete;
        self.accepted_bytes = 0;
    }

    pub fn clone_state(&self) -> Self {
        self.clone()
    }

    /// Encodes the mutable grammar matcher state with a bounded, versioned
    /// representation. The compiled grammar identity is included so state
    /// cannot be restored against a different grammar.
    pub fn snapshot(&self) -> Result<Vec<u8>, GrammarError> {
        let repeat_limits = grammar_repeat_limits(&self.grammar);
        let total_bytes = grammar_snapshot_size(&self.active, self.grammar.states.len())?;
        if total_bytes > MAX_GRAMMAR_RUNTIME_STATE_BYTES
            || self.active.is_empty()
            || self.active.len() > MAX_GRAMMAR_ACTIVE_STATES
        {
            return Err(GrammarError::RuntimeStateTooLarge);
        }
        let accepted_bytes =
            u64::try_from(self.accepted_bytes).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
        let mut bytes = Vec::with_capacity(total_bytes);
        bytes.extend_from_slice(&GRAMMAR_RUNTIME_MAGIC);
        bytes.extend_from_slice(&GRAMMAR_RUNTIME_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&self.grammar.identity_digest());
        bytes.extend_from_slice(&accepted_bytes.to_le_bytes());
        encode_utf8_state(&self.utf8, &mut bytes);
        bytes.extend_from_slice(&(self.active.len() as u32).to_le_bytes());
        let mut seen = BTreeSet::new();
        for path in &self.active {
            validate_active_path(&self.grammar, path, &repeat_limits)?;
            if !seen.insert(path) {
                return Err(GrammarError::RuntimeStateMalformed);
            }
            bytes.extend_from_slice(
                &u32::try_from(path.state)
                    .map_err(|_| GrammarError::RuntimeStateTooLarge)?
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(&(path.repeats.len() as u32).to_le_bytes());
            for (&id, &count) in &path.repeats {
                bytes.extend_from_slice(
                    &u32::try_from(id)
                        .map_err(|_| GrammarError::RuntimeStateTooLarge)?
                        .to_le_bytes(),
                );
                bytes.extend_from_slice(
                    &u32::try_from(count)
                        .map_err(|_| GrammarError::RuntimeStateTooLarge)?
                        .to_le_bytes(),
                );
            }
            bytes.extend_from_slice(&(path.returns.len() as u32).to_le_bytes());
            for &return_state in &path.returns {
                bytes.extend_from_slice(
                    &u32::try_from(return_state)
                        .map_err(|_| GrammarError::RuntimeStateTooLarge)?
                        .to_le_bytes(),
                );
            }
        }
        if bytes.len() != total_bytes || bytes.len() > MAX_GRAMMAR_RUNTIME_STATE_BYTES {
            return Err(GrammarError::RuntimeStateTooLarge);
        }
        Ok(bytes)
    }

    /// Transactionally replaces this matcher state from a snapshot.
    pub fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<(), GrammarError> {
        let restored = self.grammar.restore_state(bytes)?;
        self.active = restored.active;
        self.utf8 = restored.utf8;
        self.accepted_bytes = restored.accepted_bytes;
        Ok(())
    }

    pub fn utf8_state(&self) -> &Utf8State {
        &self.utf8
    }

    pub fn accepted_bytes(&self) -> usize {
        self.accepted_bytes
    }

    pub fn is_finished(&self) -> bool {
        self.active
            .iter()
            .any(|path| path.state == self.grammar.accept)
    }

    /// True when the current byte prefix reaches an accepting grammar state.
    /// Callers may use this predicate for an EOS mask; a prefix that is only
    /// syntactically valid but incomplete must not terminate generation.
    pub fn is_accepting(&self) -> bool {
        self.is_finished()
    }

    pub fn is_utf8_boundary(&self) -> bool {
        self.utf8 == Utf8State::Complete
    }

    pub fn is_complete_utf8(&self) -> bool {
        self.is_utf8_boundary()
    }

    pub fn active_state_count(&self) -> usize {
        self.active.len()
    }

    pub fn accept(&mut self, piece: &[u8]) -> Result<(), GrammarError> {
        if piece.is_empty() {
            return Err(GrammarError::EmptyTokenPiece);
        }
        if piece.len() > MAX_TOKEN_PIECE_BYTES {
            return Err(GrammarError::TokenPieceTooLarge {
                limit: MAX_TOKEN_PIECE_BYTES,
            });
        }
        let next = advance(&self.grammar, &self.active, piece)?;
        if next.is_empty() {
            return Err(GrammarError::TokenRejected);
        }
        let mut next_utf8 = self.utf8.clone();
        next_utf8.push(piece, self.accepted_bytes)?;
        self.active = next;
        self.utf8 = next_utf8;
        self.accepted_bytes = self
            .accepted_bytes
            .checked_add(piece.len())
            .ok_or(GrammarError::CompileLimit { limit: usize::MAX })?;
        Ok(())
    }

    pub fn valid_token_mask(
        &self,
        token_pieces: &[impl AsRef<[u8]>],
    ) -> Result<Vec<bool>, GrammarError> {
        let trie = TokenTrie::new(token_pieces)?;
        self.valid_token_mask_with_trie(&trie)
    }

    pub fn valid_token_mask_with_trie(&self, trie: &TokenTrie) -> Result<Vec<bool>, GrammarError> {
        let mut mask = vec![false; trie.token_count()];
        trie.mark_valid(self, &mut mask)?;
        if !mask.iter().any(|value| *value) {
            return Err(GrammarError::AllTokensMasked);
        }
        Ok(mask)
    }
}

impl CompiledGrammar {
    pub fn compile(source: &str) -> Result<Self, GrammarError> {
        if source.is_empty() {
            return Err(GrammarError::EmptyInput);
        }
        if source.len() > MAX_GRAMMAR_BYTES {
            return Err(GrammarError::InputTooLarge {
                limit: MAX_GRAMMAR_BYTES,
            });
        }
        let rules = GrammarParser::new(source).parse_rules()?;
        compile_rules(rules)
    }

    pub fn json_object() -> Result<Self, GrammarError> {
        Self::compile(&bounded_json_grammar_source(1))
    }

    pub fn from_json_schema(schema: &Value) -> Result<Self, GrammarError> {
        JsonSchemaLowerer::new(schema)?.compile()
    }

    pub fn initial_state(&self) -> GrammarState {
        let mut active = Vec::new();
        epsilon_closure(
            self,
            [ActivePath {
                state: self.start,
                repeats: BTreeMap::new(),
                returns: Vec::new(),
            }],
            &mut active,
        );
        GrammarState {
            grammar: self.clone(),
            active,
            utf8: Utf8State::Complete,
            accepted_bytes: 0,
        }
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Stable identity for the compiled grammar topology and byte predicates.
    pub fn identity_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(GRAMMAR_RUNTIME_STATE_SCHEMA_V1.as_bytes());
        digest.update((self.states.len() as u64).to_le_bytes());
        digest.update((self.start as u64).to_le_bytes());
        digest.update((self.accept as u64).to_le_bytes());
        for state in &self.states {
            digest.update([u8::from(state.accept)]);
            digest.update((state.edges.len() as u64).to_le_bytes());
            for edge in &state.edges {
                hash_edge(&mut digest, edge);
            }
        }
        digest.finalize().into()
    }

    /// Decodes a grammar matcher state after validating identity and all
    /// bounded NFA stack/state references.
    pub fn restore_state(&self, bytes: &[u8]) -> Result<GrammarState, GrammarError> {
        let mut cursor = GrammarRuntimeCursor::new(bytes)?;
        let version = cursor.u16()?;
        if version != GRAMMAR_RUNTIME_VERSION {
            return Err(GrammarError::RuntimeStateVersionUnsupported { version });
        }
        if cursor.u16()? != 0 {
            return Err(GrammarError::RuntimeStateMalformed);
        }
        let mut expected_digest = [0u8; 32];
        expected_digest.copy_from_slice(cursor.take(32)?);
        if expected_digest != self.identity_digest() {
            return Err(GrammarError::RuntimeStateIdentityMismatch);
        }
        let accepted_bytes =
            usize::try_from(cursor.u64()?).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
        let utf8 = decode_utf8_state(cursor.take(4)?)?;
        let active_len =
            usize::try_from(cursor.u32()?).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
        if active_len == 0 || active_len > MAX_GRAMMAR_ACTIVE_STATES {
            return Err(GrammarError::RuntimeStateTooLarge);
        }
        let repeat_limits = grammar_repeat_limits(self);
        let mut active = Vec::with_capacity(active_len);
        let mut seen = BTreeSet::new();
        for _ in 0..active_len {
            let state =
                usize::try_from(cursor.u32()?).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
            let repeat_len =
                usize::try_from(cursor.u32()?).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
            if repeat_len > MAX_GRAMMAR_STACK {
                return Err(GrammarError::RuntimeStateTooLarge);
            }
            let mut repeats = BTreeMap::new();
            for _ in 0..repeat_len {
                let id = usize::try_from(cursor.u32()?)
                    .map_err(|_| GrammarError::RuntimeStateTooLarge)?;
                let count = usize::try_from(cursor.u32()?)
                    .map_err(|_| GrammarError::RuntimeStateTooLarge)?;
                if repeats.insert(id, count).is_some() {
                    return Err(GrammarError::RuntimeStateMalformed);
                }
            }
            let returns_len =
                usize::try_from(cursor.u32()?).map_err(|_| GrammarError::RuntimeStateTooLarge)?;
            if returns_len > MAX_GRAMMAR_STACK {
                return Err(GrammarError::RuntimeStateTooLarge);
            }
            let mut returns = Vec::with_capacity(returns_len);
            for _ in 0..returns_len {
                returns.push(
                    usize::try_from(cursor.u32()?)
                        .map_err(|_| GrammarError::RuntimeStateTooLarge)?,
                );
            }
            let path = ActivePath {
                state,
                repeats,
                returns,
            };
            validate_active_path(self, &path, &repeat_limits)?;
            if !seen.insert(path.clone()) {
                return Err(GrammarError::RuntimeStateMalformed);
            }
            active.push(path);
        }
        cursor.finish()?;
        Ok(GrammarState {
            grammar: self.clone(),
            active,
            utf8,
            accepted_bytes,
        })
    }
}

fn hash_edge(digest: &mut Sha256, edge: &Edge) {
    match edge {
        Edge::Epsilon(target) => {
            digest.update([0]);
            digest.update((*target as u64).to_le_bytes());
        }
        Edge::Bytes(set, target) => {
            digest.update([1]);
            for word in set.0 {
                digest.update(word.to_le_bytes());
            }
            digest.update((*target as u64).to_le_bytes());
        }
        Edge::Call {
            target,
            return_state,
        } => {
            digest.update([2]);
            digest.update((*target as u64).to_le_bytes());
            digest.update((*return_state as u64).to_le_bytes());
        }
        Edge::Return => digest.update([3]),
        Edge::RepeatEnter { target, id } => {
            digest.update([4]);
            digest.update((*target as u64).to_le_bytes());
            digest.update((*id as u64).to_le_bytes());
        }
        Edge::RepeatExit {
            target,
            body_start,
            id,
            min,
            max,
        } => {
            digest.update([5]);
            digest.update((*target as u64).to_le_bytes());
            digest.update((*body_start as u64).to_le_bytes());
            digest.update((*id as u64).to_le_bytes());
            digest.update((*min as u64).to_le_bytes());
            digest.update((*max as u64).to_le_bytes());
        }
    }
}

fn grammar_repeat_limits(grammar: &CompiledGrammar) -> BTreeMap<usize, usize> {
    let mut limits = BTreeMap::new();
    for state in &grammar.states {
        for edge in &state.edges {
            if let Edge::RepeatExit { id, max, .. } = edge {
                limits.insert(*id, *max);
            }
        }
    }
    limits
}

fn validate_active_path(
    grammar: &CompiledGrammar,
    path: &ActivePath,
    repeat_limits: &BTreeMap<usize, usize>,
) -> Result<(), GrammarError> {
    if path.state >= grammar.states.len()
        || path.returns.len() > MAX_GRAMMAR_STACK
        || path.repeats.len() > MAX_GRAMMAR_STACK
        || path
            .returns
            .iter()
            .any(|&state| state >= grammar.states.len())
    {
        return Err(GrammarError::RuntimeStateMalformed);
    }
    for (&id, &count) in &path.repeats {
        let Some(&max) = repeat_limits.get(&id) else {
            return Err(GrammarError::RuntimeStateMalformed);
        };
        if count > max || count > MAX_GRAMMAR_REPEAT {
            return Err(GrammarError::RuntimeStateMalformed);
        }
    }
    Ok(())
}

fn grammar_snapshot_size(active: &[ActivePath], state_count: usize) -> Result<usize, GrammarError> {
    if active.len() > MAX_GRAMMAR_ACTIVE_STATES {
        return Err(GrammarError::RuntimeStateTooLarge);
    }
    let mut size = 8usize
        .checked_add(2 + 2 + 32 + 8 + 4 + 4)
        .ok_or(GrammarError::RuntimeStateTooLarge)?;
    for path in active {
        if path.state >= state_count
            || path.repeats.len() > MAX_GRAMMAR_STACK
            || path.returns.len() > MAX_GRAMMAR_STACK
        {
            return Err(GrammarError::RuntimeStateMalformed);
        }
        size = size
            .checked_add(8)
            .and_then(|size| size.checked_add(path.repeats.len().checked_mul(8)?))
            .and_then(|size| size.checked_add(4))
            .and_then(|size| size.checked_add(path.returns.len().checked_mul(4)?))
            .ok_or(GrammarError::RuntimeStateTooLarge)?;
        if size > MAX_GRAMMAR_RUNTIME_STATE_BYTES {
            return Err(GrammarError::RuntimeStateTooLarge);
        }
    }
    Ok(size)
}

fn encode_utf8_state(state: &Utf8State, bytes: &mut Vec<u8>) {
    match state {
        Utf8State::Complete => bytes.extend_from_slice(&[0, 0, 0, 0]),
        Utf8State::Partial { expected, seen } => bytes.extend_from_slice(&[1, *expected, *seen, 0]),
    }
}

fn decode_utf8_state(bytes: &[u8]) -> Result<Utf8State, GrammarError> {
    if bytes.len() != 4 || bytes[3] != 0 {
        return Err(GrammarError::RuntimeStateMalformed);
    }
    match bytes[0] {
        0 if bytes[1] == 0 && bytes[2] == 0 => Ok(Utf8State::Complete),
        1 if (2..=4).contains(&bytes[1]) && (1..bytes[1]).contains(&bytes[2]) => {
            Ok(Utf8State::Partial {
                expected: bytes[1],
                seen: bytes[2],
            })
        }
        _ => Err(GrammarError::RuntimeStateMalformed),
    }
}

struct GrammarRuntimeCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> GrammarRuntimeCursor<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, GrammarError> {
        if bytes.len() > MAX_GRAMMAR_RUNTIME_STATE_BYTES {
            return Err(GrammarError::RuntimeStateTooLarge);
        }
        if bytes.len() < GRAMMAR_RUNTIME_MAGIC.len()
            || bytes[..GRAMMAR_RUNTIME_MAGIC.len()] != GRAMMAR_RUNTIME_MAGIC
        {
            return Err(GrammarError::RuntimeStateMalformed);
        }
        Ok(Self {
            bytes,
            offset: GRAMMAR_RUNTIME_MAGIC.len(),
        })
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], GrammarError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(GrammarError::RuntimeStateTooLarge)?;
        if end > self.bytes.len() {
            return Err(GrammarError::RuntimeStateMalformed);
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u16(&mut self) -> Result<u16, GrammarError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, GrammarError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, GrammarError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn finish(self) -> Result<(), GrammarError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(GrammarError::RuntimeStateMalformed)
        }
    }
}

#[derive(Clone, Debug)]
struct TrieNode {
    edges: BTreeMap<u8, usize>,
    token_ids: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TokenTrie {
    nodes: Vec<TrieNode>,
    token_count: usize,
}

impl TokenTrie {
    pub fn new<I, B>(pieces: I) -> Result<Self, GrammarError>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let pieces: Vec<Vec<u8>> = pieces
            .into_iter()
            .map(|piece| piece.as_ref().to_vec())
            .collect();
        Self::new_indexed(
            pieces.len(),
            pieces
                .into_iter()
                .enumerate()
                .map(|(token_id, piece)| (token_id, Some(piece))),
        )
    }

    /// Builds a vocabulary-sized trie while retaining rows for tokens that do
    /// not have a grammar-visible byte piece (special/reserved tokenizer rows).
    /// Such rows remain `false` in the returned mask rather than shifting later
    /// token IDs.
    pub fn new_optional<I, B>(pieces: I) -> Result<Self, GrammarError>
    where
        I: IntoIterator<Item = Option<B>>,
        B: AsRef<[u8]>,
    {
        let pieces: Vec<Option<Vec<u8>>> = pieces
            .into_iter()
            .map(|piece| piece.map(|piece| piece.as_ref().to_vec()))
            .collect();
        Self::new_indexed(pieces.len(), pieces.into_iter().enumerate())
    }

    /// Builds a trie from sparse, explicitly indexed token rows.  Missing IDs
    /// are retained in the mask and can never become valid by accident.
    pub fn new_indexed<I, B>(token_count: usize, pieces: I) -> Result<Self, GrammarError>
    where
        I: IntoIterator<Item = (usize, Option<B>)>,
        B: AsRef<[u8]>,
    {
        let mut trie = Self {
            nodes: vec![TrieNode {
                edges: BTreeMap::new(),
                token_ids: Vec::new(),
            }],
            token_count,
        };
        for (token_id, piece) in pieces {
            if token_id >= token_count {
                return Err(GrammarError::InvalidTokenId {
                    token_id,
                    token_count,
                });
            }
            let Some(piece) = piece else { continue };
            let piece = piece.as_ref();
            if piece.is_empty() {
                return Err(GrammarError::EmptyTokenPiece);
            }
            if piece.len() > MAX_TOKEN_PIECE_BYTES {
                return Err(GrammarError::TokenPieceTooLarge {
                    limit: MAX_TOKEN_PIECE_BYTES,
                });
            }
            let mut node = 0;
            for &byte in piece {
                let next = if let Some(&next) = trie.nodes[node].edges.get(&byte) {
                    next
                } else {
                    if trie.nodes.len() >= MAX_TOKEN_TRIE_NODES {
                        return Err(GrammarError::TokenTrieLimit {
                            limit: MAX_TOKEN_TRIE_NODES,
                        });
                    }
                    let next = trie.nodes.len();
                    trie.nodes.push(TrieNode {
                        edges: BTreeMap::new(),
                        token_ids: Vec::new(),
                    });
                    trie.nodes[node].edges.insert(byte, next);
                    next
                };
                node = next;
            }
            trie.nodes[node].token_ids.push(token_id);
        }
        Ok(trie)
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    fn mark_valid(&self, state: &GrammarState, mask: &mut [bool]) -> Result<(), GrammarError> {
        let mut stack = vec![(0_usize, state.active.clone(), state.utf8.clone())];
        let mut visited = BTreeSet::new();
        while let Some((node, active, utf8)) = stack.pop() {
            if !visited.insert((node, active.clone(), utf8.clone())) {
                continue;
            }
            if visited.len() > MAX_GRAMMAR_ACTIVE_STATES {
                return Err(GrammarError::ActiveStateLimit {
                    limit: MAX_GRAMMAR_ACTIVE_STATES,
                });
            }
            for &token_id in &self.nodes[node].token_ids {
                mask[token_id] = true;
            }
            for (&byte, &child) in &self.nodes[node].edges {
                let next = advance_byte(&state.grammar, &active, byte)?;
                let mut next_utf8 = utf8.clone();
                if next_utf8.push(&[byte], state.accepted_bytes).is_ok() && !next.is_empty() {
                    stack.push((child, next, next_utf8));
                }
            }
        }
        Ok(())
    }
}

struct GrammarParser<'a> {
    source: &'a [u8],
    offset: usize,
    depth: usize,
}

impl<'a> GrammarParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            offset: 0,
            depth: 0,
        }
    }

    fn parse_rules(mut self) -> Result<Vec<Rule>, GrammarError> {
        let mut rules = Vec::new();
        while self.skip_space_and_comments() {
            let start = self.offset;
            let name = self.parse_identifier()?;
            if name.len() > MAX_GRAMMAR_NAME_BYTES {
                return Err(GrammarError::RuleNameTooLong {
                    limit: MAX_GRAMMAR_NAME_BYTES,
                });
            }
            self.skip_inline_space();
            if !self.consume_bytes(b"::=") {
                return Err(self.error("expected ::= after rule name"));
            }
            let expr = self.parse_alternation(true)?;
            if rules.iter().any(|rule: &Rule| rule.name == name) {
                return Err(GrammarError::DuplicateRule(name));
            }
            rules.push(Rule { name, expr });
            if rules.len() > MAX_GRAMMAR_RULES {
                return Err(GrammarError::RuleLimit {
                    limit: MAX_GRAMMAR_RULES,
                });
            }
            if self.offset == start {
                return Err(self.error("parser made no progress"));
            }
            self.skip_space_and_comments();
        }
        if rules.is_empty() {
            return Err(GrammarError::EmptyInput);
        }
        Ok(rules)
    }

    fn parse_alternation(&mut self, line_end: bool) -> Result<Expr, GrammarError> {
        let mut alternatives = vec![self.parse_sequence(line_end)?];
        loop {
            self.skip_inline_space();
            if !self.consume_byte(b'|') {
                break;
            }
            alternatives.push(self.parse_sequence(line_end)?);
            if alternatives.len() > MAX_GRAMMAR_ALTERNATIVES {
                return Err(GrammarError::AlternativeLimit {
                    limit: MAX_GRAMMAR_ALTERNATIVES,
                });
            }
        }
        if alternatives.len() == 1 {
            Ok(alternatives.remove(0))
        } else {
            Ok(Expr::Alternation(alternatives))
        }
    }

    fn parse_sequence(&mut self, line_end: bool) -> Result<Expr, GrammarError> {
        let mut values = Vec::new();
        loop {
            self.skip_inline_space();
            if self.at_end()
                || self.peek_byte() == Some(b'|')
                || (line_end && self.peek_byte() == Some(b'\n'))
                || self.peek_byte() == Some(b')')
            {
                break;
            }
            if self.peek_byte() == Some(b'#') {
                break;
            }
            values.push(self.parse_postfix()?);
        }
        if values.is_empty() {
            Ok(Expr::Empty)
        } else if values.len() == 1 {
            Ok(values.remove(0))
        } else {
            Ok(Expr::Sequence(values))
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, GrammarError> {
        let mut expr = self.parse_primary()?;
        self.skip_inline_space();
        if self.consume_byte(b'?') {
            expr = Expr::Repeat {
                expr: Box::new(expr),
                min: 0,
                max: 1,
            };
        } else if self.consume_byte(b'*') || self.consume_byte(b'+') {
            return Err(GrammarError::UnboundedRepetition);
        } else if self.consume_byte(b'{') {
            self.skip_inline_space();
            let min = self.parse_number()?;
            self.skip_inline_space();
            let max = if self.consume_byte(b',') {
                self.skip_inline_space();
                let value = self.parse_number()?;
                self.skip_inline_space();
                value
            } else {
                min
            };
            if !self.consume_byte(b'}') {
                return Err(self.error("expected } in repetition"));
            }
            if min > max {
                return Err(self.error("repetition minimum exceeds maximum"));
            }
            if max > MAX_GRAMMAR_REPEAT {
                return Err(GrammarError::RepeatLimit {
                    limit: MAX_GRAMMAR_REPEAT,
                });
            }
            expr = Expr::Repeat {
                expr: Box::new(expr),
                min,
                max,
            };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, GrammarError> {
        self.skip_inline_space();
        let byte = self
            .peek_byte()
            .ok_or_else(|| self.error("expected expression"))?;
        match byte {
            b'"' => Ok(Expr::Literal(self.parse_string()?)),
            b'[' => self.parse_class(),
            b'.' => {
                self.offset += 1;
                Ok(Expr::Bytes(ByteSet::any()))
            }
            b'(' => {
                self.offset += 1;
                self.depth += 1;
                if self.depth > MAX_GRAMMAR_NESTING {
                    return Err(GrammarError::NestingLimit {
                        limit: MAX_GRAMMAR_NESTING,
                    });
                }
                let expr = self.parse_alternation(false)?;
                self.skip_inline_space();
                if !self.consume_byte(b')') {
                    return Err(self.error("expected )"));
                }
                self.depth -= 1;
                Ok(expr)
            }
            _ => Ok(Expr::Ref(self.parse_identifier()?)),
        }
    }

    fn parse_class(&mut self) -> Result<Expr, GrammarError> {
        self.offset += 1;
        let negated = self.consume_byte(b'^');
        let mut set = ByteSet::empty();
        let mut saw_value = false;
        while let Some(byte) = self.peek_byte() {
            if byte == b']' {
                self.offset += 1;
                if !saw_value {
                    return Err(self.error("empty byte class"));
                }
                if negated {
                    for word in &mut set.0 {
                        *word = !*word;
                    }
                }
                return Ok(Expr::Bytes(set));
            }
            let start = self.parse_class_byte()?;
            saw_value = true;
            if self.consume_byte(b'-') {
                let end = self.parse_class_byte()?;
                if start > end {
                    return Err(GrammarError::InvalidRange);
                }
                for value in start..=end {
                    set.insert(value);
                }
            } else {
                set.insert(start);
            }
        }
        Err(self.error("unterminated byte class"))
    }

    fn parse_class_byte(&mut self) -> Result<u8, GrammarError> {
        if !self.consume_byte(b'\\') {
            let value = self
                .peek_byte()
                .ok_or_else(|| self.error("expected byte"))?;
            self.offset += 1;
            return Ok(value);
        }
        self.parse_escape_byte()
    }

    fn parse_string(&mut self) -> Result<Vec<u8>, GrammarError> {
        if !self.consume_byte(b'"') {
            return Err(self.error("expected string"));
        }
        let mut value = Vec::new();
        while let Some(byte) = self.peek_byte() {
            self.offset += 1;
            match byte {
                b'"' => return Ok(value),
                b'\\' => value.extend(self.parse_escape_bytes()?),
                0..=0x1f => return Err(self.error("control byte in string")),
                _ => value.push(byte),
            }
        }
        Err(self.error("unterminated string"))
    }

    fn parse_escape_byte(&mut self) -> Result<u8, GrammarError> {
        let bytes = self.parse_escape_bytes()?;
        if bytes.len() == 1 {
            Ok(bytes[0])
        } else {
            Err(GrammarError::InvalidEscape)
        }
    }

    fn parse_escape_bytes(&mut self) -> Result<Vec<u8>, GrammarError> {
        let byte = self
            .peek_byte()
            .ok_or_else(|| self.error("incomplete escape"))?;
        self.offset += 1;
        match byte {
            b'n' => Ok(vec![b'\n']),
            b'r' => Ok(vec![b'\r']),
            b't' => Ok(vec![b'\t']),
            b'b' => Ok(vec![0x08]),
            b'f' => Ok(vec![0x0c]),
            b'"' => Ok(vec![b'"']),
            b'\\' => Ok(vec![b'\\']),
            b'\'' => Ok(vec![b'\'']),
            b'x' => self.parse_hex(2).and_then(|value| {
                u8::try_from(value)
                    .map(|value| vec![value])
                    .map_err(|_| GrammarError::InvalidEscape)
            }),
            b'u' => {
                let value = self.parse_hex(4)?;
                let character = char::from_u32(value).ok_or(GrammarError::InvalidEscape)?;
                let mut bytes = [0; 4];
                Ok(character.encode_utf8(&mut bytes).as_bytes().to_vec())
            }
            _ => Err(GrammarError::InvalidEscape),
        }
    }

    fn parse_hex(&mut self, count: usize) -> Result<u32, GrammarError> {
        let mut value = 0_u32;
        for _ in 0..count {
            let byte = self.peek_byte().ok_or(GrammarError::InvalidEscape)?;
            self.offset += 1;
            value = value
                .checked_mul(16)
                .and_then(|value| value.checked_add(byte_to_hex(byte)?))
                .ok_or(GrammarError::InvalidEscape)?;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<usize, GrammarError> {
        let start = self.offset;
        while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
            self.offset += 1;
        }
        if self.offset == start {
            return Err(self.error("expected decimal repetition bound"));
        }
        std::str::from_utf8(&self.source[start..self.offset])
            .ok()
            .and_then(|text| text.parse().ok())
            .ok_or(GrammarError::RepeatLimit {
                limit: MAX_GRAMMAR_REPEAT,
            })
    }

    fn parse_identifier(&mut self) -> Result<String, GrammarError> {
        let start = self.offset;
        while self
            .peek_byte()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            self.offset += 1;
        }
        if self.offset == start {
            return Err(self.error("expected rule identifier"));
        }
        Ok(String::from_utf8_lossy(&self.source[start..self.offset]).into_owned())
    }

    fn skip_inline_space(&mut self) {
        while self
            .peek_byte()
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
        {
            self.offset += 1;
        }
    }

    fn skip_space_and_comments(&mut self) -> bool {
        loop {
            while self
                .peek_byte()
                .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                self.offset += 1;
            }
            if self.peek_byte() == Some(b'#') {
                while self.peek_byte().is_some_and(|byte| byte != b'\n') {
                    self.offset += 1;
                }
                continue;
            }
            return !self.at_end();
        }
    }

    fn consume_bytes(&mut self, bytes: &[u8]) -> bool {
        if self
            .source
            .get(self.offset..)
            .is_some_and(|tail| tail.starts_with(bytes))
        {
            self.offset += bytes.len();
            true
        } else {
            false
        }
    }
    fn consume_byte(&mut self, byte: u8) -> bool {
        if self.peek_byte() == Some(byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn peek_byte(&self) -> Option<u8> {
        self.source.get(self.offset).copied()
    }
    fn at_end(&self) -> bool {
        self.offset >= self.source.len()
    }
    fn error(&self, message: &str) -> GrammarError {
        GrammarError::Parse {
            offset: self.offset,
            message: message.to_owned(),
        }
    }
}

fn byte_to_hex(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some(u32::from(byte - b'0')),
        b'a'..=b'f' => Some(u32::from(byte - b'a' + 10)),
        b'A'..=b'F' => Some(u32::from(byte - b'A' + 10)),
        _ => None,
    }
}

fn compile_rules(rules: Vec<Rule>) -> Result<CompiledGrammar, GrammarError> {
    let names: BTreeSet<String> = rules.iter().map(|rule| rule.name.clone()).collect();
    let mut refs = BTreeMap::<String, BTreeSet<String>>::new();
    for rule in &rules {
        let mut outgoing = BTreeSet::new();
        collect_refs(&rule.expr, &mut outgoing);
        for reference in &outgoing {
            if !names.contains(reference) {
                return Err(GrammarError::UnknownRule(reference.clone()));
            }
        }
        refs.insert(rule.name.clone(), outgoing);
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for rule in &rules {
        detect_left_recursion(&rule.name, &refs, &mut visiting, &mut visited)?;
    }
    let root_name = rules
        .iter()
        .find(|rule| rule.name == "root")
        .map(|rule| rule.name.clone())
        .or_else(|| rules.first().map(|rule| rule.name.clone()))
        .ok_or(GrammarError::EmptyInput)?;
    let mut compiler = Compiler {
        rules: rules
            .into_iter()
            .map(|rule| (rule.name, rule.expr))
            .collect(),
        states: Vec::new(),
        stack_depth: 0,
        next_repeat_id: 0,
        rule_fragments: BTreeMap::new(),
    };
    let start = compiler.new_state()?;
    let accept = compiler.new_state()?;
    let (root_start, _) = compiler.compile_rule(&root_name)?;
    compiler.add_edge(
        start,
        Edge::Call {
            target: root_start,
            return_state: accept,
        },
    )?;
    compiler.states[accept].accept = true;
    Ok(CompiledGrammar {
        states: compiler.states,
        start,
        accept,
    })
}

fn collect_refs(expr: &Expr, refs: &mut BTreeSet<String>) {
    match expr {
        Expr::Ref(name) => {
            refs.insert(name.clone());
        }
        Expr::Sequence(values) | Expr::Alternation(values) => {
            for value in values {
                collect_refs(value, refs);
            }
        }
        Expr::Repeat { expr, .. } => collect_refs(expr, refs),
        Expr::Empty | Expr::Literal(_) | Expr::Bytes(_) => {}
    }
}

fn detect_left_recursion(
    name: &str,
    refs: &BTreeMap<String, BTreeSet<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), GrammarError> {
    if !visiting.insert(name.to_owned()) {
        return Err(GrammarError::LeftRecursion(name.to_owned()));
    }
    if !visited.contains(name) {
        if let Some(outgoing) = refs.get(name) {
            for child in outgoing {
                if child == name {
                    return Err(GrammarError::LeftRecursion(name.to_owned()));
                }
                if !visited.contains(child) {
                    detect_left_recursion(child, refs, visiting, visited)?;
                }
            }
        }
        visited.insert(name.to_owned());
    }
    visiting.remove(name);
    Ok(())
}

struct Fragment {
    start: usize,
    end: usize,
}

struct Compiler {
    rules: BTreeMap<String, Expr>,
    states: Vec<NfaState>,
    stack_depth: usize,
    next_repeat_id: usize,
    rule_fragments: BTreeMap<String, (usize, usize)>,
}

impl Compiler {
    fn new_state(&mut self) -> Result<usize, GrammarError> {
        if self.states.len() >= MAX_GRAMMAR_ACTIVE_STATES {
            return Err(GrammarError::CompileLimit {
                limit: MAX_GRAMMAR_ACTIVE_STATES,
            });
        }
        let index = self.states.len();
        self.states.push(NfaState::default());
        Ok(index)
    }
    fn add_edge(&mut self, from: usize, edge: Edge) -> Result<(), GrammarError> {
        self.states
            .get_mut(from)
            .ok_or(GrammarError::CompileLimit {
                limit: MAX_GRAMMAR_ACTIVE_STATES,
            })?
            .edges
            .push(edge);
        Ok(())
    }
    fn compile_ref(&mut self, name: &str) -> Result<Fragment, GrammarError> {
        let (target, _) = self.compile_rule(name)?;
        let start = self.new_state()?;
        let end = self.new_state()?;
        self.add_edge(
            start,
            Edge::Call {
                target,
                return_state: end,
            },
        )?;
        Ok(Fragment { start, end })
    }

    fn compile_rule(&mut self, name: &str) -> Result<(usize, usize), GrammarError> {
        if let Some(&fragment) = self.rule_fragments.get(name) {
            return Ok(fragment);
        }
        self.stack_depth += 1;
        if self.stack_depth > MAX_GRAMMAR_STACK {
            return Err(GrammarError::StackLimit {
                limit: MAX_GRAMMAR_STACK,
            });
        }
        let expr = self
            .rules
            .get(name)
            .ok_or_else(|| GrammarError::UnknownRule(name.to_owned()))?
            .clone();
        let start = self.new_state()?;
        let end = self.new_state()?;
        self.rule_fragments.insert(name.to_owned(), (start, end));
        let result = self.compile_expr(&expr);
        self.stack_depth -= 1;
        let body = result?;
        self.add_edge(start, Edge::Epsilon(body.start))?;
        self.add_edge(body.end, Edge::Epsilon(end))?;
        self.add_edge(end, Edge::Return)?;
        Ok((start, end))
    }
    fn compile_expr(&mut self, expr: &Expr) -> Result<Fragment, GrammarError> {
        match expr {
            Expr::Empty => {
                let start = self.new_state()?;
                let end = self.new_state()?;
                self.add_edge(start, Edge::Epsilon(end))?;
                Ok(Fragment { start, end })
            }
            Expr::Literal(bytes) => {
                let start = self.new_state()?;
                let mut current = start;
                for &byte in bytes {
                    let next = self.new_state()?;
                    self.add_edge(current, Edge::Bytes(ByteSet::one(byte), next))?;
                    current = next;
                }
                Ok(Fragment {
                    start,
                    end: current,
                })
            }
            Expr::Bytes(set) => {
                let start = self.new_state()?;
                let end = self.new_state()?;
                self.add_edge(start, Edge::Bytes(*set, end))?;
                Ok(Fragment { start, end })
            }
            Expr::Ref(name) => self.compile_ref(name),
            Expr::Sequence(values) => {
                let mut iterator = values.iter();
                let first = if let Some(expr) = iterator.next() {
                    self.compile_expr(expr)?
                } else {
                    self.compile_expr(&Expr::Empty)?
                };
                let mut end = first.end;
                for expr in iterator {
                    let next = self.compile_expr(expr)?;
                    self.add_edge(end, Edge::Epsilon(next.start))?;
                    end = next.end;
                }
                Ok(Fragment {
                    start: first.start,
                    end,
                })
            }
            Expr::Alternation(values) => {
                let start = self.new_state()?;
                let end = self.new_state()?;
                for expr in values {
                    let branch = self.compile_expr(expr)?;
                    self.add_edge(start, Edge::Epsilon(branch.start))?;
                    self.add_edge(branch.end, Edge::Epsilon(end))?;
                }
                Ok(Fragment { start, end })
            }
            Expr::Repeat { expr, min, max } => {
                let start = self.new_state()?;
                let end = self.new_state()?;
                if *max == 0 {
                    self.add_edge(start, Edge::Epsilon(end))?;
                    return Ok(Fragment { start, end });
                }
                let body = self.compile_expr(expr)?;
                let id = self.next_repeat_id;
                self.next_repeat_id =
                    self.next_repeat_id
                        .checked_add(1)
                        .ok_or(GrammarError::CompileLimit {
                            limit: MAX_GRAMMAR_ACTIVE_STATES,
                        })?;
                if *min == 0 {
                    self.add_edge(start, Edge::Epsilon(end))?;
                }
                self.add_edge(
                    start,
                    Edge::RepeatEnter {
                        target: body.start,
                        id,
                    },
                )?;
                self.add_edge(
                    body.end,
                    Edge::RepeatExit {
                        target: end,
                        body_start: body.start,
                        id,
                        min: *min,
                        max: *max,
                    },
                )?;
                Ok(Fragment { start, end })
            }
        }
    }
}

fn epsilon_closure<I>(grammar: &CompiledGrammar, initial: I, output: &mut Vec<ActivePath>)
where
    I: IntoIterator<Item = ActivePath>,
{
    output.clear();
    let mut queue = VecDeque::new();
    let mut seen = BTreeSet::new();
    for path in initial {
        if seen.insert(path.clone()) {
            queue.push_back(path);
        }
    }
    while let Some(path) = queue.pop_front() {
        output.push(path.clone());
        if output.len() > MAX_GRAMMAR_ACTIVE_STATES {
            break;
        }
        for edge in &grammar.states[path.state].edges {
            let next_paths = match edge {
                Edge::Epsilon(next) => vec![ActivePath {
                    state: *next,
                    repeats: path.repeats.clone(),
                    returns: path.returns.clone(),
                }],
                Edge::Call {
                    target,
                    return_state,
                } => {
                    if path.returns.len() >= MAX_GRAMMAR_STACK {
                        Vec::new()
                    } else {
                        let mut returns = path.returns.clone();
                        returns.push(*return_state);
                        vec![ActivePath {
                            state: *target,
                            repeats: path.repeats.clone(),
                            returns,
                        }]
                    }
                }
                Edge::Return => path.returns.last().map_or_else(Vec::new, |return_state| {
                    let mut returns = path.returns.clone();
                    returns.pop();
                    vec![ActivePath {
                        state: *return_state,
                        repeats: path.repeats.clone(),
                        returns,
                    }]
                }),
                Edge::RepeatEnter { target, id, .. } => {
                    let mut repeats = path.repeats.clone();
                    repeats.insert(*id, 0);
                    vec![ActivePath {
                        state: *target,
                        repeats,
                        returns: path.returns.clone(),
                    }]
                }
                Edge::RepeatExit {
                    target,
                    body_start,
                    id,
                    min,
                    max,
                } => {
                    let count = path.repeats.get(id).copied().unwrap_or(0).saturating_add(1);
                    let mut paths = Vec::new();
                    if count >= *min {
                        let mut repeats = path.repeats.clone();
                        repeats.remove(id);
                        paths.push(ActivePath {
                            state: *target,
                            repeats,
                            returns: path.returns.clone(),
                        });
                    }
                    if count < *max {
                        let mut repeats = path.repeats.clone();
                        repeats.insert(*id, count);
                        paths.push(ActivePath {
                            state: *body_start,
                            repeats,
                            returns: path.returns.clone(),
                        });
                    }
                    paths
                }
                Edge::Bytes(_, _) => Vec::new(),
            };
            for next in next_paths {
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
    }
    output.sort_unstable();
}

fn advance(
    grammar: &CompiledGrammar,
    active: &[ActivePath],
    bytes: &[u8],
) -> Result<Vec<ActivePath>, GrammarError> {
    let mut current = active.to_vec();
    for &byte in bytes {
        current = advance_byte(grammar, &current, byte)?;
        if current.is_empty() {
            break;
        }
    }
    Ok(current)
}

fn advance_byte(
    grammar: &CompiledGrammar,
    active: &[ActivePath],
    byte: u8,
) -> Result<Vec<ActivePath>, GrammarError> {
    let mut next = Vec::new();
    for path in active {
        for edge in &grammar.states[path.state].edges {
            if let Edge::Bytes(set, target) = edge {
                if set.contains(byte) {
                    next.push(ActivePath {
                        state: *target,
                        repeats: path.repeats.clone(),
                        returns: path.returns.clone(),
                    });
                }
            }
        }
    }
    let mut closed = Vec::new();
    epsilon_closure(grammar, next, &mut closed);
    if closed.len() > MAX_GRAMMAR_ACTIVE_STATES {
        return Err(GrammarError::ActiveStateLimit {
            limit: MAX_GRAMMAR_ACTIVE_STATES,
        });
    }
    Ok(closed)
}

// Generic json_object mode remains deliberately bounded, but must be useful
// for real responses rather than accepting only toy three-field objects.
// JSON Schema mode can express larger fixed objects under the global 1024
// property cap; this generic lane permits four members/items per container.
const JSON_REPEAT_LIMIT: usize = 3;

fn bounded_json_grammar_source(max_depth: usize) -> String {
    let mut source = String::from("root ::= ws json_object_0 ws\n");
    for depth in 0..=max_depth {
        source.push_str(&format!(
            "json_object_{depth} ::= \"{{\" ws json_members_{depth}? ws \"}}\"\n"
        ));
        source.push_str(&format!(
            "json_members_{depth} ::= json_member_{depth} (ws \",\" ws json_member_{depth}){{0,{JSON_REPEAT_LIMIT}}}\n"
        ));
        source.push_str(&format!(
            "json_member_{depth} ::= json_string ws \":\" ws json_value_{depth}\n"
        ));
        source.push_str(&format!(
            "json_array_{depth} ::= \"[\" ws (json_value_{depth} (ws \",\" ws json_value_{depth}){{0,{JSON_REPEAT_LIMIT}}})? ws \"]\"\n"
        ));
        if depth == max_depth {
            source.push_str(&format!(
                "json_value_{depth} ::= json_string | json_number | \"true\" | \"false\" | \"null\"\n"
            ));
        } else {
            source.push_str(&format!(
                "json_value_{depth} ::= json_string | json_number | json_object_{} | json_array_{} | \"true\" | \"false\" | \"null\"\n",
                depth + 1,
                depth + 1
            ));
        }
    }
    source.push_str(JSON_OBJECT_COMMON_RULES);
    source
}

pub struct JsonSchemaLowerer<'a> {
    schema: &'a Value,
    defs: Map<String, Value>,
    property_count: usize,
}

impl<'a> JsonSchemaLowerer<'a> {
    pub fn new(schema: &'a Value) -> Result<Self, GrammarError> {
        let bytes = serde_json::to_vec(schema)
            .map_err(|error| GrammarError::InvalidSchema(error.to_string()))?;
        if bytes.len() > MAX_GRAMMAR_BYTES {
            return Err(GrammarError::JsonTooLarge {
                limit: MAX_GRAMMAR_BYTES,
            });
        }
        let object = schema.as_object().ok_or(GrammarError::JsonNotObject)?;
        let defs = match object.get("$defs") {
            Some(Value::Object(values)) => values.clone(),
            Some(_) => {
                return Err(GrammarError::InvalidSchema(
                    "$defs must be an object".to_owned(),
                ));
            }
            None => Map::new(),
        };
        for key in object.keys() {
            if key != "$defs" {
                validate_schema_keyword(key)?;
            }
        }
        Ok(Self {
            schema,
            defs,
            property_count: 0,
        })
    }

    pub fn compile(mut self) -> Result<CompiledGrammar, GrammarError> {
        let source = self.lower_schema(self.schema, 0, &mut BTreeSet::new())?;
        CompiledGrammar::compile(&format!("root ::= ws {source} ws\n{}", JSON_COMMON_RULES))
    }

    fn lower_schema(
        &mut self,
        schema: &Value,
        depth: usize,
        refs: &mut BTreeSet<String>,
    ) -> Result<String, GrammarError> {
        if depth > MAX_GRAMMAR_NESTING {
            return Err(GrammarError::JsonDepthLimit {
                limit: MAX_GRAMMAR_NESTING,
            });
        }
        let object = schema.as_object().ok_or_else(|| {
            GrammarError::InvalidSchema("schema node must be an object".to_owned())
        })?;
        for key in object.keys() {
            if key != "$defs" {
                validate_schema_keyword(key)?;
            }
        }
        if let Some(reference) = object.get("$ref") {
            let reference = reference
                .as_str()
                .ok_or_else(|| GrammarError::InvalidSchema("$ref must be a string".to_owned()))?;
            if !reference.starts_with("#/$defs/") {
                return Err(GrammarError::RemoteReference(reference.to_owned()));
            }
            let name = &reference[8..];
            if name.is_empty() {
                return Err(GrammarError::InvalidSchema("empty local $ref".to_owned()));
            }
            if !refs.insert(name.to_owned()) {
                return Err(GrammarError::RecursiveReference(reference.to_owned()));
            }
            let value =
                self.defs.get(name).cloned().ok_or_else(|| {
                    GrammarError::InvalidSchema(format!("unknown $ref {reference}"))
                })?;
            let result = self.lower_schema(&value, depth + 1, refs);
            refs.remove(name);
            return result;
        }
        if let Some(enum_values) = object.get("enum") {
            let values = enum_values
                .as_array()
                .ok_or_else(|| GrammarError::InvalidSchema("enum must be an array".to_owned()))?;
            if values.is_empty() || values.len() > MAX_JSON_ENUM {
                return Err(GrammarError::JsonEnumLimit {
                    limit: MAX_JSON_ENUM,
                });
            }
            let mut literals = Vec::with_capacity(values.len());
            for value in values {
                literals.push(json_literal(value)?);
            }
            return Ok(format!("({})", literals.join(" | ")));
        }
        if let Some(constant) = object.get("const") {
            return json_literal(constant);
        }
        if let Some(any_of) = object.get("anyOf") {
            let values = any_of
                .as_array()
                .ok_or_else(|| GrammarError::InvalidSchema("anyOf must be an array".to_owned()))?;
            if values.is_empty() || values.len() > MAX_GRAMMAR_ALTERNATIVES {
                return Err(GrammarError::AlternativeLimit {
                    limit: MAX_GRAMMAR_ALTERNATIVES,
                });
            }
            let mut branches = Vec::with_capacity(values.len());
            for value in values {
                branches.push(self.lower_schema(value, depth + 1, refs)?);
            }
            return Ok(format!("({})", branches.join(" | ")));
        }
        let type_name = object.get("type").and_then(Value::as_str).ok_or_else(|| {
            GrammarError::InvalidSchema("schema requires a supported string type".to_owned())
        })?;
        match type_name {
            "object" => self.lower_object(object, depth, refs),
            "array" => self.lower_array(object, depth, refs),
            "string" => Ok("json_string".to_owned()),
            "number" => Ok("json_number".to_owned()),
            "integer" => Ok("json_integer".to_owned()),
            "boolean" => Ok("(\"true\" | \"false\")".to_owned()),
            "null" => Ok("\"null\"".to_owned()),
            other => Err(GrammarError::InvalidSchema(format!(
                "unsupported type {other}"
            ))),
        }
    }

    fn lower_object(
        &mut self,
        object: &Map<String, Value>,
        depth: usize,
        refs: &mut BTreeSet<String>,
    ) -> Result<String, GrammarError> {
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| GrammarError::InvalidSchema("object requires properties".to_owned()))?;
        self.property_count = self.property_count.checked_add(properties.len()).ok_or(
            GrammarError::JsonPropertyLimit {
                limit: MAX_JSON_PROPERTIES,
            },
        )?;
        if self.property_count > MAX_JSON_PROPERTIES {
            return Err(GrammarError::JsonPropertyLimit {
                limit: MAX_JSON_PROPERTIES,
            });
        }
        if object.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(GrammarError::UnsupportedSchemaKeyword(
                "additionalProperties (must be false)".to_owned(),
            ));
        }
        let required: BTreeSet<String> = match object.get("required") {
            Some(Value::Array(values)) => values
                .iter()
                .map(|value| {
                    value.as_str().map(str::to_owned).ok_or_else(|| {
                        GrammarError::InvalidSchema("required entries must be strings".to_owned())
                    })
                })
                .collect::<Result<_, _>>()?,
            Some(_) => {
                return Err(GrammarError::InvalidSchema(
                    "required must be an array".to_owned(),
                ));
            }
            None => BTreeSet::new(),
        };
        if required.iter().any(|name| !properties.contains_key(name)) {
            return Err(GrammarError::InvalidSchema(
                "required property is not declared".to_owned(),
            ));
        }
        let mut members = Vec::new();
        for (name, value) in properties {
            let key = json_literal(&Value::String(name.clone()))?;
            let lowered = self.lower_schema(value, depth + 1, refs)?;
            members.push((
                format!("{key} ws \":\" ws {lowered}"),
                required.contains(name),
            ));
        }
        let mut branches = Vec::new();
        if required.is_empty() {
            branches.push(String::new());
        }
        for first in 0..members.len() {
            if members[..first].iter().any(|(_, required)| *required) {
                continue;
            }
            let mut branch = members[first].0.clone();
            for (member, is_required) in members.iter().skip(first + 1) {
                if *is_required {
                    branch.push_str(" ws \",\" ws ");
                    branch.push_str(member);
                } else {
                    branch.push_str(" (ws \",\" ws ");
                    branch.push_str(member);
                    branch.push_str(")?");
                }
            }
            branches.push(branch);
        }
        if branches.is_empty() {
            return Err(GrammarError::InvalidSchema(
                "object has no satisfiable property order".to_owned(),
            ));
        }
        let contents = if branches.len() == 1 {
            branches.remove(0)
        } else {
            format!("({})", branches.join(" | "))
        };
        Ok(format!("\"{{\" ws {contents} ws \"}}\""))
    }

    fn lower_array(
        &mut self,
        object: &Map<String, Value>,
        depth: usize,
        refs: &mut BTreeSet<String>,
    ) -> Result<String, GrammarError> {
        let items = object
            .get("items")
            .ok_or_else(|| GrammarError::InvalidSchema("array requires items".to_owned()))?;
        let lowered = self.lower_schema(items, depth + 1, refs)?;
        Ok(format!(
            "\"[\" ws ({lowered} (ws \",\" ws {lowered}){{0,{JSON_REPEAT_LIMIT}}})? ws \"]\""
        ))
    }
}

fn validate_schema_keyword(keyword: &str) -> Result<(), GrammarError> {
    const SUPPORTED: &[&str] = &[
        "$ref",
        "$defs",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
    ];
    if SUPPORTED.contains(&keyword) {
        Ok(())
    } else {
        Err(GrammarError::UnsupportedSchemaKeyword(keyword.to_owned()))
    }
}

fn json_literal(value: &Value) -> Result<String, GrammarError> {
    let serialized = serde_json::to_string(value)
        .map_err(|error| GrammarError::InvalidSchema(error.to_string()))?;
    Ok(gbnf_literal(serialized.as_bytes()))
}

fn gbnf_literal(bytes: &[u8]) -> String {
    let mut literal = String::with_capacity(bytes.len() + 2);
    literal.push('"');
    for &byte in bytes {
        match byte {
            b'"' => literal.push_str("\\\""),
            b'\\' => literal.push_str("\\\\"),
            b'\n' => literal.push_str("\\n"),
            b'\r' => literal.push_str("\\r"),
            b'\t' => literal.push_str("\\t"),
            0x20..=0x7e => literal.push(char::from(byte)),
            _ => literal.push_str(&format!("\\x{byte:02x}")),
        }
    }
    literal.push('"');
    literal
}

const JSON_COMMON_RULES: &str = r#"
json_string ::= "\"" json_chars "\""
json_chars ::= [^"\\\x00-\x1f]{0,4096} ("\\" json_escape [^\x00-\x1f]){0,4096}
json_escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F]{4}
json_number ::= "-"? json_integer json_fraction? json_exponent?
json_integer ::= "0" | [1-9][0-9]{0,4096}
json_fraction ::= "." [0-9]{1,4096}
json_exponent ::= [eE] [+-]? [0-9]{1,4096}
json_fractional_integer ::= "-"? ("0" | [1-9][0-9]{0,4096})
ws ::= [ \t\r\n]{0,4096}
"#;

const JSON_OBJECT_COMMON_RULES: &str = r#"
json_string ::= "\"" json_chars "\""
json_chars ::= [^"\\\x00-\x1f]{0,64} ("\\" json_escape [^\x00-\x1f]){0,64}
json_escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F]{4}
json_number ::= "-"? json_integer json_fraction? json_exponent?
json_integer ::= "0" | [1-9][0-9]{0,64}
json_fraction ::= "." [0-9]{1,64}
json_exponent ::= [eE] [+-]? [0-9]{1,64}
json_fractional_integer ::= "-"? ("0" | [1-9][0-9]{0,64})
ws ::= [ \t\r\n]{0,16}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_literal_and_mask() {
        let grammar = CompiledGrammar::compile("root ::= (\"ab\" | [0-9]{2})").unwrap();
        let state = grammar.initial_state();
        let mask = state
            .valid_token_mask(&[b"a".as_slice(), b"b", b"ab", b"12", b"x"])
            .unwrap();
        assert_eq!(mask, [true, false, true, true, false]);
    }

    #[test]
    fn partial_utf8_is_retained() {
        let grammar = CompiledGrammar::compile("root ::= \"é\"").unwrap();
        let mut state = grammar.initial_state();
        state.accept(&[0xc3]).unwrap();
        assert_eq!(
            state.utf8_state(),
            &Utf8State::Partial {
                expected: 2,
                seen: 1
            }
        );
        state.accept(&[0xa9]).unwrap();
        assert!(state.is_finished());
    }

    #[test]
    fn json_schema_rejects_unbounded_keywords() {
        let schema = serde_json::json!({"type":"string", "pattern":"x"});
        assert!(
            matches!(CompiledGrammar::from_json_schema(&schema), Err(GrammarError::UnsupportedSchemaKeyword(keyword)) if keyword == "pattern")
        );
    }

    #[test]
    fn json_object_compiles() {
        let grammar = CompiledGrammar::json_object().expect("bounded JSON grammar");
        assert!(grammar.state_count() > 0);
        for (case, value) in [
            br#"{"message":"structured output","items":[1,2,3,4]}"#.as_slice(),
            br#"{"a":{"b":1}}"#,
            br#"{"a":[true,null,-12.5]}"#,
        ]
        .into_iter()
        .enumerate()
        {
            let mut state = grammar.initial_state();
            state
                .accept(value)
                .unwrap_or_else(|error| panic!("bounded nested JSON case {case}: {error}"));
            assert!(state.is_finished());
        }
    }

    #[test]
    fn reused_rules_do_not_jump_between_call_sites() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "boolean"}},
            "required": ["x"],
            "additionalProperties": false
        });
        let grammar = CompiledGrammar::from_json_schema(&schema).expect("schema compiles");
        let mut state = grammar.initial_state();
        state
            .accept(b" {")
            .expect("leading whitespace and object start");

        assert_eq!(state.accept(b" {"), Err(GrammarError::TokenRejected));
    }

    #[test]
    fn repeated_rule_calls_keep_their_caller_and_counter() {
        let grammar = CompiledGrammar::compile(
            "root ::= value (ws \",\" ws value){0,3}\nvalue ::= \"a\" | \"b\" | object\nobject ::= \"{\" \"c\" \"}\"\nws ::= [ ]{0,4}\n",
        )
        .expect("grammar compiles");
        let mut state = grammar.initial_state();
        state
            .accept(b"a,b,{c}")
            .expect("three repeated values with a nested rule");
        assert!(state.is_finished());
    }

    #[test]
    fn grammar_snapshot_roundtrip_preserves_utf8_and_active_paths() {
        let grammar = CompiledGrammar::compile("root ::= \"é\"").unwrap();
        let mut state = grammar.initial_state();
        state.accept(&[0xc3]).unwrap();
        let snapshot = state.snapshot().unwrap();
        let mut restored = grammar.restore_state(&snapshot).unwrap();
        assert_eq!(restored.accepted_bytes(), state.accepted_bytes());
        assert_eq!(restored.utf8_state(), state.utf8_state());
        assert_eq!(restored.active_state_count(), state.active_state_count());
        state.accept(&[0xa9]).unwrap();
        restored.accept(&[0xa9]).unwrap();
        assert_eq!(restored.is_finished(), state.is_finished());
        let debug = format!("{restored:?}");
        assert!(debug.contains("GrammarState"));
        assert!(!debug.contains("195"));
    }

    #[test]
    fn grammar_snapshot_rejects_identity_corruption_and_bounds() {
        let grammar = CompiledGrammar::compile("root ::= \"a\"").unwrap();
        let state = grammar.initial_state();
        let snapshot = state.snapshot().unwrap();
        let different = CompiledGrammar::compile("root ::= \"b\"").unwrap();
        assert!(matches!(
            different.restore_state(&snapshot),
            Err(GrammarError::RuntimeStateIdentityMismatch)
        ));
        assert!(matches!(
            grammar.restore_state(&snapshot[..snapshot.len() - 1]),
            Err(GrammarError::RuntimeStateMalformed)
        ));
        let mut unsupported = snapshot.clone();
        unsupported[8] = 2;
        assert!(matches!(
            grammar.restore_state(&unsupported),
            Err(GrammarError::RuntimeStateVersionUnsupported { version: 2 })
        ));
        assert!(matches!(
            grammar.restore_state(&vec![0_u8; MAX_GRAMMAR_RUNTIME_STATE_BYTES + 1]),
            Err(GrammarError::RuntimeStateTooLarge)
        ));
    }

    #[test]
    fn compiled_grammar_identity_is_stable_and_structure_sensitive() {
        let first = CompiledGrammar::compile("root ::= \"a\"").unwrap();
        let same = CompiledGrammar::compile("root ::= \"a\"").unwrap();
        let different = CompiledGrammar::compile("root ::= \"b\"").unwrap();
        assert_eq!(first.identity_digest(), same.identity_digest());
        assert_ne!(first.identity_digest(), different.identity_digest());
    }
}
