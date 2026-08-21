//! Backend-independent embedding pooling and embedding-rerank contracts.
//!
//! The model executors publish token hidden rows as either BF16 bits or F32
//! values.  This module deliberately keeps the post-processing on the host:
//! it defines the profile arithmetic, validates the bounded shape, and
//! produces a finite F32 vector that can be transported by an API layer.

use std::cmp::Ordering;
use std::fmt;

/// The pinned profile implemented by [`EmbeddingPoolV1`].
pub const EMBEDDING_POOL_PROFILE_V1: &str = "embedding-pool-v1";

/// The pinned rerank profile for already pooled, L2-normalized embeddings.
pub const COSINE_EMBEDDING_RERANK_PROFILE_V1: &str = "cosine-embedding-v1";

/// Errors returned while validating or pooling token hidden rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddingPoolError {
    /// A token or hidden dimension was zero.
    EmptyShape,
    /// The `[tokens, hidden]` product overflowed `usize`.
    ShapeOverflow,
    /// The flat row buffer length does not match the declared shape.
    ShapeMismatch {
        tokens: usize,
        hidden: usize,
        values: usize,
    },
    /// An input element was NaN or infinite.
    NonFiniteInput { row: usize, column: usize },
    /// The arithmetic mean was not finite.
    NonFiniteMean,
    /// The mean vector has no finite, positive L2 norm.
    ZeroNorm,
    /// A normalized output element could not be represented as finite F32.
    NonFiniteOutput { index: usize },
    /// A byte-vector allocation would overflow `usize`.
    ByteLengthOverflow,
}

impl fmt::Display for EmbeddingPoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => {
                formatter.write_str("embedding rows must have nonzero tokens and hidden dimensions")
            }
            Self::ShapeOverflow => formatter.write_str("embedding row shape overflows usize"),
            Self::ShapeMismatch {
                tokens,
                hidden,
                values,
            } => write!(
                formatter,
                "embedding row shape [{tokens},{hidden}] requires {expected} values, got {values}",
                expected = tokens.saturating_mul(*hidden)
            ),
            Self::NonFiniteInput { row, column } => {
                write!(
                    formatter,
                    "embedding input at [{row},{column}] is not finite"
                )
            }
            Self::NonFiniteMean => formatter.write_str("embedding mean is not finite"),
            Self::ZeroNorm => formatter.write_str("embedding mean has zero or non-finite L2 norm"),
            Self::NonFiniteOutput { index } => {
                write!(formatter, "embedding output at index {index} is not finite")
            }
            Self::ByteLengthOverflow => {
                formatter.write_str("embedding byte length overflows usize")
            }
        }
    }
}

/// Errors returned by the cosine-embedding reranker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddingRerankError {
    /// The query vector has no dimensions.
    EmptyQuery,
    /// No documents were supplied.
    EmptyDocuments,
    /// A document vector has no dimensions.
    EmptyDocument { index: usize },
    /// A document dimension differs from the query dimension.
    DimensionMismatch {
        index: usize,
        query: usize,
        document: usize,
    },
    /// A query or document element is NaN or infinite.
    NonFiniteInput { index: usize, dimension: usize },
    /// `top_n` is zero or exceeds the document count.  The profile does not
    /// silently clamp caller mistakes.
    InvalidTopN { top_n: usize, documents: usize },
    /// The f64 dot product or its F32 conversion was not finite.
    NonFiniteScore { index: usize },
}

impl fmt::Display for EmbeddingRerankError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => {
                formatter.write_str("cosine rerank requires a nonempty query vector")
            }
            Self::EmptyDocuments => {
                formatter.write_str("cosine rerank requires at least one document")
            }
            Self::EmptyDocument { index } => {
                write!(formatter, "cosine rerank document {index} is empty")
            }
            Self::DimensionMismatch {
                index,
                query,
                document,
            } => write!(
                formatter,
                "cosine rerank document {index} has dimension {document}, query has {query}"
            ),
            Self::NonFiniteInput { index, dimension } => write!(
                formatter,
                "cosine rerank vector {index} has non-finite value at dimension {dimension}"
            ),
            Self::InvalidTopN { top_n, documents } => {
                write!(formatter, "top_n={top_n} must be in 1..={documents}")
            }
            Self::NonFiniteScore { index } => {
                write!(
                    formatter,
                    "cosine rerank score for document {index} is not finite"
                )
            }
        }
    }
}

/// A typed view of one flat `[tokens, hidden]` hidden-state matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmbeddingRowsV1<'a> {
    /// BF16 values represented by their IEEE-754 high 16 bits.
    Bf16 {
        values: &'a [u16],
        tokens: usize,
        hidden: usize,
    },
    /// Native F32 hidden values.
    F32 {
        values: &'a [f32],
        tokens: usize,
        hidden: usize,
    },
}

impl<'a> EmbeddingRowsV1<'a> {
    /// Construct a BF16 row view. Shape validation is deferred to `pool` so
    /// callers can preserve the typed input while reporting one contract
    /// error through the normal processing path.
    pub const fn bf16(values: &'a [u16], tokens: usize, hidden: usize) -> Self {
        Self::Bf16 {
            values,
            tokens,
            hidden,
        }
    }

    /// Construct an F32 row view. Shape validation is deferred to `pool`.
    pub const fn f32(values: &'a [f32], tokens: usize, hidden: usize) -> Self {
        Self::F32 {
            values,
            tokens,
            hidden,
        }
    }

    pub const fn tokens(self) -> usize {
        match self {
            Self::Bf16 { tokens, .. } | Self::F32 { tokens, .. } => tokens,
        }
    }

    pub const fn hidden(self) -> usize {
        match self {
            Self::Bf16 { hidden, .. } | Self::F32 { hidden, .. } => hidden,
        }
    }

    fn values_len(self) -> usize {
        match self {
            Self::Bf16 { values, .. } => values.len(),
            Self::F32 { values, .. } => values.len(),
        }
    }
}

/// A finite F32 embedding vector produced by the v1 pooling profile.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingVectorV1 {
    values: Vec<f32>,
}

impl EmbeddingVectorV1 {
    fn new(values: Vec<f32>) -> Result<Self, EmbeddingPoolError> {
        if values.is_empty() {
            return Err(EmbeddingPoolError::EmptyShape);
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(EmbeddingPoolError::NonFiniteOutput { index });
            }
        }
        Ok(Self { values })
    }

    /// Reconstitutes a finite vector at a trusted transport boundary.
    ///
    /// Runtime backends use this after the pooling profile has already
    /// normalized the vector and the bounded scheduler has preserved it. The
    /// constructor deliberately revalidates emptiness and finiteness; it does
    /// not silently renormalize or alter public embedding bytes.
    pub fn from_finite_f32(values: Vec<f32>) -> Result<Self, EmbeddingPoolError> {
        Self::new(values)
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// Return the vector as little-endian IEEE-754 F32 bytes.  Base64
    /// encoding belongs to the transport layer and is intentionally absent
    /// from `sllm-core`.
    pub fn to_little_endian_bytes(&self) -> Result<Vec<u8>, EmbeddingPoolError> {
        let byte_len = self
            .values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(EmbeddingPoolError::ByteLengthOverflow)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for value in &self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Arithmetic mean followed by L2 normalization, profile v1.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmbeddingPoolV1;

impl EmbeddingPoolV1 {
    pub const fn new() -> Self {
        Self
    }

    /// Pool one typed hidden-state matrix into a finite, L2-normalized F32
    /// embedding.
    pub fn pool(&self, rows: EmbeddingRowsV1<'_>) -> Result<EmbeddingVectorV1, EmbeddingPoolError> {
        let tokens = rows.tokens();
        let hidden = rows.hidden();
        if tokens == 0 || hidden == 0 {
            return Err(EmbeddingPoolError::EmptyShape);
        }
        let expected = tokens
            .checked_mul(hidden)
            .ok_or(EmbeddingPoolError::ShapeOverflow)?;
        if rows.values_len() != expected {
            return Err(EmbeddingPoolError::ShapeMismatch {
                tokens,
                hidden,
                values: rows.values_len(),
            });
        }

        let mut mean = vec![0.0_f64; hidden];
        match rows {
            EmbeddingRowsV1::Bf16 { values, .. } => {
                for (flat_index, bits) in values.iter().copied().enumerate() {
                    let value = f32::from_bits(u32::from(bits) << 16);
                    if !value.is_finite() {
                        return Err(EmbeddingPoolError::NonFiniteInput {
                            row: flat_index / hidden,
                            column: flat_index % hidden,
                        });
                    }
                    mean[flat_index % hidden] += f64::from(value);
                }
            }
            EmbeddingRowsV1::F32 { values, .. } => {
                for (flat_index, value) in values.iter().copied().enumerate() {
                    if !value.is_finite() {
                        return Err(EmbeddingPoolError::NonFiniteInput {
                            row: flat_index / hidden,
                            column: flat_index % hidden,
                        });
                    }
                    mean[flat_index % hidden] += f64::from(value);
                }
            }
        }

        let token_count = tokens as f64;
        for value in &mut mean {
            *value /= token_count;
            if !value.is_finite() {
                return Err(EmbeddingPoolError::NonFiniteMean);
            }
        }

        let squared_norm = mean.iter().try_fold(0.0_f64, |accumulator, value| {
            let next = accumulator + value * value;
            next.is_finite().then_some(next)
        });
        let squared_norm = squared_norm.ok_or(EmbeddingPoolError::ZeroNorm)?;
        let norm = squared_norm.sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(EmbeddingPoolError::ZeroNorm);
        }

        let normalized = mean
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let output = (value / norm) as f32;
                output
                    .is_finite()
                    .then_some(output)
                    .ok_or(EmbeddingPoolError::NonFiniteOutput { index })
            })
            .collect::<Result<Vec<_>, _>>()?;
        EmbeddingVectorV1::new(normalized)
    }

    pub fn pool_bf16(
        &self,
        values: &[u16],
        tokens: usize,
        hidden: usize,
    ) -> Result<EmbeddingVectorV1, EmbeddingPoolError> {
        self.pool(EmbeddingRowsV1::bf16(values, tokens, hidden))
    }

    pub fn pool_f32(
        &self,
        values: &[f32],
        tokens: usize,
        hidden: usize,
    ) -> Result<EmbeddingVectorV1, EmbeddingPoolError> {
        self.pool(EmbeddingRowsV1::f32(values, tokens, hidden))
    }
}

/// One score/index pair returned by the cosine reranker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingRerankResultV1 {
    index: usize,
    relevance_score: f32,
}

impl EmbeddingRerankResultV1 {
    pub const fn index(self) -> usize {
        self.index
    }

    pub const fn relevance_score(self) -> f32 {
        self.relevance_score
    }
}

/// Cosine-embedding-v1 ranking over pooled, L2-normalized vectors.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CosineEmbeddingRerankV1;

impl CosineEmbeddingRerankV1 {
    pub const fn new() -> Self {
        Self
    }

    /// Compute one f64 dot product and return its finite F32 score.
    pub fn score(
        &self,
        query: &EmbeddingVectorV1,
        document: &EmbeddingVectorV1,
    ) -> Result<f32, EmbeddingRerankError> {
        let scores = self.rank(query, std::slice::from_ref(document), None)?;
        Ok(scores[0].relevance_score)
    }

    /// Rank documents by descending score. `top_n` is strict: zero and values
    /// above the document count are rejected rather than silently clamped.
    pub fn rank(
        &self,
        query: &EmbeddingVectorV1,
        documents: &[EmbeddingVectorV1],
        top_n: Option<usize>,
    ) -> Result<Vec<EmbeddingRerankResultV1>, EmbeddingRerankError> {
        validate_vector(query.as_slice(), 0, true)?;
        if documents.is_empty() {
            return Err(EmbeddingRerankError::EmptyDocuments);
        }
        if let Some(top_n) = top_n {
            if top_n == 0 || top_n > documents.len() {
                return Err(EmbeddingRerankError::InvalidTopN {
                    top_n,
                    documents: documents.len(),
                });
            }
        }

        let query_dimension = query.dimension();
        let mut ranked = Vec::with_capacity(documents.len());
        for (index, document) in documents.iter().enumerate() {
            if document.dimension() == 0 {
                return Err(EmbeddingRerankError::EmptyDocument { index });
            }
            if document.dimension() != query_dimension {
                return Err(EmbeddingRerankError::DimensionMismatch {
                    index,
                    query: query_dimension,
                    document: document.dimension(),
                });
            }
            validate_vector(document.as_slice(), index, false)?;
            let score = query.as_slice().iter().zip(document.as_slice()).fold(
                0.0_f64,
                |sum, (query_value, document_value)| {
                    sum + f64::from(*query_value) * f64::from(*document_value)
                },
            );
            let score = score as f32;
            if !score.is_finite() {
                return Err(EmbeddingRerankError::NonFiniteScore { index });
            }
            ranked.push(EmbeddingRerankResultV1 {
                index,
                relevance_score: score,
            });
        }
        ranked.sort_by(|left, right| {
            right
                .relevance_score
                .total_cmp(&left.relevance_score)
                .then_with(|| left.index.cmp(&right.index))
        });
        if let Some(top_n) = top_n {
            ranked.truncate(top_n);
        }
        Ok(ranked)
    }
}

fn validate_vector(values: &[f32], index: usize, query: bool) -> Result<(), EmbeddingRerankError> {
    if values.is_empty() {
        return Err(if query {
            EmbeddingRerankError::EmptyQuery
        } else {
            EmbeddingRerankError::EmptyDocument { index }
        });
    }
    for (dimension, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(EmbeddingRerankError::NonFiniteInput { index, dimension });
        }
    }
    Ok(())
}

impl PartialOrd for EmbeddingRerankResultV1 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.relevance_score.total_cmp(&other.relevance_score))
    }
}
