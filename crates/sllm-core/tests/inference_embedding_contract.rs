use sllm_core::{
    CosineEmbeddingRerankV1, EmbeddingPoolError, EmbeddingPoolV1, EmbeddingRerankError,
    EmbeddingRowsV1,
};

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

#[test]
fn profile_v1_pools_f32_with_f64_mean_and_l2() {
    let rows = [1.0_f32, 2.0, 3.0, 5.0, 7.0, 11.0];
    let embedding = EmbeddingPoolV1::new()
        .pool(EmbeddingRowsV1::f32(&rows, 2, 3))
        .expect("valid rows");
    let expected = [3.0_f32, 4.5, 7.0];
    let norm = expected
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    for (actual, expected) in embedding.as_slice().iter().zip(expected) {
        assert!((f64::from(*actual) - f64::from(expected) / norm).abs() < 1e-7);
    }
    let norm = embedding
        .as_slice()
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-7);
}

#[test]
fn profile_v1_accepts_bf16_and_little_endian_bytes() {
    let rows = [
        bf16_bits(1.0),
        bf16_bits(2.0),
        bf16_bits(3.0),
        bf16_bits(1.0),
        bf16_bits(2.0),
        bf16_bits(3.0),
    ];
    let embedding = EmbeddingPoolV1::new()
        .pool_bf16(&rows, 2, 3)
        .expect("valid bf16 rows");
    for (actual, expected) in
        embedding
            .as_slice()
            .iter()
            .zip([0.26726124_f32, 0.5345225, 0.8017837])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    let bytes = embedding.to_little_endian_bytes().expect("finite bytes");
    assert_eq!(bytes.len(), 12);
    for (chunk, value) in bytes.chunks_exact(4).zip(embedding.as_slice()) {
        assert_eq!(chunk, value.to_le_bytes());
    }
}

#[test]
fn profile_v1_covers_non_aligned_dimensions_and_token_lengths() {
    for tokens in [1, 2, 3, 255, 256, 257] {
        for hidden in [1, 3, 17, 255, 256, 257] {
            let rows = (0..tokens * hidden)
                .map(|index| (index % hidden) as f32 + 1.0)
                .collect::<Vec<_>>();
            let embedding = EmbeddingPoolV1::new()
                .pool_f32(&rows, tokens, hidden)
                .unwrap_or_else(|error| panic!("{tokens}x{hidden}: {error}"));
            assert_eq!(embedding.dimension(), hidden);
            assert!(embedding.as_slice().iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn profile_v1_rejects_empty_shape_mismatch_overflow_and_nonfinite() {
    let pool = EmbeddingPoolV1::new();
    assert_eq!(
        pool.pool_f32(&[], 0, 3),
        Err(EmbeddingPoolError::EmptyShape)
    );
    assert_eq!(
        pool.pool_f32(&[1.0], 1, 2),
        Err(EmbeddingPoolError::ShapeMismatch {
            tokens: 1,
            hidden: 2,
            values: 1,
        })
    );
    assert_eq!(
        pool.pool_f32(&[1.0], usize::MAX, 2),
        Err(EmbeddingPoolError::ShapeOverflow)
    );
    assert!(matches!(
        pool.pool_f32(&[1.0, f32::NAN], 1, 2),
        Err(EmbeddingPoolError::NonFiniteInput { row: 0, column: 1 })
    ));
    assert_eq!(
        pool.pool_f32(&[0.0, 0.0], 1, 2),
        Err(EmbeddingPoolError::ZeroNorm)
    );
}

#[test]
fn cosine_rerank_uses_f64_dot_and_stable_original_index_ties() {
    let pool = EmbeddingPoolV1::new();
    let query = pool.pool_f32(&[1.0, 0.0, 0.0], 1, 3).unwrap();
    let duplicate = query.clone();
    let orthogonal = pool.pool_f32(&[0.0, 1.0, 0.0], 1, 3).unwrap();
    let rerank = CosineEmbeddingRerankV1::new();
    let ranked = rerank
        .rank(
            &query,
            &[orthogonal.clone(), duplicate.clone(), duplicate],
            None,
        )
        .unwrap();
    assert_eq!(
        ranked
            .iter()
            .map(|result| result.index())
            .collect::<Vec<_>>(),
        vec![1, 2, 0]
    );
    assert!((ranked[0].relevance_score() - 1.0).abs() < 1e-7);
    assert!((rerank.score(&query, &query).unwrap() - 1.0).abs() < 1e-7);
}

#[test]
fn cosine_rerank_rejects_invalid_top_n_and_nonfinite_vectors() {
    let pool = EmbeddingPoolV1::new();
    let query = pool.pool_f32(&[1.0, 0.0], 1, 2).unwrap();
    let document = pool.pool_f32(&[0.0, 1.0], 1, 2).unwrap();
    let rerank = CosineEmbeddingRerankV1::new();
    assert_eq!(
        rerank.rank(&query, std::slice::from_ref(&document), Some(0)),
        Err(EmbeddingRerankError::InvalidTopN {
            top_n: 0,
            documents: 1,
        })
    );
    assert_eq!(
        rerank.rank(&query, std::slice::from_ref(&document), Some(2)),
        Err(EmbeddingRerankError::InvalidTopN {
            top_n: 2,
            documents: 1,
        })
    );
    assert_eq!(
        rerank.rank(&query, &[], None),
        Err(EmbeddingRerankError::EmptyDocuments)
    );
}
