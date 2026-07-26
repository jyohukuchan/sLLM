// Extracted from clean uLLM runtime HEAD
// 0216b131cf5377d90125abd9c1c49c5a8a210511,
// crates/ullm-engine/src/sq8_serving_runtime.rs.
// This excerpt documents the actual fixed-M128 tail behavior used by the
// measured serving path.

pub enum Sq8ServingPrefillMode {
    SequentialM1,
    FixedM8Chunks,
    FixedM32Chunks,
    FixedM128Chunks,
}

impl Sq8ServingPrefillMode {
    fn chunk_tokens(self) -> Option<usize> {
        match self {
            Self::SequentialM1 => None,
            Self::FixedM8Chunks => Some(QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS),
            Self::FixedM32Chunks => Some(32),
            Self::FixedM128Chunks => Some(128),
        }
    }
}

// Exact planner rule (comments and validation text omitted for brevity):
let remaining = prompt_tokens - start_position;
let width = mode
    .chunk_tokens()
    .filter(|chunk_tokens| remaining >= *chunk_tokens)
    .unwrap_or(1);
let end = start_position.checked_add(width)?;
let unit = Sq8PrefillUnit {
    start_position,
    width,
    is_final: end == prompt_tokens,
};

// Therefore N=4095 in FixedM128Chunks is 31 x M=128 followed by
// 127 x M=1, for 158 calls.
