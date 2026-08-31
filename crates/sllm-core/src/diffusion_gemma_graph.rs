//! Container-neutral execution schedule for DiffusionGemma's canvas loop.
//!
//! This module fixes state ownership and attention modes without claiming that
//! the 51.6 GB official checkpoint is resident or executable in production.

use std::fmt;

pub const DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH: u64 = 256;
pub const DIFFUSION_GEMMA_GRAPH_TEXT_LAYER_COUNT: u32 = 30;
pub const DIFFUSION_GEMMA_GRAPH_MAX_DENOISING_STEPS: u32 = 48;
pub const DIFFUSION_GEMMA_GRAPH_MAX_CONTEXT_TOKENS: u64 = 262_144;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaGraphAttentionMode {
    /// Appends prompt or a previously finalized canvas to the request KV cache.
    CausalEncoder,
    /// Reads the encoder cache and the complete current canvas without updating KV.
    BidirectionalCanvasDecoder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaGraphStage {
    IncrementalPrefill,
    UniformCanvasInitialization,
    BidirectionalDenoising,
    SelfConditioning,
    EntropyBoundAcceptance,
    UniformRenoising,
    AdaptiveStopping,
    CanvasPublication,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaCanvasPlan {
    pub canvas_index: u64,
    pub encoder_append_tokens: u64,
    pub cache_tokens_before_encoder: u64,
    pub cache_tokens_after_encoder: u64,
    pub decoder_position_start: u64,
    pub decoder_position_end_exclusive: u64,
    pub maximum_denoising_steps: u32,
    pub final_canvas: bool,
}

impl DiffusionGemmaCanvasPlan {
    pub const fn encoder_attention_mode(self) -> DiffusionGemmaGraphAttentionMode {
        DiffusionGemmaGraphAttentionMode::CausalEncoder
    }

    pub const fn decoder_attention_mode(self) -> DiffusionGemmaGraphAttentionMode {
        DiffusionGemmaGraphAttentionMode::BidirectionalCanvasDecoder
    }

    /// The decoder only borrows the encoder cache. It must never publish its
    /// per-step K/V projections into the request state.
    pub const fn decoder_writes_kv(self) -> bool {
        false
    }

    pub const fn stages(self) -> [DiffusionGemmaGraphStage; 8] {
        [
            DiffusionGemmaGraphStage::IncrementalPrefill,
            DiffusionGemmaGraphStage::UniformCanvasInitialization,
            DiffusionGemmaGraphStage::BidirectionalDenoising,
            DiffusionGemmaGraphStage::SelfConditioning,
            DiffusionGemmaGraphStage::EntropyBoundAcceptance,
            DiffusionGemmaGraphStage::UniformRenoising,
            DiffusionGemmaGraphStage::AdaptiveStopping,
            DiffusionGemmaGraphStage::CanvasPublication,
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaGraph {
    pub prompt_token_count: u64,
    pub requested_output_tokens: u64,
    pub padded_canvas_tokens: u64,
    pub canvases: Vec<DiffusionGemmaCanvasPlan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaGraphError {
    EmptyPrompt,
    EmptyOutput,
    ContextExceeded {
        prompt_tokens: u64,
        padded_canvas_tokens: u64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for DiffusionGemmaGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => formatter.write_str("DiffusionGemma requires a non-empty prompt"),
            Self::EmptyOutput => {
                formatter.write_str("DiffusionGemma requires at least one requested output token")
            }
            Self::ContextExceeded {
                prompt_tokens,
                padded_canvas_tokens,
            } => write!(
                formatter,
                "DiffusionGemma prompt {prompt_tokens} plus padded canvas {padded_canvas_tokens} exceeds 262144 tokens"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("DiffusionGemma graph arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for DiffusionGemmaGraphError {}

/// Builds the outer autoregressive canvas schedule. The final canvas is not
/// appended to KV because no later canvas consumes it. Earlier finalized
/// canvases are appended exactly once at the next incremental-prefill stage.
pub fn build_diffusion_gemma_graph(
    prompt_token_count: u64,
    requested_output_tokens: u64,
) -> Result<DiffusionGemmaGraph, DiffusionGemmaGraphError> {
    if prompt_token_count == 0 {
        return Err(DiffusionGemmaGraphError::EmptyPrompt);
    }
    if requested_output_tokens == 0 {
        return Err(DiffusionGemmaGraphError::EmptyOutput);
    }
    let canvas_count = requested_output_tokens
        .checked_add(DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH - 1)
        .ok_or(DiffusionGemmaGraphError::ArithmeticOverflow)?
        / DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH;
    let padded_canvas_tokens = canvas_count
        .checked_mul(DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH)
        .ok_or(DiffusionGemmaGraphError::ArithmeticOverflow)?;
    let padded_total = prompt_token_count
        .checked_add(padded_canvas_tokens)
        .ok_or(DiffusionGemmaGraphError::ArithmeticOverflow)?;
    if padded_total > DIFFUSION_GEMMA_GRAPH_MAX_CONTEXT_TOKENS {
        return Err(DiffusionGemmaGraphError::ContextExceeded {
            prompt_tokens: prompt_token_count,
            padded_canvas_tokens,
        });
    }
    let canvas_capacity =
        usize::try_from(canvas_count).map_err(|_| DiffusionGemmaGraphError::ArithmeticOverflow)?;
    let mut canvases = Vec::with_capacity(canvas_capacity);
    let mut cache_tokens = 0_u64;
    for canvas_index in 0..canvas_count {
        let encoder_append_tokens = if canvas_index == 0 {
            prompt_token_count
        } else {
            DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH
        };
        let cache_tokens_before_encoder = cache_tokens;
        cache_tokens = cache_tokens
            .checked_add(encoder_append_tokens)
            .ok_or(DiffusionGemmaGraphError::ArithmeticOverflow)?;
        let decoder_position_end_exclusive = cache_tokens
            .checked_add(DIFFUSION_GEMMA_GRAPH_CANVAS_LENGTH)
            .ok_or(DiffusionGemmaGraphError::ArithmeticOverflow)?;
        canvases.push(DiffusionGemmaCanvasPlan {
            canvas_index,
            encoder_append_tokens,
            cache_tokens_before_encoder,
            cache_tokens_after_encoder: cache_tokens,
            decoder_position_start: cache_tokens,
            decoder_position_end_exclusive,
            maximum_denoising_steps: DIFFUSION_GEMMA_GRAPH_MAX_DENOISING_STEPS,
            final_canvas: canvas_index + 1 == canvas_count,
        });
    }
    Ok(DiffusionGemmaGraph {
        prompt_token_count,
        requested_output_tokens,
        padded_canvas_tokens,
        canvases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_255_256_257_outputs_have_exact_canvas_boundaries() {
        for (requested, expected_canvases, expected_padded) in
            [(1, 1, 256), (255, 1, 256), (256, 1, 256), (257, 2, 512)]
        {
            let graph = build_diffusion_gemma_graph(33, requested).expect("valid graph");
            assert_eq!(graph.canvases.len(), expected_canvases);
            assert_eq!(graph.padded_canvas_tokens, expected_padded);
        }
    }

    #[test]
    fn second_canvas_encodes_the_first_once_and_decoder_never_writes_kv() {
        let graph = build_diffusion_gemma_graph(31, 257).expect("two canvases");
        assert_eq!(graph.canvases[0].cache_tokens_before_encoder, 0);
        assert_eq!(graph.canvases[0].cache_tokens_after_encoder, 31);
        assert_eq!(graph.canvases[0].decoder_position_start, 31);
        assert_eq!(graph.canvases[0].decoder_position_end_exclusive, 287);
        assert_eq!(graph.canvases[1].cache_tokens_before_encoder, 31);
        assert_eq!(graph.canvases[1].encoder_append_tokens, 256);
        assert_eq!(graph.canvases[1].cache_tokens_after_encoder, 287);
        assert_eq!(graph.canvases[1].decoder_position_start, 287);
        assert!(!graph.canvases[0].decoder_writes_kv());
        assert!(!graph.canvases[1].decoder_writes_kv());
        assert!(graph.canvases[1].final_canvas);
    }

    #[test]
    fn stage_order_and_attention_modes_are_not_autoregressive_decoder_aliases() {
        let graph = build_diffusion_gemma_graph(32, 33).expect("valid graph");
        let canvas = graph.canvases[0];
        assert_eq!(
            canvas.encoder_attention_mode(),
            DiffusionGemmaGraphAttentionMode::CausalEncoder
        );
        assert_eq!(
            canvas.decoder_attention_mode(),
            DiffusionGemmaGraphAttentionMode::BidirectionalCanvasDecoder
        );
        assert_eq!(
            canvas.stages(),
            [
                DiffusionGemmaGraphStage::IncrementalPrefill,
                DiffusionGemmaGraphStage::UniformCanvasInitialization,
                DiffusionGemmaGraphStage::BidirectionalDenoising,
                DiffusionGemmaGraphStage::SelfConditioning,
                DiffusionGemmaGraphStage::EntropyBoundAcceptance,
                DiffusionGemmaGraphStage::UniformRenoising,
                DiffusionGemmaGraphStage::AdaptiveStopping,
                DiffusionGemmaGraphStage::CanvasPublication,
            ]
        );
    }

    #[test]
    fn context_and_arithmetic_boundaries_fail_before_allocation() {
        assert_eq!(
            build_diffusion_gemma_graph(0, 1),
            Err(DiffusionGemmaGraphError::EmptyPrompt)
        );
        assert_eq!(
            build_diffusion_gemma_graph(1, 0),
            Err(DiffusionGemmaGraphError::EmptyOutput)
        );
        assert!(matches!(
            build_diffusion_gemma_graph(262_143, 1),
            Err(DiffusionGemmaGraphError::ContextExceeded { .. })
        ));
        assert_eq!(
            build_diffusion_gemma_graph(1, u64::MAX),
            Err(DiffusionGemmaGraphError::ArithmeticOverflow)
        );
    }

    #[test]
    fn exact_padded_context_boundary_is_accepted() {
        let graph =
            build_diffusion_gemma_graph(262_144 - 512, 257).expect("exact padded context boundary");
        assert_eq!(graph.canvases.len(), 2);
        assert_eq!(
            graph.canvases[1].decoder_position_end_exclusive,
            DIFFUSION_GEMMA_GRAPH_MAX_CONTEXT_TOKENS
        );
    }
}
