// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Load-once, request-resettable Qwen3.5 AQ4 model runtime.
//!
//! This module is the ownership boundary between package-derived model geometry and request
//! execution. Resident weights and request state are loaded once; request reset never reloads or
//! clones a weight buffer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::loader::{
    PassthroughF32Data, effective_rmsnorm_weight_values, read_named_passthrough_f32,
};
use crate::qwen35_aq4_head_runtime::{
    PackageEmbeddingRuntime, PackageFinalNormRuntime, PackageLmHeadMode, PackageLmHeadRuntime,
    PackageTokenLogit, QWEN3_FINAL_NORM_TENSOR, package_embedding_shape,
};
use crate::qwen35_aq4_layer_runtime::{
    PackageLinearAttnComponentStepMs, PackageLinearAttnResidentStepLayer,
    PackageSelfAttnComponentStepMs, PackageSelfAttnResidentStepLayer,
};
use crate::qwen35_package_contract::{
    PackageDecoderLayerKind, PackageManifestLayerEntry, package_manifest_layer_entries,
};

/// Product context length used by the Qwen3.5 9B served-model contract.
pub const QWEN35_AQ4_CONTEXT_LENGTH: usize = 4096;
pub const QWEN35_AQ4_KV_BLOCK_SIZE: usize = 256;

pub type Qwen35Aq4LayerKind = PackageDecoderLayerKind;
pub type Qwen35Aq4LayerSpec = PackageManifestLayerEntry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4SelfAttentionGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub q_projection_layout: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4ModelGeometry {
    pub vocab: usize,
    pub hidden: usize,
    pub context_length: usize,
    pub block_size: usize,
    pub cache_blocks: usize,
    pub block_table: Vec<u32>,
    pub layers: Vec<Qwen35Aq4LayerSpec>,
    pub self_attention: Option<Qwen35Aq4SelfAttentionGeometry>,
}

impl Qwen35Aq4ModelGeometry {
    fn new(
        vocab: usize,
        hidden: usize,
        context_length: usize,
        block_size: usize,
        layers: Vec<Qwen35Aq4LayerSpec>,
    ) -> Result<Self, String> {
        if vocab == 0 || hidden == 0 || context_length == 0 || block_size == 0 {
            return Err("Qwen3.5 AQ4 model geometry must be nonzero".to_string());
        }
        if layers.is_empty() {
            return Err("Qwen3.5 AQ4 model requires at least one decoder layer".to_string());
        }
        let cache_blocks = context_length.div_ceil(block_size);
        let block_table = (0..cache_blocks)
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| format!("Qwen3.5 AQ4 KV block index {index} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab,
            hidden,
            context_length,
            block_size,
            cache_blocks,
            block_table,
            layers,
            self_attention: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Aq4ModelLoadConfig {
    pub package_dir: PathBuf,
    pub device_index: u32,
    pub chunk_bytes: usize,
    pub context_length: usize,
    pub kv_block_size: usize,
    /// `None` loads every package-manifest layer. A selection must retain manifest order.
    pub layer_indices: Option<Vec<usize>>,
    pub lm_head_mode: PackageLmHeadMode,
    pub lm_head_chunk_rows: usize,
}

#[derive(Clone)]
pub struct Qwen35Aq4StackStep {
    pub final_layer_position: usize,
    pub layer_step_ms: Vec<f64>,
    pub linear_attention_components: Vec<Option<PackageLinearAttnComponentStepMs>>,
    pub self_attention_components: Vec<Option<PackageSelfAttnComponentStepMs>>,
}

pub enum Qwen35Aq4ResidentLayer {
    LinearAttention(PackageLinearAttnResidentStepLayer),
    SelfAttention(PackageSelfAttnResidentStepLayer),
}

impl Qwen35Aq4ResidentLayer {
    pub fn kind(&self) -> Qwen35Aq4LayerKind {
        match self {
            Self::LinearAttention(_) => Qwen35Aq4LayerKind::LinearAttention,
            Self::SelfAttention(_) => Qwen35Aq4LayerKind::SelfAttention,
        }
    }

    pub fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        match self {
            Self::LinearAttention(layer) => layer.output_buffer(),
            Self::SelfAttention(layer) => layer.output_buffer(),
        }
    }

    fn step_from_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: &ullm_runtime_sys::RuntimeBuffer,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => layer.step_from_device_to_device(stream, input, label),
            Self::SelfAttention(layer) => layer.step_from_device_to_device(
                stream,
                input,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                label,
            ),
        }
    }

    fn reset_synchronized(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => layer.reset_request_state_synchronized(stream),
            Self::SelfAttention(layer) => layer.reset_request_state_synchronized(stream),
        }
    }

    fn take_components(
        &mut self,
    ) -> (
        Option<PackageLinearAttnComponentStepMs>,
        Option<PackageSelfAttnComponentStepMs>,
    ) {
        match self {
            Self::LinearAttention(layer) => (layer.take_last_component_step_ms(), None),
            Self::SelfAttention(layer) => (None, layer.take_last_component_step_ms()),
        }
    }
}

/// Owns one device context and every resident allocation required by one Qwen3.5 AQ4 model.
///
/// Rust drops fields in declaration order. GPU allocation holders are therefore declared before
/// `stream`, and `stream` before `context`; the context can never be destroyed while a holder or
/// stream still exists. `Drop` synchronizes outstanding work before that ordered destruction.
pub struct Qwen35Aq4ModelRuntime {
    // GPU allocation holders -- do not move these below stream/context.
    embedding: Option<PackageEmbeddingRuntime>,
    layers: Vec<Qwen35Aq4ResidentLayer>,
    final_norm: PassthroughF32Data,
    final_norm_runtime: Option<PackageFinalNormRuntime>,
    lm_head: PackageLmHeadRuntime,
    // Runtime handles -- stream must precede context for destruction order.
    stream: ullm_runtime_sys::RuntimeStream,
    _context: ullm_runtime_sys::RuntimeContext,
    package_dir: PathBuf,
    geometry: Qwen35Aq4ModelGeometry,
    device_name: String,
    backend: String,
    device_total_global_mem: u64,
}

impl Qwen35Aq4ModelRuntime {
    pub fn load(config: Qwen35Aq4ModelLoadConfig) -> Result<Self, String> {
        if config.chunk_bytes == 0 {
            return Err("Qwen3.5 AQ4 load chunk bytes must be positive".to_string());
        }
        let path = package_path_text(&config.package_dir)?;
        let manifest_layers = package_manifest_layer_entries(&config.package_dir)?;
        let layers = select_manifest_layers(&manifest_layers, config.layer_indices.as_deref())?;
        let (vocab, hidden) = package_embedding_shape(path)?;
        let mut geometry = Qwen35Aq4ModelGeometry::new(
            vocab,
            hidden,
            config.context_length,
            config.kv_block_size,
            layers,
        )?;

        let mut context = ullm_runtime_sys::RuntimeContext::create(config.device_index)
            .map_err(|err| format!("failed to create Qwen3.5 AQ4 runtime context: {err}"))?;
        let info = context
            .device_info()
            .map_err(|err| format!("failed to query Qwen3.5 AQ4 runtime device: {err}"))?;
        let mut stream = context
            .create_stream()
            .map_err(|err| format!("failed to create Qwen3.5 AQ4 runtime stream: {err}"))?;

        let mut resident_layers = Vec::with_capacity(geometry.layers.len());
        for spec in &geometry.layers {
            let layer = match spec.kind {
                Qwen35Aq4LayerKind::LinearAttention => Qwen35Aq4ResidentLayer::LinearAttention({
                    let layer = PackageLinearAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        config.chunk_bytes,
                        spec.layer_index,
                    )
                    .map_err(|err| {
                        format!(
                            "failed to load Qwen3.5 AQ4 linear layer {}: {err}",
                            spec.layer_index
                        )
                    })?;
                    if layer.hidden != hidden {
                        return Err(format!(
                            "Qwen3.5 AQ4 linear-attention layer {} hidden {} does not match embedding hidden {hidden}",
                            spec.layer_index, layer.hidden
                        ));
                    }
                    layer
                }),
                Qwen35Aq4LayerKind::SelfAttention => Qwen35Aq4ResidentLayer::SelfAttention({
                    let layer = PackageSelfAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        config.chunk_bytes,
                        spec.layer_index,
                        &geometry.block_table,
                        geometry.block_size,
                        geometry.cache_blocks,
                    )
                    .map_err(|err| {
                        format!(
                            "failed to load Qwen3.5 AQ4 self-attention layer {}: {err}",
                            spec.layer_index
                        )
                    })?;
                    if layer.hidden != hidden {
                        return Err(format!(
                            "Qwen3.5 AQ4 self-attention layer {} hidden {} does not match embedding hidden {hidden}",
                            spec.layer_index, layer.hidden
                        ));
                    }
                    let layer_geometry = Qwen35Aq4SelfAttentionGeometry {
                        q_heads: layer.q_heads,
                        kv_heads: layer.kv_heads,
                        head_dim: layer.head_dim,
                        value_dim: layer.value_dim,
                        q_projection_layout: layer.q_projection_layout.as_str(),
                    };
                    match &geometry.self_attention {
                        Some(previous) if previous != &layer_geometry => {
                            return Err(format!(
                                "Qwen3.5 AQ4 self-attention geometry changed at layer {}: previous={previous:?} current={layer_geometry:?}",
                                spec.layer_index
                            ));
                        }
                        None => geometry.self_attention = Some(layer_geometry),
                        Some(_) => {}
                    }
                    layer
                }),
            };
            resident_layers.push(layer);
        }

        let mut final_norm =
            read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, config.chunk_bytes)
                .map_err(|err| format!("failed to read Qwen3.5 final RMSNorm: {err}"))?;
        final_norm.values =
            effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
        if final_norm.values.len() != hidden {
            return Err(format!(
                "Qwen3.5 final RMSNorm length {} does not match hidden {hidden}",
                final_norm.values.len()
            ));
        }
        let lm_head = PackageLmHeadRuntime::load(
            config.lm_head_mode,
            &mut context,
            &mut stream,
            path,
            config.chunk_bytes,
            hidden,
            config.lm_head_chunk_rows,
        )?;
        let final_norm_runtime = if lm_head.supports_device_input() {
            Some(PackageFinalNormRuntime::load(
                &mut context,
                &mut stream,
                &final_norm,
                hidden,
            )?)
        } else {
            None
        };
        let embedding = if lm_head.supports_device_input() {
            PackageEmbeddingRuntime::load_if_available(
                &mut context,
                &mut stream,
                path,
                config.chunk_bytes,
                hidden,
            )?
        } else {
            None
        };
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model load: {err}"))?;

        Ok(Self {
            embedding,
            layers: resident_layers,
            final_norm,
            final_norm_runtime,
            lm_head,
            stream,
            _context: context,
            package_dir: config.package_dir,
            geometry,
            device_name: info.name,
            backend: info.backend.to_string(),
            device_total_global_mem: info.total_global_mem,
        })
    }

    pub fn geometry(&self) -> &Qwen35Aq4ModelGeometry {
        &self.geometry
    }

    pub fn package_dir(&self) -> &Path {
        &self.package_dir
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_total_global_mem(&self) -> u64 {
        self.device_total_global_mem
    }

    pub fn has_resident_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    pub fn supports_device_logits(&self) -> bool {
        self.final_norm_runtime.is_some() && self.lm_head.supports_device_input()
    }

    /// Gathers one token and dispatches it through the complete resident decoder stack.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Qwen35Aq4StackStep, String> {
        if cache_position >= self.geometry.context_length {
            return Err(format!(
                "{label} cache position {cache_position} exceeds context length {}",
                self.geometry.context_length
            ));
        }
        if let Some(attention) = &self.geometry.self_attention
            && (rotary_dim == 0 || rotary_dim > attention.head_dim || rotary_dim % 2 != 0)
        {
            return Err(format!(
                "{label} rotary dimension {rotary_dim} must be a positive even value at most {}",
                attention.head_dim
            ));
        }
        let embedding = self.embedding.as_mut().ok_or_else(|| {
            "Qwen3.5 AQ4 token dispatch requires a resident embedding".to_string()
        })?;
        embedding.gather_token(&mut self.stream, token_id, label)?;
        let input = embedding.output_buffer();
        dispatch_layer_stack(
            &mut self.layers,
            &mut self.stream,
            input,
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            sync_each_layer_for_timing,
            label,
        )
    }

    pub fn top_logits_from_last_layer(
        &mut self,
        top_k: usize,
        label: &str,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        let last = self
            .layers
            .last()
            .ok_or_else(|| "Qwen3.5 AQ4 model has no final layer".to_string())?;
        let final_norm = self.final_norm_runtime.as_mut().ok_or_else(|| {
            "Qwen3.5 AQ4 device logits require resident final RMSNorm".to_string()
        })?;
        final_norm.normalize_device(&mut self.stream, last.output_buffer(), label)?;
        self.lm_head.top_logits_from_device_buffer(
            &mut self.stream,
            final_norm.output_buffer(),
            top_k,
        )
    }

    pub fn final_norm(&self) -> &PassthroughF32Data {
        &self.final_norm
    }

    pub fn lm_head(&mut self) -> &mut PackageLmHeadRuntime {
        &mut self.lm_head
    }

    /// Synchronizes, clears all request-owned KV/conv/recurrent state, and retains all weights.
    pub fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
        self.stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3.5 AQ4 model before request reset: {err}")
        })?;
        for (position, layer) in self.layers.iter_mut().enumerate() {
            layer
                .reset_synchronized(&mut self.stream)
                .map_err(|err| format!("failed to reset Qwen3.5 AQ4 layer {position}: {err}"))?;
        }
        self.stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model request reset: {err}"))
    }

    pub fn synchronize(&mut self) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model runtime: {err}"))
    }

    /// Explicitly synchronizes before the owner and its context are dropped.
    pub fn shutdown_synchronized(mut self) -> Result<(), String> {
        self.synchronize()
    }
}

impl Drop for Qwen35Aq4ModelRuntime {
    fn drop(&mut self) {
        // Destructors cannot return errors; serving code should call `shutdown_synchronized` when
        // it needs an observable shutdown result. This still prevents normal outstanding work
        // from racing the ordered GPU-holder -> stream -> context destruction below.
        let _ = self.stream.synchronize();
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_layer_stack(
    layers: &mut [Qwen35Aq4ResidentLayer],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    first_input: &ullm_runtime_sys::RuntimeBuffer,
    rotary_dim: usize,
    rope_base: f32,
    rope_position: usize,
    cache_position: usize,
    sync_each_layer_for_timing: bool,
    label: &str,
) -> Result<Qwen35Aq4StackStep, String> {
    if layers.is_empty() {
        return Err(format!("{label} requires at least one resident layer"));
    }
    let mut layer_step_ms = Vec::with_capacity(layers.len());
    let mut linear_attention_components = Vec::with_capacity(layers.len());
    let mut self_attention_components = Vec::with_capacity(layers.len());
    for position in 0..layers.len() {
        let started = std::time::Instant::now();
        let layer_label = format!("{label} layer {position} position {rope_position}");
        if position == 0 {
            layers[0].step_from_device(
                stream,
                first_input,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        } else {
            let (previous, current) = layers.split_at_mut(position);
            current[0].step_from_device(
                stream,
                previous[position - 1].output_buffer(),
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                &layer_label,
            )?;
        }
        if sync_each_layer_for_timing {
            stream
                .synchronize()
                .map_err(|err| format!("failed to synchronize {layer_label}: {err}"))?;
        }
        layer_step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let (linear, self_attention) = layers[position].take_components();
        linear_attention_components.push(linear);
        self_attention_components.push(self_attention);
    }
    Ok(Qwen35Aq4StackStep {
        final_layer_position: layers.len() - 1,
        layer_step_ms,
        linear_attention_components,
        self_attention_components,
    })
}

fn select_manifest_layers(
    manifest: &[Qwen35Aq4LayerSpec],
    requested: Option<&[usize]>,
) -> Result<Vec<Qwen35Aq4LayerSpec>, String> {
    let Some(requested) = requested else {
        return Ok(manifest.to_vec());
    };
    if requested.is_empty() {
        return Err("Qwen3.5 AQ4 selected layer list is empty".to_string());
    }
    let requested_set = requested.iter().copied().collect::<BTreeSet<_>>();
    if requested_set.len() != requested.len() {
        return Err("Qwen3.5 AQ4 selected layer list contains duplicates".to_string());
    }
    let selected = manifest
        .iter()
        .copied()
        .filter(|entry| requested_set.contains(&entry.layer_index))
        .collect::<Vec<_>>();
    let selected_indices = selected
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    if selected_indices != requested {
        return Err(format!(
            "Qwen3.5 AQ4 selected layers must exist and retain manifest order: requested={requested:?} manifest_selection={selected_indices:?}"
        ));
    }
    Ok(selected)
}

fn package_path_text(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| "Qwen3.5 AQ4 package path is not valid UTF-8".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<Qwen35Aq4LayerSpec> {
        vec![
            Qwen35Aq4LayerSpec {
                layer_index: 0,
                kind: Qwen35Aq4LayerKind::LinearAttention,
            },
            Qwen35Aq4LayerSpec {
                layer_index: 1,
                kind: Qwen35Aq4LayerKind::SelfAttention,
            },
            Qwen35Aq4LayerSpec {
                layer_index: 2,
                kind: Qwen35Aq4LayerKind::LinearAttention,
            },
        ]
    }

    #[test]
    fn geometry_allocates_the_full_product_context() {
        let geometry = Qwen35Aq4ModelGeometry::new(
            248_320,
            4_096,
            QWEN35_AQ4_CONTEXT_LENGTH,
            QWEN35_AQ4_KV_BLOCK_SIZE,
            specs(),
        )
        .unwrap();
        assert_eq!(geometry.context_length, 4_096);
        assert_eq!(geometry.block_size, 256);
        assert_eq!(geometry.cache_blocks, 16);
        assert_eq!(geometry.block_table, (0_u32..16).collect::<Vec<_>>());

        let smaller = Qwen35Aq4ModelGeometry::new(100, 64, 513, 256, specs()).unwrap();
        assert_eq!(smaller.context_length, 513);
        assert_eq!(smaller.cache_blocks, 3);
        assert_eq!(smaller.block_table, vec![0, 1, 2]);
    }

    #[test]
    fn selection_requires_unique_manifest_order() {
        assert_eq!(
            select_manifest_layers(&specs(), Some(&[0, 2]))
                .unwrap()
                .iter()
                .map(|entry| entry.layer_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(select_manifest_layers(&specs(), Some(&[2, 0])).is_err());
        assert!(select_manifest_layers(&specs(), Some(&[1, 1])).is_err());
        assert!(select_manifest_layers(&specs(), Some(&[9])).is_err());
    }

    #[test]
    fn layer_kind_names_match_existing_cli_reports() {
        assert_eq!(Qwen35Aq4LayerKind::SelfAttention.as_str(), "self_attention");
        assert_eq!(
            Qwen35Aq4LayerKind::LinearAttention.as_str(),
            "linear_attention"
        );
    }
}
