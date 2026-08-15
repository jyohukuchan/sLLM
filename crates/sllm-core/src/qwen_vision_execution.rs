//! Qwen3.5 vision execution using the model-neutral HIP matmul provider.
//!
//! The 297 immutable tensors are uploaded once through `ExecutionSession`.
//! Dense projections execute through the same semantic HIP Matmul operation as
//! the text graph. Small vision-specific transforms remain deterministic host
//! calculations until they have a reusable model-neutral semantic operation.

use crate::{
    AccessMode, AllocationCategory, BoundSemanticOp, DType, DispatchEvidence, ExecutionBuffer,
    ExecutionError, ExecutionQueue, ExecutionSession, ExecutionState, QwenVisionManifest,
    SemanticOpDescriptor, SemanticOpKind, TensorView, VerifiedCache,
};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

const HIDDEN: usize = 1_024;
const INTERMEDIATE: usize = 4_096;
const OUTPUT: usize = 2_560;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const PATCH_WIDTH: usize = 1_536;
const MAX_PATCH_TOKENS: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenMultimodalImageEmbedding {
    pub grid_thw: [u32; 3],
    pub embeddings_bf16: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenMultimodalPrompt {
    pub embeddings_bf16: Vec<u16>,
    pub positions: Vec<[i32; 3]>,
}

/// Assemble selected text embedding rows, projected image rows, and the
/// official Qwen3.5 interleaved mRoPE coordinates without interpreting image
/// bytes as text. Image groups are the exact contiguous image-pad token runs.
pub fn assemble_qwen35_multimodal_prompt(
    cache: &VerifiedCache,
    token_ids: &[u32],
    image_pad_token: u32,
    images: &[QwenMultimodalImageEmbedding],
) -> Result<QwenMultimodalPrompt, QwenVisionExecutionError> {
    if token_ids.is_empty() || images.is_empty() || images.len() > 2 {
        return Err(QwenVisionExecutionError::Invalid(
            "multimodal prompt/image count differs".to_owned(),
        ));
    }
    let mut runs = Vec::new();
    let mut index = 0;
    while index < token_ids.len() {
        if token_ids[index] != image_pad_token {
            index += 1;
            continue;
        }
        let start = index;
        while index < token_ids.len() && token_ids[index] == image_pad_token {
            index += 1;
        }
        runs.push(start..index);
    }
    if runs.len() != images.len() {
        return Err(QwenVisionExecutionError::Invalid(
            "image-pad runs do not match image embeddings".to_owned(),
        ));
    }
    let mut embeddings_bf16 = Vec::with_capacity(token_ids.len() * OUTPUT);
    for token in token_ids {
        if usize::try_from(*token)
            .ok()
            .is_none_or(|token| token >= 248_320)
        {
            return Err(QwenVisionExecutionError::Invalid(
                "multimodal token is outside the fixed vocabulary".to_owned(),
            ));
        }
        let offset = u64::from(*token)
            .checked_mul((OUTPUT * 2) as u64)
            .ok_or_else(|| {
                QwenVisionExecutionError::Invalid("embedding offset overflowed".to_owned())
            })?;
        let bytes = cache
            .read_tensor_range(crate::QWEN35_EMBEDDING_TENSOR, offset, OUTPUT * 2)
            .map_err(|error| QwenVisionExecutionError::Invalid(error.to_string()))?;
        embeddings_bf16.extend(
            bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]])),
        );
    }

    let mut positions = vec![[0_i32; 3]; token_ids.len()];
    let mut current = 0_i32;
    let mut cursor = 0;
    for (image_index, run) in runs.iter().enumerate() {
        for position in &mut positions[cursor..run.start] {
            *position = [current; 3];
            current = current.checked_add(1).ok_or_else(|| {
                QwenVisionExecutionError::Invalid("text position overflowed".to_owned())
            })?;
        }
        let image = &images[image_index];
        let h = usize::try_from(image.grid_thw[1] / 2).map_err(|_| {
            QwenVisionExecutionError::Invalid("vision grid height does not fit usize".to_owned())
        })?;
        let w = usize::try_from(image.grid_thw[2] / 2).map_err(|_| {
            QwenVisionExecutionError::Invalid("vision grid width does not fit usize".to_owned())
        })?;
        if image.grid_thw[0] != 1
            || h == 0
            || w == 0
            || run.len() != h * w
            || image.embeddings_bf16.len() != run.len() * OUTPUT
        {
            return Err(QwenVisionExecutionError::Invalid(
                "projected image/grid/image-pad lengths differ".to_owned(),
            ));
        }
        for row in 0..h {
            for column in 0..w {
                let offset = row * w + column;
                let row = i32::try_from(row).map_err(|_| {
                    QwenVisionExecutionError::Invalid("vision row does not fit i32".to_owned())
                })?;
                let column = i32::try_from(column).map_err(|_| {
                    QwenVisionExecutionError::Invalid("vision column does not fit i32".to_owned())
                })?;
                positions[run.start + offset] = [
                    current,
                    current.checked_add(row).ok_or_else(|| {
                        QwenVisionExecutionError::Invalid(
                            "vision row position overflowed".to_owned(),
                        )
                    })?,
                    current.checked_add(column).ok_or_else(|| {
                        QwenVisionExecutionError::Invalid(
                            "vision column position overflowed".to_owned(),
                        )
                    })?,
                ];
            }
        }
        let target = &mut embeddings_bf16[run.start * OUTPUT..run.end * OUTPUT];
        target.copy_from_slice(&image.embeddings_bf16);
        current = current
            .checked_add(i32::try_from(h.max(w)).map_err(|_| {
                QwenVisionExecutionError::Invalid("vision position does not fit i32".to_owned())
            })?)
            .ok_or_else(|| {
                QwenVisionExecutionError::Invalid("vision position overflowed".to_owned())
            })?;
        cursor = run.end;
    }
    for position in &mut positions[cursor..] {
        *position = [current; 3];
        current = current.checked_add(1).ok_or_else(|| {
            QwenVisionExecutionError::Invalid("text position overflowed".to_owned())
        })?;
    }
    Ok(QwenMultimodalPrompt {
        embeddings_bf16,
        positions,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct QwenVisionExecutionInput {
    pub grid_thw: [u32; 3],
    pub patch_width: usize,
    pub patches: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenVisionExecutionOutput {
    embeddings_bf16: Vec<u16>,
    visual_tokens: usize,
    patch_tokens: usize,
    dispatches: u64,
    all_dispatches_hip: bool,
    fallback_used: bool,
}

impl QwenVisionExecutionOutput {
    pub fn embeddings_bf16(&self) -> &[u16] {
        &self.embeddings_bf16
    }

    pub const fn visual_tokens(&self) -> usize {
        self.visual_tokens
    }

    pub const fn patch_tokens(&self) -> usize {
        self.patch_tokens
    }

    pub const fn dispatches(&self) -> u64 {
        self.dispatches
    }

    pub const fn all_dispatches_hip(&self) -> bool {
        self.all_dispatches_hip
    }

    pub const fn fallback_used(&self) -> bool {
        self.fallback_used
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QwenVisionExecutionError {
    Invalid(String),
    Execution(String),
}

impl fmt::Display for QwenVisionExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid Qwen3.5 vision request: {message}")
            }
            Self::Execution(message) => {
                write!(formatter, "Qwen3.5 vision execution failed: {message}")
            }
        }
    }
}

impl std::error::Error for QwenVisionExecutionError {}

impl From<ExecutionError> for QwenVisionExecutionError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error.to_string())
    }
}

impl From<crate::TensorError> for QwenVisionExecutionError {
    fn from(error: crate::TensorError) -> Self {
        Self::Invalid(error.to_string())
    }
}

impl From<crate::OpError> for QwenVisionExecutionError {
    fn from(error: crate::OpError) -> Self {
        Self::Invalid(error.to_string())
    }
}

struct ResidentTensor {
    shape: Vec<u64>,
    buffer: ExecutionBuffer,
}

pub struct QwenVisionResidentModel {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    tensors: BTreeMap<String, ResidentTensor>,
    host_tensors: BTreeMap<String, Arc<[f32]>>,
    completion_timeout: Duration,
    model_fingerprint: String,
    manifest_digest: String,
}

impl fmt::Debug for QwenVisionResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenVisionResidentModel")
            .field("session", &self.session.id())
            .field("model_fingerprint", &self.model_fingerprint)
            .field("manifest_digest", &self.manifest_digest)
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl QwenVisionResidentModel {
    pub fn new(
        session: Arc<ExecutionSession>,
        cache: Arc<VerifiedCache>,
        manifest: QwenVisionManifest,
        completion_timeout: Duration,
    ) -> Result<Self, QwenVisionExecutionError> {
        if completion_timeout.is_zero() || manifest.model_fingerprint != cache.lock_fingerprint {
            return Err(QwenVisionExecutionError::Invalid(
                "timeout or verified cache identity differs".to_owned(),
            ));
        }
        let queue = session.create_queue()?;
        let mut tensors = BTreeMap::new();
        let mut host_tensors = BTreeMap::new();
        for tensor in &manifest.tensors {
            let buffer = session
                .allocate_with_category(tensor.byte_size, AllocationCategory::ModelResident)?;
            upload_tensor(
                session.as_ref(),
                &queue,
                &buffer,
                cache.as_ref(),
                &tensor.name,
                tensor.byte_size,
                completion_timeout,
            )?;
            if needs_host_copy(&tensor.name) {
                let length = usize::try_from(tensor.byte_size).map_err(|_| {
                    QwenVisionExecutionError::Invalid("host tensor is too large".to_owned())
                })?;
                let mut bytes = Vec::with_capacity(length);
                for (index, chunk_length) in (0..length)
                    .step_by(1_048_576)
                    .map(|offset| (offset, (length - offset).min(1_048_576)))
                {
                    bytes.extend_from_slice(
                        &cache
                            .read_tensor_range(&tensor.name, index as u64, chunk_length)
                            .map_err(|error| {
                                QwenVisionExecutionError::Invalid(error.to_string())
                            })?,
                    );
                }
                host_tensors.insert(tensor.name.clone(), Arc::from(decode_bf16(&bytes)?));
            }
            tensors.insert(
                tensor.name.clone(),
                ResidentTensor {
                    shape: tensor.shape.clone(),
                    buffer,
                },
            );
        }
        if tensors.len() != 297 {
            return Err(QwenVisionExecutionError::Invalid(
                "resident vision tensor set is incomplete".to_owned(),
            ));
        }
        let manifest_digest = manifest.digest_hex();
        Ok(Self {
            session,
            queue,
            tensors,
            host_tensors,
            completion_timeout,
            model_fingerprint: manifest.model_fingerprint,
            manifest_digest,
        })
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub fn manifest_digest(&self) -> &str {
        &self.manifest_digest
    }

    pub fn execute(
        &self,
        input: &QwenVisionExecutionInput,
    ) -> Result<QwenVisionExecutionOutput, QwenVisionExecutionError> {
        let [grid_t, grid_h, grid_w] = input.grid_thw;
        if grid_t != 1
            || grid_h == 0
            || grid_w == 0
            || grid_h % 2 != 0
            || grid_w % 2 != 0
            || input.patch_width != PATCH_WIDTH
        {
            return Err(QwenVisionExecutionError::Invalid(
                "grid or patch width differs from the fixed image contract".to_owned(),
            ));
        }
        let patch_tokens = usize::try_from(u64::from(grid_h) * u64::from(grid_w))
            .map_err(|_| QwenVisionExecutionError::Invalid("patch count overflowed".to_owned()))?;
        if patch_tokens == 0 || patch_tokens > MAX_PATCH_TOKENS {
            return Err(QwenVisionExecutionError::Invalid(format!(
                "patch count {patch_tokens} exceeds the bounded vision executor"
            )));
        }
        if input.patches.len() != patch_tokens * PATCH_WIDTH {
            return Err(QwenVisionExecutionError::Invalid(
                "patch payload length differs".to_owned(),
            ));
        }

        let mut audit = VisionDispatchAudit::default();
        let mut hidden = self.matmul(
            &input.patches,
            patch_tokens,
            PATCH_WIDTH,
            "model.visual.patch_embed.proj.weight",
            HIDDEN,
            &mut audit,
        )?;
        add_bias_round(
            &mut hidden,
            self.host("model.visual.patch_embed.proj.bias")?,
            HIDDEN,
        )?;
        self.add_position_embeddings(&mut hidden, grid_h as usize, grid_w as usize)?;
        let positions = vision_positions(grid_h as usize, grid_w as usize);

        for layer in 0..24 {
            let prefix = format!("model.visual.blocks.{layer}");
            let norm1 = layer_norm(
                &hidden,
                patch_tokens,
                HIDDEN,
                self.host(&format!("{prefix}.norm1.weight"))?,
                self.host(&format!("{prefix}.norm1.bias"))?,
            )?;
            let mut qkv = self.matmul(
                &norm1,
                patch_tokens,
                HIDDEN,
                &format!("{prefix}.attn.qkv.weight"),
                HIDDEN * 3,
                &mut audit,
            )?;
            add_bias_round(
                &mut qkv,
                self.host(&format!("{prefix}.attn.qkv.bias"))?,
                HIDDEN * 3,
            )?;
            let attended = vision_attention(&qkv, patch_tokens, &positions)?;
            let mut projected = self.matmul(
                &attended,
                patch_tokens,
                HIDDEN,
                &format!("{prefix}.attn.proj.weight"),
                HIDDEN,
                &mut audit,
            )?;
            add_bias_round(
                &mut projected,
                self.host(&format!("{prefix}.attn.proj.bias"))?,
                HIDDEN,
            )?;
            residual_add_round(&mut hidden, &projected)?;

            let norm2 = layer_norm(
                &hidden,
                patch_tokens,
                HIDDEN,
                self.host(&format!("{prefix}.norm2.weight"))?,
                self.host(&format!("{prefix}.norm2.bias"))?,
            )?;
            let mut mlp = self.matmul(
                &norm2,
                patch_tokens,
                HIDDEN,
                &format!("{prefix}.mlp.linear_fc1.weight"),
                INTERMEDIATE,
                &mut audit,
            )?;
            add_bias_round(
                &mut mlp,
                self.host(&format!("{prefix}.mlp.linear_fc1.bias"))?,
                INTERMEDIATE,
            )?;
            gelu_tanh_round(&mut mlp);
            let mut mlp_out = self.matmul(
                &mlp,
                patch_tokens,
                INTERMEDIATE,
                &format!("{prefix}.mlp.linear_fc2.weight"),
                HIDDEN,
                &mut audit,
            )?;
            add_bias_round(
                &mut mlp_out,
                self.host(&format!("{prefix}.mlp.linear_fc2.bias"))?,
                HIDDEN,
            )?;
            residual_add_round(&mut hidden, &mlp_out)?;
        }

        let merged_norm = layer_norm(
            &hidden,
            patch_tokens,
            HIDDEN,
            self.host("model.visual.merger.norm.weight")?,
            self.host("model.visual.merger.norm.bias")?,
        )?;
        let visual_tokens = patch_tokens / 4;
        let mut merged = self.matmul(
            &merged_norm,
            visual_tokens,
            INTERMEDIATE,
            "model.visual.merger.linear_fc1.weight",
            INTERMEDIATE,
            &mut audit,
        )?;
        add_bias_round(
            &mut merged,
            self.host("model.visual.merger.linear_fc1.bias")?,
            INTERMEDIATE,
        )?;
        gelu_tanh_round(&mut merged);
        let mut output = self.matmul(
            &merged,
            visual_tokens,
            INTERMEDIATE,
            "model.visual.merger.linear_fc2.weight",
            OUTPUT,
            &mut audit,
        )?;
        add_bias_round(
            &mut output,
            self.host("model.visual.merger.linear_fc2.bias")?,
            OUTPUT,
        )?;
        if audit.dispatches == 0 || !audit.all_hip || audit.fallback {
            return Err(QwenVisionExecutionError::Execution(
                "vision projection did not remain exact HIP/no-fallback".to_owned(),
            ));
        }
        Ok(QwenVisionExecutionOutput {
            embeddings_bf16: output.into_iter().map(f32_to_bf16_rne).collect(),
            visual_tokens,
            patch_tokens,
            dispatches: audit.dispatches,
            all_dispatches_hip: audit.all_hip,
            fallback_used: audit.fallback,
        })
    }

    fn host(&self, name: &str) -> Result<&[f32], QwenVisionExecutionError> {
        self.host_tensors
            .get(name)
            .map(|values| values.as_ref())
            .ok_or_else(|| {
                QwenVisionExecutionError::Invalid(format!("host tensor missing: {name}"))
            })
    }

    fn add_position_embeddings(
        &self,
        hidden: &mut [f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(), QwenVisionExecutionError> {
        let table = self.host("model.visual.pos_embed.weight")?;
        if table.len() != 2_304 * HIDDEN {
            return Err(QwenVisionExecutionError::Invalid(
                "position table length differs".to_owned(),
            ));
        }
        let mut token = 0;
        for block_h in 0..grid_h / 2 {
            for block_w in 0..grid_w / 2 {
                for merge_h in 0..2 {
                    for merge_w in 0..2 {
                        let h = block_h * 2 + merge_h;
                        let w = block_w * 2 + merge_w;
                        let (h0, h1, hf) = interpolation_coordinate(h, grid_h);
                        let (w0, w1, wf) = interpolation_coordinate(w, grid_w);
                        let corners = [
                            (h0 * 48 + w0, (1.0 - hf) * (1.0 - wf)),
                            (h0 * 48 + w1, (1.0 - hf) * wf),
                            (h1 * 48 + w0, hf * (1.0 - wf)),
                            (h1 * 48 + w1, hf * wf),
                        ];
                        for dim in 0..HIDDEN {
                            let position = corners
                                .iter()
                                .map(|(index, weight)| table[index * HIDDEN + dim] * weight)
                                .sum::<f32>();
                            let index = token * HIDDEN + dim;
                            hidden[index] = bf16_round(hidden[index] + position);
                        }
                        token += 1;
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        activation: &[f32],
        m: usize,
        k: usize,
        weight_name: &str,
        n: usize,
        audit: &mut VisionDispatchAudit,
    ) -> Result<Vec<f32>, QwenVisionExecutionError> {
        if activation.len() != m * k {
            return Err(QwenVisionExecutionError::Invalid(format!(
                "activation length differs for {weight_name}"
            )));
        }
        let weight = self.tensors.get(weight_name).ok_or_else(|| {
            QwenVisionExecutionError::Invalid(format!("resident tensor missing: {weight_name}"))
        })?;
        let expected_elements = n.checked_mul(k).ok_or_else(|| {
            QwenVisionExecutionError::Invalid("weight elements overflowed".to_owned())
        })?;
        if weight.buffer.size_bytes() != (expected_elements * 2) as u64 {
            return Err(QwenVisionExecutionError::Invalid(format!(
                "weight shape differs for {weight_name}: {:?}",
                weight.shape
            )));
        }
        let activation_view = TensorView::contiguous(DType::Bf16, &[m, k])?;
        let weight_view = TensorView::contiguous(DType::Bf16, &[n, k])?;
        let output_view = TensorView::contiguous(DType::Bf16, &[m, n])?;
        let activation_buffer = self.session.allocate(activation_view.payload_bytes())?;
        let output_buffer = self.session.allocate(output_view.payload_bytes())?;
        let bytes = activation
            .iter()
            .flat_map(|value| f32_to_bf16_rne(*value).to_le_bytes())
            .collect::<Vec<_>>();
        upload_bytes(
            self.session.as_ref(),
            &self.queue,
            &activation_buffer,
            &bytes,
            self.completion_timeout,
        )?;
        let descriptor = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )?);
        let operation = Arc::new(BoundSemanticOp::new(
            descriptor,
            vec![
                self.session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)?,
                self.session
                    .bind(&weight.buffer, weight_view, AccessMode::Read)?,
            ],
            vec![
                self.session
                    .bind(&output_buffer, output_view, AccessMode::Write)?,
            ],
        )?);
        let prepared = self.session.prepare(operation)?;
        let mut submission = self.session.submit(&prepared, &self.queue)?;
        record_dispatch(audit, submission.dispatch());
        require_success(submission.wait(self.completion_timeout)?, "vision matmul")?;
        let bytes = read_bytes(
            self.session.as_ref(),
            &self.queue,
            &output_buffer,
            self.completion_timeout,
        )?;
        decode_bf16(&bytes)
    }
}

#[derive(Default)]
struct VisionDispatchAudit {
    dispatches: u64,
    all_hip: bool,
    fallback: bool,
}

fn record_dispatch(audit: &mut VisionDispatchAudit, dispatch: &DispatchEvidence) {
    let first = audit.dispatches == 0;
    audit.dispatches += u64::from(dispatch.dispatch_count);
    audit.all_hip = (first || audit.all_hip) && dispatch.backend == 1;
    audit.fallback |= dispatch.fallback_used;
}

fn needs_host_copy(name: &str) -> bool {
    name.ends_with(".bias")
        || name.ends_with(".norm1.weight")
        || name.ends_with(".norm2.weight")
        || name == "model.visual.merger.norm.weight"
        || name == "model.visual.pos_embed.weight"
}

fn upload_tensor(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    cache: &VerifiedCache,
    name: &str,
    size: u64,
    timeout: Duration,
) -> Result<(), QwenVisionExecutionError> {
    let max = usize::try_from(session.max_transfer_bytes()?)
        .map_err(|_| QwenVisionExecutionError::Invalid("transfer bound is too large".to_owned()))?
        .min(16 * 1024 * 1024);
    if max == 0 {
        return Err(QwenVisionExecutionError::Invalid(
            "transfer bound must be nonzero".to_owned(),
        ));
    }
    let mut offset = 0_u64;
    while offset < size {
        let length = usize::try_from((size - offset).min(max as u64)).map_err(|_| {
            QwenVisionExecutionError::Invalid("transfer length does not fit usize".to_owned())
        })?;
        let bytes = cache
            .read_tensor_range(name, offset, length)
            .map_err(|error| QwenVisionExecutionError::Invalid(error.to_string()))?;
        let mut transfer = session.upload(
            queue,
            buffer.range(offset, length as u64)?,
            Arc::from(bytes),
        )?;
        require_success(transfer.wait(timeout)?, "vision weight upload")?;
        offset += length as u64;
    }
    Ok(())
}

fn upload_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    bytes: &[u8],
    timeout: Duration,
) -> Result<(), QwenVisionExecutionError> {
    let max = usize::try_from(session.max_transfer_bytes()?)
        .map_err(|_| QwenVisionExecutionError::Invalid("transfer bound is too large".to_owned()))?;
    for (index, chunk) in bytes.chunks(max).enumerate() {
        let offset = index * max;
        let mut transfer = session.upload(
            queue,
            buffer.range(offset as u64, chunk.len() as u64)?,
            Arc::from(chunk),
        )?;
        require_success(transfer.wait(timeout)?, "vision activation upload")?;
    }
    Ok(())
}

fn read_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    timeout: Duration,
) -> Result<Vec<u8>, QwenVisionExecutionError> {
    let max = session.max_transfer_bytes()?;
    let mut bytes = Vec::with_capacity(buffer.size_bytes() as usize);
    let mut offset = 0_u64;
    while offset < buffer.size_bytes() {
        let length = (buffer.size_bytes() - offset).min(max);
        let mut readback = session.readback(queue, buffer.range(offset, length)?)?;
        require_success(readback.wait(timeout)?, "vision output readback")?;
        let old = bytes.len();
        bytes.resize(old + length as usize, 0);
        let copied = readback.read_into(&mut bytes[old..])?;
        if copied != length {
            return Err(QwenVisionExecutionError::Execution(
                "vision output readback length differs".to_owned(),
            ));
        }
        offset += length;
    }
    Ok(bytes)
}

fn require_success(state: ExecutionState, stage: &str) -> Result<(), QwenVisionExecutionError> {
    if state == ExecutionState::Success {
        Ok(())
    } else {
        Err(QwenVisionExecutionError::Execution(format!(
            "{stage} completed as {state:?}"
        )))
    }
}

fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>, QwenVisionExecutionError> {
    if bytes.len() % 2 != 0 {
        return Err(QwenVisionExecutionError::Invalid(
            "BF16 payload length is odd".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect())
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return if bits & 0x007f_ffff == 0 {
            (bits >> 16) as u16
        } else {
            ((bits >> 16) as u16) | 0x0040
        };
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits(u32::from(f32_to_bf16_rne(value)) << 16)
}

fn add_bias_round(
    values: &mut [f32],
    bias: &[f32],
    width: usize,
) -> Result<(), QwenVisionExecutionError> {
    if bias.len() != width || values.len() % width != 0 {
        return Err(QwenVisionExecutionError::Invalid(
            "bias broadcast shape differs".to_owned(),
        ));
    }
    for row in values.chunks_exact_mut(width) {
        for (value, bias) in row.iter_mut().zip(bias) {
            *value = bf16_round(*value + *bias);
        }
    }
    Ok(())
}

fn residual_add_round(
    values: &mut [f32],
    residual: &[f32],
) -> Result<(), QwenVisionExecutionError> {
    if values.len() != residual.len() {
        return Err(QwenVisionExecutionError::Invalid(
            "residual shape differs".to_owned(),
        ));
    }
    for (value, residual) in values.iter_mut().zip(residual) {
        *value = bf16_round(*value + *residual);
    }
    Ok(())
}

fn layer_norm(
    values: &[f32],
    rows: usize,
    width: usize,
    scale: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, QwenVisionExecutionError> {
    if values.len() != rows * width || scale.len() != width || bias.len() != width {
        return Err(QwenVisionExecutionError::Invalid(
            "LayerNorm shape differs".to_owned(),
        ));
    }
    let mut output = vec![0.0; values.len()];
    for (source, target) in values
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let mean = source.iter().sum::<f32>() / width as f32;
        let variance = source
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + 1.0e-6).sqrt();
        for dim in 0..width {
            target[dim] = bf16_round((source[dim] - mean) * inverse * scale[dim] + bias[dim]);
        }
    }
    Ok(output)
}

fn gelu_tanh_round(values: &mut [f32]) {
    const FACTOR: f32 = 0.797_884_6;
    for value in values {
        let x = *value;
        *value = bf16_round(0.5 * x * (1.0 + (FACTOR * (x + 0.044_715 * x * x * x)).tanh()));
    }
}

fn interpolation_coordinate(index: usize, length: usize) -> (usize, usize, f32) {
    let value = if length == 1 {
        0.0
    } else {
        index as f32 * 47.0 / (length - 1) as f32
    };
    let floor = value as usize;
    (floor, (floor + 1).min(47), value - floor as f32)
}

fn vision_positions(grid_h: usize, grid_w: usize) -> Vec<[u32; 2]> {
    let mut positions = Vec::with_capacity(grid_h * grid_w);
    for block_h in 0..grid_h / 2 {
        for block_w in 0..grid_w / 2 {
            for merge_h in 0..2 {
                for merge_w in 0..2 {
                    positions.push([
                        (block_h * 2 + merge_h) as u32,
                        (block_w * 2 + merge_w) as u32,
                    ]);
                }
            }
        }
    }
    positions
}

fn vision_attention(
    qkv: &[f32],
    tokens: usize,
    positions: &[[u32; 2]],
) -> Result<Vec<f32>, QwenVisionExecutionError> {
    if qkv.len() != tokens * HIDDEN * 3 || positions.len() != tokens {
        return Err(QwenVisionExecutionError::Invalid(
            "vision attention shape differs".to_owned(),
        ));
    }
    let per_head = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            handles.push(scope.spawn(move || attention_head(qkv, tokens, positions, head)));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    QwenVisionExecutionError::Execution(
                        "vision attention worker panicked".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut output = vec![0.0; tokens * HIDDEN];
    for (head, values) in per_head.into_iter().enumerate() {
        for token in 0..tokens {
            output[token * HIDDEN + head * HEAD_DIM..token * HIDDEN + (head + 1) * HEAD_DIM]
                .copy_from_slice(&values[token * HEAD_DIM..(token + 1) * HEAD_DIM]);
        }
    }
    Ok(output)
}

fn attention_head(qkv: &[f32], tokens: usize, positions: &[[u32; 2]], head: usize) -> Vec<f32> {
    let mut query = vec![0.0; tokens * HEAD_DIM];
    let mut key = vec![0.0; tokens * HEAD_DIM];
    let mut value = vec![0.0; tokens * HEAD_DIM];
    for token in 0..tokens {
        let base = token * HIDDEN * 3;
        let head_offset = head * HEAD_DIM;
        query[token * HEAD_DIM..(token + 1) * HEAD_DIM]
            .copy_from_slice(&qkv[base + head_offset..base + head_offset + HEAD_DIM]);
        key[token * HEAD_DIM..(token + 1) * HEAD_DIM].copy_from_slice(
            &qkv[base + HIDDEN + head_offset..base + HIDDEN + head_offset + HEAD_DIM],
        );
        value[token * HEAD_DIM..(token + 1) * HEAD_DIM].copy_from_slice(
            &qkv[base + HIDDEN * 2 + head_offset..base + HIDDEN * 2 + head_offset + HEAD_DIM],
        );
        rotate_vision(
            &mut query[token * HEAD_DIM..(token + 1) * HEAD_DIM],
            positions[token],
        );
        rotate_vision(
            &mut key[token * HEAD_DIM..(token + 1) * HEAD_DIM],
            positions[token],
        );
    }
    let mut output = vec![0.0; tokens * HEAD_DIM];
    let mut scores = vec![0.0; tokens];
    for row in 0..tokens {
        let q = &query[row * HEAD_DIM..(row + 1) * HEAD_DIM];
        for column in 0..tokens {
            let k = &key[column * HEAD_DIM..(column + 1) * HEAD_DIM];
            scores[column] = q
                .iter()
                .zip(k)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                / 8.0;
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            total += *score;
        }
        for dim in 0..HEAD_DIM {
            let sum = (0..tokens)
                .map(|column| scores[column] / total * value[column * HEAD_DIM + dim])
                .sum::<f32>();
            output[row * HEAD_DIM + dim] = bf16_round(sum);
        }
    }
    output
}

fn rotate_vision(values: &mut [f32], position: [u32; 2]) {
    let original = values.to_vec();
    for pair in 0..32 {
        let axis = usize::from(pair >= 16);
        let frequency = pair % 16;
        let angle = position[axis] as f32 * 10_000.0_f32.powf(-((2 * frequency) as f32) / 32.0);
        let cosine = angle.cos();
        let sine = angle.sin();
        values[pair] = bf16_round(original[pair] * cosine - original[pair + 32] * sine);
        values[pair + 32] = bf16_round(original[pair] * sine + original[pair + 32] * cosine);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_positions_preserve_merge_block_order() {
        assert_eq!(
            vision_positions(2, 4),
            vec![
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
    }

    #[test]
    fn interpolation_reaches_both_table_edges() {
        assert_eq!(interpolation_coordinate(0, 17), (0, 1, 0.0));
        assert_eq!(interpolation_coordinate(16, 17), (47, 47, 0.0));
    }

    #[test]
    fn layer_norm_rounds_a_non_aligned_row() {
        let values = vec![1.0, 2.0, 4.0];
        let output = layer_norm(&values, 1, 3, &[1.0; 3], &[0.0; 3]).unwrap();
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().sum::<f32>().abs() < 0.02);
    }
}
