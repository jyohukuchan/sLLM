// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::package::{ReferencedFile, ReferencedFileRole, TensorPayloadBundle};
use std::fs::File;
use std::io::Read;
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    pub chunk_bytes: usize,
    pub verify: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            chunk_bytes: 1024 * 1024,
            verify: true,
        }
    }
}

#[derive(Debug)]
pub struct LoadedPayload {
    pub role: ReferencedFileRole,
    pub relative_path: String,
    pub bytes: u64,
    pub chunks: u64,
    pub buffer: RuntimeBuffer,
}

#[derive(Debug)]
pub struct LoadedTensorBundle {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub dtype: Option<String>,
    pub family: Option<String>,
    pub candidate_id: Option<String>,
    pub elements: u64,
    pub groups: u64,
    pub index: LoadedPayload,
    pub scale: LoadedPayload,
    pub codebook: LoadedPayload,
}

impl LoadedTensorBundle {
    pub fn total_payload_bytes(&self) -> u64 {
        self.index.bytes + self.scale.bytes + self.codebook.bytes
    }
}

#[derive(Debug, Default)]
pub struct WeightRegistry {
    tensors: Vec<LoadedTensorBundle>,
}

impl WeightRegistry {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    pub fn insert(&mut self, bundle: LoadedTensorBundle) -> usize {
        let index = self.tensors.len();
        self.tensors.push(bundle);
        index
    }

    pub fn load_and_insert(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        bundle: &TensorPayloadBundle,
        options: LoadOptions,
    ) -> Result<usize, String> {
        let loaded = load_tensor_payload_bundle(context, stream, bundle, options)?;
        Ok(self.insert(loaded))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn total_payload_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(LoadedTensorBundle::total_payload_bytes)
            .sum()
    }

    pub fn get(&self, index: usize) -> Option<&LoadedTensorBundle> {
        self.tensors.get(index)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&LoadedTensorBundle> {
        self.tensors
            .iter()
            .find(|bundle| bundle.tensor_name == name)
    }
}

pub fn load_tensor_payload_bundle(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    bundle: &TensorPayloadBundle,
    options: LoadOptions,
) -> Result<LoadedTensorBundle, String> {
    if options.chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let index = load_payload_file(context, stream, &bundle.index_file, options)?;
    let scale = load_payload_file(context, stream, &bundle.scale_file, options)?;
    let codebook = load_payload_file(context, stream, &bundle.codebook_file, options)?;

    Ok(LoadedTensorBundle {
        tensor_index: bundle.tensor_index,
        tensor_name: bundle.tensor_name.clone(),
        dtype: bundle.dtype.clone(),
        family: bundle.family.clone(),
        candidate_id: bundle.candidate_id.clone(),
        elements: bundle.elements,
        groups: bundle.groups,
        index,
        scale,
        codebook,
    })
}

fn load_payload_file(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    referenced: &ReferencedFile,
    options: LoadOptions,
) -> Result<LoadedPayload, String> {
    let buffer_bytes = usize::try_from(referenced.bytes).map_err(|_| {
        format!(
            "payload {} is too large for this host: {} bytes",
            referenced.relative_path, referenced.bytes
        )
    })?;
    if buffer_bytes == 0 {
        return Err(format!("payload {} is empty", referenced.relative_path));
    }

    let mut file = File::open(&referenced.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            referenced.absolute_path.display()
        )
    })?;
    let mut buffer = context.alloc_buffer(buffer_bytes)?;
    let staging_bytes = buffer_bytes.min(options.chunk_bytes);
    let mut input = vec![0_u8; staging_bytes];
    let mut output = if options.verify {
        vec![0_u8; staging_bytes]
    } else {
        Vec::new()
    };

    let mut offset = 0_usize;
    let mut chunks = 0_u64;
    loop {
        let read = file.read(&mut input).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                referenced.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }

        buffer.copy_from_host(offset, &input[..read], Some(stream))?;
        if options.verify {
            stream.synchronize()?;
            buffer.copy_to_host(offset, &mut output[..read], Some(stream))?;
            stream.synchronize()?;
            if input[..read] != output[..read] {
                return Err(format!(
                    "runtime load verification mismatch for {} at chunk {}",
                    referenced.relative_path, chunks
                ));
            }
        }
        offset += read;
        chunks += 1;
    }

    if !options.verify {
        stream.synchronize()?;
    }
    if offset != buffer_bytes {
        return Err(format!(
            "runtime load byte count mismatch for {}: expected {} got {}",
            referenced.relative_path, buffer_bytes, offset
        ));
    }

    Ok(LoadedPayload {
        role: referenced.role,
        relative_path: referenced.relative_path.clone(),
        bytes: referenced.bytes,
        chunks,
        buffer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::{TensorSelector, select_tensor_payload_bundle};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cpu_registry_loads_tensor_payload_bundle() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8, 2, 3, 4, 5]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [6_u8, 7, 8]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [9_u8; 10]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_test",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        let registry_index = registry
            .load_and_insert(
                &mut context,
                &mut stream,
                &bundle,
                LoadOptions {
                    chunk_bytes: 4,
                    verify: true,
                },
            )
            .unwrap();

        assert_eq!(registry_index, 0);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.total_payload_bytes(), 18);
        let loaded = registry.get(0).unwrap();
        assert_eq!(loaded.tensor_name, "layer.0.attn.q_proj.weight");
        assert_eq!(loaded.index.bytes, 5);
        assert_eq!(loaded.index.chunks, 2);
        assert_eq!(loaded.scale.bytes, 3);
        assert_eq!(loaded.scale.chunks, 1);
        assert_eq!(loaded.codebook.bytes, 10);
        assert_eq!(loaded.codebook.chunks, 3);
        assert!(registry.get_by_name("layer.0.attn.q_proj.weight").is_some());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loader_rejects_zero_chunk_size() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-zero-chunk-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [2_u8]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [3_u8]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "a",
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let err = load_tensor_payload_bundle(
            &mut context,
            &mut stream,
            &bundle,
            LoadOptions {
                chunk_bytes: 0,
                verify: true,
            },
        )
        .unwrap_err();
        assert!(err.contains("chunk bytes"));

        fs::remove_dir_all(root).unwrap();
    }
}
