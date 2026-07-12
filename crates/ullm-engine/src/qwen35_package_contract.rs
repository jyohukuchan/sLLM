// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Device-independent contracts for Qwen3/Qwen3.5 package execution.

use crate::package::{list_passthrough_payload_bundles, list_tensor_payload_bundles};
use crate::qwen3_names::qwen3_layer_index_from_tensor_suffix;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const QWEN35_9B_DEFAULT_LAYER_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PackageDecoderLayerKind {
    SelfAttention,
    LinearAttention,
}

impl PackageDecoderLayerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SelfAttention => "self_attention",
            Self::LinearAttention => "linear_attention",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageManifestLayerEntry {
    pub layer_index: usize,
    pub kind: PackageDecoderLayerKind,
}

/// Returns supported decoder layers in ascending manifest layer order.
pub fn package_manifest_layer_entries(
    path: impl AsRef<Path>,
) -> Result<Vec<PackageManifestLayerEntry>, String> {
    let path = path.as_ref();
    let mut layers = BTreeMap::<usize, BTreeSet<PackageDecoderLayerKind>>::new();
    for bundle in list_tensor_payload_bundles(path)? {
        record_layer_kind(
            &mut layers,
            &bundle.tensor_name,
            ".self_attn.q_proj.weight",
            PackageDecoderLayerKind::SelfAttention,
        );
        record_layer_kind(
            &mut layers,
            &bundle.tensor_name,
            ".linear_attn.in_proj_qkv.weight",
            PackageDecoderLayerKind::LinearAttention,
        );
    }
    for layer_index in package_self_attention_layer_indices(path)? {
        layers
            .entry(layer_index)
            .or_default()
            .insert(PackageDecoderLayerKind::SelfAttention);
    }
    if layers.is_empty() {
        return Err(format!(
            "package {} has no supported self_attn or linear_attn layer tensors",
            path.display()
        ));
    }

    layers
        .into_iter()
        .map(|(layer_index, kinds)| {
            let kind = if kinds.len() == 1 {
                *kinds.iter().next().expect("one checked layer kind")
            } else {
                let labels = kinds.iter().map(|kind| kind.as_str()).collect::<Vec<_>>();
                return Err(format!(
                    "package {} layer {layer_index} has ambiguous layer kinds: {labels:?}",
                    path.display()
                ));
            };
            Ok(PackageManifestLayerEntry { layer_index, kind })
        })
        .collect()
}

fn record_layer_kind(
    layers: &mut BTreeMap<usize, BTreeSet<PackageDecoderLayerKind>>,
    tensor_name: &str,
    suffix: &str,
    kind: PackageDecoderLayerKind,
) {
    if let Some(layer_index) = qwen3_layer_index_from_tensor_suffix(tensor_name, suffix) {
        layers.entry(layer_index).or_default().insert(kind);
    }
}

pub fn package_decoder_layer_kind(
    path: impl AsRef<Path>,
    layer_index: usize,
) -> Result<PackageDecoderLayerKind, String> {
    package_manifest_layer_entries(path)?
        .into_iter()
        .find(|entry| entry.layer_index == layer_index)
        .map(|entry| entry.kind)
        .ok_or_else(|| {
            format!(
                "package layer {layer_index} has neither supported self_attn nor linear_attn package tensors"
            )
        })
}

pub fn package_layer_entries_for_indices(
    path: impl AsRef<Path>,
    layer_indices: &[usize],
) -> Result<Vec<PackageManifestLayerEntry>, String> {
    let entries = package_manifest_layer_entries(path)?;
    layer_indices
        .iter()
        .map(|layer_index| {
            entries
                .iter()
                .copied()
                .find(|entry| entry.layer_index == *layer_index)
                .ok_or_else(|| {
                    format!(
                        "package layer {layer_index} has neither supported self_attn nor linear_attn package tensors"
                    )
                })
        })
        .collect()
}

pub fn package_layer_entries_are_contiguous(entries: &[PackageManifestLayerEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| window[0].layer_index.checked_add(1) == Some(window[1].layer_index))
}

/// Selects Qwen3.5 9B's default 32 layers, all manifest layers, or an explicit CSV.
pub fn select_package_layer_indices(
    path: impl AsRef<Path>,
    value: Option<&str>,
) -> Result<Vec<usize>, String> {
    match value.map(str::trim) {
        None | Some("") | Some("all") | Some("default") => {
            Ok((0..QWEN35_9B_DEFAULT_LAYER_COUNT).collect())
        }
        Some("manifest-all" | "manifest_all" | "all-manifest" | "all_manifest") => {
            Ok(package_manifest_layer_entries(path)?
                .into_iter()
                .map(|entry| entry.layer_index)
                .collect())
        }
        Some(raw) => parse_token_id_csv(raw, "layer list"),
    }
}

pub fn package_self_attention_layer_indices(
    path: impl AsRef<Path>,
) -> Result<BTreeSet<usize>, String> {
    let path = path.as_ref();
    let mut q_norm_layers = BTreeSet::new();
    let mut k_norm_layers = BTreeSet::new();
    for bundle in list_passthrough_payload_bundles(path)? {
        if let Some(layer_index) =
            qwen3_layer_index_from_tensor_suffix(&bundle.tensor_name, ".self_attn.q_norm.weight")
        {
            q_norm_layers.insert(layer_index);
        }
        if let Some(layer_index) =
            qwen3_layer_index_from_tensor_suffix(&bundle.tensor_name, ".self_attn.k_norm.weight")
        {
            k_norm_layers.insert(layer_index);
        }
    }
    if q_norm_layers.is_empty() && k_norm_layers.is_empty() {
        return Ok(BTreeSet::new());
    }
    if q_norm_layers != k_norm_layers {
        return Err(format!(
            "package {} has mismatched self-attention q_norm/k_norm layer sets: q_norm={q_norm_layers:?} k_norm={k_norm_layers:?}",
            path.display()
        ));
    }
    Ok(q_norm_layers)
}

pub fn parse_stop_token_ids(value: Option<&str>) -> Result<Vec<usize>, String> {
    match value.map(str::trim) {
        None | Some("") | Some("-") | Some("none" | "None" | "NONE") => Ok(Vec::new()),
        Some(raw) => parse_token_id_csv(raw, "stop token IDs"),
    }
}

pub fn parse_stop_token_sequences(value: Option<&str>) -> Result<Vec<Vec<usize>>, String> {
    match value.map(str::trim) {
        None | Some("") | Some("-") | Some("none" | "None" | "NONE") => Ok(Vec::new()),
        Some(raw) => raw
            .split(';')
            .map(|sequence| {
                let sequence = sequence.trim();
                if sequence.is_empty() {
                    Err(format!(
                        "invalid stop token sequences {raw:?}: empty sequence"
                    ))
                } else {
                    parse_token_id_csv(sequence, "stop token sequence")
                }
            })
            .collect(),
    }
}

pub fn matched_stop_token_id(
    generated_token_ids: &[usize],
    stop_token_ids: &[usize],
) -> Option<usize> {
    generated_token_ids
        .last()
        .copied()
        .filter(|token_id| stop_token_ids.contains(token_id))
}

pub fn matched_stop_token_sequence(
    generated_token_ids: &[usize],
    stop_token_sequences: &[Vec<usize>],
) -> Option<Vec<usize>> {
    stop_token_sequences
        .iter()
        .find(|sequence| !sequence.is_empty() && generated_token_ids.ends_with(sequence))
        .cloned()
}

fn parse_token_id_csv(value: &str, label: &str) -> Result<Vec<usize>, String> {
    let parsed = value
        .split(',')
        .map(|raw| {
            let entry = raw.trim();
            if entry.is_empty() {
                return Err(format!("invalid {label}: empty entry in {value:?}"));
            }
            entry
                .parse::<usize>()
                .map_err(|err| format!("invalid {label}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err(format!("invalid {label}: expected at least one entry"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_package(label: &str, quantized: &[&str], passthrough: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ullm-qwen35-contract-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("payload")).unwrap();
        for name in ["idx.bin", "scale.bin", "codebook.bin", "raw.bin"] {
            fs::write(root.join("payload").join(name), [0_u8; 4]).unwrap();
        }
        let tensors = quantized
            .iter()
            .map(|name| {
                json!({
                    "name": name,
                    "shape": [1, 1],
                    "elements": 1,
                    "groups": 1,
                    "index_file": "payload/idx.bin",
                    "scale_file": "payload/scale.bin",
                    "codebook_file": "payload/codebook.bin"
                })
            })
            .collect::<Vec<_>>();
        let passthrough_tensors = passthrough
            .iter()
            .map(|name| {
                json!({
                    "name": name,
                    "dtype": "F32",
                    "shape": [1],
                    "elements": 1,
                    "payload_bytes": 4,
                    "payload_file": "payload/raw.bin"
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&json!({
                "schema_version": "test",
                "tensors": tensors,
                "passthrough_tensors": passthrough_tensors
            }))
            .unwrap(),
        )
        .unwrap();
        root
    }

    #[test]
    fn default_and_explicit_layer_selection_are_stable() {
        let layers = select_package_layer_indices("unused", None).unwrap();
        assert_eq!(
            layers,
            (0..QWEN35_9B_DEFAULT_LAYER_COUNT).collect::<Vec<_>>()
        );
        assert_eq!(
            select_package_layer_indices("unused", Some("all")).unwrap(),
            layers
        );
        assert_eq!(
            select_package_layer_indices("unused", Some("0, 2,4")).unwrap(),
            vec![0, 2, 4]
        );
        assert!(select_package_layer_indices("unused", Some("0,,2")).is_err());
    }

    #[test]
    fn manifest_layers_are_sorted_across_qwen_namespaces() {
        let root = test_package(
            "ordered",
            &[
                "model.layers.2.linear_attn.in_proj_qkv.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.linear_attn.in_proj_qkv.weight",
            ],
            &[],
        );
        let entries = package_manifest_layer_entries(&root).unwrap();
        assert_eq!(
            entries,
            vec![
                PackageManifestLayerEntry {
                    layer_index: 0,
                    kind: PackageDecoderLayerKind::SelfAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 1,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 2,
                    kind: PackageDecoderLayerKind::LinearAttention,
                },
            ]
        );
        assert!(package_layer_entries_are_contiguous(&entries));
        assert_eq!(
            select_package_layer_indices(&root, Some("manifest-all")).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            package_decoder_layer_kind(&root, 0).unwrap(),
            PackageDecoderLayerKind::SelfAttention
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_rejects_mixed_kinds_for_one_layer() {
        let root = test_package(
            "ambiguous",
            &[
                "model.layers.4.self_attn.q_proj.weight",
                "model.layers.4.linear_attn.in_proj_qkv.weight",
            ],
            &[],
        );
        let err = package_manifest_layer_entries(&root).unwrap_err();
        assert!(err.contains("layer 4 has ambiguous layer kinds"), "{err}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paired_norm_passthrough_marks_thin_self_attention_layers() {
        let root = test_package(
            "thin",
            &[],
            &[
                "model.layers.1.self_attn.q_norm.weight",
                "model.layers.0.self_attn.q_norm.weight",
                "model.layers.1.self_attn.k_norm.weight",
                "model.layers.0.self_attn.k_norm.weight",
            ],
        );
        assert_eq!(
            package_manifest_layer_entries(&root).unwrap(),
            vec![
                PackageManifestLayerEntry {
                    layer_index: 0,
                    kind: PackageDecoderLayerKind::SelfAttention,
                },
                PackageManifestLayerEntry {
                    layer_index: 1,
                    kind: PackageDecoderLayerKind::SelfAttention,
                },
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stop_contract_parses_and_matches_only_generated_suffixes() {
        assert_eq!(parse_stop_token_ids(None).unwrap(), Vec::<usize>::new());
        assert_eq!(
            parse_stop_token_ids(Some("none")).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(parse_stop_token_ids(Some("1, 2,3")).unwrap(), vec![1, 2, 3]);
        assert!(parse_stop_token_ids(Some("1,,2")).is_err());

        let sequences = parse_stop_token_sequences(Some("1, 2; 3,4,5")).unwrap();
        assert_eq!(sequences, vec![vec![1, 2], vec![3, 4, 5]]);
        assert!(parse_stop_token_sequences(Some("1,2;;3")).is_err());
        assert_eq!(matched_stop_token_id(&[7, 3], &[2, 3]), Some(3));
        assert_eq!(matched_stop_token_id(&[3, 7], &[2, 3]), None);
        assert_eq!(
            matched_stop_token_sequence(&[9, 1, 2], &[vec![1, 2]]),
            Some(vec![1, 2])
        );
        assert_eq!(matched_stop_token_sequence(&[9, 1], &[vec![1, 2]]), None);
        assert_eq!(matched_stop_token_sequence(&[1, 2, 9], &[vec![1, 2]]), None);
    }
}
