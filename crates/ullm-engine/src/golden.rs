// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const F32_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct GoldenMetadata {
    pub format: String,
    pub format_version: String,
    pub model_dir: String,
    pub model_type: Option<String>,
    pub layer_start: Option<usize>,
    pub layer_end_exclusive: Option<usize>,
    pub fixture_kind: Option<String>,
    pub export_command: Option<String>,
    pub torch_version: Option<String>,
    pub dtype: String,
    pub token_ids: Vec<u64>,
    pub position_ids: Vec<u64>,
    pub sequence_len: usize,
    pub hidden_size: usize,
    pub layers: Vec<GoldenLayerFixture>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct GoldenLayerFixture {
    pub layer_index: usize,
    pub before_file: String,
    pub after_file: String,
    pub before_shape: Vec<usize>,
    pub after_shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoldenComparisonMetrics {
    pub mse: f64,
    pub mean_abs_diff: f64,
    pub max_abs_diff: f64,
    pub cosine_similarity: f64,
}

#[derive(Debug)]
pub struct GoldenTensorFixture {
    metadata: GoldenMetadata,
    fixture_dir: PathBuf,
}

impl GoldenTensorFixture {
    pub fn load<P: AsRef<Path>>(fixture_dir: P) -> Result<Self, String> {
        let fixture_dir = fixture_dir.as_ref();
        let metadata_path = fixture_dir.join("metadata.json");
        let mut file = File::open(&metadata_path).map_err(|err| {
            format!(
                "failed to open golden metadata {}: {err}",
                metadata_path.display()
            )
        })?;
        let mut json = String::new();
        file.read_to_string(&mut json).map_err(|err| {
            format!(
                "failed to read golden metadata {}: {err}",
                metadata_path.display()
            )
        })?;
        let metadata: GoldenMetadata = serde_json::from_str(&json).map_err(|err| {
            format!(
                "failed to parse golden metadata {}: {err}",
                metadata_path.display()
            )
        })?;

        if metadata.layers.is_empty() {
            return Err(format!(
                "golden metadata {} does not contain layers",
                metadata_path.display()
            ));
        }

        Ok(Self {
            metadata,
            fixture_dir: fixture_dir.to_path_buf(),
        })
    }

    pub fn layers(&self) -> &[GoldenLayerFixture] {
        &self.metadata.layers
    }

    pub fn metadata(&self) -> &GoldenMetadata {
        &self.metadata
    }

    pub fn select_contiguous_layers(
        &self,
        layer_start: usize,
        layer_end_exclusive: usize,
    ) -> Result<Vec<&GoldenLayerFixture>, String> {
        if layer_end_exclusive <= layer_start {
            return Err(format!(
                "invalid layer range: start={layer_start}, end_exclusive={layer_end_exclusive}"
            ));
        }

        let mut by_index = BTreeMap::<usize, &GoldenLayerFixture>::new();
        for layer in self.layers() {
            if by_index.insert(layer.layer_index, layer).is_some() {
                return Err(format!(
                    "duplicate layer entry in fixture metadata: layer_index={}",
                    layer.layer_index
                ));
            }
        }

        let mut layers = Vec::with_capacity(layer_end_exclusive - layer_start);
        for layer_index in layer_start..layer_end_exclusive {
            let layer = by_index.get(&layer_index).ok_or_else(|| {
                format!(
                    "missing layer in golden fixture: expected contiguous layer_index={layer_index} in range {layer_start}..{layer_end_exclusive}"
                )
            })?;
            layers.push(*layer);
        }

        Ok(layers)
    }

    pub fn read_initial_before_f32(&self, layer_start: usize) -> Result<Vec<f32>, String> {
        let layer = self.select_layer(layer_start)?;
        self.read_f32_payload(
            &layer.before_file,
            &layer.before_shape,
            layer_start,
            "initial before",
        )
    }

    pub fn select_layer(&self, layer_index: usize) -> Result<&GoldenLayerFixture, String> {
        self.layers()
            .iter()
            .find(|layer| layer.layer_index == layer_index)
            .ok_or_else(|| {
                format!("golden fixture has no layer entry for layer_index={layer_index}")
            })
    }

    pub fn get_layer(&self, layer_index: usize) -> Result<&GoldenLayerFixture, String> {
        self.select_layer(layer_index)
    }

    pub fn read_layer_before_f32(&self, layer_index: usize) -> Result<Vec<f32>, String> {
        let layer = self.select_layer(layer_index)?;
        self.read_f32_payload(
            &layer.before_file,
            &layer.before_shape,
            layer_index,
            "before",
        )
    }

    pub fn read_layer_after_f32(&self, layer_index: usize) -> Result<Vec<f32>, String> {
        let layer = self.select_layer(layer_index)?;
        self.read_f32_payload(&layer.after_file, &layer.after_shape, layer_index, "after")
    }

    fn read_f32_payload(
        &self,
        file_name: &str,
        shape: &[usize],
        layer_index: usize,
        position: &str,
    ) -> Result<Vec<f32>, String> {
        let path = self.fixture_dir.join(file_name);
        let mut file = File::open(&path).map_err(|err| {
            format!(
                "failed to open layer {layer_index} {position} payload {}: {err}",
                path.display()
            )
        })?;

        let expected_elements = shape_element_count(shape)
            .map_err(|err| format!("layer {layer_index} {position} shape mismatch: {err}"))?;
        let expected_bytes = expected_elements.checked_mul(F32_BYTES).ok_or_else(|| {
            format!("layer {layer_index} {position} payload element bytes overflow")
        })?;

        let mut payload = Vec::new();
        file.read_to_end(&mut payload).map_err(|err| {
            format!(
                "failed to read layer {layer_index} {position} payload {}: {err}",
                path.display()
            )
        })?;

        if !payload.len().is_multiple_of(F32_BYTES) {
            return Err(format!(
                "layer {layer_index} {position} payload byte length {} is not aligned to 4-byte f32 values",
                payload.len()
            ));
        }
        if payload.len() != expected_bytes {
            return Err(format!(
                "layer {layer_index} {position} payload byte length mismatch: expected {expected_bytes}, got {}",
                payload.len()
            ));
        }

        let mut values = Vec::with_capacity(expected_elements);
        for chunk in payload.chunks_exact(F32_BYTES) {
            values.push(f32::from_le_bytes(chunk.try_into().map_err(|_| {
                format!("layer {layer_index} {position} payload has malformed f32 chunk")
            })?));
        }
        if values.len() != expected_elements {
            return Err(format!(
                "layer {layer_index} {position} payload length mismatch: expected {} elements, got {}",
                expected_elements,
                values.len()
            ));
        }
        Ok(values)
    }
}

pub fn compare_f32_slices(
    before: &[f32],
    after: &[f32],
) -> Result<GoldenComparisonMetrics, String> {
    if before.len() != after.len() {
        return Err(format!(
            "cannot compare slices with different lengths: before={} after={}",
            before.len(),
            after.len()
        ));
    }

    if before.is_empty() {
        return Ok(GoldenComparisonMetrics {
            mse: 0.0,
            mean_abs_diff: 0.0,
            max_abs_diff: 0.0,
            cosine_similarity: 1.0,
        });
    }

    let mut sum_sq = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut left_norm_sq = 0.0_f64;
    let mut right_norm_sq = 0.0_f64;

    for (left, right) in before.iter().zip(after.iter()) {
        let delta = f64::from(*left - *right);
        sum_sq += delta * delta;
        let abs = f64::from((left - right).abs());
        sum_abs += abs;
        if abs > max_abs {
            max_abs = abs;
        }
        let left_f64 = f64::from(*left);
        let right_f64 = f64::from(*right);
        dot += left_f64 * right_f64;
        left_norm_sq += left_f64 * left_f64;
        right_norm_sq += right_f64 * right_f64;
    }

    let len = before.len() as f64;
    let cosine_similarity = cosine_similarity_from_stats(dot, left_norm_sq, right_norm_sq);

    Ok(GoldenComparisonMetrics {
        mse: sum_sq / len,
        mean_abs_diff: sum_abs / len,
        max_abs_diff: max_abs,
        cosine_similarity,
    })
}

fn cosine_similarity_from_stats(dot: f64, left_norm_sq: f64, right_norm_sq: f64) -> f64 {
    if left_norm_sq == 0.0 && right_norm_sq == 0.0 {
        1.0
    } else if left_norm_sq == 0.0 || right_norm_sq == 0.0 {
        0.0
    } else {
        dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt())
    }
}

fn shape_element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Err("shape is empty".to_string());
    }
    let mut elements = 1_usize;
    for dim in shape {
        if *dim == 0 {
            return Err("shape contains zero".to_string());
        }
        elements = elements
            .checked_mul(*dim)
            .ok_or_else(|| "shape element count overflows usize".to_string())?;
    }
    Ok(elements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{now}"))
    }

    fn write_f32_le(path: &Path, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(path, bytes).expect("write raw f32 payload");
    }

    fn write_metadata(path: &Path, metadata: &GoldenMetadata) {
        let json = serde_json::to_string_pretty(metadata).expect("serialize metadata");
        fs::write(path, json).expect("write golden metadata");
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn can_load_and_read_layer_fixtures() {
        let root = unique_temp_dir("golden-fixture-read-test");
        fs::create_dir_all(root.clone()).expect("create test dir");

        write_f32_le(
            &root.join("before.raw"),
            &[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        );
        write_f32_le(
            &root.join("after.raw"),
            &[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32],
        );

        write_metadata(
            &root.join("metadata.json"),
            &GoldenMetadata {
                format: "golden-v1".to_string(),
                format_version: "0.1".to_string(),
                model_dir: "/tmp/model".to_string(),
                model_type: Some("qwen3.5".to_string()),
                layer_start: None,
                layer_end_exclusive: None,
                fixture_kind: None,
                export_command: None,
                torch_version: None,
                dtype: "F32".to_string(),
                token_ids: vec![10, 11, 12],
                position_ids: vec![0, 1, 2],
                sequence_len: 3,
                hidden_size: 4,
                layers: vec![GoldenLayerFixture {
                    layer_index: 2,
                    before_file: "before.raw".to_string(),
                    after_file: "after.raw".to_string(),
                    before_shape: vec![1, 4],
                    after_shape: vec![1, 4],
                    dtype: "F32".to_string(),
                }],
            },
        );

        let fixture = GoldenTensorFixture::load(&root).expect("load fixture");
        let layer = fixture.get_layer(2).expect("find layer");
        assert_eq!(layer.layer_index, 2);
        let contiguous = fixture
            .select_contiguous_layers(2, 3)
            .expect("select contiguous layer range");
        assert_eq!(contiguous.len(), 1);
        assert_eq!(contiguous[0].layer_index, 2);

        let before = fixture.read_layer_before_f32(2).expect("read before");
        let initial = fixture
            .read_initial_before_f32(2)
            .expect("read initial before");
        let after = fixture.read_layer_after_f32(2).expect("read after");
        assert_eq!(before, initial);
        assert_eq!(before, vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]);
        assert_eq!(after, vec![2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32]);

        let metrics = compare_f32_slices(&before, &after).expect("compare");
        assert_eq!(metrics.max_abs_diff, 1.0_f64);
        assert!((metrics.mse - 1.0).abs() < 1e-12);
        assert!((metrics.mean_abs_diff - 1.0).abs() < 1e-12);
        assert!((metrics.cosine_similarity - 0.9938079899999065).abs() < 1e-12);

        cleanup(&root);
    }

    #[test]
    fn rejects_payload_byte_mismatch() {
        let root = unique_temp_dir("golden-fixture-byte-mismatch");
        fs::create_dir_all(root.clone()).expect("create test dir");

        write_f32_le(&root.join("before.raw"), &[1.0_f32]);
        write_f32_le(&root.join("after.raw"), &[2.0_f32, 3.0_f32]);

        write_metadata(
            &root.join("metadata.json"),
            &GoldenMetadata {
                format: "golden-v1".to_string(),
                format_version: "0.1".to_string(),
                model_dir: "/tmp/model".to_string(),
                model_type: Some("qwen3.5".to_string()),
                layer_start: None,
                layer_end_exclusive: None,
                fixture_kind: None,
                export_command: None,
                torch_version: None,
                dtype: "F32".to_string(),
                token_ids: vec![1, 2],
                position_ids: vec![0, 1],
                sequence_len: 2,
                hidden_size: 2,
                layers: vec![GoldenLayerFixture {
                    layer_index: 0,
                    before_file: "before.raw".to_string(),
                    after_file: "after.raw".to_string(),
                    before_shape: vec![1, 4],
                    after_shape: vec![1, 4],
                    dtype: "F32".to_string(),
                }],
            },
        );

        let fixture = GoldenTensorFixture::load(&root).expect("load fixture");
        let err = fixture
            .read_layer_after_f32(0)
            .expect_err("payload byte mismatch should fail");
        assert!(err.contains("payload byte length mismatch"));

        cleanup(&root);
    }

    #[test]
    fn rejects_missing_layer_in_range() {
        let root = unique_temp_dir("golden-fixture-missing-layer");
        fs::create_dir_all(root.clone()).expect("create test dir");

        write_f32_le(
            &root.join("before.raw"),
            &[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        );
        write_f32_le(
            &root.join("after.raw"),
            &[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32],
        );

        write_metadata(
            &root.join("metadata.json"),
            &GoldenMetadata {
                format: "golden-v1".to_string(),
                format_version: "0.1".to_string(),
                model_dir: "/tmp/model".to_string(),
                model_type: Some("qwen3.5".to_string()),
                layer_start: None,
                layer_end_exclusive: None,
                fixture_kind: None,
                export_command: None,
                torch_version: None,
                dtype: "F32".to_string(),
                token_ids: vec![1, 2],
                position_ids: vec![0, 1],
                sequence_len: 2,
                hidden_size: 4,
                layers: vec![
                    GoldenLayerFixture {
                        layer_index: 0,
                        before_file: "before.raw".to_string(),
                        after_file: "after.raw".to_string(),
                        before_shape: vec![1, 4],
                        after_shape: vec![1, 4],
                        dtype: "F32".to_string(),
                    },
                    GoldenLayerFixture {
                        layer_index: 2,
                        before_file: "before.raw".to_string(),
                        after_file: "after.raw".to_string(),
                        before_shape: vec![1, 4],
                        after_shape: vec![1, 4],
                        dtype: "F32".to_string(),
                    },
                ],
            },
        );

        let fixture = GoldenTensorFixture::load(&root).expect("load fixture");
        let err = fixture
            .select_layer(1)
            .expect_err("missing layer should fail");
        assert!(err.contains("no layer entry for layer_index=1"));

        let err = fixture
            .select_contiguous_layers(0, 3)
            .expect_err("non-contiguous layer range should fail");
        assert!(err.contains("missing layer in golden fixture"));

        cleanup(&root);
    }

    #[test]
    fn rejects_non_contiguous_layer_range() {
        let root = unique_temp_dir("golden-fixture-contiguous-range");
        fs::create_dir_all(root.clone()).expect("create test dir");

        write_f32_le(
            &root.join("before0.raw"),
            &[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        );
        write_f32_le(
            &root.join("after0.raw"),
            &[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32],
        );
        write_f32_le(
            &root.join("before2.raw"),
            &[9.0_f32, 8.0_f32, 7.0_f32, 6.0_f32],
        );
        write_f32_le(
            &root.join("after2.raw"),
            &[8.0_f32, 7.0_f32, 6.0_f32, 5.0_f32],
        );

        write_metadata(
            &root.join("metadata.json"),
            &GoldenMetadata {
                format: "golden-v1".to_string(),
                format_version: "0.1".to_string(),
                model_dir: "/tmp/model".to_string(),
                model_type: Some("qwen3.5".to_string()),
                layer_start: None,
                layer_end_exclusive: None,
                fixture_kind: None,
                export_command: None,
                torch_version: None,
                dtype: "F32".to_string(),
                token_ids: vec![1, 2],
                position_ids: vec![0, 1],
                sequence_len: 2,
                hidden_size: 4,
                layers: vec![
                    GoldenLayerFixture {
                        layer_index: 2,
                        before_file: "before2.raw".to_string(),
                        after_file: "after2.raw".to_string(),
                        before_shape: vec![1, 4],
                        after_shape: vec![1, 4],
                        dtype: "F32".to_string(),
                    },
                    GoldenLayerFixture {
                        layer_index: 0,
                        before_file: "before0.raw".to_string(),
                        after_file: "after0.raw".to_string(),
                        before_shape: vec![1, 4],
                        after_shape: vec![1, 4],
                        dtype: "F32".to_string(),
                    },
                ],
            },
        );

        let fixture = GoldenTensorFixture::load(&root).expect("load fixture");
        let err = fixture
            .select_contiguous_layers(0, 3)
            .expect_err("non-contiguous layer range should fail");
        assert!(err.contains("missing layer in golden fixture"));

        cleanup(&root);
    }

    #[test]
    fn compare_function_returns_expected_metrics() {
        let before = vec![1.0_f32, 0.0_f32, -1.0_f32];
        let after = vec![1.0_f32, 1.0_f32, 0.0_f32];
        let metrics = compare_f32_slices(&before, &after).expect("compare");

        assert_eq!(metrics.max_abs_diff, 1.0);
        assert!((metrics.mean_abs_diff - 0.6666666666666666).abs() < 1e-12);
        assert!((metrics.mse - 0.6666666666666666).abs() < 1e-12);
        assert!((metrics.cosine_similarity - 0.5).abs() < 1e-12);

        let zero = vec![0.0_f32; 3];
        let zero_metrics = compare_f32_slices(&zero, &zero).expect("compare");
        assert_eq!(zero_metrics.cosine_similarity, 1.0);
        let not_zero_cosine = compare_f32_slices(&zero, &after).expect("compare");
        assert_eq!(not_zero_cosine.cosine_similarity, 0.0);
    }
}
