// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use ullm_engine::sq::fp8_e4m3fn_to_f32;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_fnuz_prepack::{
    Bf16ScaleTransformError, FnuzScaleCompensation, OcpE4m3FnuzByteError, OcpE4m3FnuzMapping,
    Sq8FnuzPrepackError, bf16_bits_to_f32, fnuz_e4m3_to_f32, map_ocp_e4m3fn_byte_to_fnuz,
    prepack_bf16_scale_bits_for_fnuz, prepack_bf16_scale_payload_for_fnuz,
    prepack_ocp_e4m3fn_payload_to_fnuz, prepack_sq8_ocp_e4m3fn_tensor_to_fnuz,
    scan_sq8_canonical_artifact_for_fnuz_prepack,
};

#[test]
fn exhaustive_256_byte_oracle_enumerates_every_exception() {
    let mut finite_count = 0_u16;
    let mut rejected = Vec::new();

    for raw in 0_u8..=u8::MAX {
        match map_ocp_e4m3fn_byte_to_fnuz(raw) {
            Ok(mapping) => {
                finite_count += 1;
                let mapped = mapping.byte();
                let ocp = fp8_e4m3fn_to_f32(raw);
                let fnuz = fnuz_e4m3_to_f32(mapped);
                assert!(ocp.is_finite(), "finite mapping for raw 0x{raw:02x}");
                assert!(fnuz.is_finite(), "FNUZ mapping for raw 0x{raw:02x}");
                assert_eq!(ocp, 2.0 * fnuz, "OCP = 2 * FNUZ for 0x{raw:02x}");
                if raw == 0x80 {
                    assert_eq!(mapping, OcpE4m3FnuzMapping::NegativeZeroNormalized);
                    assert_eq!(mapped, 0x00);
                    assert_eq!(ocp.to_bits(), (-0.0_f32).to_bits());
                } else {
                    assert_eq!(mapping, OcpE4m3FnuzMapping::Exact(raw));
                }
            }
            Err(OcpE4m3FnuzByteError::OcpNaN { byte }) => rejected.push(byte),
        }
    }

    assert_eq!(finite_count, 254);
    assert_eq!(rejected, [0x7f, 0xff]);
    assert!(fnuz_e4m3_to_f32(0x80).is_nan());
    assert_eq!(fnuz_e4m3_to_f32(0x7f), 240.0);
    assert_eq!(fp8_e4m3fn_to_f32(0x7e), 448.0);
}

#[test]
fn payload_prepack_rejects_nan_and_normalizes_only_negative_zero() {
    assert_eq!(
        prepack_ocp_e4m3fn_payload_to_fnuz(&[0x00, 0x38, 0x80, 0xfe]).unwrap(),
        [0x00, 0x38, 0x00, 0xfe]
    );
    for byte in [0x7f, 0xff] {
        assert_eq!(
            prepack_ocp_e4m3fn_payload_to_fnuz(&[0x38, byte]).unwrap_err(),
            Sq8FnuzPrepackError::PayloadNonFinite { offset: 1, byte }
        );
    }
}

#[test]
fn scale_compensation_is_x2_per_operand_and_x4_for_a_pair() {
    let weight_scale = 0x3fc0_u16; // BF16 1.5
    let corrected_weight =
        prepack_bf16_scale_bits_for_fnuz(weight_scale, FnuzScaleCompensation::OneConvertedOperand)
            .unwrap();
    assert_eq!(corrected_weight, 0x4040); // BF16 3.0

    let ocp_weight = fp8_e4m3fn_to_f32(0x38); // 1.0
    let fnuz_weight = fnuz_e4m3_to_f32(0x38); // 0.5
    assert_eq!(
        ocp_weight * bf16_bits_to_f32(weight_scale),
        fnuz_weight * bf16_bits_to_f32(corrected_weight)
    );

    let pair_scale = 0x3f40_u16; // BF16 0.75
    let corrected_pair =
        prepack_bf16_scale_bits_for_fnuz(pair_scale, FnuzScaleCompensation::TwoConvertedOperands)
            .unwrap();
    assert_eq!(corrected_pair, 0x4040); // BF16 3.0 = 4 * 0.75

    let ocp_a = fp8_e4m3fn_to_f32(0x38); // 1.0
    let ocp_b = fp8_e4m3fn_to_f32(0x40); // 2.0
    let fnuz_a = fnuz_e4m3_to_f32(0x38); // 0.5
    let fnuz_b = fnuz_e4m3_to_f32(0x40); // 1.0
    assert_eq!(
        ocp_a * ocp_b * bf16_bits_to_f32(pair_scale),
        fnuz_a * fnuz_b * bf16_bits_to_f32(corrected_pair)
    );

    let scales = prepack_bf16_scale_payload_for_fnuz(
        &[0xc0, 0x3f, 0x40, 0x3f],
        FnuzScaleCompensation::OneConvertedOperand,
    )
    .unwrap();
    assert_eq!(scales, [0x40, 0x40, 0xc0, 0x3f]);
}

#[test]
fn bf16_scale_range_gate_rejects_overflow_and_has_no_valid_x2_x4_underflow() {
    assert_eq!(
        prepack_bf16_scale_bits_for_fnuz(0x0001, FnuzScaleCompensation::OneConvertedOperand),
        Ok(0x0002)
    );
    assert_eq!(
        prepack_bf16_scale_bits_for_fnuz(0x0001, FnuzScaleCompensation::TwoConvertedOperands),
        Ok(0x0004)
    );
    assert_eq!(
        prepack_bf16_scale_bits_for_fnuz(0x7eff, FnuzScaleCompensation::OneConvertedOperand),
        Ok(0x7f7f)
    );
    assert!(matches!(
        prepack_bf16_scale_bits_for_fnuz(0x7f00, FnuzScaleCompensation::OneConvertedOperand),
        Err(Bf16ScaleTransformError::Overflow { .. })
    ));
    assert_eq!(
        prepack_bf16_scale_bits_for_fnuz(0x7e7f, FnuzScaleCompensation::TwoConvertedOperands),
        Ok(0x7f7f)
    );
    assert!(matches!(
        prepack_bf16_scale_bits_for_fnuz(0x7e80, FnuzScaleCompensation::TwoConvertedOperands),
        Err(Bf16ScaleTransformError::Overflow { .. })
    ));
    for invalid in [0x0000, 0x8000, 0xbf80, 0x7f80, 0x7fc0] {
        assert!(matches!(
            prepack_bf16_scale_bits_for_fnuz(invalid, FnuzScaleCompensation::OneConvertedOperand),
            Err(Bf16ScaleTransformError::InvalidSource { .. })
        ));
    }

    for bits in 0_u16..=u16::MAX {
        let source = bf16_bits_to_f32(bits);
        if !source.is_finite() || source <= 0.0 {
            continue;
        }
        for compensation in [
            FnuzScaleCompensation::OneConvertedOperand,
            FnuzScaleCompensation::TwoConvertedOperands,
        ] {
            match prepack_bf16_scale_bits_for_fnuz(bits, compensation) {
                Ok(transformed) => {
                    let value = bf16_bits_to_f32(transformed);
                    assert!(value.is_finite() && value > 0.0);
                }
                Err(Bf16ScaleTransformError::Overflow { .. }) => {}
                Err(Bf16ScaleTransformError::Underflow { .. }) => {
                    panic!("valid BF16 0x{bits:04x} underflowed under {compensation}");
                }
                Err(Bf16ScaleTransformError::NonExactBf16 { .. }) => {
                    panic!("valid BF16 0x{bits:04x} was not exact under {compensation}");
                }
                Err(Bf16ScaleTransformError::InvalidSource { .. }) => {
                    panic!("valid BF16 0x{bits:04x} became invalid under {compensation}");
                }
            }
        }
    }
}

#[test]
fn combined_prepack_is_fail_closed_for_payload_and_scale_errors() {
    let result = prepack_sq8_ocp_e4m3fn_tensor_to_fnuz(
        &[0x38, 0x80],
        &[0x80, 0x3f],
        FnuzScaleCompensation::OneConvertedOperand,
    )
    .unwrap();
    assert_eq!(result.payload, [0x38, 0x00]);
    assert_eq!(result.scales_bf16_le, [0x00, 0x40]);

    assert_eq!(
        prepack_sq8_ocp_e4m3fn_tensor_to_fnuz(
            &[0x7f],
            &[0x80, 0x3f],
            FnuzScaleCompensation::OneConvertedOperand,
        )
        .unwrap_err(),
        Sq8FnuzPrepackError::PayloadNonFinite {
            offset: 0,
            byte: 0x7f,
        }
    );
    assert_eq!(
        prepack_sq8_ocp_e4m3fn_tensor_to_fnuz(
            &[0x38],
            &[0x80],
            FnuzScaleCompensation::OneConvertedOperand,
        )
        .unwrap_err(),
        Sq8FnuzPrepackError::OddBf16ScalePayload { bytes: 1 }
    );
}

#[test]
fn hash_checked_artifact_scan_records_frequency_and_scale_gate() {
    let fixture = TempArtifact::new("sq8-fnuz-prepack-scan");
    let weights = [0x38_u8, 0x80, 0x7e, 0xfe];
    let scales = [0x80_u8, 0x3f]; // BF16 1.0
    fs::create_dir_all(fixture.root.join("weights")).unwrap();
    fs::create_dir_all(fixture.root.join("scales")).unwrap();
    fs::write(fixture.root.join("weights/q.f8_e4m3"), weights).unwrap();
    fs::write(fixture.root.join("scales/q.bf16"), scales).unwrap();
    write_fixture_manifest(&fixture.root, &weights, &scales);

    let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
    let report = scan_sq8_canonical_artifact_for_fnuz_prepack(&artifact, 3).unwrap();
    assert_eq!(report.format_id, "SQ8_0");
    assert_eq!(report.tensor_count, 1);
    assert_eq!(report.payload_bytes, 4);
    assert_eq!(report.byte_frequency.len(), 256);
    assert_eq!(report.byte_frequency[0x38], 1);
    assert_eq!(report.byte_frequency[0x80], 1);
    assert_eq!(report.byte_frequency[0x7e], 1);
    assert_eq!(report.byte_frequency[0xfe], 1);
    assert_eq!(report.ocp_negative_zero_count, 1);
    assert_eq!(report.ocp_nan_0x7f_count, 0);
    assert_eq!(report.ocp_nan_0xff_count, 0);
    assert_eq!(report.finite_fnuz_unrepresentable_count, 0);
    assert_eq!(report.scale_count, 1);
    assert_eq!(report.min_positive_bf16_scale, Some(1.0));
    assert_eq!(report.max_positive_bf16_scale, Some(1.0));
    assert!(report.prepack_eligible());
}

struct TempArtifact {
    root: PathBuf,
}

impl TempArtifact {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("ullm-{label}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }
}

impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn write_fixture_manifest(root: &Path, weights: &[u8], scales: &[u8]) {
    let weight_sha256 = sha256_hex(weights);
    let scale_sha256 = sha256_hex(scales);
    let mut manifest = json!({
        "schema_version": "sq-fp8-artifact-v0.2",
        "artifact_kind": "canonical",
        "format_id": "SQ8_0",
        "source": {
            "model_name": "fnuz-test-model",
            "config_file": "config.json",
            "config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "index_file": null,
            "index_sha256": null,
            "quantization": {
                "quant_method": "fp8",
                "format": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_shape": [128, 128]
            }
        },
        "import": {
            "mode": "fp8_checkpoint",
            "encoding": "raw_safetensors_payload"
        },
        "integrity": {"content_sha256": ""},
        "coverage": {
            "scope": "full_model",
            "source_tensor_count": 2,
            "source_fp8_weight_count": 1,
            "source_scale_count": 1,
            "paired_tensor_count": 1,
            "selected_pair_count": 1,
            "unpaired_tensor_count": 0,
            "passthrough_tensor_count": 0
        },
        "storage": {
            "weight_payload_bytes": 4,
            "scale_payload_bytes": 2,
            "total_payload_bytes": 6
        },
        "quantized_tensors": [{
            "name": "model.layers.0.self_attn.q_proj.weight",
            "family": "attn_q",
            "shape": [2, 2],
            "elements": 4,
            "weight": {
                "dtype": "F8_E4M3",
                "encoding": "raw_safetensors_payload",
                "file": "weights/q.f8_e4m3",
                "bytes": 4,
                "sha256": weight_sha256,
                "source_file": "fixture.safetensors"
            },
            "scale": {
                "name": "model.layers.0.self_attn.q_proj.weight_scale_inv",
                "dtype": "BF16",
                "encoding": "raw_safetensors_payload",
                "file": "scales/q.bf16",
                "shape": [1, 1],
                "elements": 1,
                "bytes": 2,
                "sha256": scale_sha256,
                "source_file": "fixture.safetensors",
                "layout": "block_2d",
                "block_shape": [128, 128],
                "order": "row_major",
                "semantic": "dequant_multiplier"
            }
        }],
        "passthrough_tensors": []
    });
    let mut content = manifest.clone();
    content.as_object_mut().unwrap().remove("integrity");
    manifest["integrity"]["content_sha256"] =
        json!(sha256_hex(&serde_json::to_vec(&content).unwrap()));
    fs::write(
        root.join("sq_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
