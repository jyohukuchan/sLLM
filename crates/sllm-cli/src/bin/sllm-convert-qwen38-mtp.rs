//! Host-only converter for the reviewed Qwen3.8 MTP companion sidecar.
//!
//! The target artifact remains unchanged.  This command reads the fixed
//! Qwen3.5-27B semantic lock, verifies the Unsloth NVFP4 artifact, and emits
//! an authenticated `manifest.json` plus `payload.safetensors` directory.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::json;
use sllm_core::{
    MtpBf16RoundtripEncoding, MtpWeightEncoding, convert_qwen38_mtp_bf16_roundtrip_sidecar,
    convert_qwen38_mtp_nvfp4_sidecar, convert_qwen38_mtp_quantized_sidecar, parse_model_lock,
    verify_unsloth_qwen38_nvfp4,
};

const USAGE: &str = "usage: sllm-convert-qwen38-mtp --artifact-root ABSOLUTE_DIRECTORY --encoding mxfp8|nvfp4|bf16-roundtrip-mxfp8 --output-dir ABSOLUTE_DIRECTORY [--activation-scale-manifest ABSOLUTE_JSON for nvfp4]";

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sllm-convert-qwen38-mtp: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn parse_encoding(
    value: &str,
) -> Result<(Option<MtpWeightEncoding>, Option<MtpBf16RoundtripEncoding>), String> {
    Ok(match value {
        "mxfp8" | "mxfp8-w8a8-e4m3-block32-e8m0" => (
            Some(MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale),
            None,
        ),
        "mxfp6" | "mxfp6-w6a6-e3m2-block32-e8m0" | "bf16-roundtrip-mxfp6" => {
            return Err("retired MTP MXFP6 sidecar encoding".to_owned());
        }
        "nvfp4" | "nvfp4-w4a4-e2m1-block16-e4m3fn-f32" => {
            (Some(MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32), None)
        }
        "bf16-roundtrip-mxfp8" => (None, Some(MtpBf16RoundtripEncoding::Mxfp8)),
        value => return Err(format!("unsupported --encoding value {value:?}")),
    })
}

fn run(args: Vec<std::ffi::OsString>) -> Result<String, String> {
    let mut artifact_root = None;
    let mut encoding = None;
    let mut output_dir = None;
    let mut activation_scale_manifest = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].to_str().ok_or("arguments must be UTF-8")?;
        if flag == "--help" || flag == "-h" {
            return Ok(USAGE.to_owned());
        }
        if !matches!(
            flag,
            "--artifact-root" | "--encoding" | "--output-dir" | "--activation-scale-manifest"
        ) {
            return Err(format!("unknown argument {flag}"));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--artifact-root" => artifact_root = Some(PathBuf::from(value)),
            "--encoding" => encoding = Some(value.to_str().ok_or("encoding must be UTF-8")?),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--activation-scale-manifest" => activation_scale_manifest = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += 2;
    }
    let artifact_root = artifact_root.ok_or("--artifact-root is required")?;
    let output_dir = output_dir.ok_or("--output-dir is required")?;
    if !artifact_root.is_absolute() || !output_dir.is_absolute() {
        return Err("artifact and output paths must be absolute".to_owned());
    }
    let encoding = parse_encoding(encoding.ok_or("--encoding is required")?)?;
    let is_nvfp4 = encoding.0 == Some(MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32);
    if is_nvfp4 {
        let path = activation_scale_manifest
            .as_ref()
            .ok_or("--encoding nvfp4 requires --activation-scale-manifest")?;
        if !path.is_absolute() {
            return Err("--activation-scale-manifest must be absolute".to_owned());
        }
    } else if activation_scale_manifest.is_some() {
        return Err(
            "--activation-scale-manifest is supported only with --encoding nvfp4".to_owned(),
        );
    }
    if output_dir.exists() {
        return Err("--output-dir already exists; choose a new directory".to_owned());
    }
    let lock = parse_model_lock(include_bytes!(
        "../../../../docs/models/locks/qwen3.5-27b-bf16.json"
    ))
    .map_err(|error| format!("embedded reviewed Qwen3.5-27B lock is invalid: {error}"))?;
    let artifact = verify_unsloth_qwen38_nvfp4(&artifact_root)
        .map_err(|error| format!("Qwen3.8 NVFP4 artifact verification failed: {error}"))?;
    let verified = if is_nvfp4 {
        convert_qwen38_mtp_nvfp4_sidecar(
            &lock,
            &artifact,
            activation_scale_manifest
                .as_deref()
                .expect("NVFP4 manifest was validated above"),
            &output_dir,
        )
    } else if let Some(encoding) = encoding.0 {
        convert_qwen38_mtp_quantized_sidecar(&lock, &artifact, encoding, &output_dir)
    } else {
        convert_qwen38_mtp_bf16_roundtrip_sidecar(
            &lock,
            &artifact,
            encoding.1.expect("roundtrip encoding is present"),
            &output_dir,
        )
    }
    .map_err(|error| format!("MTP sidecar conversion failed: {error}"))?;
    Ok(json!({
        "kind": "qwen38-mtp-companion",
        "encoding": verified.manifest_encoding(),
        "output_dir": output_dir,
        "manifest_fingerprint": verified.manifest_fingerprint(),
        "combined_recipe_digest": verified.combined_recipe_digest(artifact.recipe_digest()),
        "files": ["manifest.json", "payload.safetensors"],
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removed_scale_switches_are_unknown_arguments() {
        for flag in [
            "--legacy-floor-scale",
            "--mxfp8-no-clipping-scale",
            "--mxfp6-no-clipping-scale",
        ] {
            assert_eq!(
                run(vec![flag.into()]).unwrap_err(),
                format!("unknown argument {flag}")
            );
        }
    }

    #[test]
    fn supported_mx_recipe_stays_non_saturating_and_mxfp6_is_retired() {
        for value in ["mxfp8", "mxfp8-w8a8-e4m3-block32-e8m0"] {
            assert!(matches!(
                parse_encoding(value).unwrap(),
                (
                    Some(MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale),
                    None
                )
            ));
        }
        assert!(matches!(
            parse_encoding("bf16-roundtrip-mxfp8").unwrap(),
            (None, Some(MtpBf16RoundtripEncoding::Mxfp8))
        ));
        for retired in [
            "mxfp6",
            "mxfp6-w6a6-e3m2-block32-e8m0",
            "bf16-roundtrip-mxfp6",
        ] {
            assert!(
                parse_encoding(retired)
                    .unwrap_err()
                    .contains("retired MTP MXFP6")
            );
        }
        assert!(matches!(
            parse_encoding("nvfp4").unwrap(),
            (Some(MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32), None)
        ));
    }

    #[test]
    fn nvfp4_requires_an_absolute_activation_manifest() {
        let error = run(vec![
            "--artifact-root".into(),
            "/tmp/artifact".into(),
            "--encoding".into(),
            "nvfp4".into(),
            "--output-dir".into(),
            "/tmp/output".into(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --activation-scale-manifest"));

        let error = run(vec![
            "--artifact-root".into(),
            "/tmp/artifact".into(),
            "--encoding".into(),
            "nvfp4".into(),
            "--activation-scale-manifest".into(),
            "relative.json".into(),
            "--output-dir".into(),
            "/tmp/output".into(),
        ])
        .unwrap_err();
        assert!(error.contains("must be absolute"));
    }
}
