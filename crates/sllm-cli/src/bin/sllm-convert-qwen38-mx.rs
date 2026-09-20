//! Host-only Qwen3.8-27B BF16 -> sLLM OCP MXFP6/MXFP8 GGUF converter.
//!
//! This is intentionally separate from the historical Qwen3.5 converter.  It
//! consumes the Qwen3.8 lock and source cache, preserves that identity in the
//! derived lock, and reuses only the format-independent conversion machinery.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::json;
use sllm_core::{
    DerivedGgufConverter, DerivedGgufLock, QWEN38_27B_FINGERPRINT, QWEN38_27B_REPO_ID,
    QWEN38_27B_REVISION, QwenMxWeightActivationFormat, read_model_lock,
    write_qwen35_mx_weight_activation_gguf_with_options,
};

const DEFAULT_COMMIT: &str = "0000000000000000000000000000000000000000";

struct Args {
    kind: QwenMxWeightActivationFormat,
    lock: PathBuf,
    cache: PathBuf,
    output: PathBuf,
    derived_lock: PathBuf,
    converter_commit: String,
    retain_gdn_input_gates: bool,
}

fn usage() -> &'static str {
    "usage: sllm-convert-qwen38-mx --kind mxfp8|mxfp6 --lock LOCK --cache SOURCE_DIR --output OUTPUT.gguf --derived-lock OUTPUT.derived-lock.json [--retain-gdn-in-proj-a-b] [--converter-commit SHA40]"
}

fn parse_kind(value: &str) -> Result<QwenMxWeightActivationFormat, String> {
    match value {
        "mxfp8" | "mxfp8-w8a8-e4m3-block32-e8m0" => Ok(QwenMxWeightActivationFormat::Mxfp8E4m3),
        "mxfp6" | "mxfp6-w6a6-e3m2-block32-e8m0" => Ok(QwenMxWeightActivationFormat::Mxfp6E3m2),
        _ => Err(format!("unsupported --kind {value}; {}", usage())),
    }
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut kind = None;
    let mut lock = None;
    let mut cache = None;
    let mut output = None;
    let mut derived_lock = None;
    let mut converter_commit = DEFAULT_COMMIT.to_owned();
    let mut retain_gdn_input_gates = false;
    let mut index = 0;
    while index < raw.len() {
        let flag = raw[index].as_str();
        let value = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            raw.get(*index)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--kind" => kind = Some(parse_kind(&value(&mut index)?)?),
            "--lock" => lock = Some(PathBuf::from(value(&mut index)?)),
            "--cache" => cache = Some(PathBuf::from(value(&mut index)?)),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--derived-lock" => derived_lock = Some(PathBuf::from(value(&mut index)?)),
            "--retain-gdn-in-proj-a-b" => {
                if retain_gdn_input_gates {
                    return Err("--retain-gdn-in-proj-a-b was specified more than once".to_owned());
                }
                retain_gdn_input_gates = true;
            }
            "--converter-commit" => converter_commit = value(&mut index)?,
            "--help" | "-h" => return Err(usage().to_owned()),
            other => return Err(format!("unknown argument {other}; {}", usage())),
        }
        index += 1;
    }
    if converter_commit.len() != 40
        || !converter_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("--converter-commit must be 40 lowercase hexadecimal characters".to_owned());
    }
    let kind = kind.ok_or_else(|| format!("--kind is required; {}", usage()))?;
    Ok(Args {
        kind,
        lock: lock.ok_or_else(|| format!("--lock is required; {}", usage()))?,
        cache: cache.ok_or_else(|| format!("--cache is required; {}", usage()))?,
        output: output.ok_or_else(|| format!("--output is required; {}", usage()))?,
        derived_lock: derived_lock
            .ok_or_else(|| format!("--derived-lock is required; {}", usage()))?,
        converter_commit,
        retain_gdn_input_gates,
    })
}

fn run(args: Args, raw: &[String]) -> Result<serde_json::Value, String> {
    let lock = read_model_lock(&args.lock).map_err(|error| error.to_string())?;
    if lock.model.repo_id != QWEN38_27B_REPO_ID
        || lock.model.resolved_revision != QWEN38_27B_REVISION
        || lock.fingerprint() != QWEN38_27B_FINGERPRINT
    {
        return Err("--lock is not the reviewed Qwen3.8-27B BF16 identity".to_owned());
    }
    if !args.cache.is_dir() || args.cache.is_symlink() {
        return Err("--cache must be a regular non-symlink directory".to_owned());
    }
    if args.output.exists() || args.derived_lock.exists() {
        return Err("output or derived-lock path already exists".to_owned());
    }
    let cache = lock
        .verify_cache(&args.cache)
        .map_err(|error| format!("Qwen3.8 source verification failed: {error}"))?;
    let cache = std::sync::Arc::new(cache);
    let report = write_qwen35_mx_weight_activation_gguf_with_options(
        &lock,
        &cache,
        args.kind,
        args.retain_gdn_input_gates,
        true,
        &args.output,
    )
    .map_err(|error| format!("Qwen3.8 MXFP conversion failed: {error}"))?;
    let coverage =
        QwenMxWeightActivationFormat::diagnostic_coverage_name(args.retain_gdn_input_gates);
    let semantic_model_id = args.kind.semantic_model_id_with_options(
        format!("qwen38:{QWEN38_27B_FINGERPRINT}"),
        args.retain_gdn_input_gates,
        true,
    );
    let mut effective_config = BTreeMap::from([
        (
            "architecture".to_owned(),
            "qwen35-compatible-qwen38".to_owned(),
        ),
        ("format".to_owned(), "GGUF v3 little-endian".to_owned()),
        ("tensor_mode".to_owned(), args.kind.tensor_mode().to_owned()),
    ]);
    if args.retain_gdn_input_gates {
        effective_config.insert("coverage_mode".to_owned(), coverage.to_owned());
    }
    effective_config.insert(
        "scale_mode".to_owned(),
        QwenMxWeightActivationFormat::diagnostic_scale_name(true).to_owned(),
    );
    let derived = DerivedGgufLock::new(
        semantic_model_id,
        vec![QWEN38_27B_FINGERPRINT.to_owned()],
        DerivedGgufConverter {
            repository: "sLLM".to_owned(),
            commit: args.converter_commit,
            arguments: std::iter::once("sllm-convert-qwen38-mx".to_owned())
                .chain(raw.iter().cloned())
                .collect(),
            effective_config,
            environment: BTreeMap::from([
                ("os".to_owned(), env::consts::OS.to_owned()),
                ("arch".to_owned(), env::consts::ARCH.to_owned()),
                (
                    "sllm_version".to_owned(),
                    env!("CARGO_PKG_VERSION").to_owned(),
                ),
            ]),
        },
        &report,
    )
    .map_err(|error| format!("derived lock creation failed: {error}"))?;
    fs::write(
        &args.derived_lock,
        derived
            .canonical_json()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write derived lock: {error}"))?;
    Ok(json!({
        "schema_version": "sllm-qwen38-mx-conversion-v1",
        "state": "PASS",
        "engine": "sLLM",
        "model": QWEN38_27B_REPO_ID,
        "model_revision": QWEN38_27B_REVISION,
        "model_fingerprint": QWEN38_27B_FINGERPRINT,
        "kind": args.kind.tensor_mode(),
        "coverage_mode": coverage,
        "scale_mode": QwenMxWeightActivationFormat::diagnostic_scale_name(true),
        "output": args.output,
        "derived_lock": args.derived_lock,
        "output_sha256": report.sha256,
        "output_bytes": report.size_bytes,
        "metadata_sha256": report.metadata_sha256,
        "tensor_catalog_sha256": report.tensor_catalog_sha256,
        "derived_lock_fingerprint": derived.fingerprint,
    }))
}

fn main() -> ExitCode {
    let raw = env::args().skip(1).collect::<Vec<_>>();
    match parse_args(&raw).and_then(|args| run(args, &raw)) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("report JSON")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Qwen3.8 MXFP conversion failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_args(kind: &str) -> Vec<String> {
        [
            "--kind",
            kind,
            "--lock",
            "lock.json",
            "--cache",
            "cache",
            "--output",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn no_clipping_scale_is_the_default_and_composable_with_retention() {
        let mut args = required_args("mxfp8");
        args.push("--retain-gdn-in-proj-a-b".to_owned());
        let parsed = parse_args(&args).expect("diagnostic flags parse");
        assert!(parsed.retain_gdn_input_gates);
    }

    #[test]
    fn removed_scale_switches_are_unknown_arguments() {
        let mut args = required_args("mxfp6");
        args.push("--mxfp6-no-clipping-scale".to_owned());
        assert!(parse_args(&args).is_err());
        for flag in ["--mxfp8-no-clipping-scale", "--legacy-floor-scale"] {
            let mut args = required_args("mxfp8");
            args.push(flag.to_owned());
            assert!(parse_args(&args).is_err(), "removed flag {flag}");
        }
    }
}
