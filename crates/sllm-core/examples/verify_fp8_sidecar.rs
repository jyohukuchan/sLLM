use sllm_core::{read_model_lock, verify_fp8_sidecar};
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if arguments.len() != 3 {
        eprintln!("usage: verify_fp8_sidecar SOURCE_LOCK MANIFEST ARTIFACT");
        std::process::exit(2);
    }
    let lock = read_model_lock(&arguments[0]).unwrap_or_else(|error| {
        eprintln!("FP8 sidecar verification: FAIL: {error}");
        std::process::exit(1);
    });
    let verified = verify_fp8_sidecar(&arguments[1], &arguments[2], &arguments[0], &lock)
        .unwrap_or_else(|error| {
            eprintln!("FP8 sidecar verification: FAIL: {error}");
            std::process::exit(1);
        });
    println!(
        "FP8 sidecar verification: PASS tensors={} source={} manifest={} artifact={}",
        verified.tensors().len(),
        verified.source_lock_fingerprint(),
        verified.manifest_fingerprint(),
        verified.artifact_sha256(),
    );
}
