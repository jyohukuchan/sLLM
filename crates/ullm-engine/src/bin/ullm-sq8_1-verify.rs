// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Verify a Python-packed SQ8_1 artifact with the Rust reader.

use std::env;
use ullm_engine::sq8_1::read_sq8_1_artifact;

fn main() {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let artifact = args.next().unwrap_or_else(|| {
        eprintln!(
            "usage: {} /absolute/sq8_1-artifact [tensor-name]",
            program.to_string_lossy()
        );
        std::process::exit(2);
    });
    let tensor_name = args.next();
    if args.next().is_some() {
        eprintln!(
            "usage: {} /absolute/sq8_1-artifact [tensor-name]",
            program.to_string_lossy()
        );
        std::process::exit(2);
    }
    let artifact = read_sq8_1_artifact(&artifact).unwrap_or_else(|error| {
        eprintln!("SQ8_1 verification failed: {error}");
        std::process::exit(1);
    });
    let report = artifact.checksum_report();
    let tensor = tensor_name.as_ref().map(|name| {
        artifact
            .read_tensor(&name.to_string_lossy())
            .unwrap_or_else(|error| {
                eprintln!("SQ8_1 tensor read failed: {error}");
                std::process::exit(1);
            })
    });
    let mut output = serde_json::json!({
        "format_id": "SQ8_1",
        "tensor_count": report.tensor_count,
        "payload_bytes": report.payload_bytes,
        "scale_bytes": report.scale_bytes,
    });
    if let Some(tensor) = tensor {
        output["tensor"] = serde_json::json!({
            "name": tensor.name,
            "rows": tensor.rows,
            "cols": tensor.cols,
            "payload_row_stride": tensor.payload_row_stride,
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization")
    );
}
