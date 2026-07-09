// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use ullm_engine::sq_canonical::{
    read_sq8_canonical_artifact, reconstruct_sq8_canonical_tensor_block_f32,
};

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid {label} {value:?}: {err}"))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 && args.len() != 4 {
        return Err(
            "usage: verify_sq8_canonical ARTIFACT_DIR [TENSOR_NAME BLOCK_ROW BLOCK_COL]"
                .to_string(),
        );
    }
    let artifact_dir = PathBuf::from(&args[0]);
    let artifact = read_sq8_canonical_artifact(&artifact_dir)?;
    let checksums = artifact.checksum_report();

    let reconstructed = if args.len() == 4 {
        let block_row = parse_usize(&args[2], "block row")?;
        let block_col = parse_usize(&args[3], "block col")?;
        let block =
            reconstruct_sq8_canonical_tensor_block_f32(&artifact, &args[1], block_row, block_col)?;
        let mut digest = Sha256::new();
        for value in &block.values {
            digest.update(value.to_le_bytes());
        }
        Some(json!({
            "tensor": block.tensor_name,
            "block_row": block.block_row,
            "block_col": block.block_col,
            "rows": block.rows,
            "cols": block.cols,
            "values_sha256": format!("{:x}", digest.finalize()),
        }))
    } else {
        None
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": artifact.manifest().schema_version.as_str(),
            "format_id": artifact.manifest().format_id.as_str(),
            "content_sha256": artifact.manifest().integrity.content_sha256.as_str(),
            "selected_pair_count": checksums.selected_pair_count,
            "weight_payload_bytes": checksums.weight_payload_bytes,
            "scale_payload_bytes": checksums.scale_payload_bytes,
            "reconstructed_block": reconstructed,
            "verified": true,
        }))
        .map_err(|err| format!("failed to serialize verification result: {err}"))?
    );
    Ok(())
}
