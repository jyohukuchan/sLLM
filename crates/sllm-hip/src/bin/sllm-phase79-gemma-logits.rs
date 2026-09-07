//! Fixed-input logits capture for Phase 79 kernel/execution comparisons.
//! Run separate processes with one selector override at a time. Captured
//! logits are measurement data, not an independent numerical oracle or a
//! quality PASS. Output belongs in untracked local artifacts.

use serde::{Deserialize, Serialize};
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ResidentModel,
    build_verified_gguf_gemma_weight_load_plan, parse_gemma4_model_lock, read_derived_gguf_lock,
    verify_derived_gguf,
};
use sllm_hip::HipBackend;
use std::{env, fs, path::PathBuf, process::ExitCode, sync::Arc, time::Duration};

#[derive(Deserialize, Serialize)]
struct Case {
    id: String,
    prompt: Vec<i32>,
    continuation: Vec<i32>,
}

// Only named measurement controls may enter evidence; never dump the process environment.
fn selector_environment(
    vars: impl IntoIterator<Item = (String, String)>,
) -> std::collections::BTreeMap<String, String> {
    const ALLOWED: &[&str] = &[
        "SLLM_MATMUL_SELECTOR_TRACE",
        "SLLM_NVFP4_W4A4_FORCE_BASELINE",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS",
        "SLLM_FP8_OUTER_FORCE_BASELINE",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8",
        "SLLM_PREPARED_PROJECTION_SHARING",
        "SLLM_PREPARED_DEFERRED_COMPLETION",
    ];
    vars.into_iter()
        .filter(|(key, _)| ALLOWED.contains(&key.as_str()))
        .collect()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: sllm-phase79-gemma-logits MODEL_LOCK GGUF DERIVED_LOCK TARGET FIXTURE_JSON OUTPUT_JSON (single visible GPU, device 0)".into());
    }
    if !matches!(args[3].as_str(), "gfx1030" | "gfx1201") {
        return Err("Phase79 capture requires exact gfx1030 or gfx1201".into());
    }
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(&args[4])?)?;
    if cases.is_empty() || cases.iter().any(|c| c.id.is_empty() || c.prompt.is_empty()) {
        return Err("fixture must contain named, nonempty prompts".into());
    }
    let lock = parse_gemma4_model_lock(&fs::read(&args[0])?)?;
    let verified = verify_derived_gguf(
        read_derived_gguf_lock(&PathBuf::from(&args[2]))?,
        &PathBuf::from(&args[1]),
    )?;
    let (source, plan) = build_verified_gguf_gemma_weight_load_plan(&lock, verified)?;
    let backend = HipBackend::connect()?;
    let session = Arc::new(
        backend.open_execution_session(ExecutionSessionRequest::new(0, args[3].clone())?)?,
    );
    let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let resident = Gemma4ResidentModel::new_gguf_quantized(
            Arc::clone(&session),
            lock.clone(),
            plan,
            Arc::new(source),
            Duration::from_secs(600),
        )?;
        let mut rows = Vec::new();
        for case in &cases {
            let capacity = case
                .prompt
                .len()
                .checked_add(case.continuation.len())
                .ok_or("capacity overflow")?;
            let mut owner =
                resident.new_request(case.prompt.len().try_into()?, capacity.try_into()?)?;
            let first = owner.prefill_with_last_logits(&case.prompt)?;
            let prefill_pack_submissions =
                owner.audit_snapshot()?.projection_pack_submission_count();
            let mut positions = Vec::new();
            let logits = first.last_logits().ok_or("prefill did not return logits")?;
            if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
                return Err("invalid prefill logits".into());
            }
            positions.push(serde_json::json!({"position": case.prompt.len()-1, "logits": logits}));
            for (i, &token) in case.continuation.iter().enumerate() {
                // Teacher forcing keeps inputs identical despite top1 differences.
                let output = owner.decode_with_last_logits(token)?;
                let logits = output.last_logits().ok_or("decode did not return logits")?;
                if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
                    return Err("invalid decode logits".into());
                }
                positions
                    .push(serde_json::json!({"position": case.prompt.len()+i, "logits": logits}));
            }
            let audit = owner.audit_snapshot()?;
            if audit.target() != args[3]
                || audit.fallback_used()
                || audit.kernel_dispatch_count() == 0
            {
                return Err("capture lacks exact HIP dispatch evidence".into());
            }
            rows.push(serde_json::json!({
                "id": case.id, "positions": positions,
                "kernel_dispatch_count": audit.kernel_dispatch_count(),
                "segment_count": audit.segment_count(), "boundary_count": audit.boundary_count(),
                "projection_pack_submission_count": audit.projection_pack_submission_count(),
                "prefill_projection_pack_submission_count": prefill_pack_submissions,
                "fallback_used": audit.fallback_used()
            }));
        }
        Ok(serde_json::json!({
            "schema_version": "phase79-gemma-fixed-logits-v1", "state": "CAPTURED",
            "quality_verdict": "not_evaluated", "target": args[3], "device_index": 0,
            "model_fingerprint": lock.fingerprint(), "kv_encoding": "fp16",
            "fixture": cases, "rows": rows,
            "peak_accounted_bytes": session.memory_snapshot().high_water_bytes(),
            "selector_environment": selector_environment(env::vars())
        }))
    })();
    let cleanup = session.shutdown(Duration::from_secs(60))?;
    let current = session.memory_snapshot().current_bytes();
    if current != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("capture cleanup retained runtime resources".into());
    }
    let mut report = result?;
    report["cleanup_current_bytes"] = current.into();
    report["cleanup_retryable"] = cleanup.retryable_cleanup.into();
    report["cleanup_durable"] = cleanup.durable_quarantine.into();
    fs::write(&args[5], serde_json::to_vec(&report)?)?;
    println!("CAPTURED {}", args[5]);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Phase79 logits capture failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::selector_environment;

    #[test]
    fn evidence_keeps_selector_controls_and_excludes_credentials_and_unknowns() {
        let vars = [
            ("SLLM_HIP_COMPILER_BROKER_TOKEN", "sensitive-sentinel"),
            ("SLLM_API_KEY", "sensitive-sentinel"),
            ("SLLM_UNKNOWN_SETTING", "sensitive-sentinel"),
            ("PATH", "private-path"),
            ("SLLM_PREPARED_PROJECTION_SHARING", "0"),
            ("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4", "1"),
        ];
        let captured = selector_environment(vars.map(|(k, v)| (k.into(), v.into())));
        assert_eq!(captured.len(), 2);
        assert_eq!(captured["SLLM_PREPARED_PROJECTION_SHARING"], "0");
        assert_eq!(captured["SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4"], "1");
        let json = serde_json::to_string(&captured).unwrap();
        assert!(!json.contains("sensitive-sentinel"));
        assert!(!json.contains("private-path"));
    }
}
