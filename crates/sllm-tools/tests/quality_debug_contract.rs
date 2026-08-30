use serde_json::json;
use sllm_tools::{
    DebugDumpConfig, DebugDumpError, DebugDumpWriter, QUALITY_INPUT_SCHEMA_VERSION, QualityError,
    TOOL_JSON_CANONICALIZATION_V1, TOOL_RUN_SCHEMA_VERSION_V1, TOOL_RUN_STRUCT_SIZE_V1,
    ToolFileIdentityV1, ToolIdentityV1, ToolRecipeIdentityV1, ToolRunManifestV1, ToolRunStateV1,
    evaluate_quality, sha256_bytes,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

fn tool_manifest() -> ToolRunManifestV1 {
    ToolRunManifestV1 {
        schema_version: TOOL_RUN_SCHEMA_VERSION_V1.to_owned(),
        struct_size: TOOL_RUN_STRUCT_SIZE_V1,
        canonicalization: TOOL_JSON_CANONICALIZATION_V1.to_owned(),
        operation: "debug-dump".to_owned(),
        state: ToolRunStateV1::Pass,
        selected_count: 1,
        tool: ToolIdentityV1 {
            repository: "https://github.com/89chin/sLLM".to_owned(),
            commit: "1".repeat(40),
            package: "sllm-tools".to_owned(),
            version: "0.1.0".to_owned(),
            executable_sha256: sha256_bytes(b"fixture executable"),
            arguments: vec!["debug-dump".to_owned()],
            environment: BTreeMap::from([("offline".to_owned(), "true".to_owned())]),
        },
        recipe: ToolRecipeIdentityV1 {
            id: "debug-dump".to_owned(),
            version: "v1".to_owned(),
            config_sha256: sha256_bytes(b"fixture recipe"),
        },
        sources: vec![ToolFileIdentityV1::for_bytes("submission", "fixture", b"input").unwrap()],
        outputs: vec![ToolFileIdentityV1::for_bytes("dump", "debug.json", b"planned").unwrap()],
        raw_evidence: Vec::new(),
        identities: BTreeMap::new(),
        metrics: BTreeMap::new(),
        extensions: BTreeMap::new(),
    }
}

fn quality_input(mut input: serde_json::Value) -> serde_json::Value {
    input
        .as_object_mut()
        .unwrap()
        .insert("manifest".to_owned(), json!(tool_manifest()));
    input
}

fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "sllm-tools-quality-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn perplexity_reports_loss_sum_and_token_count() {
    let input = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "perplexity": {"losses": [1.0, 2.0, 3.0]}
    }));
    let output = evaluate_quality(&input).unwrap();
    assert_eq!(output["metric"], "perplexity");
    assert_eq!(output["result"]["loss_sum"], 6.0);
    assert_eq!(output["result"]["token_count"], 3);
    assert!((output["result"]["perplexity"].as_f64().unwrap() - 7.389056).abs() < 1e-5);
}

#[test]
fn logits_report_kld_top1_quantiles_and_first_divergence() {
    let input = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "kind": "kld",
        "kld": {"samples": [
            {"position": 10, "baseline": [3.0, 1.0], "candidate": [3.0, 1.0]},
            {"position": 11, "baseline": [1.0, 3.0], "candidate": [3.0, 1.0]}
        ]}
    }));
    let result = evaluate_quality(&input).unwrap()["result"].clone();
    assert_eq!(result["sample_count"], 2);
    assert_eq!(result["top1_matches"], 1);
    assert_eq!(result["first_divergence_position"], 11);
    assert!(result["kld_max"].as_f64().unwrap() > 0.0);
    assert!(result["logit_abs_diff"]["p99"].as_f64().unwrap() > 0.0);
}

#[test]
fn task_supports_exact_match_and_numeric_multiple_choice() {
    let input = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "task": {"task_version": "fixture-v1", "samples": [
            {"prediction": "yes", "reference": "yes"},
            {"prediction": "no", "reference": "yes"},
            {"choice_logits": [0.2, 0.8], "answer_index": 1},
            {"choices": [{"logit": 0.9}, {"logit": 0.1}], "answer": 0}
        ]}
    }));
    let result = evaluate_quality(&input).unwrap()["result"].clone();
    assert_eq!(result["exact_match_count"], 1);
    assert_eq!(result["multiple_choice_correct"], 2);
    assert_eq!(result["multiple_choice_count"], 2);
}

#[test]
fn long_context_requires_early_middle_tail_and_kv_coverage() {
    let input = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "long_context": {"capacity": 300, "samples": [
            {"position": 1, "band": "early", "kv_plane": "K", "layer": 0, "kv_head": 0},
            {"position": 120, "band": "middle", "kv_plane": "V", "layer": 1, "kv_head": 1},
            {"position": 299, "band": "tail", "kv_plane": "K", "block_tail": true, "layer": 2, "kv_head": 0}
        ]}
    }));
    let result = evaluate_quality(&input).unwrap()["result"].clone();
    assert_eq!(result["early"], 1);
    assert_eq!(result["middle"], 1);
    assert_eq!(result["tail"], 1);
    assert_eq!(result["key_samples"], 2);
    assert_eq!(result["value_samples"], 1);
}

#[test]
fn empty_and_over_limit_inputs_fail_closed() {
    let empty = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "perplexity": {"losses": []}
    }));
    assert!(matches!(
        evaluate_quality(&empty),
        Err(QualityError::Empty(_))
    ));
    let over = quality_input(json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "limits": {"max_samples": 1, "max_logit_width": 2, "max_task_choices": 2, "max_context_tokens": 10, "max_input_bytes": 1024},
        "logit_comparison": {"samples": [
            {"baseline": [0.0, 1.0, 2.0], "candidate": [0.0, 1.0, 2.0]}
        ]}
    }));
    assert!(matches!(
        evaluate_quality(&over),
        Err(QualityError::OverLimit(_))
    ));
}

#[test]
fn quality_rejects_schema_invalid_and_unknown_fields() {
    for input in [
        quality_input(json!({
            "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
            "perplexity": {"loss_sum": -1.0, "token_count": 1}
        })),
        quality_input(json!({
            "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
            "task": {"task_version": "", "samples": [{"prediction": "x", "reference": "x"}]}
        })),
        quality_input(json!({
            "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
            "long_context": {"capacity": 3, "samples": [
                {"position": 0, "band": "early", "kv_plane": "K", "layer": 0, "kv_head": 0},
                {"position": 1, "band": "middle", "kv_plane": "V", "layer": 0, "kv_head": 0},
                {"position": 3, "band": "tail", "kv_plane": "K", "block_tail": true, "layer": 0, "kv_head": 0}
            ]}
        })),
        quality_input(json!({
            "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
            "perplexity": {"losses": [1.0], "typo": true}
        })),
    ] {
        assert!(matches!(
            evaluate_quality(&input),
            Err(QualityError::Invalid(_) | QualityError::Unsupported(_))
        ));
    }

    let no_manifest = json!({
        "schema_version": QUALITY_INPUT_SCHEMA_VERSION,
        "perplexity": {"losses": [1.0]}
    });
    assert!(matches!(
        evaluate_quality(&no_manifest),
        Err(QualityError::Invalid(_))
    ));
}

#[test]
fn disabled_debug_dump_is_side_effect_free() {
    let root = temp_dir("disabled");
    let config = DebugDumpConfig {
        output_dir: root.clone(),
        ..DebugDumpConfig::default()
    };
    let mut writer = DebugDumpWriter::new(config).unwrap();
    assert!(!writer.is_enabled());
    writer.add_tokens(&[1, 2, 3]).unwrap();
    let artifact = writer.finish().unwrap();
    assert!(artifact.path.is_none());
    assert_eq!(fs::read_dir(root).unwrap().count(), 0);
}

#[test]
fn debug_dump_rejects_forbidden_metadata_and_cleans_partial() {
    let root = temp_dir("forbidden");
    let config = DebugDumpConfig {
        enabled: true,
        output_dir: root.clone(),
        file_name: "result.json".to_owned(),
        ..DebugDumpConfig::default()
    };
    let mut writer = DebugDumpWriter::new(config).unwrap();
    let error = writer
        .set_metadata(json!({"prompt": "do not store"}))
        .unwrap_err();
    assert!(matches!(error, DebugDumpError::Forbidden(_)));
    drop(writer);
    assert_eq!(fs::read_dir(root).unwrap().count(), 0);
}

#[test]
fn debug_dump_publishes_atomically_and_enforces_limits() {
    let root = temp_dir("publish");
    let config = DebugDumpConfig {
        enabled: true,
        output_dir: root.clone(),
        file_name: "result.json".to_owned(),
        max_tokens: 2,
        ..DebugDumpConfig::default()
    };
    let mut writer = DebugDumpWriter::new(config).unwrap();
    writer.set_manifest(&tool_manifest()).unwrap();
    writer
        .set_metadata(json!({
            "run_id": "fixture",
            "dtype": "bf16",
            "layout": "token-major",
            "endianness": "little",
            "token_count": 2,
            "token_digest": format!("sha256:{}", "a".repeat(64))
        }))
        .unwrap();
    writer.add_tokens(&[11, 12]).unwrap();
    writer.add_logits(0, 0, &[1.0, 3.0, 2.0], 2).unwrap();
    let artifact = writer.finish().unwrap();
    assert_eq!(artifact.tensor_count, 0);
    assert_eq!(artifact.token_count, 2);
    assert!(artifact.path.as_ref().unwrap().is_file());
    assert!(artifact.digest.unwrap().starts_with("sha256:"));
    assert!(fs::read_dir(&root).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")
    }));
}

#[test]
fn debug_dump_rejects_schema_invalid_metadata_and_repeated_logit_positions() {
    let root = temp_dir("schema-boundaries");
    let config = DebugDumpConfig {
        enabled: true,
        output_dir: root.clone(),
        file_name: "result.json".to_owned(),
        max_positions: 1,
        ..DebugDumpConfig::default()
    };
    let mut writer = DebugDumpWriter::new(config).unwrap();
    assert!(matches!(
        writer.set_metadata(json!({"token_digest": "sha256:fixture"})),
        Err(DebugDumpError::Invalid(_))
    ));
    assert!(matches!(
        writer.set_metadata(json!({"token_count": -1})),
        Err(DebugDumpError::Invalid(_))
    ));
    assert!(matches!(
        writer.set_metadata(json!({"target": true})),
        Err(DebugDumpError::Invalid(_))
    ));
    assert!(matches!(
        writer.set_metadata(json!({"provider": ""})),
        Err(DebugDumpError::Invalid(_))
    ));
    assert!(matches!(
        writer.add_tensor("", "BF16", &[1], "row-major", "little", &[0.0], None, None),
        Err(DebugDumpError::Invalid(_))
    ));
    writer.add_logits(0, 0, &[1.0, 2.0], 1).unwrap();
    assert!(matches!(
        writer.add_logits(0, 0, &[1.0, 2.0], 1),
        Err(DebugDumpError::OverLimit(_))
    ));
    drop(writer);
    assert_eq!(fs::read_dir(root).unwrap().count(), 0);
}
