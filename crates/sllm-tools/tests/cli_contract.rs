use serde_json::{Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!(
            "sllm-tools-cli-{label}-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create CLI test temporary directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn executable(name: &str) -> PathBuf {
    let variable = format!("CARGO_BIN_EXE_{name}");
    PathBuf::from(env::var_os(variable).expect("Cargo must provide the integration-test binary"))
}

fn command(name: &str) -> Command {
    Command::new(executable(name))
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "expected success, status={:?}, stdout={}, stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_failure(output: &Output) {
    assert!(
        !output.status.success(),
        "expected failure, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_json(path: &Path, value: &Value) {
    fs::write(
        path,
        serde_json::to_vec(value).expect("serialize test JSON"),
    )
    .expect("write test JSON");
}

fn digest(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn manifest_json() -> Value {
    json!({
        "schema_version": "sllm-phase46-tool-run-v1",
        "struct_size": 13,
        "canonicalization": "sllm-sorted-json-v1",
        "operation": "fixture",
        "state": "PASS",
        "selected_count": 1,
        "tool": {
            "repository": "https://github.com/89chin/sLLM",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "package": "sllm-tools",
            "version": "0.1.0",
            "executable_sha256": digest('a'),
            "arguments": ["fixture"],
            "environment": {"offline": "true"}
        },
        "recipe": {"id": "fixture", "version": "v1", "config_sha256": digest('b')},
        "sources": [{"role": "source", "logical_name": "source.bin", "size_bytes": 1, "sha256": digest('c')}],
        "outputs": [{"role": "output", "logical_name": "result.bin", "size_bytes": 1, "sha256": digest('d')}],
        "raw_evidence": [],
        "identities": {"model": "fixture"},
        "metrics": {"selected": 1},
        "extensions": {}
    })
}

fn benchmark_input(warmups: Value, measured: Value) -> Value {
    let resource = json!({"status": "measured", "bytes": 0});
    json!({
        "schema_version": "sllm-phase46-benchmark-input-v1",
        "model_lock": "missing-model-lock.json",
        "tokenizer_files": ["missing-tokenizer.json"],
        "dataset_files": ["missing-dataset.json"],
        "configuration": {
            "request_count": 1,
            "parallelism": 1,
            "context_tokens": 17,
            "sampling": "greedy",
            "kv_encoding": "fp16",
            "gpu_identity": "fixture",
            "provider": "hip",
            "fallback": false,
            "cleanup": true
        },
        "warmups": warmups,
        "measured": measured,
        "resources": {
            "hbm_before": resource,
            "hbm_peak": resource,
            "hbm_settled": resource,
            "gtt_before": resource,
            "gtt_peak": resource,
            "gtt_settled": resource,
            "model_resident": resource,
            "kv_logical": resource,
            "kv_physical": resource,
            "workspace": resource
        }
    })
}

#[test]
fn help_and_capability_surfaces_are_available_offline() {
    let artifact_help = command("sllm-artifact")
        .arg("--help")
        .output()
        .expect("run sllm-artifact --help");
    assert_success(&artifact_help);
    assert!(String::from_utf8_lossy(&artifact_help.stdout).contains("capabilities"));

    let capability = command("sllm-artifact")
        .args(["capabilities", "--architecture", "qwen35"])
        .output()
        .expect("run sllm-artifact capabilities");
    assert_success(&capability);
    let capability_json: Value =
        serde_json::from_slice(&capability.stdout).expect("capability JSON");
    assert_eq!(capability_json["schema_version"], "sllm-capability-v1");
    assert_eq!(capability_json["architecture"], "qwen35");
    assert!(
        capability_json["recipes"]
            .as_array()
            .expect("capability recipes")
            .iter()
            .any(|recipe| recipe == "mxfp4-e2m1-block32-e8m0")
    );

    let bench_help = command("sllm-bench")
        .arg("--help")
        .output()
        .expect("run sllm-bench --help");
    assert_success(&bench_help);
    assert!(String::from_utf8_lossy(&bench_help.stdout).contains("aggregate"));

    let eval_help = command("sllm-eval")
        .arg("--help")
        .output()
        .expect("run sllm-eval --help");
    assert_success(&eval_help);
    assert!(String::from_utf8_lossy(&eval_help.stdout).contains("phase46-quality-input-v1"));
}

#[test]
fn unknown_and_unsupported_cli_inputs_fail_closed() {
    let artifact_unknown = command("sllm-artifact")
        .arg("unknown-command")
        .output()
        .expect("run unknown artifact command");
    assert_failure(&artifact_unknown);

    let artifact_unsupported = command("sllm-artifact")
        .args(["capabilities", "--architecture", "unreviewed"])
        .output()
        .expect("run unsupported artifact architecture");
    assert_failure(&artifact_unsupported);

    let bench_unknown = command("sllm-bench")
        .arg("unknown-command")
        .output()
        .expect("run unknown benchmark command");
    assert_failure(&bench_unknown);

    let eval_unknown = command("sllm-eval")
        .arg("--unknown")
        .output()
        .expect("run unknown evaluator option");
    assert_failure(&eval_unknown);

    let temp = TempDir::new("unsupported");
    let input = temp.path().join("unsupported.json");
    let manifest = temp.path().join("manifest.json");
    write_json(
        &input,
        &json!({
            "schema_version": "sllm-phase46-quality-input-unsupported-v1",
            "perplexity": {"losses": [1.0]}
        }),
    );
    write_json(&manifest, &manifest_json());
    let eval_unsupported = command("sllm-eval")
        .args(["--input"])
        .arg(&input)
        .args(["--manifest"])
        .arg(&manifest)
        .output()
        .expect("run unsupported evaluator schema");
    assert_failure(&eval_unsupported);
    assert!(String::from_utf8_lossy(&eval_unsupported.stderr).contains("unsupported"));
}

#[test]
fn zero_selected_inputs_are_rejected_without_outputs() {
    let temp = TempDir::new("zero");

    let artifact_input = temp.path().join("zero-values.json");
    let artifact_output = temp.path().join("zero-quantized.bundle");
    write_json(&artifact_input, &json!({"values": []}));
    let artifact_zero = command("sllm-artifact")
        .args([
            "quantize",
            "--recipe",
            "mxfp4-e2m1-block32-e8m0",
            "--input-json",
        ])
        .arg(&artifact_input)
        .args(["--rows", "1", "--columns", "0", "--output-dir"])
        .arg(&artifact_output)
        .output()
        .expect("run zero quantization");
    assert_failure(&artifact_zero);
    assert!(!artifact_output.exists());

    let bench_input = temp.path().join("zero-benchmark.json");
    let bench_output = temp.path().join("zero-benchmark.bundle");
    write_json(&bench_input, &benchmark_input(json!([]), json!([])));
    let bench_zero = command("sllm-bench")
        .args(["aggregate", "--input"])
        .arg(&bench_input)
        .args(["--output-bundle"])
        .arg(&bench_output)
        .args(["--tool-commit", &"0".repeat(40)])
        .output()
        .expect("run zero benchmark");
    assert_failure(&bench_zero);
    assert!(!bench_output.exists());

    let eval_input = temp.path().join("zero-quality.json");
    let eval_manifest = temp.path().join("quality-manifest.json");
    let eval_output = temp.path().join("zero-quality-result.json");
    write_json(
        &eval_input,
        &json!({
            "schema_version": "sllm-phase46-quality-input-v1",
            "perplexity": {"losses": []}
        }),
    );
    write_json(&eval_manifest, &manifest_json());
    let eval_zero = command("sllm-eval")
        .args(["--input"])
        .arg(&eval_input)
        .args(["--manifest"])
        .arg(&eval_manifest)
        .args(["--output"])
        .arg(&eval_output)
        .output()
        .expect("run zero quality evaluation");
    assert_failure(&eval_zero);
    assert!(!eval_output.exists());
}

#[test]
fn quantize_success_publishes_boundary_bundle_atomically() {
    let temp = TempDir::new("success");
    let input = temp.path().join("values.json");
    let output = temp.path().join("quantized.bundle");
    let values: Vec<f64> = (0..33).map(|index| (index as f64) / 7.0).collect();
    write_json(&input, &json!({"values": values}));

    let result = command("sllm-artifact")
        .args([
            "quantize",
            "--recipe",
            "mxfp4-e2m1-block32-e8m0",
            "--input-json",
        ])
        .arg(&input)
        .args(["--rows", "1", "--columns", "33", "--output-dir"])
        .arg(&output)
        .output()
        .expect("run boundary quantization");
    assert_success(&result);
    assert!(output.is_dir());
    assert!(output.join("quantized-tensor.json").is_file());
    assert!(output.join("run-manifest.json").is_file());

    let run_manifest: Value = serde_json::from_slice(
        &fs::read(output.join("run-manifest.json")).expect("read published run manifest"),
    )
    .expect("parse published run manifest");
    assert_eq!(run_manifest["state"], "PASS");
    assert_eq!(run_manifest["selected_count"], 33);
    assert_eq!(
        run_manifest["tool"]["executable_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert!(!run_manifest["outputs"].as_array().unwrap().is_empty());
    assert!(
        fs::read_dir(temp.path())
            .expect("read test output directory")
            .all(|entry| {
                let name = entry.expect("read output entry").file_name();
                let name = name.to_string_lossy();
                !name.contains("sllm-stage") && !name.contains("phase46-partial")
            })
    );
}
