use std::process::Command;

fn sllm(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_sllm"))
        .args(args)
        .output()
        .expect("sllm CLI must start")
}

#[test]
fn help_lists_offline_model_frontend_commands() {
    let output = sllm(&["help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for command in ["verify-model", "tokenize", "render", "decode", "generate"] {
        assert!(stdout.contains(command), "help omitted {command}");
    }
    for legacy_option in ["--model-lock", "--fp8-manifest", "--fp8-provider"] {
        assert!(
            !stdout.contains(legacy_option),
            "help retained legacy option {legacy_option}"
        );
    }
    assert!(output.stderr.is_empty());
}

#[test]
fn model_frontend_failures_use_stderr_and_exit_two() {
    for command in ["verify-model", "tokenize", "render", "decode", "generate"] {
        let output = sllm(&[command]);
        assert_eq!(output.status.code(), Some(2), "{command}");
        assert!(
            output.stdout.is_empty(),
            "{command} emitted partial success"
        );
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .starts_with("sllm: "),
            "{command} did not emit a stable CLI error"
        );
    }
}

#[test]
fn invalid_derived_lock_never_falls_through_to_hip_or_json_success() {
    let output = sllm(&[
        "verify-model",
        "--gguf",
        "/definitely/not/a/model.gguf",
        "--derived-lock",
        "/definitely/not/a/derived-lock.json",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("derived GGUF lock is invalid"));
    assert!(!stderr.contains("HIP"));
}
