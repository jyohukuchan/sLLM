// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "rocm-ck-gfx1201")]

use serde_json::Value;
use std::process::Command;

fn worker_binary() -> &'static str {
    env!("CARGO_BIN_EXE_ullm-sq8-worker")
}

fn json_lines(bytes: &[u8]) -> Vec<Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

#[test]
fn help_exits_zero_without_writing_protocol_stdout() {
    let output = Command::new(worker_binary())
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .starts_with("Usage: ullm-sq8-worker")
    );
}

#[test]
fn invalid_cli_exits_one_with_structured_stderr_and_empty_stdout() {
    let output = Command::new(worker_binary())
        .arg("--unknown")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let logs = json_lines(&output.stderr);
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["schema_version"], "ullm.worker.log.v1");
    assert_eq!(logs[0]["event"], "cli_failed");
    assert_eq!(logs[0]["error_code"], "invalid_cli");
}

#[test]
fn load_failure_exits_one_and_stdout_contains_only_protocol_json() {
    let output = Command::new(worker_binary())
        .args([
            "--artifact",
            "/definitely-not-present/ullm-sq8-artifact",
            "--package",
            "/definitely-not-present/ullm-sq8-package",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let events = json_lines(&output.stdout);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["schema_version"], "ullm.worker.v1");
    assert_eq!(events[0]["type"], "error");
    assert_eq!(events[0]["code"], "load_failed");
    assert_eq!(events[0]["recoverable"], false);

    let logs = json_lines(&output.stderr);
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["schema_version"], "ullm.worker.log.v1");
    assert_eq!(logs[0]["event"], "process_failed");
    assert_eq!(logs[0]["error_code"], "process_failed");
}
