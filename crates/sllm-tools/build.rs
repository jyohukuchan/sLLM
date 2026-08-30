use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");
    println!("cargo:rerun-if-env-changed=SLLM_GIT_COMMIT");
    if let Ok(head) = fs::read_to_string("../../.git/HEAD") {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            let reference_path = Path::new("../../.git").join(reference);
            println!("cargo:rerun-if-changed={}", reference_path.display());
        }
    }
    let explicit = std::env::var("SLLM_GIT_COMMIT").ok();
    let discovered = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned());
    let commit = explicit
        .or(discovered)
        .expect("SLLM_GIT_COMMIT or a readable Git HEAD is required");
    assert!(
        commit.len() == 40
            && commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "SLLM_GIT_COMMIT must be a lowercase 40-hex commit"
    );
    assert!(
        commit.bytes().any(|byte| byte != b'0'),
        "SLLM_GIT_COMMIT must identify a real commit"
    );
    println!("cargo:rustc-env=SLLM_GIT_COMMIT={commit}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_output = Command::new(rustc)
        .arg("--version")
        .arg("--verbose")
        .output()
        .expect("run rustc --version --verbose");
    assert!(rustc_output.status.success(), "rustc version query failed");
    let rustc_verbose = String::from_utf8(rustc_output.stdout)
        .expect("rustc version output must be UTF-8")
        .lines()
        .collect::<Vec<_>>()
        .join("; ");
    assert!(!rustc_verbose.is_empty(), "rustc version output is empty");
    println!("cargo:rustc-env=SLLM_RUSTC_VERBOSE={rustc_verbose}");
}
