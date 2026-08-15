use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use sllm_core::parse_gemma4_model_lock;

fn main() -> ExitCode {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 2 {
        eprintln!("usage: verify_gemma4_lock LOCK CACHE");
        return ExitCode::from(2);
    }
    let lock_path = PathBuf::from(&arguments[0]);
    let cache = PathBuf::from(&arguments[1]);
    let result = fs::read(&lock_path)
        .map_err(|error| format!("read lock: {error}"))
        .and_then(|bytes| parse_gemma4_model_lock(&bytes).map_err(|error| error.to_string()))
        .and_then(|lock| {
            lock.verify_cache(&cache)
                .map(|verified| (lock, verified))
                .map_err(|error| error.to_string())
        });
    match result {
        Ok((lock, verified)) => {
            println!(
                "Gemma lock: PASS repo={} revision={} fingerprint={} files={}",
                lock.model.repo_id,
                lock.model.resolved_revision,
                lock.fingerprint(),
                verified.files.len()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Gemma lock: FAIL: {error}");
            ExitCode::from(1)
        }
    }
}
