use std::env;
use std::process::ExitCode;

mod benchmark;
mod model;

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("sllm: {message}");
            ExitCode::from(2)
        }
    }
}

fn run(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    match arguments.next().as_deref() {
        None | Some("help") | Some("--help") | Some("-h") => {
            print_help();
            Ok(())
        }
        Some("version") | Some("--version") | Some("-V") => print_version(),
        Some("doctor") => print_doctor(),
        Some(
            command
            @ ("verify-model" | "tokenize" | "render" | "decode" | "generate" | "benchmark"),
        ) => {
            let output = model::run(command, arguments)?;
            println!("{output}");
            Ok(())
        }
        Some(command) => Err(format!("unknown command `{command}`; try `sllm help`")),
    }
}

fn print_help() {
    println!("sLLM");
    println!();
    println!("Usage: sllm <command>");
    println!();
    println!("Commands:");
    println!("  version  Print the package and ABI version");
    println!("  doctor   Probe the Phase 1 host backend boundary");
    println!("  verify-model  Verify a derived GGUF artifact");
    println!("  tokenize      Encode text with the verified tokenizer");
    println!("  render        Render Qwen3.5 chat messages");
    println!("  decode        Decode token IDs with the verified tokenizer");
    println!("  generate      Run Qwen3.5 text generation on one exact HIP target");
    println!("  benchmark     Run the bounded Phase 5 engine benchmark lanes");
    println!();
    println!("model source: --gguf PATH --derived-lock PATH");
    println!("benchmark lane: --lane render-tokenize --model-size 2B|4B|9B");
    println!("  --case-id ID --message ROLE:CONTENT --max-new-tokens N --device-index N");
    println!("  --target gfx1030|gfx1201|gfx942 --greedy [--warmups N] [--measured N]");
    println!("  requires exactly 3 warmup and 10 measured requests");
    println!();
    println!("generate: --prompt TEXT | --message ROLE:CONTENT --max-new-tokens N");
    println!(
        "  --device-index N --target gfx1030|gfx1201|gfx942 [--greedy | --temperature F32] [--seed U64]"
    );
    println!("  [--top-p F32] [--presence-penalty F32] [--frequency-penalty F32]");
    println!("  [--stop TEXT] (repeat --stop at most four times)");
    println!("  [--image PATH] (Qwen3.5 BF16 chat only; at most two, before final user text)");
    println!("  [--fp8-provider native|native-fnuz|emulation|converted-bf16]");
}

fn print_version() -> Result<(), String> {
    let output = version_output(sllm_hip::version())?;
    print!("{output}");
    Ok(())
}

fn version_output(
    version: Result<sllm_hip::Version, sllm_hip::HipError>,
) -> Result<String, String> {
    let version = version.map_err(|error| format!("ABI version query failed: {error}"))?;
    Ok(format!(
        "sllm {}\nabi {} (native {}.{}.{})\n",
        env!("CARGO_PKG_VERSION"),
        version.abi_version,
        version.major,
        version.minor,
        version.patch
    ))
}

fn print_doctor() -> Result<(), String> {
    println!("sLLM doctor");
    println!("phase: 1 host stub");

    let backend = sllm_hip::backend_probe().map_err(|error| error.to_string())?;
    println!(
        "HIP: {}",
        if backend.available {
            "available"
        } else {
            "unavailable (no HIP/ROCm runtime is linked in Phase 1)"
        }
    );
    println!("HIP runtime present: {}", backend.hip_runtime_present);
    println!("diagnostic: {}", backend.diagnostic);

    let context = sllm_hip::context_probe().map_err(|error| error.to_string())?;
    println!(
        "context: {}",
        if context.hip_available {
            "available"
        } else {
            "unavailable"
        }
    );
    println!("context diagnostic: {}", context.diagnostic);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_abi_failure_is_a_cli_error() {
        let error = version_output(Err(sllm_hip::HipError::Status {
            status: sllm_hip::Status::InvalidAbiVersion,
            message: "test ABI mismatch".to_owned(),
        }))
        .expect_err("ABI failures must not be printed as successful versions");
        assert!(error.contains("ABI version query failed"));
        assert!(error.contains("test ABI mismatch"));
    }
}
