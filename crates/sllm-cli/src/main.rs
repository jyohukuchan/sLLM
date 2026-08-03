use std::env;
use std::process::ExitCode;

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
        Some(command) => Err(format!("unknown command `{command}`; try `sllm help`")),
    }
}

fn print_help() {
    println!("sLLM Phase 1");
    println!();
    println!("Usage: sllm <command>");
    println!();
    println!("Commands:");
    println!("  version  Print the package and ABI version");
    println!("  doctor   Probe the Phase 1 host backend boundary");
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
