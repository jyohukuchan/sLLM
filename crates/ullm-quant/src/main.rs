use std::env;
use std::num::NonZeroUsize;
use std::process::ExitCode;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct KernelVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

unsafe extern "C" {
    fn ullm_aq_get_kernel_version() -> KernelVersion;
    fn ullm_aq_pack_nibbles(
        low: *const u8,
        high: *const u8,
        output: *mut u8,
        len: usize,
    ) -> usize;
}

#[derive(Debug)]
struct Options {
    threads: usize,
    io_threads: usize,
    max_working_memory_mib: usize,
    dry_run: bool,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .map(|threads| threads.min(64))
        .unwrap_or(1)
}

fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("{flag} requires a value"))?;
    let parsed = raw
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be >= 1"));
    }
    Ok(parsed)
}

fn parse_options() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mut options = Options {
        threads: default_threads(),
        io_threads: 2,
        max_working_memory_mib: 4096,
        dry_run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => options.threads = parse_usize("--threads", args.next())?,
            "--io-threads" => options.io_threads = parse_usize("--io-threads", args.next())?,
            "--max-working-memory-mib" => {
                options.max_working_memory_mib =
                    parse_usize("--max-working-memory-mib", args.next())?;
            }
            "--dry-run" => options.dry_run = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(options)
}

fn print_help() {
    println!("ullm-quant");
    println!();
    println!("Options:");
    println!("  --threads <N>                 compute worker threads");
    println!("  --io-threads <N>              read/write helper threads");
    println!("  --max-working-memory-mib <N>  working-memory budget");
    println!("  --dry-run                     print the current skeleton plan");
}

fn run_pack_smoke() -> Result<Vec<u8>, String> {
    let low = [0x00, 0x01, 0x0f, 0x08];
    let high = [0x01, 0x02, 0x00, 0x07];
    let mut output = [0u8; 4];
    let written = unsafe {
        ullm_aq_pack_nibbles(low.as_ptr(), high.as_ptr(), output.as_mut_ptr(), output.len())
    };
    if written != output.len() {
        return Err(format!("pack smoke wrote {written}, expected {}", output.len()));
    }
    let expected = [0x10, 0x21, 0x0f, 0x78];
    if output != expected {
        return Err(format!("pack smoke output mismatch: {output:?} != {expected:?}"));
    }
    Ok(output.to_vec())
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    let version = unsafe { ullm_aq_get_kernel_version() };
    let packed = run_pack_smoke()?;

    println!("ullm-quant skeleton");
    println!(
        "kernel_version={}.{}.{}",
        version.major, version.minor, version.patch
    );
    println!("threads={}", options.threads);
    println!("io_threads={}", options.io_threads);
    println!("max_working_memory_mib={}", options.max_working_memory_mib);
    println!("dry_run={}", options.dry_run);
    println!("pack_smoke=ok {packed:?}");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::default_threads;

    #[test]
    fn default_thread_count_is_nonzero() {
        assert!(default_threads() >= 1);
    }
}

