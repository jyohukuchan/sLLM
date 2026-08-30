fn main() {
    if let Err(error) = sllm_tools::run_bench_cli(std::env::args().skip(1)) {
        eprintln!("sllm-bench: {error}");
        std::process::exit(2);
    }
}
