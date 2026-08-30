fn main() {
    if let Err(error) = sllm_tools::run_eval_cli(std::env::args().skip(1)) {
        eprintln!("sllm-eval: {error}");
        std::process::exit(2);
    }
}
