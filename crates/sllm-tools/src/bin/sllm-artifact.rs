fn main() {
    match sllm_tools::run_artifact_cli(std::env::args().skip(1)) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("sllm-artifact: {error}");
            std::process::exit(2);
        }
    }
}
