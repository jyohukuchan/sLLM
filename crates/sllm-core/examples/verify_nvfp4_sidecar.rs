use sllm_core::{read_model_lock, verify_nvfp4_sidecar};
use std::path::Path;

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 3 {
        eprintln!("usage: verify_nvfp4_sidecar SOURCE_LOCK MANIFEST ARTIFACT");
        std::process::exit(2);
    }
    let result = (|| -> Result<_, String> {
        let lock = read_model_lock(Path::new(&arguments[0])).map_err(|error| error.to_string())?;
        verify_nvfp4_sidecar(
            Path::new(&arguments[1]),
            Path::new(&arguments[2]),
            Path::new(&arguments[0]),
            &lock,
        )
        .map_err(|error| error.to_string())
    })();
    match result {
        Ok(sidecar) => println!(
            "NVFP4 sidecar verification: PASS tensors={} source={} fingerprint={} artifact={}",
            sidecar.tensors().len(),
            sidecar.source_lock_fingerprint(),
            sidecar.manifest_fingerprint(),
            sidecar.artifact_sha256(),
        ),
        Err(error) => {
            eprintln!("NVFP4 sidecar verification: FAIL: {error}");
            std::process::exit(1);
        }
    }
}
