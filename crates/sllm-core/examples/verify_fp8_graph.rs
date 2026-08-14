use sllm_core::{
    build_qwen35_fp8_graph, build_verified_weight_load_plan, read_model_lock, verify_fp8_sidecar,
    verify_model_cache,
};
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if arguments.len() != 4 {
        eprintln!("usage: verify_fp8_graph SOURCE_LOCK CACHE MANIFEST ARTIFACT");
        std::process::exit(2);
    }
    let result = (|| -> Result<_, String> {
        let lock = read_model_lock(&arguments[0]).map_err(|error| error.to_string())?;
        let cache = verify_model_cache(&lock, &arguments[1]).map_err(|error| error.to_string())?;
        let plan =
            build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
        let sidecar = verify_fp8_sidecar(&arguments[2], &arguments[3], &arguments[0], &lock)
            .map_err(|error| error.to_string())?;
        let graph = build_qwen35_fp8_graph(&lock, &plan, &sidecar, 3, 129)
            .map_err(|error| error.to_string())?;
        let fp8_weights = graph
            .tensor_metadata()
            .iter()
            .filter(|tensor| tensor.view().dtype() == sllm_core::DType::F8E4M3Fn)
            .count();
        if fp8_weights != sidecar.tensors().len() {
            return Err("FP8 graph weight coverage differs from the sidecar".to_owned());
        }
        Ok((graph, sidecar, fp8_weights))
    })();
    match result {
        Ok((graph, sidecar, fp8_weights)) => println!(
            "FP8 graph verification: PASS fp8_weights={} nodes={} source={} sidecar={}",
            fp8_weights,
            graph.nodes().len(),
            graph.model_fingerprint(),
            sidecar.manifest_fingerprint(),
        ),
        Err(error) => {
            eprintln!("FP8 graph verification: FAIL: {error}");
            std::process::exit(1);
        }
    }
}
