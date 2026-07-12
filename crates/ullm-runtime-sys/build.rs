// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .expect("workspace root");

    let header = root.join("runtime/include/ullm_runtime.h");
    let source = root.join("runtime/src/ullm_runtime.cpp");
    let include_sources = [
        root.join("runtime/src/ullm_runtime_hiprtc_sources.inc"),
        root.join("runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc"),
        root.join("runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc"),
        root.join("runtime/src/ullm_runtime_parts/part_00.inc"),
        root.join("runtime/src/ullm_runtime_parts/part_01.inc"),
        root.join("runtime/src/ullm_runtime_api.inc"),
        root.join("runtime/src/ullm_runtime_api_core.inc"),
        root.join("runtime/src/ullm_runtime_api_sq8_ck.inc"),
        root.join("runtime/src/sq8_ck_gfx1201.h"),
        root.join("runtime/src/sq8_ck_gfx1201.hip.cpp"),
        root.join("runtime/src/ullm_runtime_api_aq4.inc"),
        root.join("runtime/src/ullm_runtime_api_linear_attn_prepare.inc"),
        root.join("runtime/src/ullm_runtime_api_primitives.inc"),
        root.join("runtime/src/ullm_runtime_api_sq8_0.inc"),
        root.join("runtime/src/ullm_runtime_api_attention.inc"),
        root.join("runtime/src/ullm_runtime_api_linear_attn.inc"),
        root.join("runtime/src/ullm_runtime_api_smoke.inc"),
    ];

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", source.display());
    for include_source in include_sources {
        println!("cargo:rerun-if-changed={}", include_source.display());
    }

    let ck_enabled = std::env::var_os("CARGO_FEATURE_ROCM_CK_GFX1201").is_some();
    let rocm_path =
        PathBuf::from(std::env::var_os("ROCM_PATH").unwrap_or_else(|| "/opt/rocm".into()));
    let mut runtime = cc::Build::new();
    runtime
        .cpp(true)
        .include(root.join("runtime/include"))
        .file(source)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O2")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra");
    if rocm_path.join("include/hip/hip_runtime_api.h").is_file() {
        runtime
            .include(rocm_path.join("include"))
            .define("__HIP_PLATFORM_AMD__", "1")
            .define("ULLM_HAVE_HIP_RUNTIME_API", "1");
    }
    if ck_enabled {
        runtime.define("ULLM_RUNTIME_ROCM_CK_GFX1201", "1");
    }
    runtime.compile("ullm_runtime");

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=GPU_ARCH");
    if ck_enabled {
        let gpu_arch = std::env::var("GPU_ARCH").unwrap_or_else(|_| "gfx1201".to_string());
        if gpu_arch != "gfx1201" {
            panic!("Cargo feature rocm-ck-gfx1201 requires GPU_ARCH=gfx1201");
        }
        let hipcc = rocm_path.join("bin/hipcc");
        if !hipcc.is_file() {
            panic!("ROCm hipcc was not found at {}", hipcc.display());
        }

        cc::Build::new()
            .cpp(true)
            .compiler(hipcc)
            .include(root.join("runtime/src"))
            .include(rocm_path.join("include"))
            .file(root.join("runtime/src/sq8_ck_gfx1201.hip.cpp"))
            .define("CK_USE_OCP_FP8", "1")
            .define("CK_ENABLE_FP8", "1")
            .define("CK_ENABLE_BF16", "1")
            .flag("-std=c++20")
            .flag("-O3")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .flag("-ffunction-sections")
            .flag("-fdata-sections")
            .flag(&format!("--offload-arch={gpu_arch}"))
            .compile("ullm_runtime_sq8_ck_gfx1201");

        println!(
            "cargo:rustc-link-search=native={}",
            rocm_path.join("lib").display()
        );
        println!("cargo:rustc-link-lib=static=device_gemm_operations");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
