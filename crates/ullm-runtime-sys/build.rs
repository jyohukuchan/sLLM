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

    cc::Build::new()
        .cpp(true)
        .include(root.join("runtime/include"))
        .file(source)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O2")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("ullm_runtime");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
