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

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", source.display());

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
