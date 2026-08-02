use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to start {description}: {error}"));
    assert!(
        status.success(),
        "{description} failed with status {status}"
    );
}

fn capture(command: &mut Command, description: &str) -> String {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to start {description}: {error}"));
    assert!(
        output.status.success(),
        "{description} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("{description} returned non-UTF-8 output: {error}"))
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_dir = manifest_dir.join("../../native/hip");
    let header = manifest_dir.join("../../include/ullm/hip.h");
    let umbrella_header = manifest_dir.join("../../include/ullm/ullm.h");
    let source = source_dir.join("src/hip_stub.cpp");
    let layout_probe = source_dir.join("src/abi_layout_probe.cpp");
    let header_c_compile = source_dir.join("src/header_c_compile.c");
    let header_cpp_compile = source_dir.join("src/header_cpp_compile.cpp");
    let bindings = manifest_dir.join("src/bindings.rs");
    let cmake_file = source_dir.join("CMakeLists.txt");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let build_dir = out_dir.join("native-hip-build");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", umbrella_header.display());
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", layout_probe.display());
    println!("cargo:rerun-if-changed={}", header_c_compile.display());
    println!("cargo:rerun-if-changed={}", header_cpp_compile.display());
    println!("cargo:rerun-if-changed={}", bindings.display());
    println!("cargo:rerun-if-changed={}", cmake_file.display());
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_HIP_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=ULLM_HIP_CODEGEN_FEATURES");
    println!("cargo:rerun-if-env-changed=CXX");

    let profile = env::var("PROFILE").expect("Cargo must provide PROFILE");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={profile}"))
        .arg(format!(
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
            build_dir.display()
        ));

    for variable in [
        "ROCM_PATH",
        "CMAKE_HIP_ARCHITECTURES",
        "ULLM_HIP_CODEGEN_FEATURES",
    ] {
        if let Some(value) = env::var_os(variable) {
            configure.arg(format!("-D{variable}={}", value.to_string_lossy()));
        }
    }
    if let Some(cxx) = env::var_os("CXX") {
        configure.arg(format!("-DCMAKE_CXX_COMPILER={}", cxx.to_string_lossy()));
    }
    run(&mut configure, "CMake configure");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("ullm_hip_stub");
    run(&mut build, "CMake build");

    let archive = static_archive(&build_dir);
    assert!(
        archive.is_file(),
        "native archive was not produced: {}",
        archive.display()
    );
    verify_checked_in_bindings(&manifest_dir, &layout_probe, &bindings, &out_dir);
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=ullm_hip_stub");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn verify_checked_in_bindings(
    manifest_dir: &Path,
    layout_probe: &Path,
    bindings: &Path,
    out_dir: &Path,
) {
    let cxx = env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let cxx_probe = out_dir.join("ullm-abi-layout-cxx");
    let cxx_output = capture(
        Command::new(&cxx)
            .arg("-std=c++17")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg("-I")
            .arg(manifest_dir.join("../../include"))
            .arg(layout_probe)
            .arg("-o")
            .arg(&cxx_probe),
        "C++ ABI layout probe compilation",
    );
    assert!(cxx_output.is_empty(), "C++ ABI probe compiler wrote stdout");
    let c_layout = capture(&mut Command::new(&cxx_probe), "C++ ABI layout probe");

    let rust_probe_source = out_dir.join("ullm-abi-layout-rust.rs");
    let rust_probe_binary = out_dir.join("ullm-abi-layout-rust");
    let bindings_path = bindings
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", bindings.display()));
    let rust_source = format!(
        "#[path = {:?}] mod bindings;\n\
         use std::mem::{{align_of, offset_of, size_of}};\n\
         fn main() {{\n\
             println!(\"const ULLM_HIP_ABI_VERSION={{}}\", bindings::ULLM_HIP_ABI_VERSION);\n\
             println!(\"const ULLM_HIP_LIBRARY_VERSION_MAJOR={{}}\", bindings::ULLM_HIP_LIBRARY_VERSION_MAJOR);\n\
             println!(\"const ULLM_HIP_LIBRARY_VERSION_MINOR={{}}\", bindings::ULLM_HIP_LIBRARY_VERSION_MINOR);\n\
             println!(\"const ULLM_HIP_LIBRARY_VERSION_PATCH={{}}\", bindings::ULLM_HIP_LIBRARY_VERSION_PATCH);\n\
             println!(\"const ULLM_STATUS_OK={{}}\", bindings::ULLM_STATUS_OK);\n\
             println!(\"const ULLM_STATUS_INVALID_ARGUMENT={{}}\", bindings::ULLM_STATUS_INVALID_ARGUMENT);\n\
             println!(\"const ULLM_STATUS_BUFFER_TOO_SMALL={{}}\", bindings::ULLM_STATUS_BUFFER_TOO_SMALL);\n\
             println!(\"const ULLM_STATUS_UNSUPPORTED={{}}\", bindings::ULLM_STATUS_UNSUPPORTED);\n\
             println!(\"const ULLM_STATUS_HIP_UNAVAILABLE={{}}\", bindings::ULLM_STATUS_HIP_UNAVAILABLE);\n\
             println!(\"const ULLM_STATUS_INVALID_ABI_VERSION={{}}\", bindings::ULLM_STATUS_INVALID_ABI_VERSION);\n\
             println!(\"const ULLM_STATUS_RESERVED_NONZERO={{}}\", bindings::ULLM_STATUS_RESERVED_NONZERO);\n\
             println!(\"const ULLM_STATUS_INTERNAL_ERROR={{}}\", bindings::ULLM_STATUS_INTERNAL_ERROR);\n\
             println!(\"const ULLM_BACKEND_HIP={{}}\", bindings::ULLM_BACKEND_HIP);\n\
             println!(\"const ULLM_ACCESS_READ={{}}\", bindings::ULLM_ACCESS_READ);\n\
             println!(\"const ULLM_ACCESS_WRITE={{}}\", bindings::ULLM_ACCESS_WRITE);\n\
             println!(\"const ULLM_ACCESS_READ_WRITE={{}}\", bindings::ULLM_ACCESS_READ_WRITE);\n\
             println!(\"layout ullm_error_sink_t size={{}} align={{}} struct_size={{}} abi_version={{}} message={{}} message_capacity={{}} message_length={{}} reserved={{}}\", size_of::<bindings::ullm_error_sink_t>(), align_of::<bindings::ullm_error_sink_t>(), offset_of!(bindings::ullm_error_sink_t, struct_size), offset_of!(bindings::ullm_error_sink_t, abi_version), offset_of!(bindings::ullm_error_sink_t, message), offset_of!(bindings::ullm_error_sink_t, message_capacity), offset_of!(bindings::ullm_error_sink_t, message_length), offset_of!(bindings::ullm_error_sink_t, reserved));\n\
             println!(\"layout ullm_version_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} major={{}} minor={{}} patch={{}} reserved={{}}\", size_of::<bindings::ullm_version_info_t>(), align_of::<bindings::ullm_version_info_t>(), offset_of!(bindings::ullm_version_info_t, struct_size), offset_of!(bindings::ullm_version_info_t, abi_version), offset_of!(bindings::ullm_version_info_t, major), offset_of!(bindings::ullm_version_info_t, minor), offset_of!(bindings::ullm_version_info_t, patch), offset_of!(bindings::ullm_version_info_t, reserved));\n\
             println!(\"layout ullm_backend_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} backend={{}} available={{}} hip_runtime_present={{}} reserved={{}}\", size_of::<bindings::ullm_backend_probe_result_t>(), align_of::<bindings::ullm_backend_probe_result_t>(), offset_of!(bindings::ullm_backend_probe_result_t, struct_size), offset_of!(bindings::ullm_backend_probe_result_t, abi_version), offset_of!(bindings::ullm_backend_probe_result_t, backend), offset_of!(bindings::ullm_backend_probe_result_t, available), offset_of!(bindings::ullm_backend_probe_result_t, hip_runtime_present), offset_of!(bindings::ullm_backend_probe_result_t, reserved));\n\
             println!(\"layout ullm_context_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} context_present={{}} hip_available={{}} reserved={{}}\", size_of::<bindings::ullm_context_probe_result_t>(), align_of::<bindings::ullm_context_probe_result_t>(), offset_of!(bindings::ullm_context_probe_result_t, struct_size), offset_of!(bindings::ullm_context_probe_result_t, abi_version), offset_of!(bindings::ullm_context_probe_result_t, context_present), offset_of!(bindings::ullm_context_probe_result_t, hip_available), offset_of!(bindings::ullm_context_probe_result_t, reserved));\n\
         }}\n",
        bindings_path.display()
    );
    fs::write(&rust_probe_source, rust_source)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", rust_probe_source.display()));
    run(
        Command::new("rustc")
            .arg("--edition=2024")
            .arg(&rust_probe_source)
            .arg("-o")
            .arg(&rust_probe_binary),
        "Rust ABI layout probe compilation",
    );
    let rust_layout = capture(
        &mut Command::new(&rust_probe_binary),
        "Rust ABI layout probe",
    );

    assert_eq!(
        c_layout.trim(),
        rust_layout.trim(),
        "checked-in Rust bindings do not match include/ullm/hip.h ABI layout/constants\nC++:\n{}\nRust:\n{}",
        c_layout,
        rust_layout
    );
}

fn static_archive(build_dir: &Path) -> PathBuf {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build_dir.join("ullm_hip_stub.lib")
    } else {
        build_dir.join("libullm_hip_stub.a")
    }
}
