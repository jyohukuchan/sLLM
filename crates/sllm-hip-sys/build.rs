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
    let header = manifest_dir.join("../../include/sllm/hip.h");
    let umbrella_header = manifest_dir.join("../../include/sllm/sllm.h");
    let source = source_dir.join("src/hip_stub.cpp");
    let evidence_header = source_dir.join("src/evidence_abi.h");
    let evidence_stub = source_dir.join("src/hip_evidence_stub.cpp");
    let evidence_runtime = source_dir.join("src/hip_evidence_runtime.hip.cpp");
    let layout_probe = source_dir.join("src/abi_layout_probe.cpp");
    let header_c_compile = source_dir.join("src/header_c_compile.c");
    let header_cpp_compile = source_dir.join("src/header_cpp_compile.cpp");
    let bindings = manifest_dir.join("src/bindings.rs");
    let evidence_bindings = manifest_dir.join("src/evidence_bindings.rs");
    let cmake_file = source_dir.join("CMakeLists.txt");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", umbrella_header.display());
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", evidence_header.display());
    println!("cargo:rerun-if-changed={}", evidence_stub.display());
    println!("cargo:rerun-if-changed={}", evidence_runtime.display());
    println!("cargo:rerun-if-changed={}", layout_probe.display());
    println!("cargo:rerun-if-changed={}", header_c_compile.display());
    println!("cargo:rerun-if-changed={}", header_cpp_compile.display());
    println!("cargo:rerun-if-changed={}", bindings.display());
    println!("cargo:rerun-if-changed={}", evidence_bindings.display());
    println!("cargo:rerun-if-changed={}", cmake_file.display());
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_HIP_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_CODEGEN_FEATURES");
    println!("cargo:rerun-if-env-changed=SLLM_ENABLE_HIP_COMPILE_PROBE");
    println!("cargo:rerun-if-env-changed=SLLM_ENABLE_HIP_RUNTIME");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER");
    println!("cargo:rerun-if-env-changed=CXX");

    let profile = env::var("PROFILE").expect("Cargo must provide PROFILE");
    let hip_probe = match env::var("SLLM_ENABLE_HIP_COMPILE_PROBE") {
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("on") => true,
        Ok(value) if value == "0" || value.eq_ignore_ascii_case("off") => false,
        Ok(value) => {
            panic!("SLLM_ENABLE_HIP_COMPILE_PROBE must be unset, 0/OFF, or 1/ON; got {value}")
        }
        Err(env::VarError::NotPresent) => false,
        Err(error) => panic!("cannot read SLLM_ENABLE_HIP_COMPILE_PROBE: {error}"),
    };
    let hip_configuration = if hip_probe {
        Some(validate_hip_environment(&profile, "H3 compile probe"))
    } else {
        None
    };
    let hip_runtime = match env::var("SLLM_ENABLE_HIP_RUNTIME") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => panic!("SLLM_ENABLE_HIP_RUNTIME must be unset, 0, or exactly 1; got {value}"),
        Err(env::VarError::NotPresent) => false,
        Err(error) => panic!("cannot read SLLM_ENABLE_HIP_RUNTIME: {error}"),
    };
    let hip_configuration = if hip_runtime {
        Some(hip_configuration.unwrap_or_else(|| validate_hip_environment(&profile, "HIP runtime")))
    } else {
        hip_configuration
    };
    let build_dir = match &hip_configuration {
        Some(configuration) => out_dir.join(format!("native-hip-build-{}", configuration.target)),
        None => out_dir.join("native-hip-build-stub"),
    };
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-G")
        .arg("Unix Makefiles")
        .arg(format!("-DCMAKE_BUILD_TYPE={profile}"))
        .arg(format!(
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
            build_dir.display()
        ));

    if let Some(configuration) = &hip_configuration {
        configure
            .arg(format!("-DROCM_PATH={}", configuration.rocm_path.display()))
            .arg(format!(
                "-DCMAKE_HIP_COMPILER={}",
                configuration.compiler.display()
            ))
            .arg(format!(
                "-DCMAKE_HIP_ARCHITECTURES={}",
                configuration.target
            ))
            .arg(format!(
                "-DSLLM_HIP_COMPILE_TARGET={}",
                configuration.target
            ))
            .arg(format!(
                "-DSLLM_HIP_CODEGEN_FEATURES={}",
                configuration.codegen_features
            ));
        configure.arg(if hip_probe {
            "-DSLLM_ENABLE_HIP_COMPILE_PROBE=ON"
        } else {
            "-DSLLM_ENABLE_HIP_COMPILE_PROBE=OFF"
        });
        configure.arg(if hip_runtime {
            "-DSLLM_ENABLE_HIP_RUNTIME=ON"
        } else {
            "-DSLLM_ENABLE_HIP_RUNTIME=OFF"
        });
    } else {
        configure.arg("-DSLLM_ENABLE_HIP_COMPILE_PROBE=OFF");
        configure.arg("-DSLLM_ENABLE_HIP_RUNTIME=OFF");
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
        .arg("sllm_hip_stub");
    run(&mut build, "CMake build");
    if hip_probe {
        let mut probe_build = Command::new("cmake");
        probe_build
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("sllm_hip_compile_probe_link");
        run(&mut probe_build, "HIP compile/link probe build");
    }

    let archive = static_archive(&build_dir);
    assert!(
        archive.is_file(),
        "native archive was not produced: {}",
        archive.display()
    );
    verify_checked_in_bindings(&manifest_dir, &layout_probe, &bindings, &out_dir);
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=sllm_hip_stub");
    if hip_runtime {
        let runtime_rocm_lib = hip_configuration
            .as_ref()
            .expect("runtime configuration")
            .rocm_path
            .join("lib");
        println!(
            "cargo:rustc-link-search=native={}",
            runtime_rocm_lib.display()
        );
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

struct HipConfiguration {
    rocm_path: PathBuf,
    compiler: PathBuf,
    target: String,
    codegen_features: String,
}

fn validate_hip_environment(profile: &str, purpose: &str) -> HipConfiguration {
    assert_eq!(profile, "release", "{purpose} requires Cargo --release");
    let rocm_path = required_absolute_path("ROCM_PATH");
    assert_eq!(
        rocm_path,
        Path::new("/opt/rocm"),
        "{purpose} requires the logical ROCm root /opt/rocm"
    );
    let canonical_rocm = rocm_path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "cannot canonicalize ROCM_PATH {}: {error}",
            rocm_path.display()
        )
    });

    let compiler = required_absolute_path("SLLM_HIP_COMPILER");
    assert_eq!(
        compiler,
        rocm_path.join("bin/amdclang++"),
        "{purpose} requires the logical ROCM_PATH/bin/amdclang++ entry point"
    );
    let compiler_real = compiler.canonicalize().unwrap_or_else(|error| {
        panic!(
            "cannot canonicalize HIP compiler {}: {error}",
            compiler.display()
        )
    });
    assert!(
        path_within(&compiler_real, &canonical_rocm),
        "HIP compiler must resolve inside ROCM_PATH: {}",
        compiler_real.display()
    );
    assert_eq!(
        compiler.file_name().and_then(|name| name.to_str()),
        Some("amdclang++"),
        "{purpose} requires the ROCm amdclang++ entry point"
    );
    verify_rocm_release(&canonical_rocm, purpose);
    let compiler_version = capture(
        Command::new(&compiler).arg("--version"),
        "ROCm amdclang++ version probe",
    );
    let version_line = compiler_version.lines().next().unwrap_or_default();
    assert!(
        version_line.starts_with("AMD clang version 23."),
        "{purpose} requires LLVM major 23 from ROCm amdclang++; got {version_line}"
    );

    let target = env::var("CMAKE_HIP_ARCHITECTURES")
        .unwrap_or_else(|_| panic!("H3 requires CMAKE_HIP_ARCHITECTURES"));
    assert!(
        matches!(target.as_str(), "gfx1030" | "gfx1201"),
        "{purpose} requires exactly one exact gfx1030 or gfx1201 target"
    );
    assert!(
        !target.contains(';') && !target.contains(',') && !target.contains(' '),
        "H3 target must not contain multiple or generic architectures"
    );

    let codegen_features = env::var("SLLM_HIP_CODEGEN_FEATURES")
        .unwrap_or_else(|_| panic!("H3 requires SLLM_HIP_CODEGEN_FEATURES"));
    assert_eq!(
        codegen_features,
        "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0",
        "HIP codegen features are not the pinned tuple"
    );
    HipConfiguration {
        rocm_path: canonical_rocm,
        compiler,
        target,
        codegen_features,
    }
}

fn verify_rocm_release(rocm_path: &Path, purpose: &str) {
    let mut markers = Vec::new();
    let direct_marker = rocm_path.join(".info/version");
    if direct_marker.is_file() {
        markers.push(direct_marker);
    }
    let entries = fs::read_dir(rocm_path).unwrap_or_else(|error| {
        panic!(
            "{purpose} cannot inspect ROCm root {}: {error}",
            rocm_path.display()
        )
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "{purpose} cannot inspect ROCm root {}: {error}",
                rocm_path.display()
            )
        });
        let marker = entry.path().join(".info/version");
        if marker.is_file() {
            markers.push(marker);
        }
    }
    assert!(
        !markers.is_empty(),
        "{purpose} requires a ROCm .info/version release marker under {}",
        rocm_path.display()
    );
    for marker in markers {
        let release = fs::read_to_string(&marker).unwrap_or_else(|error| {
            panic!(
                "{purpose} cannot read the ROCm release marker {}: {error}",
                marker.display()
            )
        });
        assert_eq!(
            release.trim(),
            "7.14.0",
            "{purpose} requires every discovered ROCm release marker to be 7.14.0: {}",
            marker.display()
        );
    }
}

fn path_within(path: &Path, root: &Path) -> bool {
    path == root || path.strip_prefix(root).is_ok()
}

fn required_absolute_path(name: &str) -> PathBuf {
    let value = env::var_os(name).unwrap_or_else(|| panic!("H3 requires {name}"));
    let path = PathBuf::from(value);
    assert!(path.is_absolute(), "{name} must be an absolute path");
    path
}

fn verify_checked_in_bindings(
    manifest_dir: &Path,
    layout_probe: &Path,
    bindings: &Path,
    out_dir: &Path,
) {
    let cxx = env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let cxx_probe = out_dir.join("sllm-abi-layout-cxx");
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

    let rust_probe_source = out_dir.join("sllm-abi-layout-rust.rs");
    let rust_probe_binary = out_dir.join("sllm-abi-layout-rust");
    let bindings_path = bindings
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", bindings.display()));
    let rust_source = format!(
        "#[path = {:?}] mod bindings;\n\
         use std::mem::{{align_of, offset_of, size_of}};\n\
         fn main() {{\n\
             println!(\"const SLLM_HIP_ABI_VERSION={{}}\", bindings::SLLM_HIP_ABI_VERSION);\n\
             println!(\"const SLLM_HIP_LIBRARY_VERSION_MAJOR={{}}\", bindings::SLLM_HIP_LIBRARY_VERSION_MAJOR);\n\
             println!(\"const SLLM_HIP_LIBRARY_VERSION_MINOR={{}}\", bindings::SLLM_HIP_LIBRARY_VERSION_MINOR);\n\
             println!(\"const SLLM_HIP_LIBRARY_VERSION_PATCH={{}}\", bindings::SLLM_HIP_LIBRARY_VERSION_PATCH);\n\
             println!(\"const SLLM_STATUS_OK={{}}\", bindings::SLLM_STATUS_OK);\n\
             println!(\"const SLLM_STATUS_INVALID_ARGUMENT={{}}\", bindings::SLLM_STATUS_INVALID_ARGUMENT);\n\
             println!(\"const SLLM_STATUS_BUFFER_TOO_SMALL={{}}\", bindings::SLLM_STATUS_BUFFER_TOO_SMALL);\n\
             println!(\"const SLLM_STATUS_UNSUPPORTED={{}}\", bindings::SLLM_STATUS_UNSUPPORTED);\n\
             println!(\"const SLLM_STATUS_HIP_UNAVAILABLE={{}}\", bindings::SLLM_STATUS_HIP_UNAVAILABLE);\n\
             println!(\"const SLLM_STATUS_INVALID_ABI_VERSION={{}}\", bindings::SLLM_STATUS_INVALID_ABI_VERSION);\n\
             println!(\"const SLLM_STATUS_RESERVED_NONZERO={{}}\", bindings::SLLM_STATUS_RESERVED_NONZERO);\n\
             println!(\"const SLLM_STATUS_INTERNAL_ERROR={{}}\", bindings::SLLM_STATUS_INTERNAL_ERROR);\n\
             println!(\"const SLLM_BACKEND_HIP={{}}\", bindings::SLLM_BACKEND_HIP);\n\
             println!(\"const SLLM_ACCESS_READ={{}}\", bindings::SLLM_ACCESS_READ);\n\
             println!(\"const SLLM_ACCESS_WRITE={{}}\", bindings::SLLM_ACCESS_WRITE);\n\
             println!(\"const SLLM_ACCESS_READ_WRITE={{}}\", bindings::SLLM_ACCESS_READ_WRITE);\n\
             println!(\"layout sllm_error_sink_t size={{}} align={{}} struct_size={{}} abi_version={{}} message={{}} message_capacity={{}} message_length={{}} reserved={{}}\", size_of::<bindings::sllm_error_sink_t>(), align_of::<bindings::sllm_error_sink_t>(), offset_of!(bindings::sllm_error_sink_t, struct_size), offset_of!(bindings::sllm_error_sink_t, abi_version), offset_of!(bindings::sllm_error_sink_t, message), offset_of!(bindings::sllm_error_sink_t, message_capacity), offset_of!(bindings::sllm_error_sink_t, message_length), offset_of!(bindings::sllm_error_sink_t, reserved));\n\
             println!(\"layout sllm_version_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} major={{}} minor={{}} patch={{}} reserved={{}}\", size_of::<bindings::sllm_version_info_t>(), align_of::<bindings::sllm_version_info_t>(), offset_of!(bindings::sllm_version_info_t, struct_size), offset_of!(bindings::sllm_version_info_t, abi_version), offset_of!(bindings::sllm_version_info_t, major), offset_of!(bindings::sllm_version_info_t, minor), offset_of!(bindings::sllm_version_info_t, patch), offset_of!(bindings::sllm_version_info_t, reserved));\n\
             println!(\"layout sllm_backend_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} backend={{}} available={{}} hip_runtime_present={{}} reserved={{}}\", size_of::<bindings::sllm_backend_probe_result_t>(), align_of::<bindings::sllm_backend_probe_result_t>(), offset_of!(bindings::sllm_backend_probe_result_t, struct_size), offset_of!(bindings::sllm_backend_probe_result_t, abi_version), offset_of!(bindings::sllm_backend_probe_result_t, backend), offset_of!(bindings::sllm_backend_probe_result_t, available), offset_of!(bindings::sllm_backend_probe_result_t, hip_runtime_present), offset_of!(bindings::sllm_backend_probe_result_t, reserved));\n\
             println!(\"layout sllm_context_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} context_present={{}} hip_available={{}} reserved={{}}\", size_of::<bindings::sllm_context_probe_result_t>(), align_of::<bindings::sllm_context_probe_result_t>(), offset_of!(bindings::sllm_context_probe_result_t, struct_size), offset_of!(bindings::sllm_context_probe_result_t, abi_version), offset_of!(bindings::sllm_context_probe_result_t, context_present), offset_of!(bindings::sllm_context_probe_result_t, hip_available), offset_of!(bindings::sllm_context_probe_result_t, reserved));\n\
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
        "checked-in Rust bindings do not match include/sllm/hip.h ABI layout/constants\nC++:\n{}\nRust:\n{}",
        c_layout,
        rust_layout
    );
}

fn static_archive(build_dir: &Path) -> PathBuf {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build_dir.join("sllm_hip_stub.lib")
    } else {
        build_dir.join("libsllm_hip_stub.a")
    }
}
