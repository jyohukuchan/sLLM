use std::env;
use std::fs;
#[cfg(target_os = "linux")]
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use sha2::{Digest, Sha256};

const G2_BUILD_INPUTS_PATH: &str = "ci/matrix/rmsnorm-g2-build-inputs-v1.json";
const G2_IDENTITY_SCHEMA: &str = "rmsnorm-g2-build-identity-v1";
const G2_ROLE: &str = "dedicated-g2-runtime";
const G2_BINARY: &str = "sllm-rmsnorm-g2-evidence";
const G2_SOURCE_PATH: &str = "crates/sllm-hip/src/bin/sllm-rmsnorm-g2-evidence.rs";
const G2_IDENTITY_MARKER: &str = "SLLM_G2_BUILD_IDENTITY_V1:";

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn canonical_json(value: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string(value).expect("canonical JSON serialization")
    )
}

fn source_bytes(repo: &Path, relative: &str) -> Vec<u8> {
    let path = Path::new(relative);
    assert!(
        path.is_relative(),
        "G2 build input must be relative: {relative}"
    );
    assert!(
        path.components()
            .all(|component| !matches!(component, std::path::Component::ParentDir)),
        "G2 build input may not escape the repository: {relative}"
    );
    let mut full = repo.to_path_buf();
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            full.push(name);
            let metadata = fs::symlink_metadata(&full)
                .unwrap_or_else(|error| panic!("cannot stat G2 build input {relative}: {error}"));
            assert!(
                !metadata.file_type().is_symlink(),
                "G2 build input contains a symlink: {relative}"
            );
        }
    }
    let metadata = fs::symlink_metadata(&full)
        .unwrap_or_else(|error| panic!("cannot stat G2 build input {relative}: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "G2 build input is not regular: {relative}"
    );
    fs::read(&full).unwrap_or_else(|error| panic!("cannot read G2 build input {relative}: {error}"))
}

fn generate_g2_build_identity(manifest_dir: &Path, out_dir: &Path) {
    let repo = manifest_dir
        .join("../..")
        .canonicalize()
        .expect("cannot canonicalize repository root for G2 build identity");
    let manifest_path = repo.join(G2_BUILD_INPUTS_PATH);
    let manifest_bytes = source_bytes(&repo, G2_BUILD_INPUTS_PATH);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .expect("G2 build-input manifest must be valid JSON");
    let object = manifest
        .as_object()
        .expect("G2 build-input manifest must be an object");
    let required_fields = [
        "schema_version",
        "identity_schema",
        "role",
        "binary_name",
        "source_path",
        "source_order_sha256",
        "source_paths",
    ];
    assert_eq!(
        object.len(),
        required_fields.len(),
        "G2 build-input manifest has unknown fields"
    );
    for field in required_fields {
        assert!(
            object.contains_key(field),
            "G2 build-input manifest is missing {field}"
        );
    }
    assert_eq!(
        object.get("schema_version").and_then(Value::as_str),
        Some("rmsnorm-g2-build-inputs-v1")
    );
    assert_eq!(
        object.get("identity_schema").and_then(Value::as_str),
        Some(G2_IDENTITY_SCHEMA)
    );
    assert_eq!(object.get("role").and_then(Value::as_str), Some(G2_ROLE));
    assert_eq!(
        object.get("binary_name").and_then(Value::as_str),
        Some(G2_BINARY)
    );
    assert_eq!(
        object.get("source_path").and_then(Value::as_str),
        Some(G2_SOURCE_PATH)
    );

    let entries = object
        .get("source_paths")
        .and_then(Value::as_array)
        .expect("G2 build-input manifest source_paths must be an array");
    assert!(
        !entries.is_empty(),
        "G2 build-input manifest source_paths must not be empty"
    );
    let mut paths = Vec::with_capacity(entries.len());
    for (order, entry) in entries.iter().enumerate() {
        let entry = entry
            .as_object()
            .expect("G2 source path entry must be an object");
        assert_eq!(entry.len(), 2, "G2 source path entry has unknown fields");
        assert_eq!(
            entry.get("order").and_then(Value::as_u64),
            Some(order as u64),
            "G2 source path order is not canonical"
        );
        let path = entry
            .get("path")
            .and_then(Value::as_str)
            .expect("G2 source path entry is missing path");
        assert!(
            !paths.iter().any(|item| item == path),
            "G2 source paths are duplicated: {path}"
        );
        paths.push(path.to_owned());
    }
    assert_eq!(paths.first().map(String::as_str), Some(G2_SOURCE_PATH));
    assert_eq!(
        paths.last().map(String::as_str),
        Some("native/hip/src/rmsnorm_kernel_internal.hpp")
    );
    let path_value = Value::Array(paths.iter().cloned().map(Value::String).collect());
    let order_sha = sha256(canonical_json(&path_value).as_bytes());
    assert_eq!(
        object.get("source_order_sha256").and_then(Value::as_str),
        Some(order_sha.as_str()),
        "G2 source order digest is stale"
    );

    let files: Vec<Value> = paths
        .iter()
        .map(|path| {
            let bytes = source_bytes(&repo, path);
            let mut file = serde_json::Map::new();
            file.insert("path".to_owned(), Value::String(path.clone()));
            file.insert("sha256".to_owned(), Value::String(sha256(&bytes)));
            Value::Object(file)
        })
        .collect();
    let file_value = Value::Array(files);
    let source_set_sha = sha256(canonical_json(&file_value).as_bytes());
    let source_sha = source_bytes(&repo, G2_SOURCE_PATH);
    let manifest_sha = sha256(&manifest_bytes);
    let identity_json = format!(
        "{{\"binary_name\":\"{G2_BINARY}\",\"identity_schema\":\"{G2_IDENTITY_SCHEMA}\",\"role\":\"{G2_ROLE}\",\"source_order_sha256\":\"{order_sha}\",\"source_path\":\"{G2_SOURCE_PATH}\",\"source_set_manifest_sha256\":\"{manifest_sha}\",\"source_set_sha256\":\"{source_set_sha}\",\"source_sha256\":\"{}\"}}\n",
        sha256(&source_sha),
    );
    let identity_payload = format!("{G2_IDENTITY_MARKER}{identity_json}");
    assert!(
        manifest_path.is_file(),
        "G2 build-input manifest disappeared"
    );
    let generated = format!(
        "pub const IDENTITY_PAYLOAD: &[u8] = b{identity_payload:?};\n\
         pub const IDENTITY_JSON: &str = {identity_json:?};\n\
         pub const IDENTITY_SCHEMA: &str = \"{G2_IDENTITY_SCHEMA}\";\n\
         pub const ROLE: &str = \"{G2_ROLE}\";\n\
         pub const BINARY_NAME: &str = \"{G2_BINARY}\";\n\
         pub const SOURCE_PATH: &str = \"{G2_SOURCE_PATH}\";\n\
         pub const SOURCE_SET_SHA256: &str = \"{source_set_sha}\";\n\
         pub const SOURCE_SHA256: &str = \"{}\";\n",
        sha256(&source_bytes(&repo, G2_SOURCE_PATH)),
    );
    fs::write(out_dir.join("rmsnorm_g2_build_identity.rs"), generated)
        .expect("cannot write generated G2 build identity");
}

#[cfg(target_os = "linux")]
const REQUIRED_CLIENT_SEALS: c_int = 0x0001 | 0x0002 | 0x0004 | 0x0008;

#[cfg(target_os = "linux")]
const F_GET_SEALS: c_int = 1034;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
}

#[cfg(target_os = "linux")]
fn assert_sealed_client_fd(fd: i32, purpose: &str) {
    // The broker passes this descriptor through Cargo/CMake; a regular file
    // and a digest string alone do not prove immutability.  Require the Linux
    // memfd seal set before any version probe or CMake invocation.
    let seals = unsafe { fcntl(fd, F_GET_SEALS) };
    assert_eq!(
        seals & REQUIRED_CLIENT_SEALS,
        REQUIRED_CLIENT_SEALS,
        "{purpose} requires the complete sealed compiler-client FD seal set"
    );
}

#[cfg(not(target_os = "linux"))]
fn assert_sealed_client_fd(_fd: i32, purpose: &str) {
    panic!("{purpose} requires Linux sealed compiler-client FD handoff");
}

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
    let hip_compile_probe = source_dir.join("src/hip_compile_probe.hip.cpp");
    let public_runtime_internal = source_dir.join("src/public_runtime_internal.hpp");
    let public_runtime_stub = source_dir.join("src/public_runtime_stub.cpp");
    let public_runtime = source_dir.join("src/public_runtime.hip.cpp");
    let rmsnorm_api_header = source_dir.join("src/rmsnorm_api.hpp");
    let rmsnorm_api = source_dir.join("src/rmsnorm_api.cpp");
    let rmsnorm_kernel_internal = source_dir.join("src/rmsnorm_kernel_internal.hpp");
    let rmsnorm_kernel = source_dir.join("src/rmsnorm_kernel.hip.cpp");
    let layout_probe = source_dir.join("src/abi_layout_probe.cpp");
    let header_c_compile = source_dir.join("src/header_c_compile.c");
    let header_cpp_compile = source_dir.join("src/header_cpp_compile.cpp");
    let bindings = manifest_dir.join("src/bindings.rs");
    let evidence_bindings = manifest_dir.join("src/evidence_bindings.rs");
    let cmake_file = source_dir.join("CMakeLists.txt");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    generate_g2_build_identity(&manifest_dir, &out_dir);
    let build_inputs = manifest_dir.join("../../ci/matrix/rmsnorm-g2-build-inputs-v1.json");
    println!("cargo:rerun-if-changed={}", build_inputs.display());
    let build_inputs_value: Value = serde_json::from_slice(
        &fs::read(&build_inputs).expect("cannot read G2 build-input manifest for rerun paths"),
    )
    .expect("G2 build-input manifest must remain valid JSON");
    for entry in build_inputs_value["source_paths"]
        .as_array()
        .expect("G2 build-input manifest source_paths must be an array")
    {
        let path = entry["path"]
            .as_str()
            .expect("G2 source path must be a string");
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join("../..").join(path).display()
        );
    }
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", umbrella_header.display());
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", evidence_header.display());
    println!("cargo:rerun-if-changed={}", evidence_stub.display());
    println!("cargo:rerun-if-changed={}", evidence_runtime.display());
    println!("cargo:rerun-if-changed={}", hip_compile_probe.display());
    println!(
        "cargo:rerun-if-changed={}",
        public_runtime_internal.display()
    );
    println!("cargo:rerun-if-changed={}", public_runtime_stub.display());
    println!("cargo:rerun-if-changed={}", public_runtime.display());
    println!("cargo:rerun-if-changed={}", rmsnorm_api_header.display());
    println!("cargo:rerun-if-changed={}", rmsnorm_api.display());
    println!(
        "cargo:rerun-if-changed={}",
        rmsnorm_kernel_internal.display()
    );
    println!("cargo:rerun-if-changed={}", rmsnorm_kernel.display());
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
    println!("cargo:rerun-if-env-changed=SLLM_ENABLE_PUBLIC_HIP_RUNTIME");
    println!("cargo:rerun-if-env-changed=SLLM_SEMANTIC_G1_AUTHORITY");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_SOCKET");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_SESSION");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_CLIENT");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_CLIENT_FD");
    println!("cargo:rerun-if-env-changed=SLLM_HIP_COMPILER_BROKER_TOKEN");
    println!("cargo:rerun-if-env-changed=SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR");
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
    let public_runtime_enabled = match env::var("SLLM_ENABLE_PUBLIC_HIP_RUNTIME") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => {
            panic!("SLLM_ENABLE_PUBLIC_HIP_RUNTIME must be unset, 0, or exactly 1; got {value}")
        }
        Err(env::VarError::NotPresent) => false,
        Err(error) => panic!("cannot read SLLM_ENABLE_PUBLIC_HIP_RUNTIME: {error}"),
    };
    let semantic_g1_authority = semantic_g1_authority_enabled();
    let hip_configuration = if hip_probe {
        Some(validate_hip_environment(
            &profile,
            "H3 compile probe",
            semantic_g1_authority,
        ))
    } else if public_runtime_enabled {
        Some(validate_hip_environment(
            &profile,
            "public HIP runtime",
            semantic_g1_authority,
        ))
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
        Some(hip_configuration.unwrap_or_else(|| {
            validate_hip_environment(&profile, "HIP runtime", semantic_g1_authority)
        }))
    } else {
        hip_configuration
    };
    let build_dir = if semantic_g1_authority {
        let value = env::var_os("SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR").unwrap_or_else(|| {
            panic!("semantic G1 requires SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR")
        });
        let path = PathBuf::from(value);
        assert!(
            path.is_absolute(),
            "semantic G1 native HIP build directory must be absolute"
        );
        assert!(
            path.is_dir(),
            "semantic G1 native HIP build directory must be parent-created"
        );
        path
    } else {
        match &hip_configuration {
            Some(configuration) => {
                out_dir.join(format!("native-hip-build-{}", configuration.target))
            }
            None => out_dir.join("native-hip-build-stub"),
        }
    };
    let mut configure = Command::new("/usr/bin/cmake");
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
                configuration.compiler.client_path().display()
            ))
            .arg(format!(
                "-DSLLM_HIP_COMPILER_LOGICAL={}",
                configuration.compiler.logical_path.display()
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
            ))
            .arg(if semantic_g1_authority {
                "-DSLLM_SEMANTIC_G1_AUTHORITY=ON"
            } else {
                "-DSLLM_SEMANTIC_G1_AUTHORITY=OFF"
            });
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
        configure.arg(if public_runtime_enabled {
            "-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=ON"
        } else {
            "-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=OFF"
        });
        configuration.compiler.apply_environment(&mut configure);
    } else {
        configure.arg("-DSLLM_ENABLE_HIP_COMPILE_PROBE=OFF");
        configure.arg("-DSLLM_ENABLE_HIP_RUNTIME=OFF");
        configure.arg("-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=OFF");
    }
    configure.arg("-DCMAKE_CXX_COMPILER=/usr/bin/c++");
    run(&mut configure, "CMake configure");

    let mut build = Command::new("/usr/bin/cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("sllm_hip_stub")
        .arg("--parallel")
        .arg("1");
    if let Some(configuration) = &hip_configuration {
        configuration.compiler.apply_environment(&mut build);
    }
    run(&mut build, "CMake build");
    if hip_probe {
        let mut probe_build = Command::new("/usr/bin/cmake");
        probe_build
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("sllm_hip_compile_probe_link");
        if let Some(configuration) = &hip_configuration {
            configuration.compiler.apply_environment(&mut probe_build);
        }
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
    if hip_runtime || public_runtime_enabled {
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
    compiler: PinnedCompiler,
    target: String,
    codegen_features: String,
}

struct PinnedCompiler {
    logical_path: PathBuf,
    client_path: Option<PathBuf>,
    client_sha256: Option<String>,
    client_fd: Option<i32>,
}

impl PinnedCompiler {
    fn client_path(&self) -> &Path {
        self.client_path.as_deref().unwrap_or(&self.logical_path)
    }

    fn apply_environment(&self, command: &mut Command) {
        if self.client_path.is_some() {
            // Cargo adds build-script-only variables such as LD_LIBRARY_PATH.
            // A semantic G1 compiler observation must instead inherit exactly
            // the controller-owned build environment and broker credentials.
            const SEMANTIC_G1_COMMAND_ENVIRONMENT: &[&str] = &[
                "PATH",
                "HOME",
                "LC_ALL",
                "LANG",
                "RUSTUP_HOME",
                "CARGO_HOME",
                "RUSTUP_TOOLCHAIN",
                "RUSTC",
                "CXX",
                "ROCM_PATH",
                "HIP_PATH",
                "SLLM_HIP_COMPILER",
                "CMAKE_HIP_ARCHITECTURES",
                "SLLM_HIP_CODEGEN_FEATURES",
                "SLLM_ENABLE_HIP_RUNTIME",
                "SLLM_ENABLE_PUBLIC_HIP_RUNTIME",
                "SLLM_ENABLE_HIP_COMPILE_PROBE",
                "SLLM_SEMANTIC_G1_AUTHORITY",
                "CARGO_TARGET_DIR",
                "SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR",
                "SLLM_HIP_COMPILER_BROKER_SOCKET",
                "SLLM_HIP_COMPILER_BROKER_TOKEN",
                "SLLM_HIP_COMPILER_BROKER_SESSION",
                "SLLM_HIP_COMPILER_BROKER_CLIENT",
                "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256",
                "SLLM_HIP_COMPILER_BROKER_CLIENT_FD",
            ];
            command.env_clear();
            for name in SEMANTIC_G1_COMMAND_ENVIRONMENT {
                let value = env::var_os(name).unwrap_or_else(|| {
                    panic!("semantic G1 closed command environment requires {name}")
                });
                command.env(name, value);
            }
            return;
        }
        command.env("SLLM_HIP_COMPILER_LOGICAL", self.logical_path.as_os_str());
        if let Some(client_path) = &self.client_path {
            command.env("SLLM_HIP_COMPILER_BROKER_CLIENT", client_path);
        }
        if let Some(client_sha256) = &self.client_sha256 {
            command.env("SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256", client_sha256);
        }
        if let Some(client_fd) = self.client_fd {
            command.env("SLLM_HIP_COMPILER_BROKER_CLIENT_FD", client_fd.to_string());
        }
    }

    fn version_probe(&self) -> Command {
        let executable = self.client_path.as_ref().unwrap_or(&self.logical_path);
        let mut command = Command::new(executable);
        self.apply_environment(&mut command);
        command.arg("--version");
        command
    }
}

fn semantic_g1_authority_enabled() -> bool {
    match env::var("SLLM_SEMANTIC_G1_AUTHORITY") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => {
            panic!("SLLM_SEMANTIC_G1_AUTHORITY must be unset, 0, or exactly 1; got {value}")
        }
        Err(env::VarError::NotPresent) => false,
        Err(error) => panic!("cannot read SLLM_SEMANTIC_G1_AUTHORITY: {error}"),
    }
}

fn validate_hip_environment(
    profile: &str,
    purpose: &str,
    semantic_g1_authority: bool,
) -> HipConfiguration {
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
    let (client_path, client_sha256, client_fd) = if semantic_g1_authority {
        let value = env::var_os("SLLM_HIP_COMPILER_BROKER_CLIENT")
            .unwrap_or_else(|| panic!("{purpose} requires SLLM_HIP_COMPILER_BROKER_CLIENT"));
        let path = PathBuf::from(value);
        let fd_text = env::var("SLLM_HIP_COMPILER_BROKER_CLIENT_FD")
            .unwrap_or_else(|_| panic!("{purpose} requires SLLM_HIP_COMPILER_BROKER_CLIENT_FD"));
        let fd = fd_text
            .parse::<i32>()
            .unwrap_or_else(|_| panic!("{purpose} compiler broker client FD is malformed"));
        assert!(
            fd >= 3,
            "semantic G1 compiler broker client FD must be >= 3"
        );
        let sealed_path = PathBuf::from(format!("/proc/self/fd/{fd}"));
        assert_eq!(
            path, sealed_path,
            "semantic G1 compiler broker client must be the exact sealed FD path"
        );
        assert!(
            fs::metadata(&sealed_path)
                .map(|metadata| metadata.is_file())
                .unwrap_or(false),
            "semantic G1 compiler broker client FD must name a regular sealed object"
        );
        assert_sealed_client_fd(fd, purpose);
        let digest = env::var("SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256").unwrap_or_else(|_| {
            panic!("{purpose} requires SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256")
        });
        assert!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "semantic G1 compiler broker client SHA-256 must be lowercase hexadecimal"
        );
        (Some(path), Some(digest), Some(fd))
    } else {
        (None, None, None)
    };
    let compiler = PinnedCompiler {
        logical_path: compiler,
        client_path,
        client_sha256,
        client_fd,
    };
    if !semantic_g1_authority {
        let mut compiler_probe = compiler.version_probe();
        let compiler_version = capture(&mut compiler_probe, "sealed ROCm amdclang++ version probe");
        let version_line = compiler_version.lines().next().unwrap_or_default();
        assert!(
            version_line.starts_with("AMD clang version 23."),
            "{purpose} requires LLVM major 23 from the brokered ROCm amdclang++; got {version_line}"
        );
    }

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
    let cxx = PathBuf::from("/usr/bin/c++");
    assert!(
        cxx.is_file(),
        "fixed /usr/bin/c++ is unavailable for ABI layout verification"
    );
    let cxx_probe = out_dir.join("sllm-abi-layout-cxx");
    let cxx_output = capture(
        Command::new(&cxx)
            .arg("-std=c++17")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg("-I")
            .arg(manifest_dir.join("../../include"))
            .arg("-I")
            .arg(manifest_dir.join("../../native/hip/src"))
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
             println!(\"const SLLM_STATUS_PUBLIC_PENDING={{}}\", bindings::SLLM_STATUS_PUBLIC_PENDING);\n\
             println!(\"const SLLM_STATUS_PUBLIC_TIMEOUT={{}}\", bindings::SLLM_STATUS_PUBLIC_TIMEOUT);\n\
             println!(\"const SLLM_STATUS_PUBLIC_INVALID_HANDLE={{}}\", bindings::SLLM_STATUS_PUBLIC_INVALID_HANDLE);\n\
             println!(\"const SLLM_STATUS_PUBLIC_DEVICE_MISMATCH={{}}\", bindings::SLLM_STATUS_PUBLIC_DEVICE_MISMATCH);\n\
             println!(\"const SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR={{}}\", bindings::SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR);\n\
             println!(\"const SLLM_STATUS_PUBLIC_BUSY={{}}\", bindings::SLLM_STATUS_PUBLIC_BUSY);\n\
             println!(\"const SLLM_STATUS_PUBLIC_NOT_READY={{}}\", bindings::SLLM_STATUS_PUBLIC_NOT_READY);\n\
             println!(\"const SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR={{}}\", bindings::SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR);\n\
             println!(\"const SLLM_STATUS_INVALID_TENSOR_BINDING={{}}\", bindings::SLLM_STATUS_INVALID_TENSOR_BINDING);\n\
             println!(\"const SLLM_STATUS_ZERO_EXTENT={{}}\", bindings::SLLM_STATUS_ZERO_EXTENT);\n\
             println!(\"const SLLM_STATUS_SHAPE_MISMATCH={{}}\", bindings::SLLM_STATUS_SHAPE_MISMATCH);\n\
             println!(\"const SLLM_STATUS_STRIDE_MISMATCH={{}}\", bindings::SLLM_STATUS_STRIDE_MISMATCH);\n\
             println!(\"const SLLM_STATUS_METADATA_OVERFLOW={{}}\", bindings::SLLM_STATUS_METADATA_OVERFLOW);\n\
             println!(\"const SLLM_STATUS_BUFFER_OUT_OF_BOUNDS={{}}\", bindings::SLLM_STATUS_BUFFER_OUT_OF_BOUNDS);\n\
             println!(\"const SLLM_STATUS_MISALIGNED_OFFSET={{}}\", bindings::SLLM_STATUS_MISALIGNED_OFFSET);\n\
             println!(\"const SLLM_STATUS_UNSUPPORTED_DTYPE={{}}\", bindings::SLLM_STATUS_UNSUPPORTED_DTYPE);\n\
             println!(\"const SLLM_STATUS_UNSUPPORTED_ENCODING={{}}\", bindings::SLLM_STATUS_UNSUPPORTED_ENCODING);\n\
             println!(\"const SLLM_STATUS_INVALID_EPSILON={{}}\", bindings::SLLM_STATUS_INVALID_EPSILON);\n\
             println!(\"const SLLM_STATUS_UNSUPPORTED_SCALE_MODE={{}}\", bindings::SLLM_STATUS_UNSUPPORTED_SCALE_MODE);\n\
             println!(\"const SLLM_STATUS_ALIAS_OVERLAP={{}}\", bindings::SLLM_STATUS_ALIAS_OVERLAP);\n\
             println!(\"const SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH={{}}\", bindings::SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH);\n\
             println!(\"const SLLM_BACKEND_HIP={{}}\", bindings::SLLM_BACKEND_HIP);\n\
             println!(\"const SLLM_ACCESS_READ={{}}\", bindings::SLLM_ACCESS_READ);\n\
             println!(\"const SLLM_ACCESS_WRITE={{}}\", bindings::SLLM_ACCESS_WRITE);\n\
             println!(\"const SLLM_ACCESS_READ_WRITE={{}}\", bindings::SLLM_ACCESS_READ_WRITE);\n\
             println!(\"const SLLM_HIP_MAX_DEVICE_NAME={{}}\", bindings::SLLM_HIP_MAX_DEVICE_NAME);\n\
             println!(\"const SLLM_HIP_MAX_GCN_ARCH_NAME={{}}\", bindings::SLLM_HIP_MAX_GCN_ARCH_NAME);\n\
             println!(\"const SLLM_HIP_MAX_TRANSFER_BYTES={{}}\", bindings::SLLM_HIP_MAX_TRANSFER_BYTES);\n\
             println!(\"const SLLM_HIP_RMSNORM_VERSION={{}}\", bindings::SLLM_HIP_RMSNORM_VERSION);\n\
             println!(\"const SLLM_HIP_TENSOR_MAX_RANK={{}}\", bindings::SLLM_HIP_TENSOR_MAX_RANK);\n\
             println!(\"const SLLM_TENSOR_DTYPE_BF16={{}}\", bindings::SLLM_TENSOR_DTYPE_BF16);\n\
             println!(\"const SLLM_TENSOR_DTYPE_F32={{}}\", bindings::SLLM_TENSOR_DTYPE_F32);\n\
             println!(\"const SLLM_TENSOR_ENCODING_UNQUANTIZED={{}}\", bindings::SLLM_TENSOR_ENCODING_UNQUANTIZED);\n\
             println!(\"const SLLM_RMSNORM_ACCUMULATION_F32={{}}\", bindings::SLLM_RMSNORM_ACCUMULATION_F32);\n\
             println!(\"const SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE={{}}\", bindings::SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE);\n\
             println!(\"const SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP={{}}\", bindings::SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP);\n\
             println!(\"const SLLM_COMPLETION_STATE_PENDING={{}}\", bindings::SLLM_COMPLETION_STATE_PENDING);\n\
             println!(\"const SLLM_COMPLETION_STATE_SUCCESS={{}}\", bindings::SLLM_COMPLETION_STATE_SUCCESS);\n\
             println!(\"const SLLM_COMPLETION_STATE_FAILURE={{}}\", bindings::SLLM_COMPLETION_STATE_FAILURE);\n\
             println!(\"layout sllm_error_sink_t size={{}} align={{}} struct_size={{}} abi_version={{}} message={{}} message_capacity={{}} message_length={{}} reserved={{}}\", size_of::<bindings::sllm_error_sink_t>(), align_of::<bindings::sllm_error_sink_t>(), offset_of!(bindings::sllm_error_sink_t, struct_size), offset_of!(bindings::sllm_error_sink_t, abi_version), offset_of!(bindings::sllm_error_sink_t, message), offset_of!(bindings::sllm_error_sink_t, message_capacity), offset_of!(bindings::sllm_error_sink_t, message_length), offset_of!(bindings::sllm_error_sink_t, reserved));\n\
             println!(\"layout sllm_version_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} major={{}} minor={{}} patch={{}} reserved={{}}\", size_of::<bindings::sllm_version_info_t>(), align_of::<bindings::sllm_version_info_t>(), offset_of!(bindings::sllm_version_info_t, struct_size), offset_of!(bindings::sllm_version_info_t, abi_version), offset_of!(bindings::sllm_version_info_t, major), offset_of!(bindings::sllm_version_info_t, minor), offset_of!(bindings::sllm_version_info_t, patch), offset_of!(bindings::sllm_version_info_t, reserved));\n\
             println!(\"layout sllm_backend_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} backend={{}} available={{}} hip_runtime_present={{}} reserved={{}}\", size_of::<bindings::sllm_backend_probe_result_t>(), align_of::<bindings::sllm_backend_probe_result_t>(), offset_of!(bindings::sllm_backend_probe_result_t, struct_size), offset_of!(bindings::sllm_backend_probe_result_t, abi_version), offset_of!(bindings::sllm_backend_probe_result_t, backend), offset_of!(bindings::sllm_backend_probe_result_t, available), offset_of!(bindings::sllm_backend_probe_result_t, hip_runtime_present), offset_of!(bindings::sllm_backend_probe_result_t, reserved));\n\
             println!(\"layout sllm_context_probe_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} context_present={{}} hip_available={{}} reserved={{}}\", size_of::<bindings::sllm_context_probe_result_t>(), align_of::<bindings::sllm_context_probe_result_t>(), offset_of!(bindings::sllm_context_probe_result_t, struct_size), offset_of!(bindings::sllm_context_probe_result_t, abi_version), offset_of!(bindings::sllm_context_probe_result_t, context_present), offset_of!(bindings::sllm_context_probe_result_t, hip_available), offset_of!(bindings::sllm_context_probe_result_t, reserved));\n\
             println!(\"layout sllm_device_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} device_index={{}} visible_device_count={{}} total_memory_bytes={{}} wavefront_size={{}} reserved0={{}} name={{}} gcn_arch_name={{}} reserved={{}}\", size_of::<bindings::sllm_device_info_t>(), align_of::<bindings::sllm_device_info_t>(), offset_of!(bindings::sllm_device_info_t, struct_size), offset_of!(bindings::sllm_device_info_t, abi_version), offset_of!(bindings::sllm_device_info_t, device_index), offset_of!(bindings::sllm_device_info_t, visible_device_count), offset_of!(bindings::sllm_device_info_t, total_memory_bytes), offset_of!(bindings::sllm_device_info_t, wavefront_size), offset_of!(bindings::sllm_device_info_t, reserved0), offset_of!(bindings::sllm_device_info_t, name), offset_of!(bindings::sllm_device_info_t, gcn_arch_name), offset_of!(bindings::sllm_device_info_t, reserved));\n\
             println!(\"layout sllm_context_create_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} device_index={{}} flags={{}} expected_gcn_arch_name={{}} reserved={{}}\", size_of::<bindings::sllm_context_create_info_t>(), align_of::<bindings::sllm_context_create_info_t>(), offset_of!(bindings::sllm_context_create_info_t, struct_size), offset_of!(bindings::sllm_context_create_info_t, abi_version), offset_of!(bindings::sllm_context_create_info_t, device_index), offset_of!(bindings::sllm_context_create_info_t, flags), offset_of!(bindings::sllm_context_create_info_t, expected_gcn_arch_name), offset_of!(bindings::sllm_context_create_info_t, reserved));\n\
             println!(\"layout sllm_queue_create_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} flags={{}} reserved={{}}\", size_of::<bindings::sllm_queue_create_info_t>(), align_of::<bindings::sllm_queue_create_info_t>(), offset_of!(bindings::sllm_queue_create_info_t, struct_size), offset_of!(bindings::sllm_queue_create_info_t, abi_version), offset_of!(bindings::sllm_queue_create_info_t, flags), offset_of!(bindings::sllm_queue_create_info_t, reserved));\n\
             println!(\"layout sllm_buffer_create_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} size_bytes={{}} alignment_bytes={{}} flags={{}} reserved={{}}\", size_of::<bindings::sllm_buffer_create_info_t>(), align_of::<bindings::sllm_buffer_create_info_t>(), offset_of!(bindings::sllm_buffer_create_info_t, struct_size), offset_of!(bindings::sllm_buffer_create_info_t, abi_version), offset_of!(bindings::sllm_buffer_create_info_t, size_bytes), offset_of!(bindings::sllm_buffer_create_info_t, alignment_bytes), offset_of!(bindings::sllm_buffer_create_info_t, flags), offset_of!(bindings::sllm_buffer_create_info_t, reserved));\n\
             println!(\"layout sllm_transfer_desc_t size={{}} align={{}} struct_size={{}} abi_version={{}} host_pointer={{}} buffer_offset_bytes={{}} size_bytes={{}} reserved={{}}\", size_of::<bindings::sllm_transfer_desc_t>(), align_of::<bindings::sllm_transfer_desc_t>(), offset_of!(bindings::sllm_transfer_desc_t, struct_size), offset_of!(bindings::sllm_transfer_desc_t, abi_version), offset_of!(bindings::sllm_transfer_desc_t, host_pointer), offset_of!(bindings::sllm_transfer_desc_t, buffer_offset_bytes), offset_of!(bindings::sllm_transfer_desc_t, size_bytes), offset_of!(bindings::sllm_transfer_desc_t, reserved));\n\
             println!(\"layout sllm_completion_result_t size={{}} align={{}} struct_size={{}} abi_version={{}} state={{}} reserved0={{}} transfer_size_bytes={{}} available_bytes={{}} reserved={{}}\", size_of::<bindings::sllm_completion_result_t>(), align_of::<bindings::sllm_completion_result_t>(), offset_of!(bindings::sllm_completion_result_t, struct_size), offset_of!(bindings::sllm_completion_result_t, abi_version), offset_of!(bindings::sllm_completion_result_t, state), offset_of!(bindings::sllm_completion_result_t, reserved0), offset_of!(bindings::sllm_completion_result_t, transfer_size_bytes), offset_of!(bindings::sllm_completion_result_t, available_bytes), offset_of!(bindings::sllm_completion_result_t, reserved));\n\
             println!(\"layout sllm_tensor_binding_t size={{}} align={{}} struct_size={{}} abi_version={{}} buffer={{}} byte_offset={{}} dtype={{}} encoding={{}} rank={{}} reserved0={{}} shape={{}} stride_elements={{}} reserved={{}}\", size_of::<bindings::sllm_tensor_binding_t>(), align_of::<bindings::sllm_tensor_binding_t>(), offset_of!(bindings::sllm_tensor_binding_t, struct_size), offset_of!(bindings::sllm_tensor_binding_t, abi_version), offset_of!(bindings::sllm_tensor_binding_t, buffer), offset_of!(bindings::sllm_tensor_binding_t, byte_offset), offset_of!(bindings::sllm_tensor_binding_t, dtype), offset_of!(bindings::sllm_tensor_binding_t, encoding), offset_of!(bindings::sllm_tensor_binding_t, rank), offset_of!(bindings::sllm_tensor_binding_t, reserved0), offset_of!(bindings::sllm_tensor_binding_t, shape), offset_of!(bindings::sllm_tensor_binding_t, stride_elements), offset_of!(bindings::sllm_tensor_binding_t, reserved));\n\
             println!(\"layout sllm_rmsnorm_desc_t size={{}} align={{}} struct_size={{}} abi_version={{}} op_version={{}} accumulation_dtype={{}} scale_mode={{}} alias_policy={{}} epsilon_bits={{}} reserved={{}} activation={{}} raw_scale={{}} output={{}}\", size_of::<bindings::sllm_rmsnorm_desc_t>(), align_of::<bindings::sllm_rmsnorm_desc_t>(), offset_of!(bindings::sllm_rmsnorm_desc_t, struct_size), offset_of!(bindings::sllm_rmsnorm_desc_t, abi_version), offset_of!(bindings::sllm_rmsnorm_desc_t, op_version), offset_of!(bindings::sllm_rmsnorm_desc_t, accumulation_dtype), offset_of!(bindings::sllm_rmsnorm_desc_t, scale_mode), offset_of!(bindings::sllm_rmsnorm_desc_t, alias_policy), offset_of!(bindings::sllm_rmsnorm_desc_t, epsilon_bits), offset_of!(bindings::sllm_rmsnorm_desc_t, reserved), offset_of!(bindings::sllm_rmsnorm_desc_t, activation), offset_of!(bindings::sllm_rmsnorm_desc_t, raw_scale), offset_of!(bindings::sllm_rmsnorm_desc_t, output));\n\
             println!(\"layout sllm_rmsnorm_dispatch_info_t size={{}} align={{}} struct_size={{}} abi_version={{}} info_version={{}} backend={{}} dispatch_id={{}} dispatch_count={{}} kernel_id={{}} workgroup_size_x={{}} grid_size_x={{}} row_count={{}} normalized_size={{}} fallback_allowed={{}} fallback_used={{}} kernel_symbol={{}} device_symbol={{}} gcn_arch_name={{}} reserved={{}}\", size_of::<bindings::sllm_rmsnorm_dispatch_info_t>(), align_of::<bindings::sllm_rmsnorm_dispatch_info_t>(), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, struct_size), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, abi_version), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, info_version), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, backend), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, dispatch_id), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, dispatch_count), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, kernel_id), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, workgroup_size_x), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, grid_size_x), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, row_count), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, normalized_size), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, fallback_allowed), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, fallback_used), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, kernel_symbol), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, device_symbol), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, gcn_arch_name), offset_of!(bindings::sllm_rmsnorm_dispatch_info_t, reserved));\n\
         }}\n",
        bindings_path.display()
    );
    fs::write(&rust_probe_source, rust_source)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", rust_probe_source.display()));
    let rustc = required_absolute_path("RUSTC");
    assert!(
        rustc.is_file(),
        "RUSTC must name the controller-fixed absolute Rust compiler"
    );
    run(
        Command::new(&rustc)
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
