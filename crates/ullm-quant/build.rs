fn main() {
    println!("cargo:rerun-if-changed=src/cpu/aq_kernels.cpp");
    println!("cargo:rerun-if-changed=src/cpu/aq_kernels.h");

    cc::Build::new()
        .cpp(true)
        .file("src/cpu/aq_kernels.cpp")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("ullm_aq_kernels");
}
