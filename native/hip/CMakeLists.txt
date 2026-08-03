cmake_minimum_required(VERSION 3.21)

option(ULLM_ENABLE_HIP_COMPILE_PROBE
    "Build the explicit HIP compile/link-only probe"
    OFF
)

if(ULLM_ENABLE_HIP_COMPILE_PROBE)
    if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
        message(FATAL_ERROR "ULLM_ENABLE_HIP_COMPILE_PROBE requires an explicit ROCM_PATH")
    endif()
    if(NOT IS_ABSOLUTE "${ROCM_PATH}")
        message(FATAL_ERROR "ROCM_PATH must be an absolute path")
    endif()
    get_filename_component(ULLM_ROCM_LOGICAL "${ROCM_PATH}" ABSOLUTE)
    if(NOT ULLM_ROCM_LOGICAL STREQUAL "/opt/rocm")
        message(FATAL_ERROR "H3 requires the logical ROCm root /opt/rocm")
    endif()
    get_filename_component(ULLM_ROCM_REAL "${ROCM_PATH}" REALPATH)
    if(ULLM_ROCM_REAL STREQUAL "" OR NOT IS_DIRECTORY "${ULLM_ROCM_REAL}")
        message(FATAL_ERROR "H3 ROCm root does not resolve to a directory")
    endif()

    function(ullm_require_rocm_path path label)
        if(NOT EXISTS "${path}")
            message(FATAL_ERROR "required ROCm path is missing: ${label}=${path}")
        endif()
        get_filename_component(ULLM_REQUIRED_REAL "${path}" REALPATH)
        string(FIND "${ULLM_REQUIRED_REAL}" "${ULLM_ROCM_REAL}/" ULLM_REQUIRED_ROOT_OFFSET)
        if(NOT ULLM_REQUIRED_REAL STREQUAL ULLM_ROCM_REAL AND
           NOT ULLM_REQUIRED_ROOT_OFFSET EQUAL 0)
            message(FATAL_ERROR "required ROCm path resolves outside the ROCm root: ${label}=${ULLM_REQUIRED_REAL}")
        endif()
    endfunction()

    if(NOT DEFINED CMAKE_HIP_COMPILER OR CMAKE_HIP_COMPILER STREQUAL "")
        message(FATAL_ERROR "H3 requires an explicit CMAKE_HIP_COMPILER")
    endif()
    if(NOT IS_ABSOLUTE "${CMAKE_HIP_COMPILER}")
        message(FATAL_ERROR "CMAKE_HIP_COMPILER must be an absolute path")
    endif()
    get_filename_component(ULLM_HIP_COMPILER_REAL "${CMAKE_HIP_COMPILER}" REALPATH)
    string(FIND "${ULLM_HIP_COMPILER_REAL}" "${ULLM_ROCM_REAL}/" ULLM_HIP_COMPILER_ROOT_OFFSET)
    if(ULLM_HIP_COMPILER_REAL STREQUAL "" OR
       (NOT ULLM_HIP_COMPILER_REAL STREQUAL ULLM_ROCM_REAL AND
        NOT ULLM_HIP_COMPILER_ROOT_OFFSET EQUAL 0))
        message(FATAL_ERROR "HIP compiler must resolve inside ROCM_PATH")
    endif()
    get_filename_component(ULLM_HIP_COMPILER_NAME "${CMAKE_HIP_COMPILER}" NAME)
    if(NOT ULLM_HIP_COMPILER_NAME STREQUAL "amdclang++")
        message(FATAL_ERROR "H3 requires the ROCm amdclang++ entry point")
    endif()
    if(NOT CMAKE_HIP_COMPILER STREQUAL "${ROCM_PATH}/bin/amdclang++")
        message(FATAL_ERROR "HIP compiler must use the logical ROCM_PATH/bin/amdclang++ entry point")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR
       CMAKE_HIP_ARCHITECTURES STREQUAL "" OR
       CMAKE_HIP_ARCHITECTURES MATCHES ";" OR
       NOT CMAKE_HIP_ARCHITECTURES MATCHES "^(gfx1030|gfx1201)$")
        message(FATAL_ERROR "H3 requires exactly one exact gfx1030 or gfx1201 target")
    endif()
    if(NOT DEFINED ULLM_HIP_CODEGEN_FEATURES OR
       NOT ULLM_HIP_CODEGEN_FEATURES STREQUAL
          "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0")
        message(FATAL_ERROR "H3 codegen features are not the pinned compile-only tuple")
    endif()
    foreach(ULLM_REQUIRED_PATH_AND_LABEL
        "${ROCM_PATH}/include/hip/hip_runtime.h;hip_headers"
        "${ROCM_PATH}/lib/cmake/hip;hip_cmake_package"
        "${ROCM_PATH}/amdgcn/bitcode;device_libraries"
        "${ROCM_PATH}/lib/libamdhip64.so;hip_runtime"
        "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler;clang_offload_bundler"
        "${ROCM_PATH}/lib/llvm/bin/llvm-objcopy;llvm_objcopy"
        "${ROCM_PATH}/lib/llvm/bin/llvm-readobj;llvm_readobj"
        "${ROCM_PATH}/lib/llvm/bin/llvm-objdump;llvm_objdump"
    )
        list(GET ULLM_REQUIRED_PATH_AND_LABEL 0 ULLM_REQUIRED_PATH)
        list(GET ULLM_REQUIRED_PATH_AND_LABEL 1 ULLM_REQUIRED_LABEL)
        ullm_require_rocm_path("${ULLM_REQUIRED_PATH}" "${ULLM_REQUIRED_LABEL}")
    endforeach()
    project(ullm_hip_stub VERSION 0.1.0 LANGUAGES C CXX HIP)
else()
    project(ullm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)
endif()

add_library(ullm_hip_stub STATIC
    src/hip_stub.cpp
    src/header_c_compile.c
    src/header_cpp_compile.cpp
)

target_compile_features(ullm_hip_stub PRIVATE c_std_11 cxx_std_17)
set_target_properties(ullm_hip_stub PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ullm_hip_stub
    PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
)

if(MSVC)
    target_compile_options(ullm_hip_stub PRIVATE /W4 /WX)
else()
    target_compile_options(ullm_hip_stub PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Werror
    )
endif()

if(ULLM_ENABLE_HIP_COMPILE_PROBE)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(ULLM_HIP_PROBE_SOURCE src/hip_compile_probe.hip.cpp)
    set_source_files_properties(${ULLM_HIP_PROBE_SOURCE} PROPERTIES LANGUAGE HIP)
    add_library(ullm_hip_compile_probe_object OBJECT ${ULLM_HIP_PROBE_SOURCE})
    target_compile_features(ullm_hip_compile_probe_object PRIVATE cxx_std_17)
    target_compile_options(ullm_hip_compile_probe_object PRIVATE
        -mcode-object-version=6
        -mno-wavefrontsize64
    )
    add_executable(ullm_hip_compile_probe_link
        $<TARGET_OBJECTS:ullm_hip_compile_probe_object>
    )
    set_target_properties(ullm_hip_compile_probe_link PROPERTIES
        CXX_EXTENSIONS OFF
        LINKER_LANGUAGE HIP
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_NAME "hip-compile-probe-${CMAKE_HIP_ARCHITECTURES}"
        SUFFIX ".elf"
    )
    target_link_options(ullm_hip_compile_probe_link PRIVATE
        -mcode-object-version=6
        -mno-wavefrontsize64
    )
    target_include_directories(ullm_hip_compile_probe_link PRIVATE
        "${ROCM_PATH}/include"
    )
    target_include_directories(ullm_hip_compile_probe_object PRIVATE
        "${ROCM_PATH}/include"
    )

    set(ULLM_HIP_PROBE_OBJECT_OUTPUT
        "${CMAKE_BINARY_DIR}/hip-compile-probe-${CMAKE_HIP_ARCHITECTURES}.o"
    )
    add_custom_command(
        OUTPUT "${ULLM_HIP_PROBE_OBJECT_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_OBJECTS:ullm_hip_compile_probe_object>"
            "${ULLM_HIP_PROBE_OBJECT_OUTPUT}"
        DEPENDS ullm_hip_compile_probe_object
        VERBATIM
    )
    add_custom_target(ullm_hip_compile_probe_bundle_object
        DEPENDS "${ULLM_HIP_PROBE_OBJECT_OUTPUT}"
    )
    add_dependencies(ullm_hip_compile_probe_link ullm_hip_compile_probe_bundle_object)
endif()
