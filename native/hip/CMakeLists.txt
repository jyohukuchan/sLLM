cmake_minimum_required(VERSION 3.21)

option(SLLM_ENABLE_HIP_COMPILE_PROBE
    "Build the explicit HIP compile/link-only probe"
    OFF
)

option(SLLM_ENABLE_HIP_RUNTIME
    "Build the explicit model-free HIP evidence runtime"
    OFF
)

if(SLLM_ENABLE_HIP_COMPILE_PROBE OR SLLM_ENABLE_HIP_RUNTIME)
    if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
        message(FATAL_ERROR "HIP build requires an explicit ROCM_PATH")
    endif()
    if(NOT IS_ABSOLUTE "${ROCM_PATH}")
        message(FATAL_ERROR "ROCM_PATH must be an absolute path")
    endif()
    get_filename_component(SLLM_ROCM_LOGICAL "${ROCM_PATH}" ABSOLUTE)
    if(NOT SLLM_ROCM_LOGICAL STREQUAL "/opt/rocm")
        message(FATAL_ERROR "HIP build requires the logical ROCm root /opt/rocm")
    endif()
    get_filename_component(SLLM_ROCM_REAL "${ROCM_PATH}" REALPATH)
    if(SLLM_ROCM_REAL STREQUAL "" OR NOT IS_DIRECTORY "${SLLM_ROCM_REAL}")
        message(FATAL_ERROR "H3 ROCm root does not resolve to a directory")
    endif()
    set(SLLM_ROCM_RELEASE_MARKERS "${SLLM_ROCM_REAL}/.info/version")
    file(GLOB SLLM_ROCM_CHILD_RELEASE_MARKERS
        "${SLLM_ROCM_REAL}/*/.info/version"
    )
    list(APPEND SLLM_ROCM_RELEASE_MARKERS ${SLLM_ROCM_CHILD_RELEASE_MARKERS})
    list(REMOVE_DUPLICATES SLLM_ROCM_RELEASE_MARKERS)
    set(SLLM_FOUND_ROCM_RELEASE_MARKER OFF)
    foreach(SLLM_ROCM_RELEASE_MARKER IN LISTS SLLM_ROCM_RELEASE_MARKERS)
        if(EXISTS "${SLLM_ROCM_RELEASE_MARKER}")
            set(SLLM_FOUND_ROCM_RELEASE_MARKER ON)
            file(READ "${SLLM_ROCM_RELEASE_MARKER}" SLLM_ROCM_RELEASE)
            string(STRIP "${SLLM_ROCM_RELEASE}" SLLM_ROCM_RELEASE)
            if(NOT SLLM_ROCM_RELEASE STREQUAL "7.14.0")
                message(FATAL_ERROR
                    "HIP build requires every discovered ROCm release marker to be 7.14.0: ${SLLM_ROCM_RELEASE_MARKER}=${SLLM_ROCM_RELEASE}")
            endif()
        endif()
    endforeach()
    if(NOT SLLM_FOUND_ROCM_RELEASE_MARKER)
        message(FATAL_ERROR "ROCm release marker is missing under ${SLLM_ROCM_REAL}")
    endif()

    function(sllm_require_rocm_path path label)
        if(NOT EXISTS "${path}")
            message(FATAL_ERROR "required ROCm path is missing: ${label}=${path}")
        endif()
        get_filename_component(SLLM_REQUIRED_REAL "${path}" REALPATH)
        string(FIND "${SLLM_REQUIRED_REAL}" "${SLLM_ROCM_REAL}/" SLLM_REQUIRED_ROOT_OFFSET)
        if(NOT SLLM_REQUIRED_REAL STREQUAL SLLM_ROCM_REAL AND
           NOT SLLM_REQUIRED_ROOT_OFFSET EQUAL 0)
            message(FATAL_ERROR "required ROCm path resolves outside the ROCm root: ${label}=${SLLM_REQUIRED_REAL}")
        endif()
    endfunction()

    if(NOT DEFINED CMAKE_HIP_COMPILER OR CMAKE_HIP_COMPILER STREQUAL "")
        message(FATAL_ERROR "HIP build requires an explicit CMAKE_HIP_COMPILER")
    endif()
    if(NOT IS_ABSOLUTE "${CMAKE_HIP_COMPILER}")
        message(FATAL_ERROR "CMAKE_HIP_COMPILER must be an absolute path")
    endif()
    get_filename_component(SLLM_HIP_COMPILER_REAL "${CMAKE_HIP_COMPILER}" REALPATH)
    string(FIND "${SLLM_HIP_COMPILER_REAL}" "${SLLM_ROCM_REAL}/" SLLM_HIP_COMPILER_ROOT_OFFSET)
    if(SLLM_HIP_COMPILER_REAL STREQUAL "" OR
       (NOT SLLM_HIP_COMPILER_REAL STREQUAL SLLM_ROCM_REAL AND
        NOT SLLM_HIP_COMPILER_ROOT_OFFSET EQUAL 0))
        message(FATAL_ERROR "HIP compiler must resolve inside ROCM_PATH")
    endif()
    get_filename_component(SLLM_HIP_COMPILER_NAME "${CMAKE_HIP_COMPILER}" NAME)
    if(NOT SLLM_HIP_COMPILER_NAME STREQUAL "amdclang++")
        message(FATAL_ERROR "HIP build requires the ROCm amdclang++ entry point")
    endif()
    if(NOT CMAKE_HIP_COMPILER STREQUAL "${ROCM_PATH}/bin/amdclang++")
        message(FATAL_ERROR "HIP compiler must use the logical ROCM_PATH/bin/amdclang++ entry point")
    endif()
    execute_process(
        COMMAND "${CMAKE_HIP_COMPILER}" --version
        RESULT_VARIABLE SLLM_HIP_COMPILER_STATUS
        OUTPUT_VARIABLE SLLM_HIP_COMPILER_VERSION
        ERROR_VARIABLE SLLM_HIP_COMPILER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT SLLM_HIP_COMPILER_STATUS EQUAL 0 OR
       NOT SLLM_HIP_COMPILER_VERSION MATCHES "(^|\\n)AMD clang version 23\\.")
        message(FATAL_ERROR
            "HIP build requires LLVM major 23 from ROCm amdclang++; compiler output=${SLLM_HIP_COMPILER_VERSION}; error=${SLLM_HIP_COMPILER_ERROR}")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR
       CMAKE_HIP_ARCHITECTURES STREQUAL "" OR
       CMAKE_HIP_ARCHITECTURES MATCHES ";" OR
       NOT CMAKE_HIP_ARCHITECTURES MATCHES "^(gfx1030|gfx1201)$")
        message(FATAL_ERROR "HIP build requires exactly one exact gfx1030 or gfx1201 target")
    endif()
    if(NOT DEFINED SLLM_HIP_COMPILE_TARGET OR
       NOT SLLM_HIP_COMPILE_TARGET STREQUAL CMAKE_HIP_ARCHITECTURES)
        message(FATAL_ERROR
            "HIP build requires SLLM_HIP_COMPILE_TARGET to equal the exact CMAKE_HIP_ARCHITECTURES target")
    endif()
    if(NOT DEFINED SLLM_HIP_CODEGEN_FEATURES OR
       NOT SLLM_HIP_CODEGEN_FEATURES STREQUAL
          "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0")
        message(FATAL_ERROR "HIP codegen features are not the pinned tuple")
    endif()
    sllm_require_rocm_path("${ROCM_PATH}/include/hip/hip_runtime.h" "hip_headers")
    sllm_require_rocm_path("${ROCM_PATH}/lib/cmake/hip" "hip_cmake_package")
    sllm_require_rocm_path("${ROCM_PATH}/amdgcn/bitcode" "device_libraries")
    sllm_require_rocm_path("${ROCM_PATH}/lib/libamdhip64.so" "hip_runtime")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" "clang_offload_bundler")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objcopy" "llvm_objcopy")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-readobj" "llvm_readobj")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objdump" "llvm_objdump")
    project(sllm_hip_stub VERSION 0.1.0 LANGUAGES C CXX HIP)
else()
    project(sllm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)
endif()

if(SLLM_ENABLE_HIP_COMPILE_PROBE OR SLLM_ENABLE_HIP_RUNTIME)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

set(SLLM_HIP_NATIVE_SOURCES
    src/hip_stub.cpp
    src/hip_evidence_stub.cpp
    src/header_c_compile.c
    src/header_cpp_compile.cpp
)

if(SLLM_ENABLE_HIP_RUNTIME)
    set_source_files_properties(src/hip_evidence_runtime.hip.cpp PROPERTIES LANGUAGE HIP)
    list(REMOVE_ITEM SLLM_HIP_NATIVE_SOURCES src/hip_evidence_stub.cpp)
    list(APPEND SLLM_HIP_NATIVE_SOURCES src/hip_evidence_runtime.hip.cpp)
endif()

add_library(sllm_hip_stub STATIC ${SLLM_HIP_NATIVE_SOURCES})

target_compile_features(sllm_hip_stub PRIVATE c_std_11 cxx_std_17)
set_target_properties(sllm_hip_stub PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(sllm_hip_stub
    PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
)

if(MSVC)
    target_compile_options(sllm_hip_stub PRIVATE /W4 /WX)
else()
    target_compile_options(sllm_hip_stub PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Werror
    )
endif()

if(SLLM_ENABLE_HIP_RUNTIME)
    target_compile_definitions(sllm_hip_stub PRIVATE
        SLLM_HIP_COMPILE_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_hip_stub PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>
    )
endif()

if(SLLM_ENABLE_HIP_COMPILE_PROBE)
    set(SLLM_HIP_PROBE_SOURCE src/hip_compile_probe.hip.cpp)
    set_source_files_properties(${SLLM_HIP_PROBE_SOURCE} PROPERTIES LANGUAGE HIP)
    add_library(sllm_hip_compile_probe_object OBJECT ${SLLM_HIP_PROBE_SOURCE})
    target_compile_features(sllm_hip_compile_probe_object PRIVATE cxx_std_17)
    target_compile_options(sllm_hip_compile_probe_object PRIVATE
        -mcode-object-version=6
        -mno-wavefrontsize64
    )
    add_executable(sllm_hip_compile_probe_link
        $<TARGET_OBJECTS:sllm_hip_compile_probe_object>
    )
    set_target_properties(sllm_hip_compile_probe_link PROPERTIES
        CXX_EXTENSIONS OFF
        LINKER_LANGUAGE HIP
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_NAME "hip-compile-probe-${CMAKE_HIP_ARCHITECTURES}"
        SUFFIX ".elf"
    )
    target_link_options(sllm_hip_compile_probe_link PRIVATE
        -mcode-object-version=6
        -mno-wavefrontsize64
    )
    target_include_directories(sllm_hip_compile_probe_link PRIVATE
        "${ROCM_PATH}/include"
    )
    target_include_directories(sllm_hip_compile_probe_object PRIVATE
        "${ROCM_PATH}/include"
    )

    set(SLLM_HIP_PROBE_OBJECT_OUTPUT
        "${CMAKE_BINARY_DIR}/hip-compile-probe-${CMAKE_HIP_ARCHITECTURES}.o"
    )
    add_custom_command(
        OUTPUT "${SLLM_HIP_PROBE_OBJECT_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_OBJECTS:sllm_hip_compile_probe_object>"
            "${SLLM_HIP_PROBE_OBJECT_OUTPUT}"
        DEPENDS sllm_hip_compile_probe_object
        VERBATIM
    )
    add_custom_target(sllm_hip_compile_probe_bundle_object
        DEPENDS "${SLLM_HIP_PROBE_OBJECT_OUTPUT}"
    )
    add_dependencies(sllm_hip_compile_probe_link sllm_hip_compile_probe_bundle_object)
endif()
