cmake_minimum_required(VERSION 3.21)

option(ULLM_ENABLE_HIP_COMPILE_PROBE
    "Build the explicit HIP compile/link-only probe"
    OFF
)

option(ULLM_ENABLE_HIP_RUNTIME
    "Build the explicit model-free HIP evidence runtime"
    OFF
)

if(ULLM_ENABLE_HIP_COMPILE_PROBE OR ULLM_ENABLE_HIP_RUNTIME)
    if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
        message(FATAL_ERROR "HIP build requires an explicit ROCM_PATH")
    endif()
    if(NOT IS_ABSOLUTE "${ROCM_PATH}")
        message(FATAL_ERROR "ROCM_PATH must be an absolute path")
    endif()
    get_filename_component(ULLM_ROCM_LOGICAL "${ROCM_PATH}" ABSOLUTE)
    if(NOT ULLM_ROCM_LOGICAL STREQUAL "/opt/rocm")
        message(FATAL_ERROR "HIP build requires the logical ROCm root /opt/rocm")
    endif()
    get_filename_component(ULLM_ROCM_REAL "${ROCM_PATH}" REALPATH)
    if(ULLM_ROCM_REAL STREQUAL "" OR NOT IS_DIRECTORY "${ULLM_ROCM_REAL}")
        message(FATAL_ERROR "H3 ROCm root does not resolve to a directory")
    endif()
    set(ULLM_ROCM_RELEASE_MARKERS "${ULLM_ROCM_REAL}/.info/version")
    file(GLOB ULLM_ROCM_CHILD_RELEASE_MARKERS
        "${ULLM_ROCM_REAL}/*/.info/version"
    )
    list(APPEND ULLM_ROCM_RELEASE_MARKERS ${ULLM_ROCM_CHILD_RELEASE_MARKERS})
    list(REMOVE_DUPLICATES ULLM_ROCM_RELEASE_MARKERS)
    set(ULLM_FOUND_ROCM_RELEASE_MARKER OFF)
    foreach(ULLM_ROCM_RELEASE_MARKER IN LISTS ULLM_ROCM_RELEASE_MARKERS)
        if(EXISTS "${ULLM_ROCM_RELEASE_MARKER}")
            set(ULLM_FOUND_ROCM_RELEASE_MARKER ON)
            file(READ "${ULLM_ROCM_RELEASE_MARKER}" ULLM_ROCM_RELEASE)
            string(STRIP "${ULLM_ROCM_RELEASE}" ULLM_ROCM_RELEASE)
            if(NOT ULLM_ROCM_RELEASE STREQUAL "7.14.0")
                message(FATAL_ERROR
                    "HIP build requires every discovered ROCm release marker to be 7.14.0: ${ULLM_ROCM_RELEASE_MARKER}=${ULLM_ROCM_RELEASE}")
            endif()
        endif()
    endforeach()
    if(NOT ULLM_FOUND_ROCM_RELEASE_MARKER)
        message(FATAL_ERROR "ROCm release marker is missing under ${ULLM_ROCM_REAL}")
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
        message(FATAL_ERROR "HIP build requires an explicit CMAKE_HIP_COMPILER")
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
        message(FATAL_ERROR "HIP build requires the ROCm amdclang++ entry point")
    endif()
    if(NOT CMAKE_HIP_COMPILER STREQUAL "${ROCM_PATH}/bin/amdclang++")
        message(FATAL_ERROR "HIP compiler must use the logical ROCM_PATH/bin/amdclang++ entry point")
    endif()
    execute_process(
        COMMAND "${CMAKE_HIP_COMPILER}" --version
        RESULT_VARIABLE ULLM_HIP_COMPILER_STATUS
        OUTPUT_VARIABLE ULLM_HIP_COMPILER_VERSION
        ERROR_VARIABLE ULLM_HIP_COMPILER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT ULLM_HIP_COMPILER_STATUS EQUAL 0 OR
       NOT ULLM_HIP_COMPILER_VERSION MATCHES "(^|\\n)AMD clang version 23\\.")
        message(FATAL_ERROR
            "HIP build requires LLVM major 23 from ROCm amdclang++; compiler output=${ULLM_HIP_COMPILER_VERSION}; error=${ULLM_HIP_COMPILER_ERROR}")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR
       CMAKE_HIP_ARCHITECTURES STREQUAL "" OR
       CMAKE_HIP_ARCHITECTURES MATCHES ";" OR
       NOT CMAKE_HIP_ARCHITECTURES MATCHES "^(gfx1030|gfx1201)$")
        message(FATAL_ERROR "HIP build requires exactly one exact gfx1030 or gfx1201 target")
    endif()
    if(NOT DEFINED ULLM_HIP_COMPILE_TARGET OR
       NOT ULLM_HIP_COMPILE_TARGET STREQUAL CMAKE_HIP_ARCHITECTURES)
        message(FATAL_ERROR
            "HIP build requires ULLM_HIP_COMPILE_TARGET to equal the exact CMAKE_HIP_ARCHITECTURES target")
    endif()
    if(NOT DEFINED ULLM_HIP_CODEGEN_FEATURES OR
       NOT ULLM_HIP_CODEGEN_FEATURES STREQUAL
          "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0")
        message(FATAL_ERROR "HIP codegen features are not the pinned tuple")
    endif()
    ullm_require_rocm_path("${ROCM_PATH}/include/hip/hip_runtime.h" "hip_headers")
    ullm_require_rocm_path("${ROCM_PATH}/lib/cmake/hip" "hip_cmake_package")
    ullm_require_rocm_path("${ROCM_PATH}/amdgcn/bitcode" "device_libraries")
    ullm_require_rocm_path("${ROCM_PATH}/lib/libamdhip64.so" "hip_runtime")
    ullm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" "clang_offload_bundler")
    ullm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objcopy" "llvm_objcopy")
    ullm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-readobj" "llvm_readobj")
    ullm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objdump" "llvm_objdump")
    project(ullm_hip_stub VERSION 0.1.0 LANGUAGES C CXX HIP)
else()
    project(ullm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)
endif()

if(ULLM_ENABLE_HIP_COMPILE_PROBE OR ULLM_ENABLE_HIP_RUNTIME)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

set(ULLM_HIP_NATIVE_SOURCES
    src/hip_stub.cpp
    src/hip_evidence_stub.cpp
    src/header_c_compile.c
    src/header_cpp_compile.cpp
)

if(ULLM_ENABLE_HIP_RUNTIME)
    set_source_files_properties(src/hip_evidence_runtime.hip.cpp PROPERTIES LANGUAGE HIP)
    list(REMOVE_ITEM ULLM_HIP_NATIVE_SOURCES src/hip_evidence_stub.cpp)
    list(APPEND ULLM_HIP_NATIVE_SOURCES src/hip_evidence_runtime.hip.cpp)
endif()

add_library(ullm_hip_stub STATIC ${ULLM_HIP_NATIVE_SOURCES})

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

if(ULLM_ENABLE_HIP_RUNTIME)
    target_compile_definitions(ullm_hip_stub PRIVATE
        ULLM_HIP_COMPILE_TARGET="${ULLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(ullm_hip_stub PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>
    )
endif()

if(ULLM_ENABLE_HIP_COMPILE_PROBE)
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
