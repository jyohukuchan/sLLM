cmake_minimum_required(VERSION 3.21)

option(SLLM_ENABLE_LOWP_MATMUL_GPU_TESTS
    "Build the lowp matrix multiplication oracle, example, and microbench"
    OFF)

# The library is usable both as a top-level HIP project and as a child of the
# sLLM native/hip build.  Do not call project() in the latter case: the parent
# owns compiler discovery and the exact target/codegen contract.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sllm_lowp VERSION 0.1.0 LANGUAGES C CXX HIP)
    set(LOWP_STANDALONE ON)
    set(LOWP_ENABLE_HIP ON)
else()
    set(LOWP_STANDALONE OFF)
    if(SLLM_ENABLE_HIP_RUNTIME OR SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR
       SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST OR
       SLLM_ENABLE_LOWP_MATMUL_GPU_TESTS)
        set(LOWP_ENABLE_HIP ON)
    else()
        set(LOWP_ENABLE_HIP OFF)
    endif()
endif()

set(LOWP_TARGET "${SLLM_HIP_TARGET}" CACHE STRING
    "Logical exact gfx target for the lowp HIP library")
if(LOWP_TARGET STREQUAL "")
    set(LOWP_TARGET "${SLLM_HIP_COMPILE_TARGET}")
endif()
if(LOWP_ENABLE_HIP)
    if(NOT LOWP_TARGET MATCHES "^(gfx1030|gfx1201|gfx942)$")
        message(FATAL_ERROR "lowp requires logical target gfx1030, gfx1201, or gfx942")
    endif()
elseif(LOWP_TARGET STREQUAL "")
    # Host-only parent configurations intentionally do not select a GPU.
    # Keep a stable non-HIP value for diagnostics without imposing a target
    # contract on the plan-only library.
    set(LOWP_TARGET "host")
endif()

if(LOWP_STANDALONE)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
        if(LOWP_TARGET STREQUAL "gfx942")
            set(CMAKE_HIP_ARCHITECTURES "gfx942:sramecc+:xnack-" CACHE STRING
                "Exact HIP architecture" FORCE)
        else()
            set(CMAKE_HIP_ARCHITECTURES "${LOWP_TARGET}" CACHE STRING
                "Exact HIP architecture" FORCE)
        endif()
    endif()
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

if(NOT LOWP_ENABLE_HIP)
    set(LOWP_EXPECTED_ARCH "")
    set(LOWP_WAVEFRONT_FLAG "")
elseif(LOWP_TARGET STREQUAL "gfx942")
    set(LOWP_EXPECTED_ARCH "gfx942:sramecc+:xnack-")
    set(LOWP_WAVEFRONT_FLAG -mwavefrontsize64)
else()
    set(LOWP_EXPECTED_ARCH "${LOWP_TARGET}")
    set(LOWP_WAVEFRONT_FLAG -mno-wavefrontsize64)
endif()
if(LOWP_ENABLE_HIP AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL LOWP_EXPECTED_ARCH)
    message(FATAL_ERROR
        "lowp requires exact HIP architecture ${LOWP_EXPECTED_ARCH}, got ${CMAKE_HIP_ARCHITECTURES}")
endif()

set(LOWP_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# The plan implementation is deliberately a separate host-safe target.  Host
# contract tests link this target and never pull HIP launch code or headers.
add_library(sllm_lowp_plan STATIC src/lowp_plan.cpp)
target_compile_features(sllm_lowp_plan PRIVATE cxx_std_17)
target_include_directories(sllm_lowp_plan
    PUBLIC "${LOWP_PUBLIC_INCLUDE_DIR}")
if(DEFINED ROCM_PATH AND NOT ROCM_PATH STREQUAL "")
    target_include_directories(sllm_lowp_plan PRIVATE "${ROCM_PATH}/include")
endif()
if(NOT LOWP_ENABLE_HIP)
    target_include_directories(sllm_lowp_plan PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/host_stubs")
endif()
set_target_properties(sllm_lowp_plan PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(sllm_lowp_plan PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror)
target_compile_definitions(sllm_lowp_plan PRIVATE __HIP_PLATFORM_AMD__=1)
set_target_properties(sllm_lowp_plan PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lowp")

set(LOWP_HIP_SOURCES src/lowp_kernel.hip.cpp)
if(LOWP_ENABLE_HIP)
    # The launcher is a host C++ wrapper around the device entry points. Keep
    # it out of the HIP language set so the archive contains one intentional
    # device translation unit and no empty device image from the wrapper.
    add_library(sllm_lowp STATIC src/lowp_launch.cpp ${LOWP_HIP_SOURCES})
    set_source_files_properties(${LOWP_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    target_compile_features(sllm_lowp PRIVATE cxx_std_17)
    target_include_directories(sllm_lowp
        PUBLIC "${LOWP_PUBLIC_INCLUDE_DIR}")
    target_link_libraries(sllm_lowp PUBLIC sllm_lowp_plan)
    set_target_properties(sllm_lowp PROPERTIES
        CXX_EXTENSIONS OFF
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lowp")
    target_compile_definitions(sllm_lowp PRIVATE
        __HIP_PLATFORM_AMD__=1
        LOWP_COMPILED_TARGET="${LOWP_EXPECTED_ARCH}"
        SLLM_HIP_COMPILE_TARGET="${LOWP_TARGET}")
    target_compile_options(sllm_lowp PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:${LOWP_WAVEFRONT_FLAG}>
        $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)

    if(DEFINED ROCM_PATH AND NOT ROCM_PATH STREQUAL "")
        target_include_directories(sllm_lowp PRIVATE "${ROCM_PATH}/include")
        target_link_libraries(sllm_lowp PRIVATE "${ROCM_PATH}/lib/libamdhip64.so")
    endif()
else()
    add_library(sllm_lowp INTERFACE)
    target_link_libraries(sllm_lowp INTERFACE sllm_lowp_plan)
endif()

option(SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST
    "Build the focused real-GPU reusable lowp codec oracle" OFF)
if(SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST)
    if(NOT LOWP_ENABLE_HIP)
        message(FATAL_ERROR "lowp GPU test requires LOWP_ENABLE_HIP=ON")
    endif()
    enable_testing()
    add_executable(sllm_low_precision_codec_gpu_test
        tests/low_precision_block_codec_gpu_test.hip.cpp)
    set_source_files_properties(
        tests/low_precision_block_codec_gpu_test.hip.cpp PROPERTIES LANGUAGE HIP)
    target_compile_features(sllm_low_precision_codec_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_low_precision_codec_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${LOWP_TARGET}" __HIP_PLATFORM_AMD__=1)
    target_compile_options(sllm_low_precision_codec_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        -mcode-object-version=6 ${LOWP_WAVEFRONT_FLAG} -ffp-contract=off)
    target_include_directories(sllm_low_precision_codec_gpu_test PRIVATE
        "${LOWP_PUBLIC_INCLUDE_DIR}")
    target_link_libraries(sllm_low_precision_codec_gpu_test PRIVATE sllm_lowp)
    add_test(NAME sllm_low_precision_codec_gpu_test
        COMMAND sllm_low_precision_codec_gpu_test)
    set_tests_properties(sllm_low_precision_codec_gpu_test PROPERTIES
        LABELS "gpu;lowp;codec" RESOURCE_LOCK sllm_gpu)
endif()

if(SLLM_ENABLE_LOWP_MATMUL_GPU_TESTS)
    if(NOT LOWP_ENABLE_HIP)
        message(FATAL_ERROR "lowp matrix multiplication tests require LOWP_ENABLE_HIP=ON")
    endif()
    enable_testing()

    function(sllm_lowp_add_gpu_target target source label)
        add_executable(${target} ${source})
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
        target_compile_features(${target} PRIVATE cxx_std_17)
        target_compile_definitions(${target} PRIVATE
            SLLM_TEST_EXPECTED_TARGET="${LOWP_TARGET}" __HIP_PLATFORM_AMD__=1)
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
            -mcode-object-version=6 ${LOWP_WAVEFRONT_FLAG} -ffp-contract=off)
        target_include_directories(${target} PRIVATE
            "${LOWP_PUBLIC_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        target_link_libraries(${target} PRIVATE sllm_lowp)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES
            LABELS "gpu;lowp;${label}"
            RESOURCE_LOCK sllm_gpu
            ENVIRONMENT "LD_LIBRARY_PATH=${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}")
    endfunction()

    sllm_lowp_add_gpu_target(
        sllm_lowp_matmul_oracle_gpu_test
        tests/lowp_matmul_oracle_gpu_test.hip.cpp
        matmul-oracle)
    sllm_lowp_add_gpu_target(
        sllm_lowp_matmul_example
        tests/lowp_matmul_example.hip.cpp
        matmul-example)
    sllm_lowp_add_gpu_target(
        sllm_lowp_matmul_microbench
        tests/lowp_matmul_microbench.hip.cpp
        matmul-microbench)
    sllm_lowp_add_gpu_target(
        sllm_phase87_copy_bandwidth
        tests/phase87_copy_bandwidth.hip.cpp
        phase87-bandwidth)
endif()

option(LOWP_BUILD_HOST_API_TEST
    "Build the host-only lowp public C API contract test" ON)
if(LOWP_BUILD_HOST_API_TEST)
    enable_testing()
    add_executable(sllm_lowp_api_host_test tests/lowp_api_host_test.cpp)
    target_compile_features(sllm_lowp_api_host_test PRIVATE cxx_std_17)
    target_compile_options(sllm_lowp_api_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        -UNDEBUG)
    target_include_directories(sllm_lowp_api_host_test PRIVATE
        "${LOWP_PUBLIC_INCLUDE_DIR}")
    target_link_libraries(sllm_lowp_api_host_test PRIVATE sllm_lowp_plan)
    add_test(NAME sllm_lowp_api_host_test COMMAND sllm_lowp_api_host_test)
    set_tests_properties(sllm_lowp_api_host_test PROPERTIES LABELS "host;lowp")
endif()

option(LOWP_BUILD_HOST_SELECTION_TEST
    "Build the host-only lowp selection fixture" OFF)
if(LOWP_BUILD_HOST_SELECTION_TEST)
    enable_testing()
    add_executable(sllm_lowp_selection_fixture_test
        tests/selection_fixture_test.cpp)
    target_compile_features(sllm_lowp_selection_fixture_test PRIVATE cxx_std_17)
    target_compile_options(sllm_lowp_selection_fixture_test PRIVATE -UNDEBUG)
    target_compile_definitions(sllm_lowp_selection_fixture_test PRIVATE
        LOWP_SELECTION_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/selection_fixture.csv")
    target_include_directories(sllm_lowp_selection_fixture_test PRIVATE
        "${LOWP_PUBLIC_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/host_stubs")
    target_link_libraries(sllm_lowp_selection_fixture_test PRIVATE sllm_lowp_plan)
    add_test(NAME sllm_lowp_selection_fixture_test
        COMMAND sllm_lowp_selection_fixture_test)
    set_tests_properties(sllm_lowp_selection_fixture_test PROPERTIES
        LABELS "host;lowp")
endif()
