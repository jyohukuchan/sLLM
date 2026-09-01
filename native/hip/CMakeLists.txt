cmake_minimum_required(VERSION 3.21)

set(SLLM_ENABLE_PHASE54_KV_RESEARCH "OFF" CACHE STRING
    "Compile the private Phase 54 block16 KV recipe research path (0/1 or OFF/ON)"
)
set_property(CACHE SLLM_ENABLE_PHASE54_KV_RESEARCH PROPERTY STRINGS OFF ON)
if(NOT SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "0" AND
   NOT SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "1" AND
   NOT SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "OFF" AND
   NOT SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "ON")
    message(FATAL_ERROR
        "SLLM_ENABLE_PHASE54_KV_RESEARCH must be exactly 0, 1, OFF, or ON")
endif()
if(SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "1" OR
   SLLM_ENABLE_PHASE54_KV_RESEARCH STREQUAL "ON")
    set(SLLM_PHASE54_KV_RESEARCH_ENABLED 1)
else()
    set(SLLM_PHASE54_KV_RESEARCH_ENABLED 0)
endif()

option(SLLM_ENABLE_HIP_COMPILE_PROBE
    "Build the explicit HIP compile/link-only probe"
    OFF
)

option(SLLM_ENABLE_HIP_RUNTIME
    "Build the explicit model-free HIP evidence runtime"
    OFF
)

option(SLLM_ENABLE_PUBLIC_HIP_RUNTIME
    "Build the public HIP context/resource runtime"
    OFF
)

if(SLLM_PHASE54_KV_RESEARCH_ENABLED AND
   NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
    message(FATAL_ERROR
        "Phase 54 KV research requires SLLM_ENABLE_PUBLIC_HIP_RUNTIME=ON")
endif()

option(SLLM_ENABLE_PUBLIC_RUNTIME_HOST_TEST
    "Build the production public runtime against the test-only fake HIP API"
    OFF
)

option(SLLM_ENABLE_LINEAR_ATTENTION_GPU_TEST
    "Build the focused real-GPU C4 linear-attention numerical runner"
    OFF
)

option(SLLM_ENABLE_ARGMAX_GPU_TEST
    "Build the focused real-GPU C5 argmax numerical runner"
    OFF
)

option(SLLM_ENABLE_TOKEN_SELECTOR_GPU_TEST
    "Build the focused real-GPU prepared token-selector numerical runner"
    OFF
)

option(SLLM_ENABLE_MOE_ROUTE_GPU_TEST
    "Build the focused real-GPU deterministic MoE routing runner"
    OFF
)

option(SLLM_ENABLE_GEMMA4_MOE_ROUTE_PUBLIC_GPU_TEST
    "Build the Phase55 Gemma4 public-ABI MoE routing oracle"
    OFF
)

option(SLLM_ENABLE_DEEPSEEK_V4_MOE_ROUTE_GPU_TEST
    "Build the Phase57 DeepSeek V4 score/hash MoE routing oracle"
    OFF
)

option(SLLM_ENABLE_MINIMAX_M3_MOE_ROUTE_GPU_TEST
    "Build the Phase58 MiniMax M3 sigmoid top-4 MoE routing oracle"
    OFF
)

option(SLLM_ENABLE_MOE_EXPERT_GPU_TEST
    "Build the focused real-weight Qwen3.5 MoE expert runner"
    OFF
)

option(SLLM_ENABLE_RMSNORM_DIRECT_GPU_TEST
    "Build the focused real-GPU Gemma direct-scale RMSNorm runner"
    OFF
)

option(SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST
    "Build the focused real-GPU reusable low-precision codec oracle"
    OFF
)

option(SLLM_ENABLE_ROTARY_GPU_TEST
    "Build the focused real-GPU split-half rotary runner"
    OFF
)

option(SLLM_ENABLE_ROTARY_PUBLIC_GPU_TEST
    "Build the focused real-GPU public-ABI split-half rotary runner"
    OFF
)

option(SLLM_ENABLE_MINISTRAL3_YARN_PUBLIC_GPU_TEST
    "Build the focused real-GPU public-ABI Ministral 3 YaRN runner"
    OFF
)

option(SLLM_ENABLE_BROADCAST_ADD_PUBLIC_GPU_TEST
    "Build the focused real-GPU public-ABI BroadcastAdd numerical runner"
    OFF
)

option(SLLM_ENABLE_BROADCAST_MUL_PUBLIC_GPU_TEST
    "Build the focused real-GPU public-ABI BroadcastMul numerical runner"
    OFF
)

option(SLLM_ENABLE_GEMMA_ATTENTION_GPU_TEST
    "Build the focused real-GPU Gemma sliding/full attention runner"
    OFF
)

option(SLLM_ENABLE_GEMMA_ATTENTION_PUBLIC_GPU_TEST
    "Build the focused real-GPU public-ABI Gemma sliding/full attention runner"
    OFF
)

option(SLLM_ENABLE_PHASE41_STATE_GPU_TEST
    "Build the focused real-GPU Phase 41 state fork/image runner"
    OFF
)

option(SLLM_SEMANTIC_G1_AUTHORITY
    "Require controller-held sealed compiler authority for semantic G1 only"
    OFF
)

option(SLLM_SEMANTIC_G1_DISCOVERY_TEST
    "Configure-only host contract probe for semantic G1 compiler authority"
    OFF
)

function(sllm_validate_semantic_g1_authority compiler_path)
    if(NOT DEFINED ENV{SLLM_HIP_COMPILER_BROKER_CLIENT} OR
       NOT "$ENV{SLLM_HIP_COMPILER_BROKER_CLIENT}" STREQUAL "${compiler_path}" OR
       NOT compiler_path MATCHES "^/proc/self/fd/(3|[4-9]|[1-9][0-9]+)$" OR
       NOT DEFINED ENV{SLLM_HIP_COMPILER_BROKER_CLIENT_FD} OR
       NOT "$ENV{SLLM_HIP_COMPILER_BROKER_CLIENT_FD}" MATCHES "^(3|[4-9]|[1-9][0-9]+)$" OR
       NOT compiler_path STREQUAL "/proc/self/fd/$ENV{SLLM_HIP_COMPILER_BROKER_CLIENT_FD}")
        message(FATAL_ERROR "semantic G1 requires the exact authenticated compiler broker client")
    endif()
    if(NOT DEFINED ENV{SLLM_HIP_COMPILER_BROKER_SOCKET} OR
       NOT "$ENV{SLLM_HIP_COMPILER_BROKER_SOCKET}" MATCHES "^/[^\n\r ]+$" OR
       NOT DEFINED ENV{SLLM_HIP_COMPILER_BROKER_SESSION} OR
       NOT DEFINED ENV{SLLM_HIP_COMPILER_BROKER_TOKEN})
        message(FATAL_ERROR "semantic G1 compiler broker session is missing or malformed")
    endif()
    foreach(SLLM_SEMANTIC_G1_HEX_ENV
            SLLM_HIP_COMPILER_BROKER_SESSION
            SLLM_HIP_COMPILER_BROKER_TOKEN
            SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256)
        if(NOT DEFINED ENV{${SLLM_SEMANTIC_G1_HEX_ENV}})
            message(FATAL_ERROR "semantic G1 compiler broker identity is missing")
        endif()
        string(LENGTH "$ENV{${SLLM_SEMANTIC_G1_HEX_ENV}}" SLLM_SEMANTIC_G1_HEX_LENGTH)
        if(NOT SLLM_SEMANTIC_G1_HEX_LENGTH EQUAL 64 OR
           NOT "$ENV{${SLLM_SEMANTIC_G1_HEX_ENV}}" MATCHES "^[0-9a-f]+$")
            message(FATAL_ERROR "semantic G1 compiler broker identity is malformed")
        endif()
    endforeach()
endfunction()

if(SLLM_SEMANTIC_G1_DISCOVERY_TEST)
    if(NOT SLLM_SEMANTIC_G1_AUTHORITY OR
       NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR
       SLLM_ENABLE_HIP_COMPILE_PROBE OR SLLM_ENABLE_HIP_RUNTIME OR
       SLLM_ENABLE_PUBLIC_RUNTIME_HOST_TEST OR
       SLLM_ENABLE_LINEAR_ATTENTION_GPU_TEST OR SLLM_ENABLE_ARGMAX_GPU_TEST OR
       SLLM_ENABLE_TOKEN_SELECTOR_GPU_TEST OR
       SLLM_ENABLE_DEEPSEEK_V4_MOE_ROUTE_GPU_TEST OR
       SLLM_ENABLE_MINIMAX_M3_MOE_ROUTE_GPU_TEST OR
       SLLM_ENABLE_PHASE41_STATE_GPU_TEST OR
       SLLM_ENABLE_BROADCAST_ADD_PUBLIC_GPU_TEST OR
       SLLM_ENABLE_MINISTRAL3_YARN_PUBLIC_GPU_TEST)
        message(FATAL_ERROR
            "semantic G1 discovery test requires only public HIP runtime authority")
    endif()
    sllm_validate_semantic_g1_authority("${CMAKE_HIP_COMPILER}")
    project(sllm_hip_semantic_discovery_test VERSION 0.1.0 LANGUAGES C CXX)
    return()
endif()

if(SLLM_ENABLE_HIP_COMPILE_PROBE OR SLLM_ENABLE_HIP_RUNTIME OR
   SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR
   SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST)
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
    if(NOT DEFINED SLLM_HIP_COMPILER_LOGICAL OR
       NOT SLLM_HIP_COMPILER_LOGICAL STREQUAL "${ROCM_PATH}/bin/amdclang++")
        message(FATAL_ERROR
            "HIP build requires the logical ROCM_PATH/bin/amdclang++ compiler identity")
    endif()
    if(SLLM_SEMANTIC_G1_AUTHORITY)
        # CMake and Cargo receive only the reviewed client pathname and an
        # authenticated broker session.  The pathname is a retained sealed
        # descriptor, so replacement of the original client path cannot alter
        # the compiler client that CMake executes.
        sllm_validate_semantic_g1_authority("${CMAKE_HIP_COMPILER}")
    else()
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
        if(NOT SLLM_HIP_COMPILER_NAME STREQUAL "amdclang++" OR
           NOT CMAKE_HIP_COMPILER STREQUAL "${ROCM_PATH}/bin/amdclang++")
            message(FATAL_ERROR "HIP compiler must use the logical ROCM_PATH/bin/amdclang++ entry point")
        endif()
    endif()
    if(NOT SLLM_SEMANTIC_G1_AUTHORITY)
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
    endif()
    # SLLM_HIP_TARGET is the public logical target name.  Keep the older
    # SLLM_HIP_COMPILE_TARGET spelling as the source-level compatibility alias
    # for direct CMake callers, but never use either logical name as the HIP
    # architecture when a feature suffix is required.
    if(NOT DEFINED SLLM_HIP_COMPILE_TARGET AND DEFINED SLLM_HIP_TARGET)
        set(SLLM_HIP_COMPILE_TARGET "${SLLM_HIP_TARGET}")
    endif()
    if(NOT DEFINED SLLM_HIP_COMPILE_TARGET OR
       SLLM_HIP_COMPILE_TARGET STREQUAL "" OR
       SLLM_HIP_COMPILE_TARGET MATCHES ":" OR
       NOT SLLM_HIP_COMPILE_TARGET MATCHES "^(gfx1030|gfx1201|gfx942)$")
        message(FATAL_ERROR
            "HIP build requires exactly one logical gfx1030, gfx1201, or gfx942 target")
    endif()
    if(DEFINED SLLM_HIP_TARGET AND
       NOT SLLM_HIP_TARGET STREQUAL SLLM_HIP_COMPILE_TARGET)
        message(FATAL_ERROR
            "SLLM_HIP_TARGET must equal the logical SLLM_HIP_COMPILE_TARGET target")
    endif()
    set(SLLM_HIP_TARGET "${SLLM_HIP_COMPILE_TARGET}")
    if(SLLM_HIP_TARGET STREQUAL "gfx942")
        set(SLLM_EXPECTED_HIP_ARCHITECTURE "gfx942:sramecc+:xnack-")
        set(SLLM_EXPECTED_CODEGEN_FEATURES
            "co_v6,wave64,xnack=off,sramecc=on,generic_processor_version=0")
        set(SLLM_WAVEFRONT_FLAG -mwavefrontsize64)
        set(SLLM_COMPILE_WAVE64 1)
    else()
        set(SLLM_EXPECTED_HIP_ARCHITECTURE "${SLLM_HIP_TARGET}")
        set(SLLM_EXPECTED_CODEGEN_FEATURES
            "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0")
        set(SLLM_WAVEFRONT_FLAG -mno-wavefrontsize64)
        set(SLLM_COMPILE_WAVE64 0)
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR
       CMAKE_HIP_ARCHITECTURES STREQUAL "" OR
       CMAKE_HIP_ARCHITECTURES MATCHES ";" OR
       NOT CMAKE_HIP_ARCHITECTURES STREQUAL SLLM_EXPECTED_HIP_ARCHITECTURE)
        message(FATAL_ERROR
            "HIP build requires codegen architecture ${SLLM_EXPECTED_HIP_ARCHITECTURE} for logical target ${SLLM_HIP_TARGET}")
    endif()
    if(NOT DEFINED SLLM_HIP_CODEGEN_FEATURES OR
       NOT SLLM_HIP_CODEGEN_FEATURES STREQUAL SLLM_EXPECTED_CODEGEN_FEATURES)
        message(FATAL_ERROR "HIP codegen features are not the pinned tuple")
    endif()
    # ROCm 7.14 / AMD clang 23 accepts the two code-object/wave32 driver
    # options below. Its --help exposes neither -mno-xnack nor -mno-sramecc,
    # and target IDs with :xnack- or :sramecc- are rejected for exact
    # gfx1030/gfx1201 because those processors report both features as
    # unsupported. Therefore those logical targets intentionally keep their
    # unsuffixed exact architecture. CDNA3 is different: its exact codegen
    # architecture is pinned to gfx942:sramecc+:xnack-, while the logical
    # runtime/compile target remains gfx942. No generic target is accepted.
    if(NOT SLLM_SEMANTIC_G1_AUTHORITY)
        execute_process(
            COMMAND "${CMAKE_HIP_COMPILER}" --help
            RESULT_VARIABLE SLLM_HIP_HELP_STATUS
            OUTPUT_VARIABLE SLLM_HIP_HELP
            ERROR_VARIABLE SLLM_HIP_HELP_ERROR
        )
        if(NOT SLLM_HIP_HELP_STATUS EQUAL 0 OR
           NOT SLLM_HIP_HELP MATCHES "-mcode-object-version=<value>" OR
           NOT SLLM_HIP_HELP MATCHES "-mno-wavefrontsize64" OR
           NOT SLLM_HIP_HELP MATCHES "-mwavefrontsize64")
            message(FATAL_ERROR
                "ROCm AMD clang does not expose the pinned HIP flags: ${SLLM_HIP_HELP_ERROR}")
        endif()
    endif()
    if(SLLM_HIP_TARGET STREQUAL "gfx942" AND
       NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx942:sramecc+:xnack-")
        message(FATAL_ERROR
            "gfx942 requires the exact gfx942:sramecc+:xnack- codegen tuple")
    elseif(NOT SLLM_HIP_TARGET STREQUAL "gfx942" AND
           CMAKE_HIP_ARCHITECTURES MATCHES ":")
        message(FATAL_ERROR
            "gfx1030/gfx1201 do not accept feature-suffixed HIP architectures")
    endif()
    if(SLLM_HIP_CODEGEN_FEATURES MATCHES "generic_processor_version=[1-9]")
        message(FATAL_ERROR
            "public HIP build requires an exact target with generic_processor_version=0")
    endif()
    sllm_require_rocm_path("${ROCM_PATH}/include/hip/hip_runtime.h" "hip_headers")
    sllm_require_rocm_path("${ROCM_PATH}/lib/cmake/hip" "hip_cmake_package")
    sllm_require_rocm_path("${ROCM_PATH}/amdgcn/bitcode" "device_libraries")
    sllm_require_rocm_path("${ROCM_PATH}/lib/libamdhip64.so" "hip_runtime")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" "clang_offload_bundler")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objcopy" "llvm_objcopy")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-readobj" "llvm_readobj")
    sllm_require_rocm_path("${ROCM_PATH}/lib/llvm/bin/llvm-objdump" "llvm_objdump")
    if(SLLM_SEMANTIC_G1_AUTHORITY)
        # Configure is intentionally compiler-less.  Semantic G1 uses only
        # the two fixed custom HIP actions below; no CMake compiler-id or
        # try-compile request can gain compiler authority during discovery.
        project(sllm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)
    else()
        project(sllm_hip_stub VERSION 0.1.0 LANGUAGES C CXX HIP)
    endif()
else()
    project(sllm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)
endif()

if((SLLM_ENABLE_HIP_COMPILE_PROBE OR SLLM_ENABLE_HIP_RUNTIME OR
   SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR
   SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST) AND
   NOT SLLM_SEMANTIC_G1_AUTHORITY)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

set(SLLM_HIP_NATIVE_SOURCES
    src/hip_stub.cpp
    src/public_runtime_stub.cpp
    src/embedding_api.cpp
    src/elementwise_api.cpp
    src/matmul_api.cpp
    src/argmax_api.cpp
    src/token_selector_api.cpp
    src/moe_route_api.cpp
    src/deepseek_v4_moe_route_api.cpp
    src/deepseek_v4_moe_route_stub.cpp
    src/minimax_m3_moe_route_api.cpp
    src/minimax_m3_moe_route_stub.cpp
    src/moe_expert_api.cpp
    src/rmsnorm_api.cpp
    src/residual_rmsnorm_api.cpp
    src/attention_preprocess_api.cpp
    src/rotary_api.cpp
    src/ministral3_yarn_api.cpp
    src/windowed_attention_api.cpp
    src/causal_attention_api.cpp
    src/kv_state_api.cpp
    src/linear_attention_api.cpp
    src/hip_evidence_stub.cpp
    src/header_c_compile.c
    src/header_cpp_compile.cpp
)

# Keep the private recipe controls linkable in stub and production builds.
# Only the separately compiled HIP research kernels are gated; the disabled
# setter accepts Floor/Floor and rejects every non-production recipe.
set(SLLM_PHASE54_KV_RESEARCH_SELECTOR_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/phase54_kv_research_selector.cpp")
file(GENERATE OUTPUT "${SLLM_PHASE54_KV_RESEARCH_SELECTOR_SOURCE}" CONTENT [=[
#include <atomic>
#include <cstdint>

#ifndef SLLM_ENABLE_PHASE54_KV_RESEARCH
#define SLLM_ENABLE_PHASE54_KV_RESEARCH 0
#endif

namespace {
constexpr uint32_t kFloor = 0U;
constexpr uint32_t kRecipeCount = 4U;
constexpr int32_t kOk = 0;
constexpr int32_t kInvalidRecipe = 1;
constexpr int32_t kUnsupported = 2;
constexpr uint32_t kValueShift = 8U;
constexpr uint32_t kRecipeMask = UINT32_C(0xff);
std::atomic<uint32_t> g_recipe_pair{kFloor};
} // namespace

extern "C" int32_t
sllm_phase54_kv_research_set_recipe_pair_v1(const uint32_t key_recipe,
                                             const uint32_t value_recipe) noexcept {
  if (key_recipe >= kRecipeCount || value_recipe >= kRecipeCount) {
    g_recipe_pair.store(kFloor, std::memory_order_release);
    return kInvalidRecipe;
  }
#if SLLM_ENABLE_PHASE54_KV_RESEARCH
  g_recipe_pair.store(key_recipe | (value_recipe << kValueShift),
                      std::memory_order_release);
  return kOk;
#else
  g_recipe_pair.store(kFloor, std::memory_order_release);
  return key_recipe == kFloor && value_recipe == kFloor ? kOk : kUnsupported;
#endif
}

extern "C" int32_t
sllm_phase54_kv_research_get_recipe_pair_v1(uint32_t *const key_recipe,
                                             uint32_t *const value_recipe) noexcept {
  if (key_recipe == nullptr || value_recipe == nullptr) {
    return kInvalidRecipe;
  }
  const uint32_t pair = g_recipe_pair.load(std::memory_order_acquire);
  *key_recipe = pair & kRecipeMask;
  *value_recipe = (pair >> kValueShift) & kRecipeMask;
  return kOk;
}
]=])
set_source_files_properties("${SLLM_PHASE54_KV_RESEARCH_SELECTOR_SOURCE}"
    PROPERTIES GENERATED TRUE)
list(APPEND SLLM_HIP_NATIVE_SOURCES
    "${SLLM_PHASE54_KV_RESEARCH_SELECTOR_SOURCE}")

if(SLLM_ENABLE_HIP_RUNTIME)
    set_source_files_properties(src/hip_evidence_runtime.hip.cpp PROPERTIES LANGUAGE HIP)
    list(REMOVE_ITEM SLLM_HIP_NATIVE_SOURCES src/hip_evidence_stub.cpp)
    list(APPEND SLLM_HIP_NATIVE_SOURCES src/hip_evidence_runtime.hip.cpp)
endif()

if(SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
    list(REMOVE_ITEM SLLM_HIP_NATIVE_SOURCES src/public_runtime_stub.cpp)
    list(REMOVE_ITEM SLLM_HIP_NATIVE_SOURCES
        src/deepseek_v4_moe_route_stub.cpp)
    list(REMOVE_ITEM SLLM_HIP_NATIVE_SOURCES
        src/minimax_m3_moe_route_stub.cpp)
    if(SLLM_SEMANTIC_G1_AUTHORITY)
        set(SLLM_SEMANTIC_G1_HIP_FLAGS
            -D__HIP_ROCclr__=1
            -DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=1
            -DSLLM_ENABLE_PHASE54_KV_RESEARCH=${SLLM_PHASE54_KV_RESEARCH_ENABLED}
            -O3
            -DNDEBUG
            -ffp-contract=off
            -std=gnu++17
            -fPIC
            -I "${PROJECT_SOURCE_DIR}/../../include"
            -I "${PROJECT_SOURCE_DIR}/src"
            "--offload-arch=${CMAKE_HIP_ARCHITECTURES}"
            -mcode-object-version=6
            ${SLLM_WAVEFRONT_FLAG}
            -pthread
            -x hip
            -c
        )
        set(SLLM_SEMANTIC_G1_PUBLIC_OBJECT "${CMAKE_BINARY_DIR}/public_runtime.hip.o")
        set(SLLM_SEMANTIC_G1_KERNEL_OBJECT "${CMAKE_BINARY_DIR}/rmsnorm_kernel.hip.o")
        set(SLLM_ELEMENTWISE_G1_KERNEL_OBJECT "${CMAKE_BINARY_DIR}/elementwise_kernel.hip.o")
        set(SLLM_EMBEDDING_G1_KERNEL_OBJECT "${CMAKE_BINARY_DIR}/embedding_kernel.hip.o")
        set(SLLM_MATMUL_G1_KERNEL_OBJECT "${CMAKE_BINARY_DIR}/matmul_kernel.hip.o")
        set(SLLM_MLP_GATE_UP_SILU_BUNDLE_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/mlp_gate_up_silu_bundle_kernel.hip.o")
        set(SLLM_ARGMAX_G1_KERNEL_OBJECT "${CMAKE_BINARY_DIR}/argmax_kernel.hip.o")
        set(SLLM_TOKEN_SELECTOR_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/token_selector_kernel.hip.o")
        set(SLLM_MOE_ROUTE_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/moe_route_kernel.hip.o")
        set(SLLM_DEEPSEEK_V4_MOE_ROUTE_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/deepseek_v4_moe_route_kernel.hip.o")
        set(SLLM_MINIMAX_M3_MOE_ROUTE_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/minimax_m3_moe_route_kernel.hip.o")
        set(SLLM_MOE_EXPERT_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/moe_expert_kernel.hip.o")
        set(SLLM_ATTENTION_PREPROCESS_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/attention_preprocess_kernel.hip.o")
        set(SLLM_KV_STATE_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/kv_state_kernel.hip.o")
        set(SLLM_CAUSAL_ATTENTION_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/causal_attention_kernel.hip.o")
        set(SLLM_LINEAR_ATTENTION_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/linear_attention_kernel.hip.o")
        set(SLLM_ROTARY_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/rotary_kernel.hip.o")
        set(SLLM_MINISTRAL3_YARN_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/ministral3_yarn_kernel.hip.o")
        set(SLLM_GEMMA_ATTENTION_G1_KERNEL_OBJECT
            "${CMAKE_BINARY_DIR}/gemma_attention_kernel.hip.o")
        add_custom_command(
            OUTPUT "${SLLM_SEMANTIC_G1_PUBLIC_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/minimax_m3_moe_route_public_runtime.hip.cpp"
                -o "${SLLM_SEMANTIC_G1_PUBLIC_OBJECT}"
            DEPENDS src/minimax_m3_moe_route_public_runtime.hip.cpp
                src/public_runtime.hip.cpp src/matmul_runtime.inc
                src/low_precision_matmul_provider.hpp
                src/mlp_gate_up_silu_bundle_runtime.inc
                src/mlp_gate_up_silu_bundle_kernel_internal.hpp
                src/argmax_runtime.inc
                src/token_selector_runtime.inc
                src/moe_route_runtime.inc
                src/deepseek_v4_moe_route_runtime.inc
                src/minimax_m3_moe_route_runtime.inc
                src/moe_expert_runtime.inc
                src/attention_preprocess_runtime.inc
                src/rotary_runtime.inc
                src/ministral3_yarn_runtime.inc
                src/windowed_attention_runtime.inc
                src/causal_attention_runtime.inc
                src/linear_attention_runtime.inc
                src/public_runtime_internal.hpp src/embedding_api.hpp
                src/embedding_kernel_internal.hpp src/elementwise_api.hpp
                src/elementwise_kernel_internal.hpp src/matmul_api.hpp
                src/matmul_kernel_internal.hpp src/argmax_api.hpp
                src/argmax_kernel_internal.hpp src/rmsnorm_api.hpp
                src/token_selector_api.hpp src/token_selector_kernel_internal.hpp
                src/moe_route_api.hpp src/moe_route_kernel_internal.hpp
                src/deepseek_v4_moe_route_api.hpp
                src/deepseek_v4_moe_route_kernel_internal.hpp
                src/minimax_m3_moe_route_api.hpp
                src/minimax_m3_moe_route_kernel_internal.hpp
                src/moe_expert_api.hpp src/moe_expert_kernel_internal.hpp
                src/attention_preprocess_api.hpp
                src/attention_preprocess_kernel_internal.hpp
                src/rotary_api.hpp src/rotary_kernel_internal.hpp
                src/ministral3_yarn_api.hpp
                src/ministral3_yarn_kernel_internal.hpp
                src/windowed_attention_api.hpp
                src/gemma_attention_kernel_internal.hpp
                src/causal_attention_api.hpp
                src/causal_attention_kernel_internal.hpp
                src/kv_state_api.hpp src/kv_state_kernel_internal.hpp
                src/linear_attention_api.hpp
                src/linear_attention_kernel_internal.hpp
                src/evidence_abi.h
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_SEMANTIC_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/rmsnorm_kernel.hip.cpp"
                -o "${SLLM_SEMANTIC_G1_KERNEL_OBJECT}"
            DEPENDS src/rmsnorm_kernel.hip.cpp src/rmsnorm_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_ELEMENTWISE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/elementwise_kernel.hip.cpp"
                -o "${SLLM_ELEMENTWISE_G1_KERNEL_OBJECT}"
            DEPENDS src/elementwise_kernel.hip.cpp src/elementwise_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_EMBEDDING_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/embedding_kernel.hip.cpp"
                -o "${SLLM_EMBEDDING_G1_KERNEL_OBJECT}"
            DEPENDS src/embedding_kernel.hip.cpp src/embedding_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MATMUL_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/matmul_kernel.hip.cpp"
                -o "${SLLM_MATMUL_G1_KERNEL_OBJECT}"
            DEPENDS src/matmul_kernel.hip.cpp src/matmul_kernel_internal.hpp
                src/low_precision_block_codec.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MLP_GATE_UP_SILU_BUNDLE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/mlp_gate_up_silu_bundle_kernel.hip.cpp"
                -o "${SLLM_MLP_GATE_UP_SILU_BUNDLE_G1_KERNEL_OBJECT}"
            DEPENDS src/mlp_gate_up_silu_bundle_kernel.hip.cpp
                src/mlp_gate_up_silu_bundle_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_ARGMAX_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/argmax_kernel.hip.cpp"
                -o "${SLLM_ARGMAX_G1_KERNEL_OBJECT}"
            DEPENDS src/argmax_kernel.hip.cpp src/argmax_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_TOKEN_SELECTOR_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/token_selector_kernel.hip.cpp"
                -o "${SLLM_TOKEN_SELECTOR_G1_KERNEL_OBJECT}"
            DEPENDS src/token_selector_kernel.hip.cpp
                src/token_selector_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MOE_ROUTE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/moe_route_kernel.hip.cpp"
                -o "${SLLM_MOE_ROUTE_G1_KERNEL_OBJECT}"
            DEPENDS src/moe_route_kernel.hip.cpp
                src/moe_route_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_DEEPSEEK_V4_MOE_ROUTE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/deepseek_v4_moe_route_kernel.hip.cpp"
                -o "${SLLM_DEEPSEEK_V4_MOE_ROUTE_G1_KERNEL_OBJECT}"
            DEPENDS src/deepseek_v4_moe_route_kernel.hip.cpp
                src/deepseek_v4_moe_route_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MINIMAX_M3_MOE_ROUTE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/minimax_m3_moe_route_kernel.hip.cpp"
                -o "${SLLM_MINIMAX_M3_MOE_ROUTE_G1_KERNEL_OBJECT}"
            DEPENDS src/minimax_m3_moe_route_kernel.hip.cpp
                src/minimax_m3_moe_route_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MOE_EXPERT_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/moe_expert_kernel.hip.cpp"
                -o "${SLLM_MOE_EXPERT_G1_KERNEL_OBJECT}"
            DEPENDS src/moe_expert_kernel.hip.cpp
                src/moe_expert_kernel_internal.hpp
                src/matmul_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_ATTENTION_PREPROCESS_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/attention_preprocess_kernel.hip.cpp"
                -o "${SLLM_ATTENTION_PREPROCESS_G1_KERNEL_OBJECT}"
            DEPENDS src/attention_preprocess_kernel.hip.cpp
                src/attention_preprocess_kernel_internal.hpp
                src/rmsnorm_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_KV_STATE_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/kv_state_kernel.hip.cpp"
                -o "${SLLM_KV_STATE_G1_KERNEL_OBJECT}"
            DEPENDS src/kv_state_kernel.hip.cpp
                src/kv_state_kernel_internal.hpp
                src/low_precision_block_codec.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_CAUSAL_ATTENTION_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/causal_attention_kernel.hip.cpp"
                -o "${SLLM_CAUSAL_ATTENTION_G1_KERNEL_OBJECT}"
            DEPENDS src/causal_attention_kernel.hip.cpp
                src/causal_attention_kernel_internal.hpp
                src/low_precision_block_codec.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_LINEAR_ATTENTION_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/linear_attention_kernel.hip.cpp"
                -o "${SLLM_LINEAR_ATTENTION_G1_KERNEL_OBJECT}"
            DEPENDS src/linear_attention_kernel.hip.cpp
                src/linear_attention_kernel_internal.hpp
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_ROTARY_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/rotary_kernel.hip.cpp"
                -o "${SLLM_ROTARY_G1_KERNEL_OBJECT}"
            DEPENDS src/rotary_kernel.hip.cpp src/rotary_kernel_internal.hpp
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_MINISTRAL3_YARN_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/ministral3_yarn_kernel.hip.cpp"
                -o "${SLLM_MINISTRAL3_YARN_G1_KERNEL_OBJECT}"
            DEPENDS src/ministral3_yarn_kernel.hip.cpp
                src/ministral3_yarn_kernel_internal.hpp
                "${PROJECT_SOURCE_DIR}/../../include/sllm/hip.h"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${SLLM_GEMMA_ATTENTION_G1_KERNEL_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}" ${SLLM_SEMANTIC_G1_HIP_FLAGS}
                "${PROJECT_SOURCE_DIR}/src/gemma_attention_kernel.hip.cpp"
                -o "${SLLM_GEMMA_ATTENTION_G1_KERNEL_OBJECT}"
            DEPENDS src/gemma_attention_kernel.hip.cpp
                src/gemma_attention_kernel_internal.hpp
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
        )
        set_source_files_properties(
            "${SLLM_SEMANTIC_G1_PUBLIC_OBJECT}" "${SLLM_SEMANTIC_G1_KERNEL_OBJECT}"
            "${SLLM_ELEMENTWISE_G1_KERNEL_OBJECT}"
            "${SLLM_EMBEDDING_G1_KERNEL_OBJECT}"
            "${SLLM_MATMUL_G1_KERNEL_OBJECT}"
            "${SLLM_MLP_GATE_UP_SILU_BUNDLE_G1_KERNEL_OBJECT}"
            "${SLLM_ARGMAX_G1_KERNEL_OBJECT}"
            "${SLLM_TOKEN_SELECTOR_G1_KERNEL_OBJECT}"
            "${SLLM_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_DEEPSEEK_V4_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_MINIMAX_M3_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_MOE_EXPERT_G1_KERNEL_OBJECT}"
            "${SLLM_ATTENTION_PREPROCESS_G1_KERNEL_OBJECT}"
            "${SLLM_KV_STATE_G1_KERNEL_OBJECT}"
            "${SLLM_CAUSAL_ATTENTION_G1_KERNEL_OBJECT}"
            "${SLLM_LINEAR_ATTENTION_G1_KERNEL_OBJECT}"
            "${SLLM_ROTARY_G1_KERNEL_OBJECT}"
            "${SLLM_MINISTRAL3_YARN_G1_KERNEL_OBJECT}"
            "${SLLM_GEMMA_ATTENTION_G1_KERNEL_OBJECT}"
            PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE
        )
        list(APPEND SLLM_HIP_NATIVE_SOURCES
            "${SLLM_SEMANTIC_G1_PUBLIC_OBJECT}" "${SLLM_SEMANTIC_G1_KERNEL_OBJECT}"
            "${SLLM_ELEMENTWISE_G1_KERNEL_OBJECT}"
            "${SLLM_EMBEDDING_G1_KERNEL_OBJECT}"
            "${SLLM_MATMUL_G1_KERNEL_OBJECT}"
            "${SLLM_MLP_GATE_UP_SILU_BUNDLE_G1_KERNEL_OBJECT}"
            "${SLLM_ARGMAX_G1_KERNEL_OBJECT}"
            "${SLLM_TOKEN_SELECTOR_G1_KERNEL_OBJECT}"
            "${SLLM_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_DEEPSEEK_V4_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_MINIMAX_M3_MOE_ROUTE_G1_KERNEL_OBJECT}"
            "${SLLM_MOE_EXPERT_G1_KERNEL_OBJECT}"
            "${SLLM_ATTENTION_PREPROCESS_G1_KERNEL_OBJECT}"
            "${SLLM_KV_STATE_G1_KERNEL_OBJECT}"
            "${SLLM_CAUSAL_ATTENTION_G1_KERNEL_OBJECT}"
            "${SLLM_LINEAR_ATTENTION_G1_KERNEL_OBJECT}"
            "${SLLM_ROTARY_G1_KERNEL_OBJECT}"
            "${SLLM_MINISTRAL3_YARN_G1_KERNEL_OBJECT}"
            "${SLLM_GEMMA_ATTENTION_G1_KERNEL_OBJECT}"
        )
    else()
        set_source_files_properties(
            src/minimax_m3_moe_route_public_runtime.hip.cpp
            PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/rmsnorm_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/elementwise_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/embedding_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/matmul_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/gdn_projection_bundle_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/mlp_gate_up_silu_bundle_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/argmax_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/token_selector_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/moe_route_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/deepseek_v4_moe_route_kernel.hip.cpp
                                    PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/minimax_m3_moe_route_kernel.hip.cpp
                                    PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/moe_expert_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/attention_preprocess_kernel.hip.cpp
                                    PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/kv_state_kernel.hip.cpp PROPERTIES
                                    LANGUAGE HIP)
        set_source_files_properties(src/causal_attention_kernel.hip.cpp PROPERTIES
                                    LANGUAGE HIP)
        set_source_files_properties(src/linear_attention_kernel.hip.cpp PROPERTIES
                                    LANGUAGE HIP)
        set_source_files_properties(src/rotary_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/ministral3_yarn_kernel.hip.cpp
                                    PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/gemma_attention_kernel.hip.cpp PROPERTIES
                                    LANGUAGE HIP)
        list(APPEND SLLM_HIP_NATIVE_SOURCES
            src/minimax_m3_moe_route_public_runtime.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/rmsnorm_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/elementwise_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/embedding_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/matmul_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/gdn_projection_bundle_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/mlp_gate_up_silu_bundle_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/argmax_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/token_selector_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/moe_route_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES
            src/deepseek_v4_moe_route_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES
            src/minimax_m3_moe_route_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/moe_expert_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES
            src/attention_preprocess_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/kv_state_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/causal_attention_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/linear_attention_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/rotary_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/ministral3_yarn_kernel.hip.cpp)
        list(APPEND SLLM_HIP_NATIVE_SOURCES src/gemma_attention_kernel.hip.cpp)
    endif()
endif()

add_library(sllm_hip_stub STATIC ${SLLM_HIP_NATIVE_SOURCES})

target_compile_definitions(sllm_hip_stub PRIVATE
    SLLM_ENABLE_PHASE54_KV_RESEARCH=${SLLM_PHASE54_KV_RESEARCH_ENABLED}
)

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

if(SLLM_ENABLE_HIP_RUNTIME OR SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
    target_include_directories(sllm_hip_stub PRIVATE
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_hip_stub PRIVATE
        "${ROCM_PATH}/lib/libamdhip64.so"
    )
    if(SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
        target_link_libraries(sllm_hip_stub PRIVATE
            "${ROCM_PATH}/lib/libhipblas.so"
            "${ROCM_PATH}/lib/libhipblaslt.so"
            "${ROCM_PATH}/lib/librocblas.so"
        )
    endif()
endif()

if(SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
    target_compile_definitions(sllm_hip_stub PRIVATE
        SLLM_ENABLE_PUBLIC_HIP_RUNTIME=1
    )
endif()

if(SLLM_ENABLE_HIP_RUNTIME OR SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
    target_compile_definitions(sllm_hip_stub PRIVATE
        SLLM_HIP_COMPILE_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        SLLM_HIP_COMPILE_WAVE64=${SLLM_COMPILE_WAVE64}
    )
    target_compile_options(sllm_hip_stub PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:${SLLM_WAVEFRONT_FLAG}>
        $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    )
endif()

if(SLLM_ENABLE_HIP_COMPILE_PROBE)
    set(SLLM_HIP_PROBE_SOURCE src/hip_compile_probe.hip.cpp)
    set_source_files_properties(${SLLM_HIP_PROBE_SOURCE} PROPERTIES LANGUAGE HIP)
    add_library(sllm_hip_compile_probe_object OBJECT ${SLLM_HIP_PROBE_SOURCE})
    target_compile_features(sllm_hip_compile_probe_object PRIVATE cxx_std_17)
    target_compile_options(sllm_hip_compile_probe_object PRIVATE
        -mcode-object-version=6
        ${SLLM_WAVEFRONT_FLAG}
    )
    add_executable(sllm_hip_compile_probe_link
        $<TARGET_OBJECTS:sllm_hip_compile_probe_object>
    )
    set_target_properties(sllm_hip_compile_probe_link PROPERTIES
        CXX_EXTENSIONS OFF
        LINKER_LANGUAGE HIP
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_NAME "hip-compile-probe-${SLLM_HIP_COMPILE_TARGET}"
        SUFFIX ".elf"
    )
    target_link_options(sllm_hip_compile_probe_link PRIVATE
        -mcode-object-version=6
        ${SLLM_WAVEFRONT_FLAG}
    )
    target_include_directories(sllm_hip_compile_probe_link PRIVATE
        "${ROCM_PATH}/include"
    )
    target_include_directories(sllm_hip_compile_probe_object PRIVATE
        "${ROCM_PATH}/include"
    )

    set(SLLM_HIP_PROBE_OBJECT_OUTPUT
        "${CMAKE_BINARY_DIR}/hip-compile-probe-${SLLM_HIP_COMPILE_TARGET}.o"
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

if(SLLM_ENABLE_PUBLIC_RUNTIME_HOST_TEST)
    enable_testing()
    find_package(Threads REQUIRED)
    add_executable(sllm_public_runtime_host_test
        src/minimax_m3_moe_route_public_runtime.hip.cpp
        src/embedding_api.cpp
        src/elementwise_api.cpp
        src/matmul_api.cpp
        src/argmax_api.cpp
        src/token_selector_api.cpp
        src/token_selector_kernel.hip.cpp
        src/moe_route_api.cpp
        src/deepseek_v4_moe_route_api.cpp
        src/minimax_m3_moe_route_api.cpp
        src/moe_expert_api.cpp
        src/rmsnorm_api.cpp
        src/residual_rmsnorm_api.cpp
        src/attention_preprocess_api.cpp
        src/rotary_api.cpp
        src/ministral3_yarn_api.cpp
        src/windowed_attention_api.cpp
        src/causal_attention_api.cpp
        src/kv_state_api.cpp
        src/linear_attention_api.cpp
        tests/fake_hip_runtime.cpp
        tests/public_runtime_host_test.cpp
    )
    set_source_files_properties(
        src/minimax_m3_moe_route_public_runtime.hip.cpp PROPERTIES
        LANGUAGE CXX
    )
    target_compile_features(sllm_public_runtime_host_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_public_runtime_host_test PRIVATE
        SLLM_PUBLIC_RUNTIME_HOST_TEST=1
    )
    target_include_directories(sllm_public_runtime_host_test PRIVATE
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/tests/fake_hip/include"
    )
    target_compile_options(sllm_public_runtime_host_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Werror
    )
    target_link_libraries(sllm_public_runtime_host_test PRIVATE Threads::Threads)
    add_test(NAME sllm_public_runtime_host_test
        COMMAND sllm_public_runtime_host_test
    )

    add_executable(sllm_ministral3_yarn_public_c_abi_host_test
        src/minimax_m3_moe_route_public_runtime.hip.cpp
        src/embedding_api.cpp
        src/elementwise_api.cpp
        src/matmul_api.cpp
        src/argmax_api.cpp
        src/token_selector_api.cpp
        src/token_selector_kernel.hip.cpp
        src/moe_route_api.cpp
        src/deepseek_v4_moe_route_api.cpp
        src/minimax_m3_moe_route_api.cpp
        src/moe_expert_api.cpp
        src/rmsnorm_api.cpp
        src/residual_rmsnorm_api.cpp
        src/attention_preprocess_api.cpp
        src/rotary_api.cpp
        src/ministral3_yarn_api.cpp
        src/windowed_attention_api.cpp
        src/causal_attention_api.cpp
        src/kv_state_api.cpp
        src/linear_attention_api.cpp
        tests/fake_hip_runtime.cpp
        tests/ministral3_yarn_public_c_abi_test.cpp
    )
    set_source_files_properties(
        src/minimax_m3_moe_route_public_runtime.hip.cpp PROPERTIES
        LANGUAGE CXX
    )
    target_compile_features(sllm_ministral3_yarn_public_c_abi_host_test PRIVATE
        cxx_std_17
    )
    target_compile_definitions(sllm_ministral3_yarn_public_c_abi_host_test PRIVATE
        SLLM_PUBLIC_RUNTIME_HOST_TEST=1
    )
    target_include_directories(sllm_ministral3_yarn_public_c_abi_host_test PRIVATE
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/tests/fake_hip/include"
    )
    target_compile_options(sllm_ministral3_yarn_public_c_abi_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_ministral3_yarn_public_c_abi_host_test PRIVATE
        Threads::Threads
    )
    add_test(NAME sllm_ministral3_yarn_public_c_abi_host_test
        COMMAND sllm_ministral3_yarn_public_c_abi_host_test
    )

    add_executable(sllm_argmax_public_c_abi_host_test
        src/minimax_m3_moe_route_public_runtime.hip.cpp
        src/embedding_api.cpp
        src/elementwise_api.cpp
        src/matmul_api.cpp
        src/argmax_api.cpp
        src/token_selector_api.cpp
        src/token_selector_kernel.hip.cpp
        src/moe_route_api.cpp
        src/deepseek_v4_moe_route_api.cpp
        src/minimax_m3_moe_route_api.cpp
        src/moe_expert_api.cpp
        src/rmsnorm_api.cpp
        src/residual_rmsnorm_api.cpp
        src/attention_preprocess_api.cpp
        src/rotary_api.cpp
        src/ministral3_yarn_api.cpp
        src/windowed_attention_api.cpp
        src/causal_attention_api.cpp
        src/kv_state_api.cpp
        src/linear_attention_api.cpp
        tests/fake_hip_runtime.cpp
        tests/argmax_public_c_abi_test.cpp
    )
    set_source_files_properties(
        src/minimax_m3_moe_route_public_runtime.hip.cpp PROPERTIES
        LANGUAGE CXX
    )
    target_compile_features(sllm_argmax_public_c_abi_host_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_argmax_public_c_abi_host_test PRIVATE
        SLLM_PUBLIC_RUNTIME_HOST_TEST=1
    )
    target_include_directories(sllm_argmax_public_c_abi_host_test PRIVATE
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/tests/fake_hip/include"
    )
    target_compile_options(sllm_argmax_public_c_abi_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_argmax_public_c_abi_host_test PRIVATE Threads::Threads)
    add_test(NAME sllm_argmax_public_c_abi_host_test
        COMMAND sllm_argmax_public_c_abi_host_test
    )

    add_executable(sllm_token_selector_public_c_abi_host_test
        src/minimax_m3_moe_route_public_runtime.hip.cpp
        src/embedding_api.cpp
        src/elementwise_api.cpp
        src/matmul_api.cpp
        src/argmax_api.cpp
        src/token_selector_api.cpp
        src/token_selector_kernel.hip.cpp
        src/moe_route_api.cpp
        src/deepseek_v4_moe_route_api.cpp
        src/minimax_m3_moe_route_api.cpp
        src/moe_expert_api.cpp
        src/rmsnorm_api.cpp
        src/residual_rmsnorm_api.cpp
        src/attention_preprocess_api.cpp
        src/rotary_api.cpp
        src/ministral3_yarn_api.cpp
        src/windowed_attention_api.cpp
        src/causal_attention_api.cpp
        src/kv_state_api.cpp
        src/linear_attention_api.cpp
        tests/fake_hip_runtime.cpp
        tests/token_selector_public_c_abi_test.cpp
    )
    set_source_files_properties(src/token_selector_kernel.hip.cpp PROPERTIES
        LANGUAGE CXX
    )
    target_compile_features(sllm_token_selector_public_c_abi_host_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_token_selector_public_c_abi_host_test PRIVATE
        SLLM_PUBLIC_RUNTIME_HOST_TEST=1
    )
    target_include_directories(sllm_token_selector_public_c_abi_host_test PRIVATE
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/tests/fake_hip/include"
    )
    target_compile_options(sllm_token_selector_public_c_abi_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_token_selector_public_c_abi_host_test PRIVATE Threads::Threads)
    add_test(NAME sllm_token_selector_public_c_abi_host_test
        COMMAND sllm_token_selector_public_c_abi_host_test
    )

    add_executable(sllm_private_evidence_stub_test
        tests/private_evidence_stub_test.cpp
    )
    target_compile_features(sllm_private_evidence_stub_test PRIVATE cxx_std_17)
    target_include_directories(sllm_private_evidence_stub_test PRIVATE
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/src"
    )
    target_compile_options(sllm_private_evidence_stub_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Werror
    )
    target_link_libraries(sllm_private_evidence_stub_test PRIVATE
        sllm_hip_stub Threads::Threads
    )
    add_test(NAME sllm_private_evidence_stub_test
        COMMAND sllm_private_evidence_stub_test
    )
endif()

if(SLLM_ENABLE_LINEAR_ATTENTION_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "linear-attention GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_linear_attention_gpu_test
        tests/linear_attention_gpu_test.cpp
    )
    target_compile_features(sllm_linear_attention_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_linear_attention_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_linear_attention_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_linear_attention_gpu_test PRIVATE
        sllm_hip_stub
    )
endif()

if(SLLM_ENABLE_ARGMAX_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "argmax GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_argmax_gpu_test
        tests/argmax_gpu_test.cpp
    )
    target_compile_features(sllm_argmax_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_argmax_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_argmax_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_argmax_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_TOKEN_SELECTOR_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "token-selector GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_token_selector_gpu_test
        tests/token_selector_gpu_test.cpp
    )
    target_compile_features(sllm_token_selector_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_token_selector_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_token_selector_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_token_selector_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_MOE_ROUTE_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "MoE route GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_moe_route_gpu_test
        tests/moe_route_gpu_test.cpp
    )
    target_compile_features(sllm_moe_route_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_moe_route_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_moe_route_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_moe_route_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_GEMMA4_MOE_ROUTE_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "Phase55 Gemma4 MoE route GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_phase55_gemma4_moe_route_gpu_test
        tests/phase55_gemma4_moe_route_gpu_test.cpp
    )
    target_compile_features(sllm_phase55_gemma4_moe_route_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_phase55_gemma4_moe_route_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_phase55_gemma4_moe_route_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_include_directories(sllm_phase55_gemma4_moe_route_gpu_test SYSTEM PRIVATE
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_phase55_gemma4_moe_route_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_DEEPSEEK_V4_MOE_ROUTE_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "Phase57 DeepSeek V4 MoE route GPU test requires the normal public HIP runtime build")
    endif()
    enable_testing()
    add_executable(sllm_deepseek_v4_moe_route_gpu_test
        tests/deepseek_v4_moe_route_gpu_test.cpp
    )
    target_compile_features(sllm_deepseek_v4_moe_route_gpu_test PRIVATE
        cxx_std_17)
    target_compile_definitions(sllm_deepseek_v4_moe_route_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_deepseek_v4_moe_route_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
        -Werror
    )
    target_link_libraries(sllm_deepseek_v4_moe_route_gpu_test PRIVATE
        sllm_hip_stub)
    add_test(NAME sllm_deepseek_v4_moe_route_gpu_test
        COMMAND sllm_deepseek_v4_moe_route_gpu_test)
    set_tests_properties(sllm_deepseek_v4_moe_route_gpu_test PROPERTIES
        LABELS "gpu;phase57;deepseek-v4;moe-route"
        RESOURCE_LOCK sllm_gpu)
endif()

if(SLLM_ENABLE_MINIMAX_M3_MOE_ROUTE_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "Phase58 MiniMax M3 MoE route GPU test requires the normal public HIP runtime build")
    endif()
    enable_testing()
    add_executable(sllm_minimax_m3_moe_route_gpu_test
        tests/minimax_m3_moe_route_gpu_test.cpp
    )
    target_compile_features(sllm_minimax_m3_moe_route_gpu_test PRIVATE
        cxx_std_17)
    target_compile_definitions(sllm_minimax_m3_moe_route_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_minimax_m3_moe_route_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
        -Werror
    )
    target_link_libraries(sllm_minimax_m3_moe_route_gpu_test PRIVATE
        sllm_hip_stub)
    add_test(NAME sllm_minimax_m3_moe_route_gpu_test
        COMMAND sllm_minimax_m3_moe_route_gpu_test)
    set_tests_properties(sllm_minimax_m3_moe_route_gpu_test PROPERTIES
        LABELS "gpu;phase58;minimax-m3;moe-route"
        RESOURCE_LOCK sllm_gpu)
endif()

if(SLLM_ENABLE_MOE_EXPERT_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "MoE expert GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_moe_expert_gpu_test tests/moe_expert_gpu_test.cpp)
    target_compile_features(sllm_moe_expert_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_moe_expert_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}")
    target_compile_options(sllm_moe_expert_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror)
    target_link_libraries(sllm_moe_expert_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_RMSNORM_DIRECT_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "direct-scale RMSNorm GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_rmsnorm_direct_gpu_test
        tests/rmsnorm_direct_gpu_test.cpp
    )
    target_compile_features(sllm_rmsnorm_direct_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_rmsnorm_direct_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_rmsnorm_direct_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_rmsnorm_direct_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST)
    add_executable(sllm_low_precision_codec_gpu_test
        tests/low_precision_block_codec_gpu_test.hip.cpp
    )
    set_source_files_properties(
        tests/low_precision_block_codec_gpu_test.hip.cpp
        PROPERTIES LANGUAGE HIP
    )
    target_compile_features(sllm_low_precision_codec_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_low_precision_codec_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_low_precision_codec_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:${SLLM_WAVEFRONT_FLAG}>
        $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    )
    target_include_directories(sllm_low_precision_codec_gpu_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_low_precision_codec_gpu_test PRIVATE
        "${ROCM_PATH}/lib/libamdhip64.so"
    )
endif()

if(SLLM_ENABLE_ROTARY_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "split-half rotary GPU test requires the normal public HIP runtime build")
    endif()
    set_source_files_properties(src/rotary_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
    add_executable(sllm_rotary_gpu_test
        tests/rotary_gpu_test.cpp
        src/rotary_kernel.hip.cpp
    )
    target_compile_features(sllm_rotary_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_rotary_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_rotary_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:${SLLM_WAVEFRONT_FLAG}>
        $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    )
    target_include_directories(sllm_rotary_gpu_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_rotary_gpu_test PRIVATE
        sllm_hip_stub
        "${ROCM_PATH}/lib/libamdhip64.so"
    )
endif()

if(SLLM_ENABLE_ROTARY_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "public split-half rotary GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_rotary_public_gpu_test
        tests/rotary_public_gpu_test.cpp
    )
    target_compile_features(sllm_rotary_public_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_rotary_public_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_rotary_public_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_rotary_public_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_MINISTRAL3_YARN_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "public Ministral 3 YaRN GPU test requires the normal public HIP runtime build")
    endif()
    enable_testing()
    add_executable(sllm_ministral3_yarn_public_gpu_test
        tests/ministral3_yarn_public_c_abi_test.cpp
    )
    target_compile_features(sllm_ministral3_yarn_public_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_ministral3_yarn_public_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_ministral3_yarn_public_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_ministral3_yarn_public_gpu_test PRIVATE sllm_hip_stub)
    add_test(NAME sllm_ministral3_yarn_public_gpu_test
        COMMAND sllm_ministral3_yarn_public_gpu_test
    )
    set_tests_properties(sllm_ministral3_yarn_public_gpu_test PROPERTIES
        LABELS "gpu;phase60;ministral3;yarn"
        RESOURCE_LOCK sllm_gpu
        ENVIRONMENT "LD_LIBRARY_PATH=${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    )
endif()

if(SLLM_ENABLE_BROADCAST_ADD_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "public BroadcastAdd GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_broadcast_add_public_gpu_test
        tests/phase45_broadcast_add_gpu_test.cpp
    )
    target_compile_features(sllm_broadcast_add_public_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_broadcast_add_public_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_broadcast_add_public_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_include_directories(sllm_broadcast_add_public_gpu_test PRIVATE
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_broadcast_add_public_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_BROADCAST_MUL_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "public BroadcastMul GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_broadcast_mul_public_gpu_test
        tests/phase55_broadcast_mul_gpu_test.cpp
    )
    target_compile_features(sllm_broadcast_mul_public_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_broadcast_mul_public_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_broadcast_mul_public_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_include_directories(sllm_broadcast_mul_public_gpu_test SYSTEM PRIVATE
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_broadcast_mul_public_gpu_test PRIVATE sllm_hip_stub)
endif()

if(SLLM_ENABLE_GEMMA_ATTENTION_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "Gemma attention GPU test requires the normal public HIP runtime build")
    endif()
    set_source_files_properties(src/gemma_attention_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
    add_executable(sllm_gemma_attention_gpu_test
        tests/gemma_attention_gpu_test.cpp
        src/gemma_attention_kernel.hip.cpp
    )
    target_compile_features(sllm_gemma_attention_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_gemma_attention_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(sllm_gemma_attention_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
        -ffp-contract=off
        $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
        $<$<COMPILE_LANGUAGE:HIP>:${SLLM_WAVEFRONT_FLAG}>
    )
    target_include_directories(sllm_gemma_attention_gpu_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${ROCM_PATH}/include"
    )
    target_link_libraries(sllm_gemma_attention_gpu_test PRIVATE
        sllm_hip_stub
        "${ROCM_PATH}/lib/libamdhip64.so"
    )
endif()

if(SLLM_ENABLE_GEMMA_ATTENTION_PUBLIC_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "public Gemma attention GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_gemma_attention_public_gpu_test
        tests/gemma_attention_public_gpu_test.cpp
    )
    target_compile_features(sllm_gemma_attention_public_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_gemma_attention_public_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_gemma_attention_public_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_gemma_attention_public_gpu_test PRIVATE
        sllm_hip_stub
    )
endif()

if(SLLM_ENABLE_PHASE41_STATE_GPU_TEST)
    if(NOT SLLM_ENABLE_PUBLIC_HIP_RUNTIME OR SLLM_SEMANTIC_G1_AUTHORITY)
        message(FATAL_ERROR
            "Phase 41 state GPU test requires the normal public HIP runtime build")
    endif()
    add_executable(sllm_phase41_state_gpu_test
        tests/phase41_state_gpu_test.cpp
    )
    target_compile_features(sllm_phase41_state_gpu_test PRIVATE cxx_std_17)
    target_compile_definitions(sllm_phase41_state_gpu_test PRIVATE
        SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"
    )
    target_compile_options(sllm_phase41_state_gpu_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    target_link_libraries(sllm_phase41_state_gpu_test PRIVATE sllm_hip_stub)
endif()
