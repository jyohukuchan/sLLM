// Bounded Phase 9 HIP Graph probe. This is not linked into sLLM.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct AffineParameters {
  const float *input;
  float *output;
  float scale;
  float bias;
  uint32_t count;
  uint32_t generation;
};

struct MixedParameters {
  const float *gemv_output;
  float *output;
  float scale;
  uint32_t count;
  uint32_t generation;
};

__global__ void dynamic_affine(const AffineParameters *parameters) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < parameters->count) {
    parameters->output[index] =
        parameters->input[index] * parameters->scale + parameters->bias +
        static_cast<float>(parameters->generation) * 0.001F;
  }
}

__global__ void dynamic_mixed_epilogue(const MixedParameters *parameters) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < parameters->count) {
    parameters->output[index] =
        parameters->gemv_output[index] * parameters->scale +
        static_cast<float>(parameters->generation) * 0.001F;
  }
}

bool hip_ok(const hipError_t status, const char *const label) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", label, hipGetErrorString(status));
  return false;
}

bool blas_ok(const hipblasStatus_t status, const char *const label) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s: hipBLAS status %d\n", label,
               static_cast<int>(status));
  return false;
}

double elapsed_ns(const std::chrono::steady_clock::time_point begin,
                  const std::chrono::steady_clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

float max_error(const std::vector<float> &actual,
                const std::vector<float> &expected) {
  float result = 0.0F;
  for (size_t index = 0; index != actual.size(); ++index) {
    result = std::max(result, std::fabs(actual[index] - expected[index]));
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2 || (std::strcmp(argv[1], "gfx1030") != 0 &&
                    std::strcmp(argv[1], "gfx1201") != 0)) {
    std::fprintf(stderr, "usage: phase9_hip_graph_poc gfx1030|gfx1201\n");
    return 2;
  }

  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties") ||
      std::strcmp(properties.gcnArchName, argv[1]) != 0) {
    std::fprintf(stderr, "visible device is not the requested exact target\n");
    return 1;
  }

  constexpr uint32_t count = 257U;
  constexpr int gemv_rows = 257;
  constexpr int gemv_columns = 33;
  constexpr int repetitions = 64;
  const size_t vector_bytes = count * sizeof(float);
  const size_t matrix_bytes =
      static_cast<size_t>(gemv_rows) * gemv_columns * sizeof(float);

  std::vector<float> input_a(count);
  std::vector<float> input_b(count);
  std::vector<float> matrix(static_cast<size_t>(gemv_rows) * gemv_columns);
  std::vector<float> gemv_input(gemv_columns);
  for (uint32_t index = 0; index != count; ++index) {
    input_a[index] =
        static_cast<float>(static_cast<int>(index % 19U) - 9) / 16.0F;
    input_b[index] =
        static_cast<float>(static_cast<int>(index % 23U) - 11) / 32.0F;
  }
  for (size_t index = 0; index != matrix.size(); ++index) {
    matrix[index] =
        static_cast<float>(static_cast<int>(index % 17U) - 8) / 64.0F;
  }
  for (int index = 0; index != gemv_columns; ++index) {
    gemv_input[index] = static_cast<float>(index - 16) / 32.0F;
  }

  hipStream_t stream = nullptr;
  hipblasHandle_t handle = nullptr;
  float *device_input_a = nullptr;
  float *device_input_b = nullptr;
  float *device_output_a = nullptr;
  float *device_output_b = nullptr;
  float *device_matrix = nullptr;
  float *device_gemv_input = nullptr;
  float *device_gemv_output = nullptr;
  AffineParameters *device_affine_parameters = nullptr;
  MixedParameters *device_mixed_parameters = nullptr;
  hipGraph_t kernel_graph = nullptr;
  hipGraphExec_t kernel_executable = nullptr;
  hipGraph_t mixed_graph = nullptr;
  hipGraphExec_t mixed_executable = nullptr;

  bool ok =
      hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
             "stream create") &&
      blas_ok(hipblasCreate(&handle), "hipblas create") &&
      blas_ok(hipblasSetStream(handle, stream), "hipblas set stream") &&
      hip_ok(hipMalloc(&device_input_a, vector_bytes), "input a") &&
      hip_ok(hipMalloc(&device_input_b, vector_bytes), "input b") &&
      hip_ok(hipMalloc(&device_output_a, vector_bytes), "output a") &&
      hip_ok(hipMalloc(&device_output_b, vector_bytes), "output b") &&
      hip_ok(hipMalloc(&device_matrix, matrix_bytes), "matrix") &&
      hip_ok(hipMalloc(&device_gemv_input, gemv_columns * sizeof(float)),
             "gemv input") &&
      hip_ok(hipMalloc(&device_gemv_output, vector_bytes), "gemv output") &&
      hip_ok(hipMalloc(&device_affine_parameters, sizeof(AffineParameters)),
             "affine parameters") &&
      hip_ok(hipMalloc(&device_mixed_parameters, sizeof(MixedParameters)),
             "mixed parameters") &&
      hip_ok(hipMemcpyAsync(device_input_a, input_a.data(), vector_bytes,
                            hipMemcpyHostToDevice, stream),
             "input a upload") &&
      hip_ok(hipMemcpyAsync(device_input_b, input_b.data(), vector_bytes,
                            hipMemcpyHostToDevice, stream),
             "input b upload") &&
      hip_ok(hipMemcpyAsync(device_matrix, matrix.data(), matrix_bytes,
                            hipMemcpyHostToDevice, stream),
             "matrix upload") &&
      hip_ok(hipMemcpyAsync(device_gemv_input, gemv_input.data(),
                            gemv_columns * sizeof(float), hipMemcpyHostToDevice,
                            stream),
             "gemv input upload") &&
      hip_ok(hipStreamSynchronize(stream), "upload synchronize");

  const dim3 block(128U);
  const dim3 grid((count + block.x - 1U) / block.x);
  AffineParameters affine_parameters{device_input_a, device_output_a, 1.0F,
                                     0.0F,           count,           0U};
  if (ok) {
    ok = hip_ok(hipMemcpyAsync(device_affine_parameters, &affine_parameters,
                               sizeof(affine_parameters), hipMemcpyHostToDevice,
                               stream),
                "affine warmup parameters");
    if (ok) {
      hipLaunchKernelGGL(dynamic_affine, grid, block, 0, stream,
                         device_affine_parameters);
      ok = hip_ok(hipGetLastError(), "affine warmup launch") &&
           hip_ok(hipStreamSynchronize(stream), "affine warmup synchronize");
    }
  }

  const auto kernel_capture_begin = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                "kernel capture begin");
  }
  if (ok) {
    hipLaunchKernelGGL(dynamic_affine, grid, block, 0, stream,
                       device_affine_parameters);
    ok = hip_ok(hipGetLastError(), "kernel capture launch") &&
         hip_ok(hipStreamEndCapture(stream, &kernel_graph),
                "kernel capture end");
  }
  const auto kernel_capture_end = std::chrono::steady_clock::now();
  const auto kernel_instantiate_begin = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipGraphInstantiate(&kernel_executable, kernel_graph, nullptr,
                                    nullptr, 0),
                "kernel graph instantiate");
  }
  const auto kernel_instantiate_end = std::chrono::steady_clock::now();
  size_t kernel_nodes = 0U;
  if (ok) {
    ok = hip_ok(hipGraphGetNodes(kernel_graph, nullptr, &kernel_nodes),
                "kernel graph node count");
  }

  std::vector<float> actual(count);
  std::vector<float> expected(count);
  float kernel_error = 0.0F;
  const auto kernel_replay_begin = std::chrono::steady_clock::now();
  for (int repetition = 0; ok && repetition != repetitions; ++repetition) {
    const bool alternate = (repetition & 1) != 0;
    affine_parameters =
        AffineParameters{alternate ? device_input_b : device_input_a,
                         alternate ? device_output_b : device_output_a,
                         0.75F + 0.01F * static_cast<float>(repetition),
                         -0.125F,
                         count,
                         static_cast<uint32_t>(repetition + 1)};
    ok = hip_ok(hipMemcpyAsync(device_affine_parameters, &affine_parameters,
                               sizeof(affine_parameters), hipMemcpyHostToDevice,
                               stream),
                "affine replay parameters") &&
         hip_ok(hipGraphLaunch(kernel_executable, stream),
                "kernel graph replay");
  }
  if (ok) {
    ok = hip_ok(hipStreamSynchronize(stream), "kernel replay synchronize");
  }
  const auto kernel_replay_end = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipMemcpy(actual.data(), device_output_b, vector_bytes,
                          hipMemcpyDeviceToHost),
                "kernel output readback");
  }
  if (ok) {
    const int repetition = repetitions - 1;
    for (uint32_t index = 0; index != count; ++index) {
      expected[index] =
          input_b[index] * (0.75F + 0.01F * static_cast<float>(repetition)) -
          0.125F + static_cast<float>(repetition + 1) * 0.001F;
    }
    kernel_error = max_error(actual, expected);
    ok = std::isfinite(kernel_error) && kernel_error <= 1.0e-6F;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto gemv = [&]() {
    return hipblasSgemv(handle, HIPBLAS_OP_N, gemv_rows, gemv_columns, &alpha,
                        device_matrix, gemv_rows, device_gemv_input, 1, &beta,
                        device_gemv_output, 1);
  };
  MixedParameters mixed_parameters{device_gemv_output, device_output_a, 1.0F,
                                   count, 0U};
  if (ok) {
    ok = blas_ok(gemv(), "gemv warmup") &&
         hip_ok(hipMemcpyAsync(device_mixed_parameters, &mixed_parameters,
                               sizeof(mixed_parameters), hipMemcpyHostToDevice,
                               stream),
                "mixed warmup parameters");
    if (ok) {
      hipLaunchKernelGGL(dynamic_mixed_epilogue, grid, block, 0, stream,
                         device_mixed_parameters);
      ok = hip_ok(hipGetLastError(), "mixed warmup launch") &&
           hip_ok(hipStreamSynchronize(stream), "mixed warmup synchronize");
    }
  }

  const auto mixed_capture_begin = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                "mixed capture begin") &&
         blas_ok(gemv(), "captured gemv");
  }
  if (ok) {
    hipLaunchKernelGGL(dynamic_mixed_epilogue, grid, block, 0, stream,
                       device_mixed_parameters);
    ok = hip_ok(hipGetLastError(), "mixed capture epilogue") &&
         hip_ok(hipStreamEndCapture(stream, &mixed_graph), "mixed capture end");
  }
  const auto mixed_capture_end = std::chrono::steady_clock::now();
  const auto mixed_instantiate_begin = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipGraphInstantiate(&mixed_executable, mixed_graph, nullptr,
                                    nullptr, 0),
                "mixed graph instantiate");
  }
  const auto mixed_instantiate_end = std::chrono::steady_clock::now();
  size_t mixed_nodes = 0U;
  if (ok) {
    ok = hip_ok(hipGraphGetNodes(mixed_graph, nullptr, &mixed_nodes),
                "mixed graph node count");
  }

  float mixed_error = 0.0F;
  const auto mixed_replay_begin = std::chrono::steady_clock::now();
  for (int repetition = 0; ok && repetition != repetitions; ++repetition) {
    const bool alternate = (repetition & 1) != 0;
    mixed_parameters = MixedParameters{
        device_gemv_output, alternate ? device_output_b : device_output_a,
        0.5F + 0.005F * static_cast<float>(repetition), count,
        static_cast<uint32_t>(repetition + 1)};
    ok = hip_ok(hipMemcpyAsync(device_mixed_parameters, &mixed_parameters,
                               sizeof(mixed_parameters), hipMemcpyHostToDevice,
                               stream),
                "mixed replay parameters") &&
         hip_ok(hipGraphLaunch(mixed_executable, stream), "mixed graph replay");
  }
  if (ok) {
    ok = hip_ok(hipStreamSynchronize(stream), "mixed replay synchronize");
  }
  const auto mixed_replay_end = std::chrono::steady_clock::now();
  if (ok) {
    ok = hip_ok(hipMemcpy(actual.data(), device_output_b, vector_bytes,
                          hipMemcpyDeviceToHost),
                "mixed output readback");
  }
  if (ok) {
    const int repetition = repetitions - 1;
    for (int row = 0; row != gemv_rows; ++row) {
      double sum = 0.0;
      for (int column = 0; column != gemv_columns; ++column) {
        sum += static_cast<double>(
                   matrix[static_cast<size_t>(column) * gemv_rows + row]) *
               static_cast<double>(gemv_input[column]);
      }
      expected[static_cast<size_t>(row)] =
          static_cast<float>(sum) *
              (0.5F + 0.005F * static_cast<float>(repetition)) +
          static_cast<float>(repetition + 1) * 0.001F;
    }
    mixed_error = max_error(actual, expected);
    ok = std::isfinite(mixed_error) && mixed_error <= 2.0e-5F;
  }

  bool cleanup_ok = true;
  auto cleanup_hip = [&](const hipError_t status, const char *const label) {
    if (!hip_ok(status, label)) {
      cleanup_ok = false;
    }
  };
  if (mixed_executable != nullptr) {
    cleanup_hip(hipGraphExecDestroy(mixed_executable),
                "mixed graph executable destroy");
  }
  if (mixed_graph != nullptr) {
    cleanup_hip(hipGraphDestroy(mixed_graph), "mixed graph destroy");
  }
  if (kernel_executable != nullptr) {
    cleanup_hip(hipGraphExecDestroy(kernel_executable),
                "kernel graph executable destroy");
  }
  if (kernel_graph != nullptr) {
    cleanup_hip(hipGraphDestroy(kernel_graph), "kernel graph destroy");
  }
  cleanup_hip(hipFree(device_mixed_parameters), "mixed parameters free");
  cleanup_hip(hipFree(device_affine_parameters), "affine parameters free");
  cleanup_hip(hipFree(device_gemv_output), "gemv output free");
  cleanup_hip(hipFree(device_gemv_input), "gemv input free");
  cleanup_hip(hipFree(device_matrix), "matrix free");
  cleanup_hip(hipFree(device_output_b), "output b free");
  cleanup_hip(hipFree(device_output_a), "output a free");
  cleanup_hip(hipFree(device_input_b), "input b free");
  cleanup_hip(hipFree(device_input_a), "input a free");
  if (handle != nullptr &&
      !blas_ok(hipblasDestroy(handle), "hipblas destroy")) {
    cleanup_ok = false;
  }
  if (stream != nullptr) {
    cleanup_hip(hipStreamDestroy(stream), "stream destroy");
  }
  ok = ok && cleanup_ok;

  std::printf("{\"protocol\":\"phase9-hip-graph-poc-v1\",\"state\":\"%s\","
              "\"target\":\"%s\",\"dynamic_parameter_block\":true,"
              "\"pointer_update\":true,\"scalar_update\":true,"
              "\"warmup_runs\":1,\"replays\":%d,"
              "\"kernel_only\":{\"captured\":%s,\"nodes\":%zu,"
              "\"capture_ns\":%.0f,\"instantiate_ns\":%.0f,"
              "\"average_replay_with_update_ns\":%.0f,\"max_abs_error\":%.9g},"
              "\"hipblas_mixed\":{\"captured\":%s,\"nodes\":%zu,"
              "\"capture_ns\":%.0f,\"instantiate_ns\":%.0f,"
              "\"average_replay_with_update_ns\":%.0f,\"max_abs_error\":%.9g},"
              "\"cleanup\":{\"state\":\"%s\",\"graphs_destroyed\":%s,"
              "\"resources_destroyed\":%s}}\n",
              ok ? "PASS" : "FAIL", argv[1], repetitions,
              kernel_executable != nullptr ? "true" : "false", kernel_nodes,
              elapsed_ns(kernel_capture_begin, kernel_capture_end),
              elapsed_ns(kernel_instantiate_begin, kernel_instantiate_end),
              elapsed_ns(kernel_replay_begin, kernel_replay_end) / repetitions,
              static_cast<double>(kernel_error),
              mixed_executable != nullptr ? "true" : "false", mixed_nodes,
              elapsed_ns(mixed_capture_begin, mixed_capture_end),
              elapsed_ns(mixed_instantiate_begin, mixed_instantiate_end),
              elapsed_ns(mixed_replay_begin, mixed_replay_end) / repetitions,
              static_cast<double>(mixed_error), cleanup_ok ? "PASS" : "FAIL",
              cleanup_ok ? "true" : "false", cleanup_ok ? "true" : "false");
  return ok ? 0 : 1;
}
