#include "decode_control_kernel_internal.hpp"
#include "linear_attention_kernel_internal.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    const auto status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(status));      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int run_width(const char *const expected_target,
              const unsigned checkpoint_rows) {
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (expected_target == nullptr ||
      std::strstr(properties.gcnArchName, expected_target) !=
          properties.gcnArchName)
    return 9;
  const auto target_length = std::strlen(expected_target);
  if (properties.gcnArchName[target_length] != '\0' &&
      properties.gcnArchName[target_length] != ':')
    return 9;
  if (checkpoint_rows < 2U || checkpoint_rows > 4U) {
    return 9;
  }
  constexpr unsigned conv_size = 259, recurrent_size = 257;
  const unsigned rows = checkpoint_rows + 1U;
  std::vector<uint16_t> conv(conv_size * checkpoint_rows),
      anchor_conv_seed(conv_size), other_conv_seed(conv_size),
      observed_anchor_conv(conv_size), observed_other_conv(conv_size);
  std::vector<float> recurrent(recurrent_size * checkpoint_rows),
      anchor_recurrent_seed(recurrent_size),
      other_recurrent_seed(recurrent_size),
      observed_anchor_recurrent(recurrent_size),
      observed_other_recurrent(recurrent_size);
  for (unsigned i = 0; i < conv.size(); ++i)
    conv[i] = static_cast<uint16_t>(i + 1);
  for (unsigned i = 0; i < recurrent.size(); ++i)
    recurrent[i] = static_cast<float>(i + 1);
  for (unsigned i = 0; i < conv_size; ++i) {
    anchor_conv_seed[i] = static_cast<uint16_t>(0x5000U + i);
    other_conv_seed[i] = static_cast<uint16_t>(0x6000U + i);
  }
  for (unsigned i = 0; i < recurrent_size; ++i) {
    anchor_recurrent_seed[i] = 5000.0F + static_cast<float>(i);
    other_recurrent_seed[i] = 6000.0F + static_cast<float>(i);
  }
  uint16_t *checkpoint_conv, *other_conv, *anchor_conv;
  float *checkpoint_recurrent, *other_recurrent, *anchor_recurrent;
  sllm_decode_control::ControlV1 *control;
  HIP_CHECK(hipMalloc(&checkpoint_conv, conv.size() * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&checkpoint_recurrent, recurrent.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&other_conv, conv_size * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&other_recurrent, recurrent_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&anchor_conv, conv_size * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&anchor_recurrent, recurrent_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&control, sizeof(*control)));
  HIP_CHECK(hipMemcpy(checkpoint_conv, conv.data(),
                      conv.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(checkpoint_recurrent, recurrent.data(),
                      recurrent.size() * sizeof(float), hipMemcpyHostToDevice));
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t executable;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  HIP_CHECK(sllm_linear_attention_kernel::launch_decode_state_select(
      control, other_conv, other_recurrent, checkpoint_conv,
      checkpoint_recurrent, anchor_conv, anchor_recurrent, conv_size,
      recurrent_size, rows, checkpoint_rows, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  unsigned checks = 0;
  for (unsigned generation = 1; generation <= 3; ++generation) {
    for (unsigned active_width = 1; active_width <= checkpoint_rows;
         ++active_width) {
      const unsigned active_rows = active_width + 1U;
      for (unsigned count = 0; count <= rows + 1U; ++count) {
        for (unsigned condition = 0; condition < 4; ++condition) {
          sllm_decode_control::ControlV1 host{};
          host.mode = sllm_decode_control::kModeMtp;
          host.width = checkpoint_rows;
          host.active_width = active_width;
          host.generation = generation;
          host.commit_rows = count;
          host.halted = condition == 1;
          host.status =
              condition == 2 ? static_cast<unsigned>(
                                   sllm_decode_control::Status::InvalidToken)
              : condition == 3
                  ? static_cast<unsigned>(sllm_decode_control::Status::Noop)
                  : static_cast<unsigned>(sllm_decode_control::Status::Ok);
          HIP_CHECK(hipMemcpyAsync(control, &host, sizeof(host),
                                   hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipMemcpyAsync(anchor_conv, anchor_conv_seed.data(),
                                   conv_size * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipMemcpyAsync(
              anchor_recurrent, anchor_recurrent_seed.data(),
              recurrent_size * sizeof(float), hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipMemcpyAsync(other_conv, other_conv_seed.data(),
                                   conv_size * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipMemcpyAsync(other_recurrent, other_recurrent_seed.data(),
                                   recurrent_size * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipGraphLaunch(executable, stream));
          HIP_CHECK(hipMemcpyAsync(observed_anchor_conv.data(), anchor_conv,
                                   conv_size * sizeof(uint16_t),
                                   hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipMemcpyAsync(
              observed_anchor_recurrent.data(), anchor_recurrent,
              recurrent_size * sizeof(float), hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipMemcpyAsync(observed_other_conv.data(), other_conv,
                                   conv_size * sizeof(uint16_t),
                                   hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipMemcpyAsync(
              observed_other_recurrent.data(), other_recurrent,
              recurrent_size * sizeof(float), hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipMemcpyAsync(&host, control, sizeof(host),
                                   hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipStreamSynchronize(stream));
          const bool valid_status = condition < 2;
          const bool partial = valid_status && count > 0 && count < active_rows;
          const bool copy_other = partial && (generation & 1U) != 0U;
          const bool copy_anchor = partial && !copy_other;
          for (unsigned i = 0; i < conv_size; ++i) {
            const uint16_t expected_anchor =
                copy_anchor ? conv[(count - 1) * conv_size + i]
                            : anchor_conv_seed[i];
            const uint16_t expected_other =
                copy_other ? conv[(count - 1) * conv_size + i]
                           : other_conv_seed[i];
            if (observed_anchor_conv[i] != expected_anchor ||
                observed_other_conv[i] != expected_other)
              return 2;
          }
          for (unsigned i = 0; i < recurrent_size; ++i) {
            const float expected_anchor =
                copy_anchor ? recurrent[(count - 1) * recurrent_size + i]
                            : anchor_recurrent_seed[i];
            const float expected_other =
                copy_other ? recurrent[(count - 1) * recurrent_size + i]
                           : other_recurrent_seed[i];
            if (observed_anchor_recurrent[i] != expected_anchor ||
                observed_other_recurrent[i] != expected_other)
              return 3;
          }
          if (count > active_rows && valid_status &&
              (host.status !=
                   static_cast<unsigned>(
                       sllm_decode_control::Status::InvalidDecision) ||
               !host.halted))
            return 4;
          ++checks;
        }
      }
    }
  }
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(control));
  HIP_CHECK(hipFree(anchor_recurrent));
  HIP_CHECK(hipFree(anchor_conv));
  HIP_CHECK(hipFree(other_recurrent));
  HIP_CHECK(hipFree(other_conv));
  HIP_CHECK(hipFree(checkpoint_recurrent));
  HIP_CHECK(hipFree(checkpoint_conv));
  std::printf(
      "linear_decode_state_select status=PASS width=%u checks=%u target=%s\n",
      checkpoint_rows, checks, properties.gcnArchName);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    return 9;
  }
  for (unsigned width = 2U; width <= 4U; ++width) {
    const int status = run_width(argv[1], width);
    if (status != 0) {
      return status;
    }
  }
  return 0;
}
