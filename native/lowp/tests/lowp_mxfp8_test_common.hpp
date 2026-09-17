#ifndef SLLM_LOWP_MXFP8_TEST_COMMON_HPP
#define SLLM_LOWP_MXFP8_TEST_COMMON_HPP

#include "lowp_test_common.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lowp_test {

struct Mxfp8Fixture final {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  std::vector<uint16_t> activation;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
};

inline Mxfp8Fixture make_mxfp8_fixture(const uint64_t m, const uint64_t n,
                                       const uint64_t k) {
  Mxfp8Fixture result{m,
                      n,
                      k,
                      std::vector<uint16_t>(m * k),
                      std::vector<uint8_t>(n * k, UINT8_C(0x38)),
                      std::vector<uint8_t>(n * (k / 32U), UINT8_C(127))};
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < k; ++column) {
      const float value = ((column + row * 3U) % 11U == 0U)
                              ? -2.0F
                              : ((column + row) % 5U == 0U ? 0.5F : 1.0F);
      result.activation[row * k + column] = f32_to_bf16_rne(value);
    }
  }
  return result;
}

inline std::vector<uint16_t>
oracle_output(const Mxfp8Fixture &fixture,
              const std::vector<uint8_t> &activation,
              const std::vector<uint8_t> &activation_scales) {
  const uint64_t blocks = fixture.k / 32U;
  std::vector<uint16_t> result(fixture.m * fixture.n);
  for (uint64_t row = 0U; row < fixture.m; ++row) {
    for (uint64_t column = 0U; column < fixture.n; ++column) {
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < fixture.k; ++inner) {
        const uint64_t block = inner / 32U;
        const float left = decode_e4m3fn(activation[row * fixture.k + inner]) *
                           decode_e8m0(activation_scales[row * blocks + block]);
        const float right =
            decode_e4m3fn(fixture.weight[column * fixture.k + inner]) *
            decode_e8m0(fixture.weight_scales[column * blocks + block]);
        accumulator += left * right;
      }
      result[row * fixture.n + column] = f32_to_bf16_rne(accumulator);
    }
  }
  return result;
}

inline bool check_output(const std::vector<uint16_t> &actual,
                         const std::vector<uint16_t> &expected,
                         const char *const label) {
  if (actual.size() != expected.size()) {
    std::cerr << label << " output size mismatch\n";
    return false;
  }
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (!finite_bf16(actual[index]) || actual[index] != expected[index]) {
      std::cerr << label << " oracle mismatch index=" << index << " actual=0x"
                << std::hex << actual[index] << " expected=0x"
                << expected[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

} // namespace lowp_test

#endif
