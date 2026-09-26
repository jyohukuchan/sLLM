// Phase 87 Stage10: paged MXFP8-E4 GQA6 whole-graph device-control test.
//
// The test includes the production translation unit so it remains a focused
// exact-target GPU oracle without adding a public ABI or a runtime dependency.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kMaxRows = 5U;
constexpr uint64_t kMaxPosition = 8193U;
constexpr uint64_t kMaxTokens = kMaxPosition + kMaxRows;
constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

template <typename T> struct DeviceBuffer final {
  T *ptr = nullptr;
  size_t count = 0U;

  DeviceBuffer() = default;
  explicit DeviceBuffer(const size_t elements) : count(elements) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)),
          "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }
};

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

double e4m3fn_decode(const uint8_t value) {
  const double sign = (value & 0x80U) != 0U ? -1.0 : 1.0;
  const uint32_t magnitude = value & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  }
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

struct Fixture final {
  const uint32_t pages =
      static_cast<uint32_t>((kMaxTokens + kPageTokens - 1U) / kPageTokens);
  const size_t page_value_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kHeadDim;
  const size_t page_scale_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * 8U;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scales;
  std::vector<uint8_t> value_scales;
  std::vector<uint8_t> physical_key;
  std::vector<uint8_t> physical_value;
  std::vector<uint8_t> physical_key_scales;
  std::vector<uint8_t> physical_value_scales;

  Fixture()
      : query(static_cast<size_t>(kMaxRows) * kQHeads * kHeadDim),
        key(static_cast<size_t>(kMaxTokens) * kKvHeads * kHeadDim),
        value(key.size()),
        key_scales(static_cast<size_t>(kMaxTokens) * kKvHeads * 8U, 0x7fU),
        value_scales(key_scales),
        physical_key(static_cast<size_t>(pages) * page_value_bytes),
        physical_value(physical_key.size()),
        physical_key_scales(static_cast<size_t>(pages) * page_scale_bytes),
        physical_value_scales(physical_key_scales) {
    for (size_t index = 0U; index < query.size(); ++index) {
      query[index] =
          f32_to_bf16(0.07F * static_cast<float>((index * 13U) % 17U) - 0.4F);
    }
    for (uint64_t token = 0U; token < kMaxTokens; ++token) {
      const uint32_t logical_page = static_cast<uint32_t>(token / 128U);
      const uint32_t local_token = static_cast<uint32_t>(token % 128U);
      const uint32_t physical_page = pages - 1U - logical_page;
      for (uint32_t head = 0U; head < kKvHeads; ++head) {
        const size_t row =
            (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
        const size_t physical_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * kHeadDim +
            static_cast<size_t>(head) * kHeadDim;
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
          const uint8_t key_code = static_cast<uint8_t>(
              0x08U + ((token * 17U + head * 5U + dimension) % 0x70U));
          const uint8_t value_code = static_cast<uint8_t>(
              0x10U + ((token * 7U + head * 11U + dimension * 3U) % 0x60U));
          key[row + dimension] = key_code;
          value[row + dimension] = value_code;
          physical_key[physical_row + dimension] = key_code;
          physical_value[physical_row + dimension] = value_code;
        }
        const size_t scale_row =
            (static_cast<size_t>(token) * kKvHeads + head) * 8U;
        const size_t physical_scale_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * 8U +
            static_cast<size_t>(head) * 8U;
        std::memcpy(physical_key_scales.data() + physical_scale_row,
                    key_scales.data() + scale_row, 8U);
        std::memcpy(physical_value_scales.data() + physical_scale_row,
                    value_scales.data() + scale_row, 8U);
      }
    }
  }
};

double oracle_dim0(const Fixture &fixture, const uint64_t position,
                   const uint32_t row) {
  const size_t query_row = static_cast<size_t>(row) * kQHeads * kHeadDim;
  double maximum = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(static_cast<size_t>(position) + 1U);
  for (uint64_t token = 0U; token <= position; ++token) {
    const size_t key_row = static_cast<size_t>(token) * kKvHeads * kHeadDim;
    double dot = 0.0;
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      dot += static_cast<double>(
                 bf16_to_f32(fixture.query[query_row + dimension])) *
             e4m3fn_decode(fixture.key[key_row + dimension]);
    }
    scores[token] = dot / 16.0;
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= position; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    const size_t value_row = static_cast<size_t>(token) * kKvHeads * kHeadDim;
    numerator += weight * e4m3fn_decode(fixture.value[value_row]);
  }
  return numerator / denominator;
}

void upload(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
            DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
            DeviceBuffer<uint8_t> &key_scales,
            DeviceBuffer<uint8_t> &value_scales,
            DeviceBuffer<uint8_t> &physical_key,
            DeviceBuffer<uint8_t> &physical_value,
            DeviceBuffer<uint8_t> &physical_key_scales,
            DeviceBuffer<uint8_t> &physical_value_scales) {
  check(hipMemcpy(query.ptr, fixture.query.data(),
                  fixture.query.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload query");
  check(hipMemcpy(key.ptr, fixture.key.data(), fixture.key.size(),
                  hipMemcpyHostToDevice),
        "upload key");
  check(hipMemcpy(value.ptr, fixture.value.data(), fixture.value.size(),
                  hipMemcpyHostToDevice),
        "upload value");
  check(hipMemcpy(key_scales.ptr, fixture.key_scales.data(),
                  fixture.key_scales.size(), hipMemcpyHostToDevice),
        "upload key scales");
  check(hipMemcpy(value_scales.ptr, fixture.value_scales.data(),
                  fixture.value_scales.size(), hipMemcpyHostToDevice),
        "upload value scales");
  check(hipMemcpy(physical_key.ptr, fixture.physical_key.data(),
                  fixture.physical_key.size(), hipMemcpyHostToDevice),
        "upload physical key");
  check(hipMemcpy(physical_value.ptr, fixture.physical_value.data(),
                  fixture.physical_value.size(), hipMemcpyHostToDevice),
        "upload physical value");
  check(hipMemcpy(physical_key_scales.ptr, fixture.physical_key_scales.data(),
                  fixture.physical_key_scales.size(), hipMemcpyHostToDevice),
        "upload physical key scales");
  check(hipMemcpy(physical_value_scales.ptr,
                  fixture.physical_value_scales.data(),
                  fixture.physical_value_scales.size(), hipMemcpyHostToDevice),
        "upload physical value scales");
}

void compare_replay(
    const Fixture &fixture, const uint64_t position, const uint32_t rows,
    const bool gqa_shared, DeviceBuffer<uint16_t> &query,
    DeviceBuffer<uint32_t> &logical_table,
    DeviceBuffer<uint8_t> &descriptor_table,
    DeviceBuffer<uint16_t> &device_output, DeviceBuffer<uint16_t> &eager_output,
    DeviceBuffer<float> &device_workspace, DeviceBuffer<float> &eager_workspace,
    DeviceBuffer<uint32_t> &device_status, DeviceBuffer<uint32_t> &eager_status,
    DeviceBuffer<sllm_decode_control::ControlV1> &control, hipStream_t stream) {
  const uint64_t committed = position + rows;
  const auto *const descriptors =
      reinterpret_cast<const sllm_paged_kv::BlockDescriptor *>(
          descriptor_table.ptr);
  check(hipMemset(device_output.ptr, 0xcd,
                  device_output.count * sizeof(uint16_t)),
        "clear device output");
  check(
      hipMemset(eager_output.ptr, 0xcd, eager_output.count * sizeof(uint16_t)),
      "clear eager output");

  sllm_decode_control::ControlV1 host_control{};
  host_control.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  host_control.phase_position = position;
  host_control.phase_rows = rows;
  host_control.phase_active = 1U;
  check(hipMemcpy(control.ptr, &host_control, sizeof(host_control),
                  hipMemcpyHostToDevice),
        "upload replay control");

  // The launch arguments remain fixed at the maximum five-row shape.  Only
  // ControlV1 changes between replays, so the split selection and positions
  // are read by the captured device kernels.
  check(sllm_causal_attention_kernel::launch_paged_decode_gqa6_device(
            query.ptr, logical_table.ptr, descriptors, fixture.pages,
            fixture.pages, device_status.ptr, device_workspace.ptr,
            device_workspace.count * sizeof(float), device_output.ptr, kMaxRows,
            8191U, 8191U + kMaxRows, kQHeads, kKvHeads, kHeadDim, kEncoding,
            gqa_shared, control.ptr, stream),
        "paged graph replay");
  check(sllm_causal_attention_kernel::launch_paged_decode_gqa6(
            query.ptr, logical_table.ptr, descriptors, fixture.pages,
            fixture.pages, eager_status.ptr, eager_workspace.ptr,
            eager_workspace.count * sizeof(float), eager_output.ptr, rows,
            position, committed, kQHeads, kKvHeads, kHeadDim, kEncoding,
            gqa_shared, stream),
        "paged eager control");
  check(hipStreamSynchronize(stream), "synchronize replay");

  uint32_t status = 0U;
  uint32_t eager = 0U;
  check(hipMemcpy(&status, device_status.ptr, sizeof(status),
                  hipMemcpyDeviceToHost),
        "download graph status");
  check(
      hipMemcpy(&eager, eager_status.ptr, sizeof(eager), hipMemcpyDeviceToHost),
      "download eager status");
  if (status != 0U || eager != 0U) {
    fail("paged graph status=" + std::to_string(status) +
         " eager=" + std::to_string(eager));
  }
  std::vector<uint16_t> graph(device_output.count);
  std::vector<uint16_t> reference(eager_output.count);
  check(hipMemcpy(graph.data(), device_output.ptr,
                  graph.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download graph output");
  check(hipMemcpy(reference.data(), eager_output.ptr,
                  reference.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download eager output");
  const size_t elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
  if (!std::equal(
          graph.begin(),
          graph.begin() +
              static_cast<std::vector<uint16_t>::difference_type>(elements),
          reference.begin())) {
    fail("device-control output differs from eager output at position=" +
         std::to_string(position) + " rows=" + std::to_string(rows));
  }
  const double expected = oracle_dim0(fixture, position, 0U);
  const double actual = bf16_to_f32(graph[0]);
  if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.08) {
    fail("oracle mismatch position=" + std::to_string(position) + " rows=" +
         std::to_string(rows) + " expected=" + std::to_string(expected) +
         " actual=" + std::to_string(actual));
  }
}
} // namespace

int main() {
  try {
    check(hipSetDevice(0), "set device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "device properties");
    const std::string target(properties.gcnArchName);
    const bool gqa_shared = target.rfind("gfx1030", 0U) == 0U;

    Fixture fixture;
    DeviceBuffer<uint16_t> query(fixture.query.size());
    DeviceBuffer<uint8_t> key(fixture.key.size());
    DeviceBuffer<uint8_t> value(fixture.value.size());
    DeviceBuffer<uint8_t> key_scales(fixture.key_scales.size());
    DeviceBuffer<uint8_t> value_scales(fixture.value_scales.size());
    DeviceBuffer<uint8_t> physical_key(fixture.physical_key.size());
    DeviceBuffer<uint8_t> physical_value(fixture.physical_value.size());
    DeviceBuffer<uint8_t> physical_key_scales(
        fixture.physical_key_scales.size());
    DeviceBuffer<uint8_t> physical_value_scales(
        fixture.physical_value_scales.size());
    DeviceBuffer<uint8_t> descriptor_table(
        static_cast<size_t>(fixture.pages) *
        sizeof(sllm_paged_kv::BlockDescriptor));
    DeviceBuffer<uint32_t> logical_table(fixture.pages);
    DeviceBuffer<uint32_t> device_status(1U);
    DeviceBuffer<uint32_t> eager_status(1U);
    DeviceBuffer<uint16_t> device_output(static_cast<size_t>(kMaxRows) *
                                         kQHeads * kHeadDim);
    DeviceBuffer<uint16_t> eager_output(device_output.count);
    const size_t workspace_elements =
        static_cast<size_t>(kMaxRows) * kQHeads * 128U * (kHeadDim + 2U);
    DeviceBuffer<float> device_workspace(workspace_elements);
    DeviceBuffer<float> eager_workspace(workspace_elements);
    DeviceBuffer<sllm_decode_control::ControlV1> control(1U);
    upload(fixture, query, key, value, key_scales, value_scales, physical_key,
           physical_value, physical_key_scales, physical_value_scales);

    std::vector<uint32_t> logical_pages(fixture.pages);
    std::vector<sllm_paged_kv::BlockDescriptor> descriptors(fixture.pages);
    for (uint32_t logical_page = 0U; logical_page < fixture.pages;
         ++logical_page) {
      const uint32_t physical_page = fixture.pages - 1U - logical_page;
      logical_pages[logical_page] = physical_page;
      descriptors[physical_page] = {
          physical_key.ptr +
              static_cast<size_t>(physical_page) * fixture.page_value_bytes,
          physical_value.ptr +
              static_cast<size_t>(physical_page) * fixture.page_value_bytes,
          physical_key_scales.ptr +
              static_cast<size_t>(physical_page) * fixture.page_scale_bytes,
          physical_value_scales.ptr +
              static_cast<size_t>(physical_page) * fixture.page_scale_bytes,
          nullptr,
          nullptr};
    }
    check(hipMemcpy(logical_table.ptr, logical_pages.data(),
                    logical_pages.size() * sizeof(logical_pages[0]),
                    hipMemcpyHostToDevice),
          "upload logical table");
    check(hipMemcpy(descriptor_table.ptr, descriptors.data(),
                    descriptors.size() * sizeof(descriptors[0]),
                    hipMemcpyHostToDevice),
          "upload descriptor table");

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "create stream");
    const uint64_t positions[] = {127U, 128U, 129U, 8191U, 8192U, 8193U};
    for (const uint64_t position : positions) {
      for (uint32_t rows = 1U; rows <= kMaxRows; ++rows) {
        compare_replay(fixture, position, rows, gqa_shared, query,
                       logical_table, descriptor_table, device_output,
                       eager_output, device_workspace, eager_workspace,
                       device_status, eager_status, control, stream);
      }
    }

    // An invalid logical/physical descriptor must poison the device control
    // record and status; the graph is never allowed to publish a valid result.
    const uint32_t invalid = UINT32_MAX;
    check(hipMemcpy(logical_table.ptr, &invalid, sizeof(invalid),
                    hipMemcpyHostToDevice),
          "upload invalid logical table entry");
    sllm_decode_control::ControlV1 invalid_control{};
    invalid_control.status =
        static_cast<uint32_t>(sllm_decode_control::Status::Ok);
    invalid_control.phase_position = 0U;
    invalid_control.phase_rows = 1U;
    invalid_control.phase_active = 1U;
    check(hipMemcpy(control.ptr, &invalid_control, sizeof(invalid_control),
                    hipMemcpyHostToDevice),
          "upload invalid control");
    check(sllm_causal_attention_kernel::launch_paged_decode_gqa6_device(
              query.ptr, logical_table.ptr,
              reinterpret_cast<const sllm_paged_kv::BlockDescriptor *>(
                  descriptor_table.ptr),
              fixture.pages, fixture.pages, device_status.ptr,
              device_workspace.ptr, device_workspace.count * sizeof(float),
              device_output.ptr, kMaxRows, 8191U, 8191U + kMaxRows, kQHeads,
              kKvHeads, kHeadDim, kEncoding, gqa_shared, control.ptr, stream),
          "invalid descriptor graph launch");
    check(hipStreamSynchronize(stream), "invalid descriptor synchronize");
    uint32_t invalid_status = 0U;
    sllm_decode_control::ControlV1 observed{};
    check(hipMemcpy(&invalid_status, device_status.ptr, sizeof(invalid_status),
                    hipMemcpyDeviceToHost),
          "download invalid descriptor status");
    check(hipMemcpy(&observed, control.ptr, sizeof(observed),
                    hipMemcpyDeviceToHost),
          "download invalid descriptor control");
    if (invalid_status == 0U || observed.halted == 0U ||
        observed.phase_active != 0U ||
        observed.status != static_cast<uint32_t>(
                               sllm_decode_control::Status::InvalidPosition)) {
      fail("invalid paged descriptor did not halt device control");
    }
    check(hipStreamDestroy(stream), "destroy stream");
    check(hipDeviceSynchronize(), "final synchronize");
    std::printf("phase87-stage10-paged-graph-attention target=%s "
                "positions=6 rows=1..5 state=PASS\n",
                target.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87-stage10-paged-graph-attention FAIL: %s\n",
                 error.what());
    return 1;
  }
}
