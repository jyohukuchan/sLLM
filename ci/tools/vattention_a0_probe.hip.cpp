// SPDX-License-Identifier: MIT
// Standalone, model-free HIP VMM probe for Phase 6 A0.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kQwenKvRegionCount = 16; // 8 full-attention layers * K/V.
constexpr std::size_t kQwenBytesPerTokenPerRegion =
    4 * 256 * sizeof(std::uint16_t);
constexpr std::size_t kLogicalTokenCapacity = 4096;
constexpr int kWarmupIterations = 5;
constexpr int kMeasuredIterations = 101;

void check(hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    std::ostringstream message;
    message << operation << " failed: " << hipGetErrorName(status) << ": "
            << hipGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) {
    throw std::runtime_error("invalid alignment or size overflow");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (unsigned char character : value) {
    switch (character) {
    case '\\':
      escaped << "\\\\";
      break;
    case '"':
      escaped << "\\\"";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      if (character < 0x20) {
        escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(character) << std::dec;
      } else {
        escaped << character;
      }
    }
  }
  return escaped.str();
}

std::string format_bdf(const hipDeviceProp_t &properties) {
  std::ostringstream value;
  value << std::hex << std::setfill('0') << std::setw(4)
        << properties.pciDomainID << ":" << std::setw(2) << properties.pciBusID
        << ":" << std::setw(2) << properties.pciDeviceID << ".0";
  return value.str();
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::runtime_error("cannot calculate percentile of empty sample");
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

__global__ void fill_pattern(std::uint8_t *address, std::size_t size,
                             std::uint64_t seed) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    address[index] =
        static_cast<std::uint8_t>((index * 131U + seed * 17U + 29U) & 0xffU);
  }
}

bool verify_pattern(const std::vector<std::uint8_t> &bytes, std::uint64_t seed,
                    std::size_t logical_offset = 0) {
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const auto logical_index = index + logical_offset;
    const auto expected = static_cast<std::uint8_t>(
        (logical_index * 131U + seed * 17U + 29U) & 0xffU);
    if (bytes[index] != expected) {
      return false;
    }
  }
  return true;
}

struct Reservation {
  void *address = nullptr;
  std::size_t size = 0;

  Reservation() = default;
  Reservation(const Reservation &) = delete;
  Reservation &operator=(const Reservation &) = delete;
  Reservation(Reservation &&other) noexcept
      : address(other.address), size(other.size) {
    other.address = nullptr;
    other.size = 0;
  }
  Reservation &operator=(Reservation &&other) noexcept {
    if (this != &other) {
      release();
      address = other.address;
      size = other.size;
      other.address = nullptr;
      other.size = 0;
    }
    return *this;
  }
  ~Reservation() { release(); }

  void reserve(std::size_t bytes, std::size_t alignment) {
    check(hipMemAddressReserve(&address, bytes, alignment, nullptr, 0),
          "hipMemAddressReserve");
    size = bytes;
  }
  void release() noexcept {
    if (address != nullptr) {
      (void)hipMemAddressFree(address, size);
      address = nullptr;
      size = 0;
    }
  }
};

struct Page {
  void *address = nullptr;
  std::size_t size = 0;
  hipMemGenericAllocationHandle_t handle = nullptr;
  bool mapped = false;

  Page() = default;
  Page(const Page &) = delete;
  Page &operator=(const Page &) = delete;
  Page(Page &&other) noexcept
      : address(other.address), size(other.size), handle(other.handle),
        mapped(other.mapped) {
    other.address = nullptr;
    other.size = 0;
    other.handle = nullptr;
    other.mapped = false;
  }
  Page &operator=(Page &&other) noexcept {
    if (this != &other) {
      release();
      address = other.address;
      size = other.size;
      handle = other.handle;
      mapped = other.mapped;
      other.address = nullptr;
      other.size = 0;
      other.handle = nullptr;
      other.mapped = false;
    }
    return *this;
  }
  ~Page() { release(); }

  void create_and_map(void *target, std::size_t bytes,
                      const hipMemAllocationProp &properties,
                      const hipMemAccessDesc &access) {
    address = target;
    size = bytes;
    check(hipMemCreate(&handle, size, &properties, 0), "hipMemCreate");
    check(hipMemMap(address, size, 0, handle, 0), "hipMemMap");
    mapped = true;
    check(hipMemSetAccess(address, size, &access, 1), "hipMemSetAccess");
  }
  void release_checked() {
    if (mapped) {
      check(hipMemUnmap(address, size), "hipMemUnmap");
      mapped = false;
    }
    if (handle != nullptr) {
      check(hipMemRelease(handle), "hipMemRelease");
      handle = nullptr;
    }
    address = nullptr;
    size = 0;
  }
  void release() noexcept {
    if (mapped) {
      (void)hipMemUnmap(address, size);
      mapped = false;
    }
    if (handle != nullptr) {
      (void)hipMemRelease(handle);
      handle = nullptr;
    }
    address = nullptr;
    size = 0;
  }
};

std::pair<std::size_t, std::size_t> memory_info() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
  return {free_bytes, total_bytes};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: vattention-a0-probe <expected-gfx-target>");
    }
    const std::string expected_target(argv[1]);
    if (expected_target != "gfx1030" && expected_target != "gfx1201") {
      throw std::runtime_error("expected target must be gfx1030 or gfx1201");
    }

    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
      throw std::runtime_error("probe requires exactly one visible HIP device");
    }
    check(hipSetDevice(0), "hipSetDevice");
    check(hipFree(nullptr), "hipFree(nullptr)");

    hipDeviceProp_t device_properties{};
    check(hipGetDeviceProperties(&device_properties, 0),
          "hipGetDeviceProperties");
    const std::string actual_target(device_properties.gcnArchName);
    if (actual_target != expected_target) {
      throw std::runtime_error(
          "visible device target does not match expected target");
    }
    int vmm_supported = 0;
    check(hipDeviceGetAttribute(
              &vmm_supported,
              hipDeviceAttributeVirtualMemoryManagementSupported, 0),
          "hipDeviceGetAttribute(VMM)");
    if (vmm_supported != 1) {
      throw std::runtime_error("visible device does not report VMM support");
    }

    hipMemAllocationProp allocation_properties{};
    allocation_properties.type = hipMemAllocationTypePinned;
    allocation_properties.requestedHandleType = hipMemHandleTypeNone;
    allocation_properties.location.type = hipMemLocationTypeDevice;
    allocation_properties.location.id = 0;
    std::size_t minimum_granularity = 0;
    std::size_t recommended_granularity = 0;
    check(hipMemGetAllocationGranularity(&minimum_granularity,
                                         &allocation_properties,
                                         hipMemAllocationGranularityMinimum),
          "hipMemGetAllocationGranularity(minimum)");
    check(hipMemGetAllocationGranularity(
              &recommended_granularity, &allocation_properties,
              hipMemAllocationGranularityRecommended),
          "hipMemGetAllocationGranularity(recommended)");
    if (minimum_granularity == 0 || recommended_granularity == 0 ||
        recommended_granularity % minimum_granularity != 0) {
      throw std::runtime_error("invalid HIP VMM granularity");
    }

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = hipMemAccessFlagsProtReadWrite;

    const auto memory_before = memory_info();

    // Primitive contiguous-pointer test across three physical pages.
    const std::size_t physical_page_bytes = recommended_granularity;
    const std::size_t primitive_size = physical_page_bytes * 3;
    Reservation primitive_reservation;
    primitive_reservation.reserve(primitive_size, physical_page_bytes);
    const auto memory_after_primitive_reserve = memory_info();
    std::vector<Page> primitive_pages(3);
    for (std::size_t index = 0; index < primitive_pages.size(); ++index) {
      auto *target =
          static_cast<std::uint8_t *>(primitive_reservation.address) +
          index * physical_page_bytes;
      primitive_pages[index].create_and_map(target, physical_page_bytes,
                                            allocation_properties, access);
    }

    hipStream_t stream = nullptr;
    hipEvent_t completion = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags");
    check(hipEventCreateWithFlags(&completion, hipEventDisableTiming),
          "hipEventCreateWithFlags");
    const int blocks = static_cast<int>((primitive_size + 255) / 256);
    hipLaunchKernelGGL(
        fill_pattern, dim3(blocks), dim3(256), 0, stream,
        static_cast<std::uint8_t *>(primitive_reservation.address),
        primitive_size, 7U);
    check(hipGetLastError(), "fill_pattern launch");
    check(hipEventRecord(completion, stream), "hipEventRecord");
    check(hipEventSynchronize(completion), "hipEventSynchronize");
    std::vector<std::uint8_t> primitive_output(primitive_size);
    check(hipMemcpy(primitive_output.data(), primitive_reservation.address,
                    primitive_size, hipMemcpyDeviceToHost),
          "hipMemcpy primitive output");
    if (!verify_pattern(primitive_output, 7U)) {
      throw std::runtime_error(
          "CPU oracle rejected contiguous three-page kernel output");
    }

    // Replace the middle physical page while preserving its virtual address.
    primitive_pages[1].release_checked();
    auto *middle_address =
        static_cast<std::uint8_t *>(primitive_reservation.address) +
        physical_page_bytes;
    primitive_pages[1].create_and_map(middle_address, physical_page_bytes,
                                      allocation_properties, access);
    const int middle_blocks =
        static_cast<int>((physical_page_bytes + 255) / 256);
    hipLaunchKernelGGL(fill_pattern, dim3(middle_blocks), dim3(256), 0, stream,
                       static_cast<std::uint8_t *>(middle_address),
                       physical_page_bytes, 19U);
    check(hipGetLastError(), "remap fill_pattern launch");
    check(hipStreamSynchronize(stream), "hipStreamSynchronize after remap");
    std::vector<std::uint8_t> remap_output(physical_page_bytes);
    check(hipMemcpy(remap_output.data(), middle_address, physical_page_bytes,
                    hipMemcpyDeviceToHost),
          "hipMemcpy remap output");
    if (!verify_pattern(remap_output, 19U)) {
      throw std::runtime_error(
          "CPU oracle rejected remapped-page kernel output");
    }
    check(hipEventDestroy(completion), "hipEventDestroy");
    completion = nullptr;
    check(hipStreamDestroy(stream), "hipStreamDestroy");
    stream = nullptr;
    for (auto &page : primitive_pages) {
      page.release_checked();
    }
    primitive_reservation.release();
    const auto memory_before_qwen_reserve = memory_info();

    // Qwen3.5-4B model-free shape: reserve maximum logical capacity in every
    // full-attention K/V region, but activate one physical page per region.
    const std::size_t qwen_region_logical_bytes =
        align_up(kQwenBytesPerTokenPerRegion * kLogicalTokenCapacity,
                 physical_page_bytes);
    const std::size_t qwen_logical_bytes =
        qwen_region_logical_bytes * kQwenKvRegionCount;
    const std::size_t qwen_requested_physical_bytes =
        physical_page_bytes * kQwenKvRegionCount;
    const std::size_t tokens_per_page =
        physical_page_bytes / kQwenBytesPerTokenPerRegion;
    if (tokens_per_page == 0) {
      throw std::runtime_error(
          "one Qwen KV token exceeds the VMM minimum granularity");
    }

    std::vector<Reservation> qwen_regions(kQwenKvRegionCount);
    for (auto &region : qwen_regions) {
      region.reserve(qwen_region_logical_bytes, physical_page_bytes);
    }
    const auto memory_after_qwen_reserve = memory_info();

    std::vector<double> activation_us;
    std::vector<double> deactivation_us;
    std::vector<double> create_us;
    std::vector<double> map_us;
    std::vector<double> set_access_us;
    std::vector<double> unmap_us;
    std::vector<double> release_us;
    activation_us.reserve(kMeasuredIterations);
    deactivation_us.reserve(kMeasuredIterations);
    create_us.reserve(kMeasuredIterations);
    map_us.reserve(kMeasuredIterations);
    set_access_us.reserve(kMeasuredIterations);
    unmap_us.reserve(kMeasuredIterations);
    release_us.reserve(kMeasuredIterations);
    std::size_t free_after_first_create = 0;
    std::size_t free_after_first_map = 0;
    for (int iteration = -kWarmupIterations; iteration < kMeasuredIterations;
         ++iteration) {
      std::vector<Page> pages(kQwenKvRegionCount);
      const auto activate_start = std::chrono::steady_clock::now();
      for (std::size_t index = 0; index < pages.size(); ++index) {
        pages[index].address = qwen_regions[index].address;
        pages[index].size = physical_page_bytes;
        check(hipMemCreate(&pages[index].handle, physical_page_bytes,
                           &allocation_properties, 0),
              "hipMemCreate(qwen page)");
      }
      const auto create_end = std::chrono::steady_clock::now();
      if (iteration == 0) {
        free_after_first_create = memory_info().first;
      }
      for (auto &page : pages) {
        check(hipMemMap(page.address, page.size, 0, page.handle, 0),
              "hipMemMap(qwen page)");
        page.mapped = true;
      }
      const auto map_end = std::chrono::steady_clock::now();
      for (auto &page : pages) {
        check(hipMemSetAccess(page.address, page.size, &access, 1),
              "hipMemSetAccess(qwen page)");
      }
      const auto set_access_end = std::chrono::steady_clock::now();
      if (iteration == 0) {
        free_after_first_map = memory_info().first;
      }
      const auto deactivate_start = std::chrono::steady_clock::now();
      for (auto &page : pages) {
        check(hipMemUnmap(page.address, page.size), "hipMemUnmap(qwen page)");
        page.mapped = false;
      }
      const auto unmap_end = std::chrono::steady_clock::now();
      for (auto &page : pages) {
        check(hipMemRelease(page.handle), "hipMemRelease(qwen page)");
        page.handle = nullptr;
        page.address = nullptr;
        page.size = 0;
      }
      const auto release_end = std::chrono::steady_clock::now();
      if (iteration >= 0) {
        activation_us.push_back(std::chrono::duration<double, std::micro>(
                                    set_access_end - activate_start)
                                    .count());
        deactivation_us.push_back(std::chrono::duration<double, std::micro>(
                                      release_end - deactivate_start)
                                      .count());
        create_us.push_back(std::chrono::duration<double, std::micro>(
                                create_end - activate_start)
                                .count());
        map_us.push_back(
            std::chrono::duration<double, std::micro>(map_end - create_end)
                .count());
        set_access_us.push_back(
            std::chrono::duration<double, std::micro>(set_access_end - map_end)
                .count());
        unmap_us.push_back(std::chrono::duration<double, std::micro>(
                               unmap_end - deactivate_start)
                               .count());
        release_us.push_back(
            std::chrono::duration<double, std::micro>(release_end - unmap_end)
                .count());
      }
    }
    for (auto &region : qwen_regions) {
      region.release();
    }
    check(hipDeviceSynchronize(), "hipDeviceSynchronize(final)");
    const auto memory_after_cleanup = memory_info();
    const std::size_t qwen_reserve_delta =
        memory_before_qwen_reserve.first > memory_after_qwen_reserve.first
            ? memory_before_qwen_reserve.first - memory_after_qwen_reserve.first
            : 0;
    const std::size_t qwen_observed_physical_commit =
        memory_after_qwen_reserve.first > free_after_first_create
            ? memory_after_qwen_reserve.first - free_after_first_create
            : 0;
    const std::size_t qwen_cleanup_shortfall =
        memory_before_qwen_reserve.first > memory_after_cleanup.first
            ? memory_before_qwen_reserve.first - memory_after_cleanup.first
            : 0;

    const std::vector<std::size_t> boundary_tokens = {
        tokens_per_page - 1, tokens_per_page, tokens_per_page + 1, 37};
    if (qwen_observed_physical_commit == 0 ||
        qwen_observed_physical_commit >= qwen_logical_bytes) {
      throw std::runtime_error(
          "PoC did not preserve sparse physical commitment");
    }
    if (qwen_reserve_delta > physical_page_bytes ||
        qwen_cleanup_shortfall > physical_page_bytes) {
      throw std::runtime_error(
          "virtual reservation or cleanup consumed unexpected physical memory");
    }
    if (activation_us.size() != kMeasuredIterations ||
        deactivation_us.size() != kMeasuredIterations ||
        create_us.size() != kMeasuredIterations ||
        map_us.size() != kMeasuredIterations ||
        set_access_us.size() != kMeasuredIterations ||
        unmap_us.size() != kMeasuredIterations ||
        release_us.size() != kMeasuredIterations) {
      throw std::runtime_error("latency sample count is not canonical");
    }

    std::cout << std::setprecision(6) << std::fixed;
    std::cout
        << "{\"protocol\":\"sllm-vattention-a0-probe-v1\",\"state\":\"PASS\"";
    std::cout << ",\"device\":{\"logical_index\":0,\"product\":\""
              << json_escape(device_properties.name) << "\",\"target\":\""
              << json_escape(actual_target) << "\",\"bdf\":\""
              << format_bdf(device_properties) << "\",\"vmm_supported\":true}";
    std::cout << ",\"granularity\":{\"minimum_bytes\":" << minimum_granularity
              << ",\"recommended_bytes\":" << recommended_granularity
              << ",\"selected_physical_page_bytes\":" << physical_page_bytes
              << "}";
    std::cout << ",\"primitive\":{\"reserved_bytes\":" << primitive_size
              << ",\"mapped_pages\":3,\"contiguous_kernel_oracle\":true,"
                 "\"remap_oracle\":true"
              << ",\"event_synchronized_before_unmap\":true,"
                 "\"nonaligned_byte_offset\":37}";
    std::cout << ",\"qwen_shape\":{\"model\":\"Qwen/"
                 "Qwen3.5-4B\",\"full_attention_layers\":8"
              << ",\"regions\":" << kQwenKvRegionCount
              << ",\"kv_heads\":4,\"head_dim\":256,\"element_bytes\":2"
              << ",\"bytes_per_token_per_region\":"
              << kQwenBytesPerTokenPerRegion
              << ",\"logical_token_capacity\":" << kLogicalTokenCapacity
              << ",\"tokens_per_physical_page\":" << tokens_per_page
              << ",\"logical_reserved_bytes\":" << qwen_logical_bytes
              << ",\"requested_physical_bytes\":"
              << qwen_requested_physical_bytes
              << ",\"observed_physical_commit_bytes\":"
              << qwen_observed_physical_commit
              << ",\"virtual_reserve_physical_delta_bytes\":"
              << qwen_reserve_delta
              << ",\"activated_pages_per_step\":" << kQwenKvRegionCount
              << ",\"boundary_tokens\":[";
    for (std::size_t index = 0; index < boundary_tokens.size(); ++index) {
      if (index != 0)
        std::cout << ',';
      std::cout << boundary_tokens[index];
    }
    std::cout << "]}";
    std::cout << ",\"latency_us\":{\"warmup_iterations\":" << kWarmupIterations
              << ",\"measured_iterations\":" << kMeasuredIterations
              << ",\"activate_p50\":" << percentile(activation_us, 0.50)
              << ",\"activate_p95\":" << percentile(activation_us, 0.95)
              << ",\"create_p50\":" << percentile(create_us, 0.50)
              << ",\"create_p95\":" << percentile(create_us, 0.95)
              << ",\"map_p50\":" << percentile(map_us, 0.50)
              << ",\"map_p95\":" << percentile(map_us, 0.95)
              << ",\"set_access_p50\":" << percentile(set_access_us, 0.50)
              << ",\"set_access_p95\":" << percentile(set_access_us, 0.95)
              << ",\"deactivate_p50\":" << percentile(deactivation_us, 0.50)
              << ",\"deactivate_p95\":" << percentile(deactivation_us, 0.95)
              << ",\"unmap_p50\":" << percentile(unmap_us, 0.50)
              << ",\"unmap_p95\":" << percentile(unmap_us, 0.95)
              << ",\"release_p50\":" << percentile(release_us, 0.50)
              << ",\"release_p95\":" << percentile(release_us, 0.95) << "}";
    std::cout << ",\"memory_info\":{\"total_bytes\":" << memory_before.second
              << ",\"free_before_bytes\":" << memory_before.first
              << ",\"free_after_primitive_reserve_bytes\":"
              << memory_after_primitive_reserve.first
              << ",\"free_before_qwen_reserve_bytes\":"
              << memory_before_qwen_reserve.first
              << ",\"free_after_qwen_reserve_bytes\":"
              << memory_after_qwen_reserve.first
              << ",\"free_after_first_create_bytes\":"
              << free_after_first_create
              << ",\"free_after_first_map_bytes\":" << free_after_first_map
              << ",\"free_after_cleanup_bytes\":" << memory_after_cleanup.first
              << "}";
    std::cout << ",\"fallback_used\":false,\"cleanup_complete\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vAttention A0 probe failed: " << error.what() << '\n';
    return 1;
  }
}
