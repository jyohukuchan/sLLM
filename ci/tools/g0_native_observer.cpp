#include <dlfcn.h>

#include <hip/hip_runtime_api.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char *kProviderId = "g0-native-hip-observer-v1";
constexpr const char *kProbeKind = "hip-identity-only-v1";

struct Arguments {
  std::string rocm_root;
  std::string expected_bdf;
  std::string expected_uuid;
  std::string expected_target;
  std::string expected_product;
  std::string expected_hip_library;
  std::string expected_hsa_library;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::string lowercase(std::string value) {
  for (char &character : value) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

std::string canonical_path(const std::string &value) {
  try {
    return std::filesystem::canonical(value).string();
  } catch (const std::filesystem::filesystem_error &error) {
    fail("cannot canonicalize path " + value + ": " + error.what());
  }
}

bool is_within(const std::string &path, const std::string &root) {
  return path.size() > root.size() &&
         path.compare(0U, root.size(), root) == 0 && path[root.size()] == '/';
}

std::string read_release(const std::string &root) {
  std::ifstream stream(root + "/.info/version");
  std::string value;
  if (!stream.good() || !std::getline(stream, value)) {
    fail("cannot read ROCm release file");
  }
  value = trim(value);
  if (value != "7.14.0") {
    fail("ROCm release is not 7.14.0");
  }
  return value;
}

Arguments parse_arguments(const int argc, char **argv) {
  if (argc != 15) {
    fail("expected seven named observation arguments");
  }
  Arguments result;
  for (int index = 1; index < argc; index += 2) {
    const std::string name(argv[index]);
    const std::string value(argv[index + 1]);
    if (value.empty()) {
      fail("observation argument is empty: " + name);
    }
    if (name == "--rocm-root" && result.rocm_root.empty()) {
      result.rocm_root = value;
    } else if (name == "--expected-bdf" && result.expected_bdf.empty()) {
      result.expected_bdf = lowercase(value);
    } else if (name == "--expected-uuid" && result.expected_uuid.empty()) {
      result.expected_uuid = value;
    } else if (name == "--expected-target" && result.expected_target.empty()) {
      result.expected_target = value;
    } else if (name == "--expected-product" &&
               result.expected_product.empty()) {
      result.expected_product = value;
    } else if (name == "--expected-hip-library" &&
               result.expected_hip_library.empty()) {
      result.expected_hip_library = value;
    } else if (name == "--expected-hsa-library" &&
               result.expected_hsa_library.empty()) {
      result.expected_hsa_library = value;
    } else {
      fail("unknown or duplicate observation argument: " + name);
    }
  }
  if (result.rocm_root.empty() || result.expected_bdf.empty() ||
      result.expected_uuid.empty() || result.expected_target.empty() ||
      result.expected_product.empty() || result.expected_hip_library.empty() ||
      result.expected_hsa_library.empty()) {
    fail("missing required observation argument");
  }
  return result;
}

void check_hip(const hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + " failed with HIP status " +
         std::to_string(static_cast<int>(status)));
  }
}

std::string hex_uuid(const hipUUID &uuid) {
  std::string result(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : result) {
    const auto value = static_cast<unsigned char>(character);
    if (!std::isxdigit(value)) {
      fail("HIP UUID is not the expected 16-digit ASCII hexadecimal form");
    }
    character = static_cast<char>(std::tolower(value));
  }
  return result;
}

std::string canonical_uuid(const std::string &raw_uuid) {
  if (raw_uuid.size() != 16U) {
    fail("HIP UUID does not contain exactly 16 hexadecimal digits");
  }
  return "GPU-" + raw_uuid;
}

std::string exact_target(const std::string &gcn_arch_name) {
  const auto separator = gcn_arch_name.find(':');
  const std::string result = gcn_arch_name.substr(0U, separator);
  if (result.size() < 4U || result.rfind("gfx", 0U) != 0U) {
    fail("HIP gcnArchName is empty or not an exact gfx target");
  }
  return result;
}

template <typename Function>
std::string loaded_library_path(Function function, const char *label,
                                const std::string &root) {
  Dl_info info{};
  const auto address =
      reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(function));
  if (dladdr(address, &info) == 0 || info.dli_fname == nullptr) {
    fail(std::string("cannot identify loaded ") + label + " library");
  }
  const std::string path = canonical_path(info.dli_fname);
  if (!std::filesystem::path(path).is_absolute() || !is_within(path, root)) {
    fail(std::string("loaded ") + label + " library is outside ROCm root");
  }
  return path;
}

std::string loaded_hsa_library_path(const std::string &path,
                                    const std::string &root) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    fail("cannot load the expected HSA runtime library");
  }
  void *symbol = dlsym(handle, "hsa_init");
  if (symbol == nullptr) {
    dlclose(handle);
    fail("expected HSA runtime library does not export hsa_init");
  }
  Dl_info info{};
  if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
    dlclose(handle);
    fail("cannot identify the loaded HSA runtime library");
  }
  const std::string loaded = canonical_path(info.dli_fname);
  if (!std::filesystem::path(loaded).is_absolute() ||
      !is_within(loaded, root)) {
    dlclose(handle);
    fail("loaded HSA runtime library is outside ROCm root");
  }
  dlclose(handle);
  return loaded;
}

std::string json_escape(const std::string &value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 8U);
  for (const unsigned char character : value) {
    switch (character) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character < 0x20U) {
        result += "\\u00";
        result.push_back(digits[(character >> 4U) & 0x0fU]);
        result.push_back(digits[character & 0x0fU]);
      } else {
        result.push_back(static_cast<char>(character));
      }
      break;
    }
  }
  return result;
}

void print_string(const char *name, const std::string &value,
                  const bool trailing_comma) {
  std::cout << '"' << name << "\":\"" << json_escape(value) << '"';
  if (trailing_comma) {
    std::cout << ',';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    const std::string rocm_root = canonical_path(arguments.rocm_root);
    if (rocm_root != "/opt/rocm/core-7.14") {
      fail("ROCm root is not the canonical core-7.14 root");
    }
    const std::string release = read_release(rocm_root);
    const std::string expected_hip_library =
        canonical_path(arguments.expected_hip_library);
    const std::string expected_hsa_library =
        canonical_path(arguments.expected_hsa_library);

    int runtime_version = 0;
    check_hip(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
      fail("HIP must expose exactly one device for a G0 row");
    }

    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    std::array<char, 32U> bdf_buffer{};
    check_hip(hipDeviceGetPCIBusId(bdf_buffer.data(),
                                   static_cast<int>(bdf_buffer.size()), 0),
              "hipDeviceGetPCIBusId");
    hipUUID uuid{};
    check_hip(hipDeviceGetUuid(&uuid, static_cast<hipDevice_t>(0)),
              "hipDeviceGetUuid");

    const std::string bdf = lowercase(std::string(bdf_buffer.data()));
    const std::string raw_uuid = hex_uuid(uuid);
    const std::string property_uuid = hex_uuid(properties.uuid);
    const std::string uuid_text = canonical_uuid(raw_uuid);
    const std::string gcn_arch_name(properties.gcnArchName);
    const std::string target = exact_target(gcn_arch_name);
    const std::string product(properties.name);
    if (raw_uuid != property_uuid || bdf != arguments.expected_bdf ||
        uuid_text != arguments.expected_uuid ||
        target != arguments.expected_target ||
        product != arguments.expected_product || properties.warpSize <= 0 ||
        properties.totalGlobalMem == 0U) {
      fail("HIP identity mismatch: bdf=" + bdf + " uuid=" + uuid_text +
           " property_uuid=" + property_uuid + " target=" + target +
           " product=" + product);
    }

    const std::string hip_library =
        loaded_library_path(&hipRuntimeGetVersion, "HIP runtime", rocm_root);
    const std::string hsa_library =
        loaded_hsa_library_path(expected_hsa_library, rocm_root);
    if (hip_library != expected_hip_library ||
        hsa_library != expected_hsa_library) {
      fail("loaded ROCm library path does not match the fixed tuple");
    }

    std::cout << '{';
    print_string("provider_id", kProviderId, true);
    print_string("probe_kind", kProbeKind, true);
    print_string("rocm_root", rocm_root, true);
    print_string("release", release, true);
    std::cout << "\"hip_runtime_api_version\":" << runtime_version << ',';
    print_string("hip_runtime_library_path", hip_library, true);
    print_string("hsa_runtime_library_path", hsa_library, true);
    std::cout << "\"visible_device_count\":1,\"device\":{";
    std::cout << "\"ordinal\":0,";
    print_string("bdf", bdf, true);
    print_string("uuid", uuid_text, true);
    print_string("hip_uuid_hex", raw_uuid, true);
    print_string("gcnArchName", gcn_arch_name, true);
    print_string("exact_target", target, true);
    print_string("product", product, true);
    std::cout << "\"wave_size\":" << properties.warpSize << ',';
    std::cout << "\"total_global_memory_bytes\":" << properties.totalGlobalMem;
    std::cout << "},\"scope\":{";
    std::cout << "\"allocation_count\":0,\"copy_count\":0,"
                 "\"kernel_dispatch_count\":0,\"dispatch_count\":0";
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "g0 native HIP observer: " << error.what() << '\n';
    return 2;
  }
}
