// SPDX-License-Identifier: Apache-2.0
// Measure the unprofiled host-side cost of the same HIP module-launch API used by AQ4_0.

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        fail(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

int parse_positive(const char* value, const char* label) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) fail(std::string(label) + " must be positive");
        return parsed;
    } catch (const std::exception&) {
        fail(std::string("invalid ") + label + ": " + value);
    }
}

struct Distribution {
    double min_ns;
    double median_ns;
    double p90_ns;
    double max_ns;
    double mean_ns;
};

Distribution distribution(std::vector<double> values) {
    if (values.empty()) fail("empty measurement distribution");
    std::sort(values.begin(), values.end());
    const auto quantile = [&values](double fraction) {
        const auto index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
        return values[index];
    };
    double sum = 0.0;
    for (const double value : values) sum += value;
    return {
        .min_ns = values.front(),
        .median_ns = quantile(0.50),
        .p90_ns = quantile(0.90),
        .max_ns = values.back(),
        .mean_ns = sum / static_cast<double>(values.size()),
    };
}

void print_distribution(const char* name, const Distribution& values) {
    std::cout << "\"" << name << "\":{"
              << "\"min_ns\":" << values.min_ns << ","
              << "\"median_ns\":" << values.median_ns << ","
              << "\"p90_ns\":" << values.p90_ns << ","
              << "\"max_ns\":" << values.max_ns << ","
              << "\"mean_ns\":" << values.mean_ns << "}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            fail("usage: hip-module-launch-overhead CODE_OBJECT [ITERATIONS] [REPETITIONS]");
        }
        const int iterations = argc >= 3 ? parse_positive(argv[2], "ITERATIONS") : 4096;
        const int repetitions = argc >= 4 ? parse_positive(argv[3], "REPETITIONS") : 31;

        int visible_devices = 0;
        hip_check(hipGetDeviceCount(&visible_devices), "hipGetDeviceCount");
        if (visible_devices != 1) {
            fail("expected exactly one HIP-visible GPU; set HIP_VISIBLE_DEVICES to the R9700 only");
        }
        hip_check(hipSetDevice(0), "hipSetDevice(0)");
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
        const std::string architecture(properties.gcnArchName);
        if (architecture.rfind("gfx1201", 0) != 0) {
            fail("visible GPU is not gfx1201: " + architecture);
        }

        hipStream_t stream{};
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        hipModule_t module{};
        hip_check(hipModuleLoad(&module, argv[1]), "hipModuleLoad");
        hipFunction_t function{};
        hip_check(
            hipModuleGetFunction(&function, module, "ullm_module_launch_noop_kernel"),
            "hipModuleGetFunction");

        const auto launch = [&] {
            hip_check(
                hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr),
                "hipModuleLaunchKernel");
        };

        for (int index = 0; index < 256; ++index) launch();
        hip_check(hipStreamSynchronize(stream), "warmup hipStreamSynchronize");

        std::vector<double> batched_ns_per_launch;
        batched_ns_per_launch.reserve(repetitions);
        for (int repeat = 0; repeat < repetitions; ++repeat) {
            const auto started = Clock::now();
            for (int index = 0; index < iterations; ++index) launch();
            const auto finished = Clock::now();
            hip_check(hipStreamSynchronize(stream), "batched hipStreamSynchronize");
            const auto elapsed = std::chrono::duration<double, std::nano>(finished - started).count();
            batched_ns_per_launch.push_back(elapsed / static_cast<double>(iterations));
        }

        std::vector<double> isolated_enqueue_ns;
        std::vector<double> isolated_launch_and_sync_ns;
        isolated_enqueue_ns.reserve(repetitions);
        isolated_launch_and_sync_ns.reserve(repetitions);
        for (int repeat = 0; repeat < repetitions; ++repeat) {
            const auto started = Clock::now();
            launch();
            const auto queued = Clock::now();
            hip_check(hipStreamSynchronize(stream), "isolated hipStreamSynchronize");
            const auto completed = Clock::now();
            isolated_enqueue_ns.push_back(
                std::chrono::duration<double, std::nano>(queued - started).count());
            isolated_launch_and_sync_ns.push_back(
                std::chrono::duration<double, std::nano>(completed - started).count());
        }

        const auto batched = distribution(std::move(batched_ns_per_launch));
        const auto isolated_enqueue = distribution(std::move(isolated_enqueue_ns));
        const auto isolated_total = distribution(std::move(isolated_launch_and_sync_ns));
        std::cout << std::fixed << std::setprecision(3)
                  << "{\"schema_version\":\"ullm.hip_module_launch_overhead.v1\","
                  << "\"api\":\"hipModuleLaunchKernel\","
                  << "\"visible_device_count\":" << visible_devices << ","
                  << "\"device\":{\"name\":\"" << properties.name << "\",\"gcn_arch_name\":\""
                  << architecture << "\"},"
                  << "\"iterations_per_batched_sample\":" << iterations << ","
                  << "\"repetitions\":" << repetitions << ",";
        print_distribution("batched_host_enqueue_ns_per_launch", batched);
        std::cout << ',';
        print_distribution("isolated_host_enqueue_ns", isolated_enqueue);
        std::cout << ',';
        print_distribution("isolated_launch_plus_stream_synchronize_ns", isolated_total);
        std::cout << "}\n";

        hip_check(hipModuleUnload(module), "hipModuleUnload");
        hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hip-module-launch-overhead: " << error.what() << '\n';
        return 2;
    }
}
