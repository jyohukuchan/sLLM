#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
using Op = ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<
    Row, Col, ck::Tuple<>, Row, ck::f8_t, float, ck::f8_t, float,
    ck::Tuple<>, ck::bhalf_t, 1, 128, 128, Pass, Pass, Pass>;

static void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

int main() {
    constexpr size_t m = 4096, n = 4096, k = 4096;
    check(hipSetDevice(0), "hipSetDevice");
    std::vector<std::unique_ptr<Op>> instances;
    ck::tensor_operation::device::instance::add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(instances);
    constexpr const char* selected_name =
        "DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x128x128, "
        "WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 8x16, "
        "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, "
        "BlkGemmPipelinePrefetchStages: 2";
    Op* op = nullptr;
    for (auto& candidate : instances) if (candidate->GetTypeString() == selected_name) op = candidate.get();
    if (!op) throw std::runtime_error("selected CK LDS-tiled instance unavailable");
    const size_t a_bytes = m*k, b_bytes = n*k, c_bytes = m*n*sizeof(ck::bhalf_t);
    const size_t as_bytes = (m*(k/128))*sizeof(float), bs_bytes = ((n/128)*(k/128))*sizeof(float);
    void *a=nullptr, *b=nullptr, *c=nullptr, *as=nullptr, *bs=nullptr;
    check(hipMalloc(&a,a_bytes),"hipMalloc A"); check(hipMalloc(&b,b_bytes),"hipMalloc B");
    check(hipMalloc(&c,c_bytes),"hipMalloc C"); check(hipMalloc(&as,as_bytes),"hipMalloc AS"); check(hipMalloc(&bs,bs_bytes),"hipMalloc BS");
    check(hipMemset(a,0,a_bytes),"zero A"); check(hipMemset(b,0,b_bytes),"zero B");
    check(hipMemset(as,0,as_bytes),"zero AS"); check(hipMemset(bs,0,bs_bytes),"zero BS");
    auto arg = op->MakeArgumentPointer(a,b,std::array<const void*,0>{},c,m,n,k,k,k,std::array<ck::index_t,0>{},n,as,bs,Pass{},Pass{},Pass{});
    if (!op->IsSupportedArgument(arg.get())) throw std::runtime_error("CK rejected 4096-cube argument");
    auto invoker = op->MakeInvokerPointer();
    StreamConfig verify{}; verify.time_kernel_=false;
    invoker->Run(arg.get(), verify);
    check(hipDeviceSynchronize(),"zero correctness synchronize");
    std::vector<ck::bhalf_t> sample(16); check(hipMemcpy(sample.data(),c,sample.size()*sizeof(ck::bhalf_t),hipMemcpyDeviceToHost),"copy C");
    for (auto x : sample) if (x != ck::bhalf_t{0}) throw std::runtime_error("zero input correctness failed");
    StreamConfig clock_warm{}; clock_warm.time_kernel_=true; clock_warm.cold_niters_=5; clock_warm.nrepeat_=30000;
    (void)invoker->Run(arg.get(), clock_warm);
    StreamConfig timed{}; timed.time_kernel_=true; timed.cold_niters_=5; timed.nrepeat_=20;
    const float ms = invoker->Run(arg.get(), timed);
    const double tflops = (2.0*double(m)*double(n)*double(k))/(double(ms)*1.0e9);
    std::printf("{\"schema\":\"ullm.ck-lds-gemm.v1\",\"m\":%zu,\"n\":%zu,\"k\":%zu,\"input\":\"zero FP8 payload/scales; zero-output correctness verified\",\"instance\":\"16x128x128 RCR LDS-tiled CK\",\"clock_warmup_repeats\":30000,\"warmups\":5,\"repeats\":20,\"median_or_average\":\"CK invoker average HIP-event ms\",\"milliseconds\":%.6f,\"tflops\":%.3f}\n",m,n,k,ms,tflops);
    hipFree(bs); hipFree(as); hipFree(c); hipFree(b); hipFree(a);
}
