#ifndef LOWP_API_INTERNAL_HPP
#define LOWP_API_INTERNAL_HPP
#include "lowp_provider_plan.hpp"
#include <lowp/lowp.h>
namespace sllm_lowp {
// Legacy W4A4 is intentionally absent from the public format enumeration.
lowp_status_t plan_internal(const lowp_matmul_request_t *, lowp_matmul_plan_t *,
                            bool allow_legacy_mxfp4) noexcept;
lowp_status_t launch_internal(const lowp_matmul_plan_t *,
                              const lowp_matmul_buffers_t *, void *,
                              bool allow_legacy_mxfp4) noexcept;
} // namespace sllm_lowp
#endif
