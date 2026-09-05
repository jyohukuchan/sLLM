namespace {

using Qwen38ProjectionPackPlan = ElementwisePlan;

constexpr uint64_t kQwen38ProjectionPackM = UINT64_C(1);
constexpr uint64_t kQwen38ProjectionPackK = UINT64_C(5120);
constexpr uint64_t kQwen38ProjectionPackNvfp4N = UINT64_C(17408);
constexpr uint64_t kQwen38ProjectionPackFp8GdnQkvN = UINT64_C(10240);
constexpr uint64_t kQwen38ProjectionPackFp8GdnZN = UINT64_C(6144);
constexpr uint64_t kQwen38ProjectionPackPackedBytes =
    (kQwen38ProjectionPackK + UINT64_C(1)) / UINT64_C(2);
constexpr uint64_t kQwen38ProjectionPackScaleBytes =
    (kQwen38ProjectionPackK + UINT64_C(15)) / UINT64_C(16);
constexpr uint64_t kQwen38ProjectionPackFp8GdnWorkspaceBytes =
    kQwen38ProjectionPackK + sizeof(float);
constexpr const char *kQwen38ProjectionPackLogicalKernelId =
    "qwen38.projection_pack2.nvfp4.shared_activation.v1";
constexpr const char *kQwen38ProjectionPackDeviceSymbol =
    "sllm_qwen38_projection_pack2_nvfp4_shared_activation_v1";

static_assert(kQwen38ProjectionPackPackedBytes == UINT64_C(2560));
static_assert(kQwen38ProjectionPackScaleBytes == UINT64_C(320));
static_assert(kQwen38ProjectionPackPackedBytes +
                  kQwen38ProjectionPackScaleBytes ==
              SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES);
static_assert(kQwen38ProjectionPackFp8GdnWorkspaceBytes ==
              SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES);
static_assert(sizeof("qwen38.projection_pack2.nvfp4.shared_activation.v1") <=
              SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX);
static_assert(
    sizeof("sllm_qwen38_projection_pack2_nvfp4_shared_activation_v1") <=
    SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX);

bool qwen38_projection_pack_reserved_zero(const uint32_t *const values,
                                          const std::size_t count) noexcept {
  if (values == nullptr)
    return false;
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U)
      return false;
  }
  return true;
}

bool qwen38_projection_pack_binding_overlap(
    const Buffer *const left_buffer, const sllm_matmul::TensorMetadata &left,
    const Buffer *const right_buffer,
    const sllm_matmul::TensorMetadata &right) noexcept {
  return left_buffer == right_buffer &&
         sllm_matmul::intervals_overlap(left, right);
}

uint32_t qwen38_projection_pack_compute_grid(
    const sllm_matmul_kernel::KernelVariant variant,
    const uint64_t n) noexcept {
  using sllm_matmul_kernel::KernelVariant;
  switch (variant) {
  case KernelVariant::Nvfp4W4A4DecodeColumns128:
    return static_cast<uint32_t>((n + UINT64_C(127)) / UINT64_C(128));
  case KernelVariant::Nvfp4W4A4DecodeWave4Column32:
  case KernelVariant::Nvfp4W4A4DecodeActivationShared:
  case KernelVariant::Nvfp4W4A4DecodeScaleLut:
  case KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32:
  case KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32:
  case KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32:
  case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32:
    return static_cast<uint32_t>((n + UINT64_C(31)) / UINT64_C(32));
  case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64:
    return static_cast<uint32_t>((n + UINT64_C(63)) / UINT64_C(64));
  case KernelVariant::Fp8Native:
    return static_cast<uint32_t>(n);
  case KernelVariant::Fp8Emulation:
    return static_cast<uint32_t>((n + UINT64_C(255)) / UINT64_C(256));
  case KernelVariant::Nvfp4W4A4Packed:
  case KernelVariant::Nvfp4W4A4Decode:
    return static_cast<uint32_t>(n);
  default:
    return 0U;
  }
}

bool qwen38_projection_pack_fp8_gdn_role(const uint32_t role) noexcept {
  return role == SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_FP8_GDN_QKV_Z;
}

bool qwen38_projection_pack_valid_fp8_gdn_n(const uint64_t n,
                                            const uint32_t index) noexcept {
  return (index == 0U && n == kQwen38ProjectionPackFp8GdnQkvN) ||
         (index == 1U && n == kQwen38ProjectionPackFp8GdnZN);
}

void initialize_qwen38_projection_pack_dispatch_info(
    sllm_qwen38_projection_pack2_dispatch_info_t *const info,
    const uint64_t dispatch_id, const ElementwisePlan *const plan,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  const uint32_t info_version = info->info_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = info_version;
  info->backend = 1U;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 3U;
  const bool fp8_gdn =
      qwen38_projection_pack_fp8_gdn_role(plan->qwen38_projection_pack2_role);
  info->kernel_id =
      fp8_gdn
          ? SLLM_HIP_QWEN38_PROJECTION_PACK2_KERNEL_ID_FP8_GDN_SHARED_ACTIVATION_V1
          : SLLM_HIP_QWEN38_PROJECTION_PACK2_KERNEL_ID_NVFP4_SHARED_ACTIVATION_V1;
  info->workgroup_size_x = 256U;
  const uint32_t quantize_grid =
      fp8_gdn
          ? 1U
          : static_cast<uint32_t>(
                (kQwen38ProjectionPackScaleBytes + UINT64_C(7)) / UINT64_C(8));
  const uint32_t compute_grid0 = qwen38_projection_pack_compute_grid(
      plan->qwen38_projection_pack2_kernel_variants[0],
      plan->qwen38_projection_pack2_metadata[0].n);
  const uint32_t compute_grid1 = qwen38_projection_pack_compute_grid(
      plan->qwen38_projection_pack2_kernel_variants[1],
      plan->qwen38_projection_pack2_metadata[1].n);
  info->grid_size_x = quantize_grid + compute_grid0 + compute_grid1;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  info->m = kQwen38ProjectionPackM;
  info->k = kQwen38ProjectionPackK;
  info->n = plan->qwen38_projection_pack2_metadata[0].n;
  info->output_elements =
      plan->qwen38_projection_pack2_metadata[0].output_elements +
      plan->qwen38_projection_pack2_metadata[1].output_elements;
  info->workspace_bytes =
      fp8_gdn ? SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES
              : SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES;
  info->role = plan->qwen38_projection_pack2_role;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, sizeof(info->kernel_symbol),
      fp8_gdn ? "qwen38.projection_pack2.fp8_gdn.shared_activation.v1"
              : kQwen38ProjectionPackLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, sizeof(info->device_symbol),
      fp8_gdn ? "sllm_qwen38_projection_pack2_fp8_gdn_shared_activation_v1"
              : kQwen38ProjectionPackDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(
      info->gcn_arch_name, sizeof(info->gcn_arch_name), arch_name);
}

} // namespace

extern "C" sllm_status_t sllm_qwen38_projection_pack2_prepare(
    const sllm_context_t *const raw_context,
    const sllm_qwen38_projection_pack2_desc_t *const descriptor,
    sllm_qwen38_projection_pack2_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr)
      *raw_plan = nullptr;
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (raw_context == nullptr || descriptor == nullptr ||
        raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "Qwen3.8 projection-pack context, descriptor, or output is null");
    }
    if (descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->abi_version != SLLM_HIP_ABI_VERSION ||
        descriptor->op_version != SLLM_HIP_QWEN38_PROJECTION_PACK2_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
          "Qwen3.8 projection-pack descriptor ABI/version is unsupported");
    }
    const bool fp8_gdn = qwen38_projection_pack_fp8_gdn_role(descriptor->role);
    const bool nvfp4_gate_up =
        descriptor->role ==
        SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP;
    if ((!fp8_gdn && !nvfp4_gate_up) ||
        !qwen38_projection_pack_reserved_zero(
            descriptor->reserved,
            sizeof(descriptor->reserved) / sizeof(descriptor->reserved[0]))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_RESERVED_NONZERO,
          "Qwen3.8 projection-pack role or reserved fields are invalid");
    }
    float input_global_scale = 0.0F;
    std::memcpy(&input_global_scale, &descriptor->input_global_scale_f32_bits,
                sizeof(input_global_scale));
    if (fp8_gdn && descriptor->input_global_scale_f32_bits != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "Qwen3.8 FP8 GDN projection-pack input-global scale must be zero");
    }
    if (nvfp4_gate_up &&
        (!std::isfinite(input_global_scale) || input_global_scale <= 0.0F)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "Qwen3.8 projection-pack input-global scale must be finite and "
          "positive");
    }

    std::array<sllm_matmul::DescriptorMetadata, 2> metadata{};
    const sllm_tensor_binding_t weights_desc[2] = {descriptor->gate_weight,
                                                   descriptor->up_weight};
    const sllm_tensor_binding_t outputs_desc[2] = {descriptor->gate_output,
                                                   descriptor->up_output};
    for (uint32_t index = 0U; index != 2U; ++index) {
      sllm_matmul_desc_t matmul{};
      matmul.struct_size = sizeof(matmul);
      matmul.abi_version = SLLM_HIP_ABI_VERSION;
      matmul.op_version = fp8_gdn ? SLLM_HIP_MATMUL_FP8_VERSION
                                  : SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
      matmul.activation = descriptor->activation;
      matmul.weight = weights_desc[index];
      matmul.output = outputs_desc[index];
      const sllm_status_t status = sllm_matmul::validate_and_copy_descriptor(
          &matmul, &metadata[index], error_sink);
      if (status != SLLM_STATUS_OK)
        return status;
      const bool shape_valid =
          metadata[index].m == kQwen38ProjectionPackM &&
          metadata[index].k == kQwen38ProjectionPackK &&
          (fp8_gdn ? (metadata[index].fp8_outer &&
                      qwen38_projection_pack_valid_fp8_gdn_n(metadata[index].n,
                                                             index))
                   : (metadata[index].nvfp4_w4a4 &&
                      metadata[index].n == kQwen38ProjectionPackNvfp4N));
      if (!shape_valid) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            fp8_gdn ? "Qwen3.8 projection-pack FP8 GDN requires "
                      "M=1,K=5120,N=10240/6144"
                    : "Qwen3.8 projection-pack requires exact "
                      "M=1,K=5120,N=17408 NVFP4 W4A4 members");
      }
    }

    std::unique_ptr<Qwen38ProjectionPackPlan> candidate;
    Context *context = nullptr;
    Buffer *activation = nullptr;
    std::array<Buffer *, 2> weights{};
    std::array<Buffer *, 2> outputs{};
    bool accounting_reserved = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      activation =
          lookup<Buffer>(descriptor->activation.buffer, HandleKind::Buffer);
      for (uint32_t index = 0U; index != 2U; ++index) {
        weights[index] =
            lookup<Buffer>(weights_desc[index].buffer, HandleKind::Buffer);
        outputs[index] =
            lookup<Buffer>(outputs_desc[index].buffer, HandleKind::Buffer);
      }
      if (context == nullptr || activation == nullptr ||
          weights[0] == nullptr || weights[1] == nullptr ||
          outputs[0] == nullptr || outputs[1] == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Qwen3.8 projection-pack context or buffer handle is stale");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          activation->release_active || weights[0]->release_active ||
          weights[1]->release_active || outputs[0]->release_active ||
          outputs[1]->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Qwen3.8 projection-pack owner graph is releasing");
      }
      if (activation->context != context || weights[0]->context != context ||
          weights[1]->context != context || outputs[0]->context != context ||
          outputs[1]->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "Qwen3.8 projection-pack buffers must share one context");
      }
      if (metadata[0].activation.end_offset > activation->size_bytes ||
          metadata[0].weight.end_offset > weights[0]->size_bytes ||
          metadata[1].weight.end_offset > weights[1]->size_bytes ||
          metadata[0].output.end_offset > outputs[0]->size_bytes ||
          metadata[1].output.end_offset > outputs[1]->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "Qwen3.8 projection-pack tensor interval exceeds its buffer");
      }
      const std::array<Buffer *, 5> buffers = {
          activation, weights[0], weights[1], outputs[0], outputs[1]};
      const std::array<const sllm_matmul::TensorMetadata *, 5> tensors = {
          &metadata[0].activation, &metadata[0].weight, &metadata[1].weight,
          &metadata[0].output, &metadata[1].output};
      for (uint32_t left = 0U; left != 5U; ++left) {
        for (uint32_t right = left + 1U; right != 5U; ++right) {
          if (qwen38_projection_pack_binding_overlap(
                  buffers[left], *tensors[left], buffers[right],
                  *tensors[right])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "Qwen3.8 projection-pack tensor intervals overlap");
          }
        }
      }

      const sllm_status_t device_status =
          select_context_device(context, error_sink);
      if (device_status != SLLM_STATUS_OK)
        return device_status;
      hipDeviceProp_t properties{};
      const sllm_status_t property_status =
          get_context_device_properties(context, &properties, error_sink);
      if (property_status != SLLM_STATUS_OK)
        return property_status;
      char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME]{};
      if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                         sizeof(properties.gcnArchName)) ||
          !(matches_runtime_gcn_arch(arch_name, "gfx1030") ||
            matches_runtime_gcn_arch(arch_name, "gfx1201"))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "Qwen3.8 projection-pack requires exact gfx1030 or gfx1201");
      }
      std::array<sllm_matmul_kernel::KernelVariant, 2> variants{};
      std::optional<sllm_lowp::PreparedProviderPlan> provider;
      std::optional<sllm_lowp::PreparedProviderPlan> concrete;
      if (fp8_gdn) {
        for (uint32_t index = 0U; index != 2U; ++index) {
          variants[index] = ::sllm_matmul_kernel::select_fp8_outer_variant(
              metadata[index].m, metadata[index].k, metadata[index].n,
              arch_name, false);
          if (qwen38_projection_pack_compute_grid(variants[index],
                                                  metadata[index].n) == 0U) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_UNSUPPORTED,
                "Qwen3.8 projection-pack FP8 GDN provider rejected the exact "
                "target");
          }
        }
      } else {
        const sllm_lowp::ProviderRequest request =
            sllm_lowp::make_provider_request(
                sllm_lowp::MatmulFormat::Nvfp4W4A4,
                matmul_lowp_target(arch_name), kQwen38ProjectionPackM,
                kQwen38ProjectionPackNvfp4N, kQwen38ProjectionPackK);
        provider.emplace(sllm_lowp::prepare_provider_plan(request));
        variants[0] =
            matmul_lowp_kernel_variant(metadata[0], *provider, arch_name);
        variants[1] = variants[0];
        if (provider->supported()) {
          const auto selected =
              matmul_lowp_concrete_provider_plan(*provider, variants[0]);
          if (selected.has_value())
            concrete.emplace(*selected);
        }
        if (!concrete.has_value() ||
            qwen38_projection_pack_compute_grid(
                variants[0], kQwen38ProjectionPackNvfp4N) == 0U) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_UNSUPPORTED,
              "Qwen3.8 projection-pack NVFP4 provider rejected the exact "
              "target");
        }
      }

      if (!sllm_public_runtime::AccountingState::
              reserve_five_buffer_prepared_plan(
                  context->accounting, activation->accounting,
                  weights[0]->accounting, weights[1]->accounting,
                  outputs[0]->accounting, outputs[1]->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Qwen3.8 projection-pack prepared-plan accounting exhausted");
      }
      accounting_reserved = true;
      try {
        candidate = std::make_unique<Qwen38ProjectionPackPlan>(
            context, activation, weights, outputs, metadata,
            descriptor->input_global_scale_f32_bits);
        candidate->qwen38_projection_pack2_role = descriptor->role;
        candidate->qwen38_projection_pack2_kernel_variants = variants;
        if (concrete.has_value())
          candidate->matmul_provider_plan.emplace(*concrete);
        candidate->matmul_kernel_variant = variants[0];
        candidate->matmul_workspace_bytes =
            fp8_gdn ? SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES
                    : SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES;
        const hipError_t workspace_status = hipMalloc(
            &candidate->matmul_workspace,
            static_cast<std::size_t>(candidate->matmul_workspace_bytes));
        if (workspace_status != hipSuccess) {
          (void)sllm_public_runtime::AccountingState::
              release_five_buffer_prepared_plan(
                  context->accounting, activation->accounting,
                  weights[0]->accounting, weights[1]->accounting,
                  outputs[0]->accounting, outputs[1]->accounting);
          accounting_reserved = false;
          return hip_failure(error_sink, workspace_status,
                             "hipMalloc Qwen3.8 projection-pack workspace");
        }
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
        if (fp8_gdn && matches_runtime_gcn_arch(arch_name, "gfx1201")) {
          const auto *const shared_workspace =
              static_cast<const uint8_t *>(candidate->matmul_workspace);
          auto *const activation_scales = reinterpret_cast<float *>(
              const_cast<uint8_t *>(shared_workspace) + kQwen38ProjectionPackK);
          for (uint32_t index = 0U; index != 2U; ++index) {
            const auto &member =
                candidate->qwen38_projection_pack2_metadata[index];
            const auto *const weight_scales = reinterpret_cast<const float *>(
                static_cast<const uint8_t *>(weights[index]->device_pointer) +
                member.weight_scale_offset);
            std::lock_guard<std::mutex> lt_lock(context->matmul_lt_mutex);
            const hipError_t lt_status = create_fp8_lt_plan(
                context->matmul_lt_handle,
                &candidate->qwen38_projection_pack2_fp8_lt_plans[index],
                activation_scales, weight_scales, member.fp8_dtype, member.m,
                member.k, member.n, arch_name, true, false, nullptr);
            if (lt_status != hipSuccess) {
              (void)sllm_public_runtime::AccountingState::
                  release_five_buffer_prepared_plan(
                      context->accounting, activation->accounting,
                      weights[0]->accounting, weights[1]->accounting,
                      outputs[0]->accounting, outputs[1]->accounting);
              return hip_failure(error_sink, lt_status,
                                 "prepare Qwen3.8 FP8 GDN hipBLASLt solution");
            }
          }
        }
#endif
        const uintptr_t token =
            register_handle(candidate.get(), HandleKind::MatmulPlan);
        if (token == 0U) {
          (void)sllm_public_runtime::AccountingState::
              release_five_buffer_prepared_plan(
                  context->accounting, activation->accounting,
                  weights[0]->accounting, weights[1]->accounting,
                  outputs[0]->accounting, outputs[1]->accounting);
          accounting_reserved = false;
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "Qwen3.8 projection-pack plan handle allocation failed");
        }
        *raw_plan =
            reinterpret_cast<sllm_qwen38_projection_pack2_plan_t *>(token);
        (void)candidate.release();
        return SLLM_STATUS_OK;
      } catch (...) {
        if (accounting_reserved) {
          (void)sllm_public_runtime::AccountingState::
              release_five_buffer_prepared_plan(
                  context->accounting, activation->accounting,
                  weights[0]->accounting, weights[1]->accounting,
                  outputs[0]->accounting, outputs[1]->accounting);
        }
        throw;
      }
    }
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Qwen3.8 projection-pack prepare");
  }
}

extern "C" sllm_status_t sllm_qwen38_projection_pack2_plan_release(
    sllm_qwen38_projection_pack2_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "Qwen3.8 projection-pack plan is null");
    }
    Qwen38ProjectionPackPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan =
          lookup<Qwen38ProjectionPackPlan>(*raw_plan, HandleKind::MatmulPlan);
      if (plan == nullptr || !plan->qwen38_projection_pack2) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Qwen3.8 projection-pack plan handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight || plan->graph_pins != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Qwen3.8 projection-pack plan is releasing, in flight, or owned by "
            "a graph span");
      }
      plan->release_active = true;
      bool workspace_released = true;
      if (plan->matmul_workspace != nullptr) {
        const sllm_status_t device_status =
            select_context_device(plan->context, error_sink);
        const hipError_t workspace_status =
            device_status == SLLM_STATUS_OK ? hipFree(plan->matmul_workspace)
                                            : hipErrorInvalidValue;
        if (workspace_status == hipSuccess) {
          plan->matmul_workspace = nullptr;
          plan->matmul_workspace_bytes = 0U;
        } else {
          workspace_released = false;
        }
      }
      const bool accounting_released =
          workspace_released &&
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::
              release_five_buffer_prepared_plan(
                  plan->context->accounting, plan->input0->accounting,
                  plan->qwen38_projection_pack2_weights[0]->accounting,
                  plan->qwen38_projection_pack2_weights[1]->accounting,
                  plan->qwen38_projection_pack2_outputs[0]->accounting,
                  plan->qwen38_projection_pack2_outputs[1]->accounting);
      unregister_handle(*raw_plan);
      *raw_plan = nullptr;
      if (!accounting_released) {
        plan->context->poisoned.store(true);
        quarantine_plan = true;
      }
    }
    if (quarantine_plan) {
      poison_owner.retain(plan);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "Qwen3.8 projection-pack workspace/accounting release failed; plan "
          "quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Qwen3.8 projection-pack plan release");
  }
}

extern "C" sllm_status_t sllm_qwen38_projection_pack2_execute(
    const sllm_qwen38_projection_pack2_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_qwen38_projection_pack2_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr || dispatch_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "Qwen3.8 projection-pack execute argument is null");
    }
    if (dispatch_info->struct_size != sizeof(*dispatch_info) ||
        dispatch_info->abi_version != SLLM_HIP_ABI_VERSION ||
        dispatch_info->info_version !=
            SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
          "Qwen3.8 projection-pack dispatch info ABI is unsupported");
    }
    if (dispatch_info->reserved0 != 0U ||
        !qwen38_projection_pack_reserved_zero(
            dispatch_info->reserved, sizeof(dispatch_info->reserved) /
                                         sizeof(dispatch_info->reserved[0]))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_RESERVED_NONZERO,
          "Qwen3.8 projection-pack dispatch reserved fields must be zero");
    }

    Qwen38ProjectionPackPlan *plan = nullptr;
    Queue *queue = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<Qwen38ProjectionPackPlan>(raw_plan, HandleKind::MatmulPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr ||
          !plan->qwen38_projection_pack2) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Qwen3.8 projection-pack plan or queue handle is stale");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "Qwen3.8 projection-pack plan and queue devices differ");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight ||
          plan->input0->release_active ||
          plan->qwen38_projection_pack2_weights[0]->release_active ||
          plan->qwen38_projection_pack2_weights[1]->release_active ||
          plan->qwen38_projection_pack2_outputs[0]->release_active ||
          plan->qwen38_projection_pack2_outputs[1]->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Qwen3.8 projection-pack owner graph is busy");
      }
      if (!sllm_public_runtime::AccountingState::reserve_five_buffer_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting,
              plan->qwen38_projection_pack2_weights[0]->accounting,
              plan->qwen38_projection_pack2_weights[1]->accounting,
              plan->qwen38_projection_pack2_outputs[0]->accounting,
              plan->qwen38_projection_pack2_outputs[1]->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Qwen3.8 projection-pack submission accounting exhausted");
      }
      plan->in_flight = true;
      *completion_output = nullptr;
    }
    const auto rollback = [&]() noexcept {
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      (void)
          sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting,
              plan->qwen38_projection_pack2_weights[0]->accounting,
              plan->qwen38_projection_pack2_weights[1]->accounting,
              plan->qwen38_projection_pack2_outputs[0]->accounting,
              plan->qwen38_projection_pack2_outputs[1]->accounting);
      plan->in_flight = false;
    };

    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      rollback();
      return device_status;
    }
    hipDeviceProp_t properties{};
    const sllm_status_t property_status =
        get_context_device_properties(plan->context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      rollback();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME]{};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName)) ||
        !(matches_runtime_gcn_arch(arch_name, "gfx1030") ||
          matches_runtime_gcn_arch(arch_name, "gfx1201"))) {
      rollback();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "Qwen3.8 projection-pack requires exact gfx1030 or gfx1201");
    }
    const bool fp8_gdn =
        qwen38_projection_pack_fp8_gdn_role(plan->qwen38_projection_pack2_role);
    const uint64_t expected_workspace_bytes =
        fp8_gdn ? SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES
                : SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES;
    if (plan->matmul_workspace == nullptr ||
        plan->matmul_workspace_bytes != expected_workspace_bytes ||
        qwen38_projection_pack_compute_grid(
            plan->qwen38_projection_pack2_kernel_variants[0],
            plan->qwen38_projection_pack2_metadata[0].n) == 0U ||
        qwen38_projection_pack_compute_grid(
            plan->qwen38_projection_pack2_kernel_variants[1],
            plan->qwen38_projection_pack2_metadata[1].n) == 0U) {
      rollback();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "Qwen3.8 projection-pack prepared workspace/provider is invalid");
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    try {
      candidate =
          std::make_unique<Completion>(plan->context, queue, plan->input0, 0U,
                                       false, std::vector<uint8_t>{});
      candidate->qwen38_projection_pack2 = true;
      candidate->qwen38_projection_pack2_plan = plan;
      candidate->qwen38_projection_pack2_activation = plan->input0;
      candidate->qwen38_projection_pack2_weights =
          plan->qwen38_projection_pack2_weights;
      candidate->qwen38_projection_pack2_outputs =
          plan->qwen38_projection_pack2_outputs;

      if (queue_profiles_completions(queue)) {
        hipEvent_t native_event = nullptr;
        hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
        if (event_status != hipSuccess) {
          rollback();
          return hip_failure(error_sink, event_status,
                             "Qwen3.8 projection-pack event creation");
        }
        event_guard.adopt(plan->context, native_event);
        candidate->event = native_event;
        hipEvent_t timing_start_event = nullptr;
        event_status = hipEventCreateWithFlags(&timing_start_event, 0U);
        if (event_status != hipSuccess) {
          return rollback_unpublished_submission(
              candidate, event_guard,
              "Qwen3.8 projection-pack timing event creation failed",
              error_sink);
        }
        candidate->timing_start_event = timing_start_event;
      }

      uint64_t dispatch_id = 0U;
      uintptr_t token = 0U;
      try {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        dispatch_id = plan->context->next_dispatch_id;
        if (dispatch_id != 0U) {
          plan->context->next_dispatch_id =
              dispatch_id == std::numeric_limits<uint64_t>::max()
                  ? 0U
                  : dispatch_id + 1U;
          token = register_handle(candidate.get(), HandleKind::Completion);
        }
      } catch (...) {
        return rollback_unpublished_submission(
            candidate, event_guard,
            "Qwen3.8 projection-pack completion registry allocation failed",
            error_sink);
      }
      if (dispatch_id == 0U || token == 0U) {
        return rollback_unpublished_submission(
            candidate, event_guard,
            dispatch_id == 0U
                ? "Qwen3.8 projection-pack dispatch id exhausted"
                : "Qwen3.8 projection-pack completion handle allocation failed",
            error_sink);
      }
      if (candidate->event != nullptr)
        event_guard.release();
      if (candidate->timing_start_event != nullptr) {
        const hipError_t timing_status =
            hipEventRecord(candidate->timing_start_event, queue->stream);
        if (timing_status != hipSuccess) {
          return cleanup_failed_submission(
              candidate, token, timing_status,
              "Qwen3.8 projection-pack timing event record", queue, error_sink);
        }
      }

      hipError_t launch_status = hipSuccess;
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      auto *const workspace = static_cast<uint8_t *>(plan->matmul_workspace);
      const auto *const activation = reinterpret_cast<const uint16_t *>(
          static_cast<const uint8_t *>(plan->input0->device_pointer) +
          plan->qwen38_projection_pack2_metadata[0].activation.byte_offset);
      const auto weight_pointer = [](Buffer *const buffer,
                                     const uint64_t offset) -> const uint8_t * {
        return static_cast<const uint8_t *>(buffer->device_pointer) + offset;
      };
      const auto output_pointer = [](Buffer *const buffer,
                                     const uint64_t offset) -> uint16_t * {
        return reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(buffer->device_pointer) + offset);
      };
      if (fp8_gdn) {
        auto *const quantized_activation = workspace;
        auto *const activation_scales =
            reinterpret_cast<float *>(workspace + kQwen38ProjectionPackK);
        launch_status = ::sllm_matmul_kernel::launch_fp8_quantize(
            activation, quantized_activation, activation_scales,
            kQwen38ProjectionPackM, kQwen38ProjectionPackK, false,
            queue->stream);
        for (uint32_t index = 0U; launch_status == hipSuccess && index != 2U;
             ++index) {
          const auto &member = plan->qwen38_projection_pack2_metadata[index];
          Buffer *const weight = plan->qwen38_projection_pack2_weights[index];
          const auto *const fp8_weight =
              weight_pointer(weight, member.weight.byte_offset);
          const auto *const weight_scales = reinterpret_cast<const float *>(
              weight_pointer(weight, member.weight_scale_offset));
          uint16_t *const output =
              output_pointer(plan->qwen38_projection_pack2_outputs[index],
                             member.output.byte_offset);
          const auto variant =
              plan->qwen38_projection_pack2_kernel_variants[index];
          if (variant == sllm_matmul_kernel::KernelVariant::Fp8Native) {
            if (plan->qwen38_projection_pack2_fp8_lt_plans[index] == nullptr) {
              launch_status = hipErrorInvalidHandle;
            } else {
              std::lock_guard<std::mutex> lt_lock(
                  plan->context->matmul_lt_mutex);
              launch_status = launch_fp8_lt_plan(
                  plan->qwen38_projection_pack2_fp8_lt_plans[index],
                  quantized_activation, fp8_weight, output, queue->stream);
            }
          } else if (variant == sllm_matmul_kernel::KernelVariant::
                                    Fp8OuterDecodeGfx1030Dword8Wave4Col32) {
            launch_status = ::sllm_matmul_kernel::
                launch_fp8_outer_decode_gfx1030_dword8_wave4col32(
                    quantized_activation, activation_scales, fp8_weight,
                    weight_scales, output, kQwen38ProjectionPackM,
                    kQwen38ProjectionPackK, member.n, queue->stream);
          } else if (variant == sllm_matmul_kernel::KernelVariant::
                                    Fp8OuterDecodeGfx1030LdsLutWave4Col32) {
            launch_status = ::sllm_matmul_kernel::
                launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
                    quantized_activation, activation_scales, fp8_weight,
                    weight_scales, output, kQwen38ProjectionPackM,
                    kQwen38ProjectionPackK, member.n, queue->stream);
          } else if (variant ==
                     sllm_matmul_kernel::KernelVariant::
                         Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) {
            launch_status = ::sllm_matmul_kernel::
                launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32(
                    quantized_activation, activation_scales, fp8_weight,
                    weight_scales, output, kQwen38ProjectionPackM,
                    kQwen38ProjectionPackK, member.n, queue->stream);
          } else if (variant ==
                     sllm_matmul_kernel::KernelVariant::
                         Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) {
            launch_status = ::sllm_matmul_kernel::
                launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64(
                    quantized_activation, activation_scales, fp8_weight,
                    weight_scales, output, kQwen38ProjectionPackM,
                    kQwen38ProjectionPackK, member.n, queue->stream);
          } else if (variant == sllm_matmul_kernel::KernelVariant::
                                    Fp8OuterDecodeGfx1030Half2Wave4Col32) {
            launch_status = ::sllm_matmul_kernel::
                launch_fp8_outer_decode_gfx1030_half2_wave4col32(
                    quantized_activation, activation_scales, fp8_weight,
                    weight_scales, output, kQwen38ProjectionPackM,
                    kQwen38ProjectionPackK, member.n, queue->stream);
          } else {
            launch_status = ::sllm_matmul_kernel::launch_fp8_emulation(
                quantized_activation, activation_scales, fp8_weight,
                weight_scales, output, kQwen38ProjectionPackM,
                kQwen38ProjectionPackK, member.n, queue->stream);
          }
        }
      } else {
        auto *const packed_activation = workspace;
        auto *const activation_block_scales =
            workspace + kQwen38ProjectionPackPackedBytes;
        const auto &gate = plan->qwen38_projection_pack2_metadata[0];
        const auto *const shared_input_tensor_scale =
            reinterpret_cast<const float *>(
                weight_pointer(plan->qwen38_projection_pack2_weights[0],
                               gate.input_tensor_scale_offset));
        launch_status = ::sllm_matmul_kernel::launch_nvfp4_quantize(
            activation, packed_activation, activation_block_scales,
            shared_input_tensor_scale, kQwen38ProjectionPackM,
            kQwen38ProjectionPackK, queue->stream);
        for (uint32_t index = 0U; launch_status == hipSuccess && index != 2U;
             ++index) {
          const auto &member = plan->qwen38_projection_pack2_metadata[index];
          Buffer *const weight = plan->qwen38_projection_pack2_weights[index];
          const auto *const packed_weight =
              weight_pointer(weight, member.weight.byte_offset);
          const auto *const weight_block_scales =
              packed_weight + member.weight_value_bytes;
          const auto *const weight_tensor_scale =
              reinterpret_cast<const float *>(
                  weight_pointer(weight, member.weight_tensor_scale_offset));
          launch_status = ::sllm_matmul_kernel::launch_nvfp4_w4a4(
              packed_activation, activation_block_scales, packed_weight,
              weight_block_scales, weight_tensor_scale,
              shared_input_tensor_scale,
              output_pointer(plan->qwen38_projection_pack2_outputs[index],
                             member.output.byte_offset),
              kQwen38ProjectionPackM, kQwen38ProjectionPackK,
              kQwen38ProjectionPackNvfp4N,
              plan->qwen38_projection_pack2_kernel_variants[index],
              queue->stream);
        }
      }
#endif
      if (launch_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, launch_status,
            "Qwen3.8 projection-pack ordered kernel launch", queue, error_sink);
      }
      if (candidate->event != nullptr) {
        const hipError_t event_status =
            hipEventRecord(candidate->event, queue->stream);
        if (event_status != hipSuccess) {
          return cleanup_failed_submission(
              candidate, token, event_status,
              "Qwen3.8 projection-pack event record", queue, error_sink);
        }
      }
      initialize_qwen38_projection_pack_dispatch_info(
          dispatch_info, dispatch_id, plan, arch_name);
      *completion_output = reinterpret_cast<sllm_completion_t *>(token);
      (void)candidate.release();
      return SLLM_STATUS_OK;
    } catch (...) {
      rollback();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "unexpected exception in Qwen3.8 projection-pack execution");
    }
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Qwen3.8 projection-pack execute");
  }
}
