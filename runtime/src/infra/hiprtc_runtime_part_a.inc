// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipRtcRuntime {
public:
    bool compile_wmma_fp8_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            wmma_fp8_probe_kernel_source(),
            "ullm_wmma_fp8_probe.hip",
            code,
            error);
    }

    bool compile_wmma_fp8_qk_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            wmma_fp8_qk_probe_kernel_source(),
            "ullm_wmma_fp8_qk_probe.hip",
            code,
            error);
    }

    bool compile_rocwmma_fp8_qk_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            rocwmma_fp8_qk_probe_kernel_source(),
            "ullm_rocwmma_fp8_qk_probe.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_rocwmma_fp8_attn_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            rocwmma_fp8_attn_probe_kernel_source(),
            "ullm_rocwmma_fp8_attn_probe.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, aq4_kernel_source(), "ullm_aq4_dequant_f32.hip", code, error);
    }

    bool compile_aq4_row_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, aq4_row_kernel_source(), "ullm_aq4_row_f32.hip", code, error);
    }

    bool compile_aq4_matvec_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        const std::string source = aq4_matvec_kernel_source_for_arch(arch);
        return compile_kernel(arch, source.c_str(), "ullm_aq4_matvec_f32.hip", code, error);
    }

    bool compile_aq4_matvec_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_batch_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_batch_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_tiled_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_tiled_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_tiled_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_register_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_register_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_register_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_wmma_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 WMMA production direct-output source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_ragged_m_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_ragged_m_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 WMMA ragged-M prototype source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_ragged_m_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_prototype_v3_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            aq4_matvec_wmma_prototype_v3_kernel_source(),
            "ullm_aq4_gemm_wmma_prototype_v3_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_group8_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            aq4_matvec_wmma_group8_prototype_kernel_source(),
            "ullm_aq4_gemm_wmma_group8_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_group8_ragged_m_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_group8_ragged_m_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 group8 ragged-M WMMA prototype source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_group8_ragged_m_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_top1_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_top1_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_top1_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_add_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_add_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_add_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_pair_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_pair_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_pair_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_triple_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_triple_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_triple_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_triple_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_triple_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_triple_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_qkv_z_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_qkv_z_gate_beta_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_silu_mul_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_silu_mul_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_silu_mul_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_gate_beta_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_matvec_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, matvec_kernel_source(), "ullm_matvec_f32.hip", code, error);
    }

    bool compile_matvec_bf16_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            matvec_bf16_kernel_source(),
            "ullm_matvec_bf16_f32.hip",
            code,
            error);
    }

    bool compile_gemma_bf16_matmul_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            gemma_bf16_matmul_kernel_source(),
            "ullm_gemma_bf16_matmul_f32.hip",
            code,
            error);
    }

    bool compile_bf16_row_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            bf16_row_kernel_source(),
            "ullm_bf16_row_f32.hip",
            code,
            error);
    }

    bool compile_top1_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, top1_kernel_source(), "ullm_top1_f32.hip", code, error);
    }

    bool compile_top1_pairs_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            top1_pairs_kernel_source(),
            "ullm_top1_pairs_f32.hip",
            code,
            error);
    }

    bool compile_rmsnorm_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        const std::string source = rmsnorm_kernel_source_for_arch(arch);
        return compile_kernel(arch, source.c_str(), "ullm_rmsnorm_f32.hip", code, error);
    }

    bool compile_rmsnorm_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = rmsnorm_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_rmsnorm_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_segmented_rmsnorm_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            segmented_rmsnorm_kernel_source(),
            "ullm_segmented_rmsnorm_f32.hip",
            code,
            error);
    }

    bool compile_segmented_rmsnorm_silu_mul_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = segmented_rmsnorm_silu_mul_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_segmented_rmsnorm_silu_mul_f32.hip",
            code,
            error);
    }

    bool compile_segmented_rmsnorm_silu_mul_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_segmented_rmsnorm_silu_mul_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_silu_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, silu_mul_kernel_source(), "ullm_silu_mul_f32.hip", code, error);
    }

    bool compile_gelu_tanh_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, gelu_tanh_mul_kernel_source(), "ullm_gelu_tanh_mul_f32.hip", code, error);
    }

    bool compile_sigmoid_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, sigmoid_mul_kernel_source(), "ullm_sigmoid_mul_f32.hip", code, error);
    }

    bool compile_qwen35_split_q_gate_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_split_q_gate_kernel_source(),
            "ullm_qwen35_split_q_gate_f32.hip",
            code,
            error);
    }

    bool compile_qwen35_qk_norm_rope_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_kernel_source(),
            "ullm_qwen35_qk_norm_rope_f32.hip",
            code,
            error);
    }

    bool compile_qwen35_qk_norm_rope_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_batch_kernel_source(),
            "ullm_qwen35_qk_norm_rope_batch_f32.hip",
            code,
            error);
    }

    bool compile_qwen35_qk_norm_rope_paged_kv_write_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_paged_kv_write_kernel_source(),
            "ullm_qwen35_qk_norm_rope_paged_kv_write_f32.hip",
            code,
            error);
    }

    bool compile_add_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, add_kernel_source(), "ullm_add_f32.hip", code, error);
    }

    bool compile_rope_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, rope_kernel_source(), "ullm_rope_f32.hip", code, error);
    }

    bool compile_causal_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_kernel_source(),
            "ullm_causal_attn_f32.hip",
            code,
            error);
    }

    bool compile_causal_attn_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_flash2_f32_kernel_source(),
            "ullm_causal_attn_f32_flash2.hip",
            code,
            error);
    }

    bool compile_causal_attn_batch_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_batch_kernel_source(),
            "ullm_causal_attn_batch_f32.hip",
            code,
            error);
    }

    bool compile_causal_attn_batch_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_batch_flash2_f32_kernel_source(),
            "ullm_causal_attn_batch_f32_flash2.hip",
            code,
            error);
    }

    bool compile_decode_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            decode_attn_kernel_source(),
            "ullm_decode_attn_f32.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_attn_kernel_source(),
            "ullm_cached_prefix_attn_f32.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = cached_prefix_attn_fp8_e4m3_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_cached_prefix_attn_fp8_e4m3.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_f32_kernel_source(),
            "ullm_cached_prefix_attn_f32_flash2.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            cached_prefix_flash2_f32_staged_wave32_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_sq8_0_cached_prefix_attn_f32_flash2_staged_wave32_prototype.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_f32_flash2_gqa_grouped_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_f32_gqa_grouped_serial_kernel_source(),
            "ullm_cached_prefix_attn_f32_flash2_gqa_grouped.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_fp8_e4m3_kernel_source(),
            "ullm_cached_prefix_attn_fp8_e4m3_flash2.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_rocwmma_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_rocwmma_fp8_e4m3_kernel_source(),
            "ullm_cached_prefix_attn_fp8_e4m3_rocwmma.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_sq_fp8_matvec_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_batch_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_pair_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_pair_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_triple_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_triple_f32.hip",
            code,
            error);
    }

    bool compile_sq8_1_matvec_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq8_1_matvec_kernel_source(),
            "ullm_sq8_1_matvec_f32.hip",
            code,
            error);
    }

    bool compile_paged_decode_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_decode_attn_kernel_source_for_arch(arch).c_str(),
            "ullm_paged_decode_attn_f32.hip",
            code,
            error);
    }

    bool compile_paged_kv_write_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_kv_write_kernel_source(),
            "ullm_paged_kv_write_f32.hip",
            code,
            error);
    }

    bool compile_paged_chunk_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_chunk_kernel_source(),
            "ullm_paged_chunk_f32.hip",
            code,
            error);
    }

    bool compile_paged_causal_gqa_chunk_wmma_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            paged_causal_gqa_chunk_wmma_kernel_source(),
            "ullm_paged_causal_gqa_chunk_wmma_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_depthwise_conv1d_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            depthwise_conv1d_kernel_source(),
            "ullm_depthwise_conv1d_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_qkv_prepare_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = linear_attn_qkv_prepare_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_linear_attn_qkv_prepare_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_qkv_prepare_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = linear_attn_qkv_prepare_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_linear_attn_qkv_prepare_shuffle_prototype_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_qkv_prepare_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_qkv_prepare_batch_kernel_source(),
            "ullm_linear_attn_qkv_prepare_batch_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_gate_beta_kernel_source(),
            "ullm_linear_attn_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_recurrent_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_recurrent_kernel_source(),
            "ullm_linear_attn_recurrent_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_recurrent_generic_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_recurrent_generic_kernel_source(),
            "ullm_linear_attn_recurrent_generic_f32.hip",
            code,
            error);
    }

private:
    using hiprtc_create_program_fn =
        int (*)(void **, const char *, const char *, int, const char *const *, const char *const *);
    using hiprtc_compile_program_fn = int (*)(void *, int, const char *const *);
    using hiprtc_get_code_size_fn = int (*)(void *, size_t *);
    using hiprtc_get_code_fn = int (*)(void *, char *);
    using hiprtc_get_log_size_fn = int (*)(void *, size_t *);
    using hiprtc_get_log_fn = int (*)(void *, char *);
    using hiprtc_destroy_program_fn = int (*)(void **);
    using hiprtc_get_error_string_fn = const char *(*)(int);

    bool compile_kernel(
        const std::string &arch,
        const char *source,
        const char *name,
        std::vector<char> *code,
        std::string *error,
        const std::vector<std::string> &extra_options = {}) {
        load_once();
        if (!available()) {
            append_error(error, "HIPRTC is not available");
            return false;
        }
        if (code == nullptr) {
            append_error(error, "HIPRTC output code pointer is null");
            return false;
        }

        void *program = nullptr;
        int status = hiprtc_create_program_(
            &program,
            source,
            name,
            0,
            nullptr,
            nullptr);
        if (status != 0 || program == nullptr) {
            append_error(error, "hiprtcCreateProgram failed: " + error_string(status));
            return false;
        }

        const auto destroy_program = [&]() {
            if (program != nullptr) {
                hiprtc_destroy_program_(&program);
            }
        };

        std::vector<std::string> option_storage;
        option_storage.reserve(3u + extra_options.size());
        option_storage.push_back("--offload-arch=" + arch);
        option_storage.push_back("--std=c++17");
        option_storage.push_back("-O3");
        option_storage.insert(option_storage.end(), extra_options.begin(), extra_options.end());
        std::vector<const char *> options;
        options.reserve(option_storage.size());
        for (const std::string &option : option_storage) {
            options.push_back(option.c_str());
        }
        status = hiprtc_compile_program_(
            program,
            static_cast<int>(options.size()),
            options.data());
        if (status != 0) {
            append_error(
                error,
                "hiprtcCompileProgram failed for " + arch + ": " + error_string(status) +
                    "\n" + program_log(program));
            destroy_program();
            return false;
        }

        size_t code_size = 0;
        status = hiprtc_get_code_size_(program, &code_size);
        if (status != 0 || code_size == 0) {
            append_error(error, "hiprtcGetCodeSize failed: " + error_string(status));
            destroy_program();
            return false;
        }
        code->assign(code_size, '\0');
        status = hiprtc_get_code_(program, code->data());
        if (status != 0) {
            append_error(error, "hiprtcGetCode failed: " + error_string(status));
            destroy_program();
            return false;
        }
        destroy_program();
        return true;
    }

    static std::vector<std::string> rocwmma_include_options() {
        std::vector<std::string> options;
        const char *rocm_path = std::getenv("ROCM_PATH");
        if (rocm_path != nullptr && rocm_path[0] != '\0') {
            options.push_back(std::string("-I") + rocm_path + "/include");
        }
        options.push_back("-I/opt/rocm/include");
        options.push_back("-I/opt/rocm-7.2.1/include");
        return options;
    }
