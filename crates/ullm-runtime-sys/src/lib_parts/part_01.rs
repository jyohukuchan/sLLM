#[cfg(test)]
mod tests {
    use super::*;

    include!("../test_parts/part_00.rs");
    include!("../test_parts/part_01.rs");
    include!("../test_parts/part_02.rs");
    include!("../test_parts/part_03.rs");
    include!("../test_parts/aq4_wide_load_prototype.rs");
    include!("../test_parts/aq4_matvec_add_wide_load_prototype.rs");
    include!("../test_parts/aq4_fused_wide_load_prototype.rs");
    include!("../test_parts/aq4_matvec_shuffle_prototype.rs");
    include!("../test_parts/aq4_matvec_triple_shuffle_prototype.rs");
    include!("../test_parts/rmsnorm_shuffle_prototype.rs");
    include!("../test_parts/segmented_rmsnorm_silu_mul_shuffle_prototype.rs");
    include!("../test_parts/linear_attn_qkv_prepare_shuffle_prototype.rs");
    include!("../test_parts/sq8_1_matvec.rs");
}
