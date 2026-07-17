from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase6_window_driver_is_single_use_and_binds_the_exact_path_oracle_metric() -> None:
    source = (ROOT / "tools" / "run-aq4-phase6-service-window.sh").read_text(encoding="utf-8")

    assert "--confirm-single-window" in source
    assert "systemctl restart" not in source
    assert "exec 9< \"$lock\"" in source
    assert "flock -n 9" in source
    assert "run-aq4-phase3c-r9700-guard.py" in source
    assert "/opt/rocm/bin/amd-smi" in source
    assert "path-oracle-binary-staging" in source
    assert "stage-aq4-phase6-path-oracle-binary.py" in source
    assert "compare-aq4-phase6-final-output.py" in source
    assert "export-qwen35-aq4-path-oracle.py" in source
    assert "--prefill-m 1" in source
    assert "--device-index 1" in source
    assert "--visible-devices 1" in source
    assert "0.6151289249025698" in source
    assert "RuntimeDirectoryPreserve" in source
    assert source.index("--verify") < source.index('systemctl stop "$SERVICE"')

    phase3c_guards = (
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
        "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
        "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
        "ULLM_REQUIRE_HIP_TOP1_KERNEL",
    )
    for guard in phase3c_guards:
        assert guard in source


def test_phase6_final_output_comparator_keeps_the_bounded_metric_scope_explicit() -> None:
    source = (ROOT / "tools" / "compare-aq4-phase6-final-output.py").read_text(encoding="utf-8")
    assert "intersection_of_stored_indices" in source
    assert "not_full_vocabulary" in source
    assert "0.6151289249025698" in source
    assert "refusing to overwrite existing" in source
