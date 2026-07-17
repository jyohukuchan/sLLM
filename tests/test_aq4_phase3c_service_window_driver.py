from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_service_window_driver_is_explicitly_single_use_and_preserves_lock_contract() -> None:
    source = (ROOT / "tools" / "run-aq4-phase3c-service-window.sh").read_text(encoding="utf-8")

    assert "--confirm-single-window" in source
    assert "systemctl restart" not in source
    assert "exec 9< \"$lock\"" in source
    assert "flock -n 9" in source
    assert "run-aq4-phase3c-r9700-guard.py" in source
    assert "guard-before" in source
    assert "guard-after" in source
    assert "test -x /opt/rocm/bin/amd-smi" not in source
    assert "! -x /opt/rocm/bin/amd-smi" in source
    assert 'TRACE_STAGE_DIR="$OUT/trace-binary-staging"' in source
    assert 'TRACE_BIN="$TRACE_STAGE_DIR/ullm-aq4-differential-trace"' in source
    assert "stage-aq4-phase3c-trace-binary.py" in source
    assert "--verify" in source
    assert "staged trace binary identity contract failed" in source
    assert "PHASE3C_TRACE_UNSET_ENV=(" in source
    assert "--print-phase3c-trace-guard-requirements" in source
    assert "trace-guard-diagnostic.json" in source
    assert source.index("--print-phase3c-trace-guard-requirements") < source.index(
        'systemctl stop "$SERVICE"'
    )

    required_guards = (
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
    for guard in required_guards:
        assert f"{guard}=1" in source

    assert "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL\n" not in source.split(
        "PHASE3C_TRACE_UNSET_ENV=(", 1
    )[1].split(")\nTRACE_ENV=", 1)[0]
