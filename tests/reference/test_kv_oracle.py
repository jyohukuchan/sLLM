from __future__ import annotations

import pytest

from tests.reference.oracles import KVLayout


@pytest.mark.tier_h2
def test_kv_offsets_cover_block_boundaries_without_materializing_storage(load_json_fixture) -> None:
    fixture = load_json_fixture("kv_layout.json")
    layout = KVLayout.from_mapping(fixture["layout"])
    assert layout.blocks_per_sequence == 17

    for probe in fixture["probes"]:
        offset = layout.byte_offset(**probe)
        block, in_block = divmod(probe["token"], layout.block_size)
        element = probe["layer"] * layout.kv_planes + probe["kv"]
        element = element * layout.batch_size + probe["batch"]
        element = element * layout.blocks_per_sequence + block
        element = element * layout.block_size + in_block
        element = element * layout.kv_heads + probe["head"]
        element = element * layout.head_dim + probe["channel"]
        assert offset == element * layout.dtype_bytes
        assert offset >= 0


@pytest.mark.tier_h2
def test_kv_layout_rejects_unknown_fields_and_out_of_bounds_coordinates(load_json_fixture) -> None:
    fixture = load_json_fixture("kv_layout.json")
    invalid_descriptor = dict(fixture["layout"], future_layout=True)
    with pytest.raises(ValueError, match="fields"):
        KVLayout.from_mapping(invalid_descriptor)

    layout = KVLayout.from_mapping(fixture["layout"])
    probe = fixture["probes"][0]
    with pytest.raises(ValueError, match="out of bounds"):
        layout.byte_offset(**{**probe, "token": layout.max_tokens})
    with pytest.raises(ValueError, match="out of bounds"):
        layout.byte_offset(**{**probe, "token": -1})
