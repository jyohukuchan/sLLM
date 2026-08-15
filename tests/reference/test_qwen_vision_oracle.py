from __future__ import annotations

import hashlib

import numpy as np
import pytest


@pytest.mark.tier_h2
def test_qwen35_patch_order_and_normalization_oracle() -> None:
    """Independent array-index oracle for the fixed no-resize processor case."""
    height = width = 256
    rows, columns = np.mgrid[0:height, 0:width]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = columns % 251
    rgb[:, :, 1] = rows % 241
    rgb[:, :, 2] = (rows + columns) % 239

    blocks = []
    for block_row in range(8):
        for block_column in range(8):
            for merge_row in range(2):
                for merge_column in range(2):
                    y = (block_row * 2 + merge_row) * 16
                    x = (block_column * 2 + merge_column) * 16
                    tile = rgb[y : y + 16, x : x + 16]
                    for channel in range(3):
                        normalized = (
                            tile[:, :, channel].astype(np.float32) / np.float32(255.0)
                            - np.float32(0.5)
                        ) / np.float32(0.5)
                        # Qwen temporal patch size 2 duplicates a still image.
                        blocks.extend((normalized.ravel(), normalized.ravel()))

    patches = np.concatenate(blocks).astype("<f4", copy=False)
    assert patches.shape == (256 * 1_536,)
    assert hashlib.sha256(patches.tobytes()).hexdigest() == (
        "f1e51663a9ea2832a67e5157ca11bc42206aaf186897866dab8c779d08ee3a2e"
    )
    assert np.array_equal(
        patches[:4], np.array([-1.0, -0.99215686, -0.9843137, -0.9764706], np.float32)
    )


@pytest.mark.tier_h2
def test_qwen35_pixel_area_boundaries() -> None:
    minimum = 65_536
    maximum = 16_777_216
    assert 255 * 257 == minimum - 1
    assert 256 * 256 == minimum
    assert 1 * 65_537 == minimum + 1
    assert 4_095 * 4_097 == maximum - 1
    assert 4_096 * 4_096 == maximum
    assert 1 * 16_777_217 == maximum + 1
