#!/usr/bin/env python3
"""External EXL3 GPU encoder structural checks, not an independent codec oracle."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers, quantize_tiles

    assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1201", props
    torch.manual_seed(1234)
    rows = []
    for bits in (2, 4, 6, 8):
        for count in (1, 17, 65):
            source = torch.randn(count, 256, device="cuda")
            reconstructed, indices = quantize_tiles(source, {"K": bits, "mul1": True})
            decoded = torch.empty_like(reconstructed)
            ext.decode(indices, decoded, False, True)
            torch.cuda.synchronize()
            assert torch.isfinite(reconstructed).all().item(), "Nonfinite reconstruction"
            assert torch.equal(reconstructed, decoded), "Encode/decode disagree"
            unsigned = indices.to(torch.int32) & 65535
            assert torch.equal(unsigned[:, 0] >> bits,
                               unsigned[:, -1] & ((1 << (16 - bits)) - 1)), "Trellis is not closed"
            mse = (source - reconstructed).square().mean().item()
            assert mse < source.square().mean().item(), "Worse than zero reconstruction"
            row = {"bits": bits, "tiles": count, "mse": mse,
                   "finite": True, "decode_equal": True, "tail_biting": True}
            rows.append(row)
            print(json.dumps(row), flush=True)
        get_temp_buffers.cache_clear()
    report = {"gpu": props.gcnArchName, "torch": torch.__version__,
              "torch_hip": torch.version.hip, "codebook": "mul1", "cases": rows,
              "scope": "Structural roundtrip; encoder and decoder share codebook logic."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
