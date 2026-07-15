# P3 producer family pin cascade

## Scope

- `build-aq4-p3-selection-raw.py` のfamily helper authorityを更新した。
- producer専用testへGit authority、actual-v9 full resident pair、全kernel family分類の回帰を追加した。
- capture、launcher、maintenance、actual artifactは変更していない。

## Authority

- family commit: `e4f8583a0fc710d2146f70d06b8b49eb42f04a16`
- family tree: `be5ac39ea05b0b79223d974487c6cddda8d84f0c`
- family blob: `8c318849838f85cf2f2a687aef260506bfa4097c`
- family source SHA-256: `f8d32c340231e329f004d9e16192c02378f1fd58b8ab713e8efbbd3029b052d6`
- mapping SHA-256: `d5a159dff6776fc1229d1bacf415715154fb3bb2e3d3051f59bc3dca2ec03b29`

producerはsource SHAだけでなくmapping SHAもload直後に照合する。commit/tree/blobは回帰試験で現在のGit objectと一致することを確認する。

## Actual-v9 regression

- resident raw SHA-256: `1b2effa4c0ab44159919e32691d08329dec632cf56a1b22a78efc4fc607bf6f2`
- resident summary SHA-256: `7b122428ede8e7dd5cc8780386d2f1274ac679c4206990ba45d6a334c2e66c8e`
- kernel trace SHA-256: `a9833a65cffd6cbc3e974edcfb32fdf5657a17f6e90321085bae734c51a07131`
- session ID: `3fc38e24c47e904242a3d3f12c9bd3250e53097d62dababbaec5efc4af34e0dc`
- full resident pair: accepted
- runs: 12
- device lock raw/summary binding: exact
- live preflight raw/summary binding: exact
- kernel rows: 12,263
- unknown: 0
- multiple-family match: 0

## Verification

- producer + family: 144 passed
- producer + selector + family: 170 passed
- producer/test `py_compile`: passed
- `git diff --check`: passed
- GPU、service、actualは実行していない。

## Integration note

capture側にはproducerとfamily helperの旧pinが残るため、このproducer commit確定後に別laneで連鎖更新する。
