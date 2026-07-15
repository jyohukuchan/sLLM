# P3 capture family/producer pin cascade

## Scope

- `capture-aq4-p3-diagnostic-profile.py` のverified helper closureを更新した。
- capture専用testへproducer/family Git authorityとactual-v9回帰を追加した。
- launcher、maintenance、operator、actual artifactは変更していない。

## Producer authority

- commit: `c8becac66551f216de47d0cd935929afe60b3b96`
- tree: `088ac662dc686741d3affafe9b4ecc58cccea638`
- blob: `b070361d992fddc5749dba677ecd9d81f4ac6c06`
- source SHA-256: `a589c3e644d36132fb6054afdb15b27543d8e8181e3c737dcbd071d7c52e3d20`

## Family authority

- commit: `e4f8583a0fc710d2146f70d06b8b49eb42f04a16`
- tree: `be5ac39ea05b0b79223d974487c6cddda8d84f0c`
- blob: `8c318849838f85cf2f2a687aef260506bfa4097c`
- source SHA-256: `f8d32c340231e329f004d9e16192c02378f1fd58b8ab713e8efbbd3029b052d6`
- mapping SHA-256: `d5a159dff6776fc1229d1bacf415715154fb3bb2e3d3051f59bc3dca2ec03b29`

## Actual-v9 regression through capture-loaded helpers

- raw SHA-256: `1b2effa4c0ab44159919e32691d08329dec632cf56a1b22a78efc4fc607bf6f2`
- summary SHA-256: `7b122428ede8e7dd5cc8780386d2f1274ac679c4206990ba45d6a334c2e66c8e`
- kernel trace SHA-256: `a9833a65cffd6cbc3e974edcfb32fdf5657a17f6e90321085bae734c51a07131`
- session ID: `3fc38e24c47e904242a3d3f12c9bd3250e53097d62dababbaec5efc4af34e0dc`
- producer full resident validation: passed
- runs: 12
- device lock raw/summary binding: exact
- live preflight raw/summary binding: exact
- kernel rows: 12,263
- unknown: 0
- multiple-family match: 0

## Verification

- capture full tests: 59 passed, 1 skipped
- capture source/test `py_compile`: passed
- `git diff --check`: passed
- GPU、service、actualは実行していない。

## Integration note

launcherやready/operator authorityはこのcapture commit確定後に別laneで連鎖更新する。
