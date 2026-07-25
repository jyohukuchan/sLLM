# SQ8_1 V620 kernel optimization and SQ8_0 comparison

## 前回の要点

`SQ8_1` は K=32、signed symmetric int8、separate FP16 scale plane、aligned payload の
reference runtime kernel まで実装済みだった。V620/gfx1030 では W8A8 が
`v_dot4c_i32_i8` を使える一方、`SQ8_0` は E4M3 fallback dequant を要するため、M=1 での実測勝敗を
thermal guard 下で確定する必要があった。full-model W8A8 logits gate は未通過で、W8A16 default /
explicit-only W8A8 の方針は固定されていた。

## 今回の変更点

- `SQ8_1` W8A16 を eight wave32 rows / 256-thread block に再構成し、LDS tree reduction を
  shuffle reduction に置換した。complete K=32 payload は二つの aligned `uint4` load のままである。
- explicit W8A8 は activation K=32 plane を eight rows で共有する tiled kernel にした。5120 columns
  では dynamic LDS 5,760 B、48 KiB cap を超える valid shape は fallback kernel を使う。W8A8 の
  per-output-row activation quantization work は eight-way amortization された。
- static ISA/resource audit は gfx1030 W8A16 の LDS 1024 -> 0 B / barrier 2 -> 0、W8A8 の
  VGPR/SGPR 53/59 -> 39/32 / barrier 2 -> 1 / spill 0 を確認した。gfx1030 W8A8 は
  `v_dot4c_i32_i8`、RDNA3/RDNA4 は `v_dot4_i32_iu8` signed-control path を確認した。
- CPU tail/zero/saturation boundary tests、V620 full-shape numerical gates、K=65 runtime tail
  differential、`SQ8_0` CPU regression、format artifact separation、AQ4 oracle を通過した。
- V620 `0000:03:00.0` / card0 だけを `hipDeviceGetPCIBusId` から選択した。junction sensor は
  同 BDF の `hwmon5/temp2_input`。three M=1 runs は 41–43 C、85 C guard/cooldown timeout は0件だった。
- matched `SQ8_0` M=1 reference 0.639007 ms に対して、optimized `SQ8_1` は W8A16 0.237362 ms
  （2.692× faster）、W8A8 0.249762 ms（2.558× faster）だった。`SQ8_1` resident bytes は 6.224% 多い。
  比較は同日・同一形状・同一 card0・同一 32/31 protocol の別 process evidence であり、co-dispatch
  A/B trace ではない。
- candidate/release/campaign/authorization/active manifest、`/opt/ullm`、systemd、`SQ8_0` / `AQ4_0`
  production code は変更していない。R9700 を実行していない。

## 次の行動

1. W8A8 full-model logits quality gate を別タスクで通すまで、W8A8 は explicit-only のまま維持する。
2. 必要なら M>1/prefill は card0 thermal guard 下で別途 staged に測る。今回の M>1 は未測定であり、
   85 C guard による中断ではない。
3. 旧 reference `SQ8_1` の elapsed-time baseline と profiler-derived occupancy は未測定なので、
   それらを必要とする最適化主張は別 evidence で追加する。
