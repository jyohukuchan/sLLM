# SQ8_0 CDNA3 B control オフライン閉鎖と次回 MI300X runner

## 前回の要点

- MI300X/gfx942 では A′ の fragment/lane と実形状 5 case が CPU expectation
  に対して pass した一方、B control の `k_or_v_tail_id1` は expected
  `0.53125` に対して observed `0.03125` だった。
- その A′ pass は `ULLM_SMOKE_SKIP_B_CONTROL` を使ったもので、B の pass や
  full-model の pass を意味しなかった。occupancy/residency と full-model は
  未確認のままだった。
- 初回 rental の preserved lease は約 2 時間だが、linker、Rust/Python 依存、
  model/image/download の準備により運用上の消費時間は約 5 時間だった。stage 別の
  wall-clock は保存されていない。

## 今回の変更点

- B failure の 0.5 差を CPU oracle で再現した。旧 hipBLAS の `OP_N, lda=N` は
  row-major `W[N,K]` を strided permutation として読んでおり、K=0 の `0.03125`
  だけを残して final-K の `0.5` を落としていた。
- B call を `OP_T/OP_N`、`m=N,n=M,k=K`、`lda=ldb=K,ldc=N` に修正した。完全な
  physical fixture を使う CPU test は旧 `0.03125`、修正/expected `0.53125` を
  厳密に確認して pass した。gfx942 実機での修正確認はまだ行っていない。
- A′/B native gfx942 build と、generic SQ8_0 HIPRTC 27 kernel の gfx942 compile
  audit を GPU launch なしで pass した。ISA audit は 912
  `v_mfma_f32_16x16x32_fp8_fp8`、最大 VGPR 454 / SGPR 62 / AGPR 198 / LDS
  49,152 B、spill/private 0 を記録した。実効 occupancy は未確認である。
- `tools/run-sq8-cdna3-mi300x-validation.sh` を追加した。P0 を preflight,
  CPU, HIPRTC, build, ISA, physical の順に固定し、resume stamp、source
  fingerprint、stage timing、Cargo linker/mold override、B non-skip gate を
  持たせた。ローカルで shell syntax、dry-run、HIPRTC、build stage を確認した。
- gfx942 full-model profile はまだ無く、R9700/gfx1201 identity gate も残るため、
  generic compile pass を full-model pass と扱わない。経路 A の手書き MFMA は
  着手していない。

## 次の行動

1. fixed commit と warm Cargo cache を GPU lease 前に staging し、v0.2 runner を
   `--stage all` で実行する。
2. B を skip せずに `k_or_v_tail_id1` を含む全 5 case の B/CPU/A′ differential
   を採取する。失敗なら同一条件の evidence 再現を 1 回だけ行い、実装変更は
   持ち帰る。
3. P0 が全 pass した場合のみ、実 function に結び付いた occupancy/residency
   query と、その後の full-model gfx942 integration の別 scope を判断する。
4. service、release、campaign、authorization、`/etc/ullm/served-models/active.json`
   には引き続き触れない。activation は明示的人間承認の外では行わない。
