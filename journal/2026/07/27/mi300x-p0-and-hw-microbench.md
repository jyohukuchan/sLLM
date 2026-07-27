## 前回の要点

MI300X/gfx942 の P0 は物理実機の A′ ABI/fragment と B control を確認するための
最優先 gate であり、P0 後にだけ naive WMMA/MFMA hardware microbenchmark を実行する。

## 今回の変更点

- cold rental host に Rust minimal を導入し、`cargo fetch --locked`（29 archive）後に P0 を実行した。
- `cc` linker と空 Rust flags の rental override を実機で確認した。P0 は
  preflight 0 s / CPU 82 s / HIPRTC 32 s / build 54 s / ISA 4 s / physical 2 s で pass。
- A′ の 5 形状は全て誤差ゼロ。`ULLM_SMOKE_SKIP_B_CONTROL` を解除した B control の
  実機 sentinel は `0.53125` で、旧誤読の `0.03125` ではなかった。
- MI300X hardware microbenchmark を完走し、read/copy/triad は
  4774.766 / 3850.290 / 3878.615 GB/s、実形状 BF16/FP8 は
  35.211 / 71.982 TFLOPS だった。これは LDS タイリングなしの実効値である。
- amd-smi 26.2.2 が廃止した `--violation` を telemetry command から除き、
  ROCm 7.2.4 でも clock/power/temperature を記録できるようにした。
- `cargo test` の test-binary 引数の後ろに `--offline --locked` を追加していた runner
  を修正し、B-control 単体 test を正しく再実行した。

## 次の行動

- 実効 occupancy は runtime module API query をまだ実装・取得していないため未確認。
- MI300X の短い bandwidth phase は amd-smi CLI の応答より短く、active clock は未確認。
  次回は低オーバーヘッドの telemetry または計測反復を長くして同時サンプルを得る。
- naive kernel の GEMM 比は予想 6.83×を大きく下回った。matrix-core peak と混同せず、
  tiled kernel を別ベンチとして扱う。
