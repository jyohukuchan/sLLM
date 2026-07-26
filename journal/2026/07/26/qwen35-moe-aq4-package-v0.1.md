# BN: Qwen3.5-35B-A3B AQ4_0 MoE package

Date: 2026-07-26

## 前回の要点

- BI は Qwen3.5-35B-A3B の text decoder が 40 層すべて MoE、`256 experts / top-8 /
  I=512 / shared I=512` であることを確認し、routing / gather-scatter / grouped GEMM の
  substrate を完成させた。
- raw BF16 text decoder は R9700 31.859 GiB に収まらず、expert 重みが decoder の大半を
  占めるので、既存 `AQ4_0` を使う expert-only quantization が必要だった。
- host RAM は checkpoint 全体を同時に保持できないため、全モデル load や HF full capture を
  前提にしない、shard/tensor streaming と再開可能な作業が必要だった。

## 今回の変更点

- safetensors header audit で BI の数値を追認した。complete payload は
  `71,903,655,008 B` / `66.965497 GiB`、text decoder（`lm_head`除外）は
  `68,304,112,256 B` / `63.613162 GiB`、routed + shared experts は
  `64,676,331,520 B` / `60.234528 GiB`。BI の 63.613 / 60.235 / 66.965 GiB と一致する。
- routed expert の 3-D BF16 `gate_up_proj [256,1024,2048]` と
  `down_proj [256,2048,512]` のみを対象とする narrow converter を追加した。既存
  `ullm-quant` の rank>=2 stream converter を使い、各 tensor を独立 staging/再読込検証する。
  `quantization-state.json` により resume でき、完走後の `--resume` は 0.282 秒で再利用を
  確認した。CPU 8 job / nice 10 のみを使い、GPU・service は使わなかった。
- 方式は既存 strict `AQ4_0` の `aq4_e4m3_g8_ts_flloyd16`（G8、effective 5 bpp）を採用した。
  G16 より品質がよく、G8 package は 80 quantized + 613 raw passthrough + 2 codebook、
  `25,029,380,864 B` / `23.310427 GiB` になった。`SQ8_0` は routed expert だけで約30 GiBを
  要し、non-expert/KV を残せず、既存 BF16-source path もこの rank-3 MoE を扱わないため不採用。
- router、shared expert、attention、embedding、norm、`lm_head`、その他の text tensor は raw
  passthrough とした。40 router は source/package SHA-256 が全て一致し、PyTorch BF16
  linear → FP32 softmax → top-8 の 1,280 条件付き入力で選択は 0 変化だった。boundary tie 92 は
  記録し、隠さなかった。
- codebook は expert ごとに分けず、routed down と gate/up 各一つを全40×256 expertで共有した。
  held-out comparison で global down は median/p95/max relative MSE
  `0.003997/0.004244/0.005328`、per-expert は `0.004125/0.004516/0.008668`、gate/up の
  per-expert は max `1.448716` の不安定 tail を示した。per-layer にも layer-0 tail 悪化があり、
  global が品質/size の両方で根拠ある選択になった。
- final package を全量 re-read/dequantize した。80 tensor の relative MSE は
  `0.003603673..0.004363885`（mean `0.003634245`）、max abs は
  `0.005326890..0.043730080`（mean `0.013711253`）。relative-MSE robust outlier は layer 0
  down、layer 32/33 gate/up、layer 39 down。layer 39 down は max-abs の唯一の outlier
  `0.043730080` であり、evidence に tensor ごとに残した。
- CPU-only layer-streaming forward を追加した。HF の実 decoder layer を一層ずつ作り、source
  BF16 expert rows と AQ4_0 decoded rows の二経路を separate cache で 8-token / 40-layer 実行する。
  source-vs-source control は selected set/order 0/320 change、final hidden 0 で、full checkpoint
  materialization を回避できた。
- 重要な未達を確認した。同じ streaming check では AQ4_0 G8 は ordered top-k を 238/320、
  selected expert set を 105/320 変え、final hidden relative L2 は `0.076012410` だった。
  router 重み自体は raw/完全一致でも、lossy expert 出力が後段 router 入力に累積するためである。
  product metadata を `not_passed` とし、これを serving 品質 pass と扱っていない。
- R9700 batch=1 byte ledger は 262,144 token で total `30,858,010,436 B`、headroom
  `3,350,732,988 B`。これは packed artifact、KV、linear state、selected-weight gather、attention
  workspace を含む計算値である。MoE loader/residency は未結線なので `hipMemGetInfo` による実
  allocation は未確認。product は
  `/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/` にのみ置き、
  `/opt/ullm`、active manifest、systemd、既存 9B product は変更しなかった。

## 次の行動

1. `AQ4_0` G8 package を top-k-invariant serving candidate として昇格しない。選択 set が
   105/320 変わる現状は、router payload integrity と別の end-to-end quality failure である。
2. 既存 format の範囲で top-k stability を満たす容量内の policy があるかを改めて判断する。
   G8 は既存 `AQ4_0` の高精度 candidate まで試したが、`SQ8_0` は R9700 の expert/non-expert/KV
   budget に収まらない。新フォーマットの発明や偽の route 固定は行わない。
3. Qwen3.5 MoE loader/residency integration を実装した後、R9700 上で package の実 allocation と
   batch/context policy を測る。今回の ledger は fit を示すが GPU 実測ではない。
4. full hybrid attention / mRoPE / KV / tokenizer integration ができた時点で、より広い入力集合で
   selected-set stability と output quality を再評価する。FP32 reference corpus、bitwise gate、
   campaign はこの製造 task には使わない。
