# `SQ8_0` prefill attention redesign

> 実施日: 2026-07-26--27 JST
> 対象: R9700 (`gfx1201`) 上の Qwen3-14B `SQ8_0`、F32 KV、single stream
> 状態: exact-shape の runtime fast path を実装・計測した。`AQ4_0` の active manifest、
> systemd unit、`/opt/ullm` は変更しておらず、昇格は行っていない。

## 結論

Phase 0 は、長い prompt での崩落が実際に cached-prefix Flash2 attention
に支配されることを確認した。uLLM の同 kernel の profiler 内時間比は prompt
512 で 59.873%、2048 で 86.319%、4095 で 93.070% へ上がる。したがって、
この崩落を projection や M=1 tail のせいだと仮定したまま最適化を進めてはいけない。

Phase 1 は decode の workgroup 分割を移植しなかった。prefill の既存 Flash2 は
すでに十分な CTA を発行しており、1 dispatch あたり 5,120 WG / 40,960 wave32
（R9700 の 2,048 wave-slot に対する queued supply 2,000%）である。代わりに、
1 CTA を `(new token, KV head)` に対応させ、共有される 5 Q head を CTA 内で
逐次処理する GQA staging を追加した。K または V の 20-token x 128 F32 segment を
一度だけ LDS に置き、generic Flash2 の各 Q head の 256-thread reduction tree と
64-token online-softmax 境界は保った。

同一 executable の generic fallback と比べ、full-model prefill は全長で
1.020648--1.079858x 改善した。4095 は 100.586 -> **105.040 tok/s** である。
ただし 128/4095 の throughput 比は 8.601x から 8.407x へしか下がらず、崩落曲線を
実質的に平坦化したとは言えない。llama.cpp Q8_0/F32-KV の 4095-token 1,008.683
tok/s との差はなお 9.603x である。この変更は有効な局所改善だが、長文 prefill の
残差を解決したという結論ではない。

## Phase 0: profiler 内訳と launch geometry

次表の時間は rocprof kernel trace の合計であり、throughput ではない。uLLM は driver
の selected region の全 kernel row、llama.cpp は最終の長い HIP synchronization 2 点で
囲んだ terminal interval を集計した。従って二つの ms 値を直接速度比較には使わない。
tok/s は後段の synchronized `Instant` 計測だけから報告する。

`wave supply` は `launch wave32 / (64 CU x 32 wave slots)` の queued-work proxy であり、
resident occupancy や実際の同時実行 wave 数ではない。

| prompt | uLLM kernel 合計 | uLLM Flash2 | uLLM Flash2 の geometry / wave supply | llama.cpp F32-KV kernel 合計 | llama.cpp attention | llama.cpp attention の geometry / wave supply |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 512 | 949.872 ms / 67,111 dispatch | 160 calls, 568.719 ms, **59.873%** | grid `1,310,720`, block 256; 5,120 WG, 40,960 wave32, **2,000%** | 103.647 ms / 1,085 dispatch | 40 calls, 3.905 ms, **3.768%** | grid `(128,4,40)`, block `(32,4,1)`; 160 WG, 640 wave32, **31.25%** |
| 2048 | 10,547.436 ms / 268,435 dispatch | 640 calls, 9,104.419 ms, **86.319%** | 同上 | 116.702 ms / 1,085 dispatch | 40 calls, 13.158 ms, **11.275%** | 同上 |
| 4095 | 40,177.812 ms / 536,867 dispatch | 1,280 calls, 37,393.302 ms, **93.070%** | 同上 | 138.906 ms / 1,085 dispatch | 40 calls, 26.779 ms, **19.280%** | 同上 |

uLLM の generic Flash2 は `(token, Q head)` ごとに CTA を発行するため、40 Q head / 8 KV
head = 5:1 の GQA で同じ KV head を semantic に 5 回読む構造である。物理 HBM bytes は
PMC で取得していないため、5 分の 1 になったとは主張しない。一方 llama.cpp trace の
attention symbol は `flash_attn_ext_f16<...>` と表示されるが、この symbol 名だけでは
F32-KV 計測条件を覆さない。llama.cpp の最大行列 kernel `mul_mat_q` は同順で
85.364%、75.718%、65.723% を占めた。

この evidence は「長い uLLM prefill で attention が主残差」であることを支持する。
一方で、メモリ帯域律速、physical cache hit、occupancy は未確認である。
生 trace と集計は [`raw/profiles/`](raw/profiles/) および
[`analysis/`](analysis/) に保存した。

## 実装

`ullm_cached_prefix_attn_f32_flash2_gqa_grouped_kernel` は次の厳密な形状だけを自動選択する。

- `gfx1201`、F32 KV、`q_heads / kv_heads == 5`、`head_dim == value_dim == 128`
- `new_tokens * kv_heads` CTA。各 CTA は `(token, KV head)` を処理する。
- K と V をそれぞれ 20-token stage として共有 LDS へ一回だけ load し、5 Q head を
  deterministic に serial 処理する。K/V は同じ staging allocation を再利用する。
- 任意の別形状・別 GPU は generic Flash2 に戻る。既存 staged-wave32 prototype は
  明示 opt-in の独立実験経路として優先される。
  `ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED=1` は A/B control である。

prefill には decode のような split-C/KV partition を追加していない。Phase 0 の 2,000%
queued supply は split で workgroup 数を増やす根拠を否定するからである。20-token stage は
CTA 内の reuse 粒度であり、decode の 52-way partial reduction ではない。これが「decode の
設計をそのまま貼らない」判断である。

BK の cursor-rewind tail path を含む `sq8_serving_runtime.rs` は変更していない。4095 の
oracle で全 cache length が期待値、最終 prompt の prefill advance は 32 回であることを
記録した。

静的 HIPRTC audit は generic が VGPR 21 / LDS 1,292 B、serial GQA が VGPR 42 / LDS
12,628 B、いずれも spill なしと示す。これは resource metadata であって、occupancy や
HBM traffic の測定ではない。詳細は
[`analysis/serial-gqa-static-audit.md`](analysis/serial-gqa-static-audit.md) を参照。

## Full-model prefill throughput

R9700 のみを `HIP_VISIBLE_DEVICES=1` で使った。各 row は同じ M=128 chunk、同一 prompt、
unprofiled warm-up 1 回の後に timed repeat 5 回である。driver の同期済み `Instant` は
model load、warm-up、request construction、finish/reset を除外する。generic control は同じ
candidate executable に `ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED=1` を与えた。

llama.cpp 列は AY/BK の同一会計・同一 F32-KV 条件で保存済みの reference row であり、
この serial candidate window 中に再計測した値ではない。両 model は weight encoding が
異なるため、これは engine-loop prefill throughput の比較であり logits 同一性の主張ではない。

| prompt tokens | generic tok/s | serial GQA tok/s | GQA / generic | llama.cpp Q8_0 F32-KV tok/s | llama.cpp / serial GQA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 865.157 | **883.021** | **1.020648x** | 1,165.756 | 1.320x |
| 512 | 520.351 | **561.905** | **1.079858x** | 1,195.722 | 2.128x |
| 1024 | 338.308 | **358.745** | **1.060409x** | 1,145.351 | 3.193x |
| 2048 | 189.737 | **196.585** | **1.036094x** | 1,058.379 | 5.384x |
| 4095 | 100.586 | **105.040** | **1.044275x** | 1,008.683 | 9.603x |

128/4095 curve ratio は generic 8.601136x、serial GQA 8.406534x（2.262510% のみ平坦化）である。
全長で改善しているため kernel-only probe ではなく full-model 結果で retain するが、次の
候補はこの残る attention 支配を対象に別設計を探す必要がある。

## 数値・decode regression

128/512/1024/2048/4095 の各 prompt で final hidden state と logits の F32 little-endian
bytes は generic control と serial GQA で完全一致した。全 row で `max_abs=0`、
`relative_l2=0`、non-finite 0、top-1 と生成 token も一致している。従って今回の観測は
1 ULP 規模を上回る exact 一致である。ただし
[`lightweight-promotion-policy-v0.1.md`](../../../docs/plans/lightweight-promotion-policy-v0.1.md)
に従い、これは scalar threshold による合否ではなく review evidence として保存する。

BH の decode grouped tile-20 selector を current BR worktree build に明示指定して再測定した。
16 decode step x 5 repeat は **27.411786 tok/s**、BH reference 27.378731 tok/s に対し
1.001207x であり、通常の run variance 内で維持された。serial prefill path は decode
kernel dispatch を変更しない。

## 安全・運用記録

GPU は必ず `/run/ullm/r9700.lock` の owner を preflight で確認し、`ullm-openai.service` を
停止した後にのみ `flock` 取得した。Phase 0、初期 prototype、exact-tile64 rejection、final
serial measurement、current-source decode regression の計 5 isolated window は全て lock を
解放して service を復旧した。final serial window は 00:19:57--01:05:02、decode window は
01:10:20--01:11:20 JST であり、各 restore は return 0 / active を記録している。

各 full-model condition の開始前 thermal gate は edge <= 40 C、hotspot <= 42 C、socket
power <= 35 W を満たした。4095 の負荷中 sample は generic が gfx 3311 MHz / 99% activity /
edge 73 C / hotspot 94 C、candidate が 3345 MHz / 100% / edge 52 C / hotspot 69 C だった。
これらは単発・非同期 sample であり温度正規化比較には用いない。いずれも
`THROTTLED` 表示だが、実 gfx clock は高い DPM を維持し、violation 詳細は N/A なので、
文字列だけから clock 低下とは結論しない。

`llama-qwen35-udq4.service` は全 window で inactive/disabled を確認し、起動していない。
`active.json` の pre/post SHA-256 は全て
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` で一致した。最終監査時の
`ullm-openai.service` は active/running、`NRestarts=0` である。

## Evidence map

- Throughput summary: [`serial-gqa-throughput-summary.json`](serial-gqa-throughput-summary.json)
- Timing/configuration: [`serial-gqa-throughput-run-configuration.json`](serial-gqa-throughput-run-configuration.json)
- Numerical comparison: [`numerical/serial-gqa-comparison.json`](numerical/serial-gqa-comparison.json)
- Phase 0 aggregates: [`analysis/`](analysis/)
- Lock/service/thermal records: [`service/`](service/) and [`raw/serial-gqa-throughput/`](raw/serial-gqa-throughput/)
- Current-source BH decode record: [`raw/current-head-bh-decode-regression/stdout.log`](raw/current-head-bh-decode-regression/stdout.log)
- Source/build relationship: [`source-provenance.md`](source-provenance.md)
- Rejected experiments and their scope: [`attempts.md`](attempts.md)

The full rocprof CSV traces are retained under `raw/profiles/`; they are intentionally not treated as
throughput outputs. The profiling data is approximately 576 MiB, while the derived JSON aggregates above
are the compact review surface.
