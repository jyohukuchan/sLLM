# R9700 SQ8_0 対 llama.cpp Q8_0 prefill 比較

Date: 2026-07-26

## 今回の要点

- R9700 gfx1201 のみで、Qwen3-14B 系の uLLM SQ8_0 と公式 GGUF Q8_0 の
  single-stream prefill を測定した。長さは 128 / 512 / 1024 / 2048 / 4095、
  各 5 回、同一長の unprofiled warm-up 後である。4095 は、uLLM request が
  最小の output token を一つ予約するための operational 4096-point である。
- uLLM は F32 KV、llama.cpp は F32 KV と F16 KV を測った。llama.cpp は
  `-p N -n 0 -r 5 -b N -ub 128 -ngl 999 -fa on -t 1` を用い、source で
  prompt-only timer が model load/warm-up の後、`llama_synchronize()` を
  含むことを確認した。uLLM は `Instant` で測った synchronized prefill loop
  を使い、profiler range の時間を throughput に使っていない。
- 結果は uLLM が勝った長さなしである。tok/s は順に次の通りだった。

| prompt | uLLM SQ8_0 F32 KV | llama.cpp Q8_0 F32 KV | llama.cpp Q8_0 F16 KV |
| ---: | ---: | ---: | ---: |
| 128 | 851.659 | 1165.756 | 1189.076 |
| 512 | 513.676 | 1195.722 | 1187.016 |
| 1024 | 335.996 | 1145.351 | 1174.539 |
| 2048 | 188.425 | 1058.379 | 1127.775 |
| 4095 | 71.576 | 1008.683 | 1054.871 |

- llama.cpp F32 / uLLM は 1.369x, 2.328x, 3.409x, 5.617x, 14.092x、
  F16 / uLLM は 1.396x, 2.311x, 3.496x, 5.985x, 14.738x である。
  N=1024 の uLLM 335.996 tok/s は先行 prefill 観測の 337.132 tok/s に近く、
  この harness の同期計時と大きく矛盾しない。

## 会計と解釈

- decode の既存方針を引き継ぎ、SQ8 projection payload / BF16 scale /
  BF16 LM head と、F32 相当の Q-head-expanded causal GQA KV を全 row に
  同じ logical numerator として使った。N=1024 の causal KV read は
  859,832,320,000 B と先行 attention 計測と一致する。
- uLLM の logical GB/s は 188.550, 270.629, 317.435, 335.936, 247.610、
  lower-bound TFLOP/s は 22.560, 13.683, 9.020, 5.137, 2.011 である。
  llama.cpp の corresponding logical GB/s が 640 GB/s nominal roof を越える
  row は、同じ logical GQA denominator が physical HBM bytes ではないためである。
  これは physical-bandwidth efficiency と呼ばない。
- N=512--4095 で common logical work の 79.8--97.0% は causal KV 項である。
  したがって uLLM の長文 prefill は logical work mix 上は attention/KV
  dominated である。ただし HBM/TCC counter を取っていないので、厳密な
  physical memory-bound / compute-bound 判定は未確認とする。
- N=4095 では uLLM fixed-M128 planner が 31 個の M=128 と 127 個の M=1
  tail に分かれ、各 repeat は 158 advances だった。llama.cpp の 32 ubatch
  と非対称であり、4095 の大差を 128 の倍数長全般の主張に拡張しない。

## 熱・サービス上の注意

- 各プロセス開始前の gate は edge 38--40 C、hotspot 38--42 C、memory
  36--40 C、socket 7--16 W で通過した。一方 warm-up は gate の後なので
  timed start を厳密に揃えたとは言えない。uLLM / llama F32 の最寄り
  post-warm-up sample は N=2048 で 57/78 C 対 46/62 C、N=4095 で
  69/90 C 対 49/66 C だった。このため長文 row は temperature-normalized
  comparison ではない。
- 意図的な `ullm-openai.service` stop は 19:12:23--19:59:57 の一窓だけ。
  wrapper の restore sudo ticket が失効して return 1 となり、20:00:24 の
  start initiator は未確認だった。さらに 20:05:57 に worker stdout EOF で
  service が停止したため、20:06:13 に明示的に start し、20:06:14 に
  active/running を確認した。その後の通常 service operation 中にも
  20:08:11 / 20:09:06 / 20:11:56 / 20:12:42 に worker EOF/restart が
  journal に出たが、原因は未確認である。20:16:26 の audit は active/running。
  測定の追加実行・二度目の stop は行っていない。`llama-qwen35-udq4.service` は
  終始 inactive/disabled で起動していない。

## 保存先

- 生データ、各 repeat、full command、source excerpts、logical accounting、
  thermal history、service restore audit は
  `benchmarks/results/2026-07-26/r9700-prefill-comparison/` に保存した。
- SQ8_0 と GGUF Q8_0 は encoding が同一ではないため、品質比較ではなく
  8-bit 級の speed positioning として扱う。
