# Gemma4 E2B resident BF16 v0.1

Implementation revision: `0c6ae998` (`feat(gemma4): add resident BF16 text executor`).

対象は `google/gemma-4-E2B` の text decoder である。量子化は行わない。単一の
`model.safetensors` の全 BF16 payload（vision/audio を含む）を一度だけ R9700 に載せ、
text decoder はその resident weight を直接読む。serving 統合はこの artifact の範囲外である。

## 実行条件

- GPU: AMD Radeon Graphics / gfx1201 (R9700), reported 34,208,743,424 bytes / 31.859 GiB
- Visibility: `HIP_VISIBLE_DEVICES=1`, `ULLM_HIP_VISIBLE_DEVICES=1`; V620 は可視化も実行もしていない。
- Fallback rejection: `ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1`, `ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1`, `ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1`。
- GPU timing の直前に指定の `pgrep`、`systemctl is-active ullm-openai.service`、`amd-smi process --gpu 2` を実行した。service は `failed`（非稼働）、R9700 process は `No running processes detected` であった。
- throughput は `std::time::Instant` または llama-bench の wall-clock 測定のみで数えた。profiler range 時間は使っていない。

## VRAM 収支

`model.safetensors` は 10,246,621,918 B (9.542910 GiB)、payload は
10,246,357,958 B (9.542664 GiB, 2,011 BF16 tensor) である。実行する text subset は
9,294,899,782 B (600 tensor) であり、その中の PLE は 4,780,303,872 B (4.452005 GiB)。
実行しない multimodal payload 951,458,176 B も、complete-checkpoint resident 要件に従い
resident allocation に含めた。

KV は config の 35 layer / 20 shared K/V layer をそのまま使う。非共有 source は local 12層
（layer 0--3, 5--8, 10--13）と full 3層（4, 9, 14）だけである。local は window=512 の
F32 K/V ring で、K/V と二つの table を合わせ固定 12,632,064 B。full は F32 K/V 3 source
で 12,288 B/token、page table を含め 12,300 B/token である。temporary buffer は実測
1,170,432 B だった。

| Context tokens | K/V demand (B) | Resident weight + K/V + transient (B) | R9700 headroom (B) | Fit |
| ---: | ---: | ---: | ---: | --- |
| 1 | 12,644,364 | 10,260,172,754 | 23,948,570,670 | yes |
| 512 | 18,929,664 | 10,266,458,054 | 23,942,285,370 | yes |
| 4,096 | 63,012,864 | 10,310,541,254 | 23,898,202,170 | yes |
| 32,768 | 415,678,464 | 10,663,206,854 | 23,545,536,570 | yes |
| 131,072 (config maximum) | 1,624,817,664 | 11,872,346,054 (11.056984 GiB) | 22,336,397,370 (20.802391 GiB) | yes |

File header 263,960 B を加えた conservative source-file total も
11,872,610,014 B で fit する。動的 full-KV 部だけで割る算術上限は 1,947,039 token だが、
これは `max_position_embeddings=131072` を超えるため運用上の最大文脈は **131,072 token**
である。GPU live telemetry でも resident allocation 中は 11,752--11,754 MB used /
20,870--20,872 MB free だった。RuntimeBuffer accounting は HIP context/allocator overhead を
含まないので、両方を記録した。

## 実装と正しさ

- `load_resident` は checkpoint の 2,011 BF16 tensor を chunk upload して resident weight table を作る。projection ごとの source-matrix stream は resident path にない。
- device K/V は source layer にだけ確保する。layer 15--34 は attention kind ごとに layer 13 (local) / layer 14 (full) を参照する。shared layer 用に K/V を二重確保・再投影しない。
- local cache は HF の shared-KV timing に合わせ、現 token の後には full 512-window を残し、次 append の直前に 511 に縮める。full cache は config maximum まで保持する。
- `prefill(&[token])` は M=N の public operation、`decode(token)` は M=1 で分けた。既存 BF16×F32 primitive は matvec なので、現在の prefill projection dispatch は因果 token 順の M=1 matvec を N 回発行する。これは batch GEMM 未実装という性能上の限定であり、K/V transition と計測上の prefill は supplied N token 全体である。

`validation-with-vram-telemetry.json` は BL trace を再生した。cached decode と token 0 からの full re-prefill は、両 case で全4 token が一致した。

| Prompt | BL / cached / full-reprefill greedy IDs | Decoded continuation |
| --- | --- | --- |
| `The capital of France is` | `9079, 236761, 108, 818` | `The capital of France is Paris.\n\nThe` |
| `Once upon a time,` | `528, 496, 1902, 1298` | `Once upon a time, in a world where` |

`sliding-boundary.json` は config window 512 を越える 513 token (`2` repeated) を、M=1 cache
route と full M=N re-prefill route の双方で評価した。top-1 は双方 `184`、logit は双方
`14.404961585998535`。12 local source cache は全て `capacity=512, cache_len=512,
absolute_len=513` になった。従って窓外の token は保持していない。

共有の確認では 20 layer の snapshot が layer 15--34 を 13/14 に全て対応付けた。対照として
shared physical K/V を不正に再投影する diagnostic（serving path ではない）を実行すると
capital prompt の greedy IDs は `506, 236789, 500, 236772` となり、正常 shared path の
`9079, 236761, 108, 818` と一致しなかった。source selection が実際に出力へ効いていることを
示す確認である。

## Throughput

resident benchmark は warmup と weight upload を除外し、各3回の総 token / 総 monotonic
wall time とした。prefill は six-token BL capital-France prompt を M=N として数え、decode は
untimed six-token prefill 後の M=1 を4 tokenずつ数えた。logical traffic は各 run で BF16
weight read + F32 K/V unique read/write の下限であり、物理 HBM byte や profiler time ではない。

| Engine / condition | Prefill | Decode | Accounting |
| --- | ---: | ---: | --- |
| uLLM resident BF16 | 18.296336 tok/s (54.655751 ms/token), 18 tokens / 0.983804 s | 15.613216 tok/s (64.048305 ms/token), 12 tokens / 0.768580 s | 3 × 6-token M=N prefill; 3 × 4-token M=1 decode |
| llama.cpp `68a5592`, ROCm, BF16 GGUF, F32 K/V, FA off | 218.955938 tok/s (4.567129 ms/token) | 69.959983 tok/s (14.293886 ms/token) | `-p 6 -n 4 -r 3 -b 6 -ub 6 -ngl 999 -ctk f32 -ctv f32 -fa off` |

llama.cpp は Gemma4 support を持ち、export を `gemma4 E2B BF16` と識別して実行した。上表で
llama.cpp は uLLM の 11.967× prefill、4.481× decode throughput である。GGUF は text-only
export で complete source checkpoint とは resident payload の範囲が異なるため、VRAM footprint
比較には使わない。詳細な command、sample、pre/post telemetry は
`llama-cpp-benchmark.json` にある。

uLLM benchmark の cooldown 後 snapshot は edge/hotspot/memory = 45/45/44 C、after workload は
47/49/46 C だった。validation の active after-workload snapshot は 46/49/46 C, 106 W,
3,275 MHz gfx だった。amd-smi の aggregate `throttle_status` は idle でも `THROTTLED` を返す
snapshot があり、per-cause violation field は全て `N/A` だったため、thermal/power throttle の
原因は未確認として扱う。llama.cpp run は開始 52/53/52 C、終了 50/51/52 C、いずれも
`UNTHROTTLED`。温度を揃えた開始 snapshot と終端 snapshot を artifact に残した。

## Kernel / collision decision

新 kernel は追加していない。既存 BF16×F32 matvec と既存 paged F32 K/V write/decode attention
を利用し、required environment variable で host fallback を fail-closed にした。BH が編集中の
`runtime/src/ullm_runtime_parts/part_01.inc` と
`runtime/src/ullm_runtime_hiprtc_sources.inc`、BK が編集中の
`crates/ullm-engine/src/sq8_serving_runtime.rs`、既存 `AQ4_0` / `SQ8_0` production code は変更していない。

## Artifact index

- `validation-with-vram-telemetry.json`: BL trace、cache/no-cache、shared source と live VRAM。
- `sliding-boundary.json`: 513-token window-boundary cache/no-cache evidence。
- `benchmark.json`: uLLM resident throughput, exact accounting, cooldown and telemetry。
- `llama-cpp-benchmark.json`: llama.cpp command, output, and comparison telemetry。
