# MI300X×1 レンタル結果（2026-07-26）

このディレクトリは、Hot Aisle の MI300X VF 1 枚（host:
enc1-gpuvm008）で行った約 2 時間の借用結果を保全する。約 2 時間は
借用時の記録であり、回収済みログには各 stage の開始・終了時刻は残って
いない。したがって、stage ごとの所要時間は**未確認**である。

生データの要約は [env.txt](env.txt) にあり、数値の根拠となるログはこの
ディレクトリにそのまま残す。この README は結果を再計算・補完したもの
ではなく、確認できた値、未確認事項、比較の限界を分けて記録する。

## 1. 証跡と環境

| 項目 | 確認できた値 |
| --- | --- |
| 記録時刻 | 2026-07-26T08:23:48+00:00 |
| host | enc1-gpuvm008 |
| GPU | AMD Instinct MI300X VF ×1 |
| 実行時 arch | gfx942:sramecc+:xnack- |
| partition | NPS1 / SPX |
| VRAM | 196,288 MB |
| ROCm | 7.2.4 |
| hipcc | HIP 7.2.53211-97f5574fe2 |
| CPU / RAM | 13 vCPU / 220 GB |

| component | image または build | digest / build ID |
| --- | --- | --- |
| vLLM | rocm/vllm:latest | sha256:e7f02dd2ce3824959658bc0391296f6158638e3ebce164f6c019c4eca8150ec7 |
| SGLang | lmsysorg/sglang:v0.5.16-rocm720-mi30x | sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee |
| llama.cpp | ghcr.io/ggml-org/llama.cpp:full-rocm | sha256:190a068ef56255d7cca27ee93063e644fdf239bad9db55b85f83855aca957b6d |
| llama.cpp binary | b10133 | ff067f76d |

イメージ digest は [pull-vllm.log](pull-vllm.log)、
[pull-sglang.log](pull-sglang.log)、[pull-llama.log](pull-llama.log) で
確認した。llama.cpp の HIP backend と MI300X 1 台の認識、および補助
microbenchmark は [lbench.log](lbench.log) にある。

## 2. モデル artifact と revision

| 用途 | model / artifact | 確認できた revision・識別子 | 注記 |
| --- | --- | --- | --- |
| 14B vLLM / SGLang | Qwen/Qwen3-14B-FP8 | 9a283b4a5efbc09ce247e0ae5b02b744739e525a | [env.txt](env.txt) に記録済み。 |
| 14B llama.cpp | Qwen3-14B-Q8_0.gguf | SHA-256: a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d | 15,698,533,728 byte。取得した GGUF repository revision は回収ログだけからは**未確認**。計画書にあった revision を実行 artifact の revision とみなさない。 |
| MoE | Qwen/Qwen3-Coder-Next-FP8 | revision **未確認** | 80.4 GB、52 file の size verification は [env.txt](env.txt) に記録。 |
| MoE | Qwen3-30B-A3B-FP8 | revision **未確認** | 32.5 GB、17 file の size verification は [env.txt](env.txt) に記録。 |
| MoE | Qwen3.6-35B-A3B-FP8 | revision **未確認** | 37.5 GB、56 file の size verification は [env.txt](env.txt) に記録。mtp.safetensors を含む。 |

llama.cpp の Q8_0 は vLLM/SGLang の FP8 safetensors と同じ format では
ない。これらの数字は同一ハードウェア・同一クライアントでの engine
挙動の比較であり、量子化 format だけを切り出した比較ではない。

## 3. 測定方法と限界

- 通常 14B sweep は同一クライアント、各 request が prompt 1,010 token と
  output 16 token である。ログ中の ptok=1010 と生成 token 数で確認した。
  表の output throughput は 16 × 同時実行数を client wall 時間で割った値で
  ある。
- この wall 時間は 1,010 token の prefill を含むため、prefill が支配的で
  ある。ここにある tok/s は decode 専用の数字ではない。
- C=1 の主値は warmup 5 回後の 10 回測定の p50 で、初回 JIT 汚染を除く。
  通常 14B のその 10 個の個別観測値はこのディレクトリには残っていないため、
  p50 の独立した再計算はできない。値は [env.txt](env.txt) の要約を根拠と
  する。
- C=2 以上の sweep は回収された各 sweep log の 1 回の aggregate 値であり、
  計画していた 3 trial median ではない。C=1 の生 sweep は vLLM と SGLang
  で JIT 汚染しているため、主比較に使わない。
- vLLM は --enforce-eager で実行され、torch.compile は無効だった。従って
  この結果は vLLM を過小評価する。
- partition は NPS1/SPX の 1 条件だけである。他 partition は未測定である。
  温度、clock、power、thermal throttling の telemetry は回収済みファイル
  からは**未確認**であり、熱条件を揃えた比較とは主張しない。
- server の完全な起動 argv、cache 無効化の実測、raw HTTP response、TTFT、
  ITL、physical HBM/L2 counter はこのディレクトリからは**未確認**である。
  計画書の contract を満たしたと読み替えない。
- 計画時 workload は 1,024 prompt / 20 output token だったが、実測は
  1,010 / 16 token である。このため計画した 1,024/20 の結果や uLLM との
  直接速度比較を達成したものではない。

## 4. SQ8_0 CDNA3 A′ 実機検証

[aprime-smoke.log](aprime-smoke.log) は runtime device 1 の
MI300X VF / gfx942:sramecc+:xnack- を記録し、次を確認した。

- one-wave fragment/lane probe は pass: max_abs=0.007812、
  max_rel=0.000000、256 lane/register coordinate は全単射。
- A′ FNUZ/CK は、実モデル寸法の 5 形状すべてで CPU expectation に対して
  max_abs=0.000000 を記録した。
- 成功 run では ULLM_SMOKE_SKIP_B_CONTROL を設定して B comparison を skip
  している。出力中の B=0 および A′-B=0 は self-comparison の結果であり、
  B が正しい根拠ではない。A′ 対 CPU の結果だけを A′ の物理検証結果として
  扱う。

| case | A′ instance | M / N / K | A′ 対 CPU max_abs |
| --- | --- | ---: | ---: |
| k_or_v_tail_id1 | DefaultTile16x128x128 | 1 / 1,024 / 5,120 | 0.000000 |
| q_or_o_full_id1 | DefaultTile16x128x128 | 16 / 5,120 / 5,120 | 0.000000 |
| gate_or_up_tail_id2 | KPaddingTile16x128x256 | 1 / 17,408 / 5,120 | 0.000000 |
| gate_or_up_full_id3 | DefaultTile16x256x128 | 128 / 17,408 / 5,120 | 0.000000 |
| down_tail_id4 | DefaultTile16x128x256 | 1 / 5,120 / 17,408 | 0.000000 |

[aprime-timing.log](aprime-timing.log) は A′ projection の 3 warmup 後、
200 回 repeat の値である。これは full-model prefill/decode、実効
occupancy、実測 HBM bandwidth ではない。

| case | M / N / K | per call (ms) | TFLOPS | weight GB/s |
| --- | ---: | ---: | ---: | ---: |
| k_or_v_tail_id1 | 1 / 1,024 / 5,120 | 0.023266 | 0.451 | 225.6 |
| q_or_o_full_id1 | 16 / 5,120 / 5,120 | 0.022844 | 36.722 | 1,151.1 |
| gate_or_up_tail_id2 | 1 / 17,408 / 5,120 | 0.029517 | 6.039 | 3,019.8 |
| gate_or_up_full_id3 | 128 / 17,408 / 5,120 | 0.091482 | 249.415 | 981.4 |
| down_tail_id4 | 1 / 5,120 / 17,408 | 0.065408 | 2.725 | 1,362.9 |

したがって、M=128 の gate_or_up_full_id3 は 249.4 TFLOPS、M=1 の
gate_or_up_tail_id2 は 3,020 GB/s であった。後者は MI300X の公称 HBM3
5.3 TB/s を分母にした約 57% であり、NPS1/SPX 上の実測 HBM 効率ではない。

### 4.1 実機で判明した不具合

1. **smoke test の device selection guard（修正済み）**

   旧 guard は device_count()==1 を要求していた。しかし uLLM runtime は
   index 0 に CPU device を常に置くため、GPU が 1 枚だけ見えても runtime
   count は 2 になる。これはこの構造ではどの実機でも通らない。

   保存済みの
   [sq8_gfx942_aprime_physical_smoke.rs.mi300x-patched](sq8_gfx942_aprime_physical_smoke.rs.mi300x-patched)
   と本体の source は byte-for-byte で一致することを確認した。修正版は
   HIP_VISIBLE_DEVICES が 1 token であることをなお要求しつつ、すべての
   runtime device を列挙し、fail-closed gfx942 selector が受理する device
   がちょうど 1 台であることを要求して、その runtime index を返す。

2. **B 対照経路（未修正）**

   raw OCP から BF16 に dequant する B control は
   k_or_v_tail_id1 で期待値 0.53125 に対して観測値 0.03125 だった。
   差はちょうど 0.5 である。同じ case の A′ は 0.53125 を返したため、
   この観測から壊れているのは B control 側である。

   tail 処理の取りこぼしが疑われるが、原因は**未確認**であり、修正して
   いない。ULLM_SMOKE_SKIP_B_CONTROL は A′ 対 CPU を判定する一時的な
   escape hatch であって、B を直したものではない。B を必須対照に戻す
   再現・修正・再実機検証が必要である。

## 5. Qwen3-14B 外部 engine 結果

### 5.1 C=1 clean

warmup 5 回 + 10 回測定の p50 は次の通りである。

| engine | output throughput (tok/s) |
| --- | ---: |
| vLLM eager | 41.16 |
| SGLang v0.5.16 | 35.45 |
| llama.cpp b10133 | 49.06 |

### 5.2 報告する比較曲線

C=1 は上の clean p50、C=8 以上は sweep aggregate の値である。

| C | vLLM eager | SGLang v0.5.16 | llama.cpp b10133 |
| ---: | ---: | ---: | ---: |
| 1 | 41.16 | 35.45 | 49.06 |
| 8 | 253.73 | 159.29 | 280.51 |
| 16 | 471.01 | 286.65 | 147.05 |
| 32 | 848.70 | 529.14 | 144.75 |
| 64 | 1,419.20 | 688.52 | 143.50 |
| 128 | 2,526.96 | 1,158.50 | 140.09 |

llama.cpp は C<=8 で最速だが、C>=16 ではおよそ 140 tok/s 台に飽和した。
vLLM は C=128 まで伸びた。これは上記の E2E wall throughput の観測であり、
decode 専用性能の順位付けではない。

### 5.3 回収された sweep の全行

以下は [sweep-vllm.log](sweep-vllm.log)、
[sweep-sglang.log](sweep-sglang.log)、[sweep-llama.log](sweep-llama.log)
の aggregate output throughput を全行転記したもの。C=1 の vLLM/SGLang
は初回 JIT 汚染値であり、5.2 節の C=1 値に置き換えている。

| C | vLLM eager | SGLang | llama.cpp | 注記 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 4.20 | 0.12 | 49.06 | vLLM/SGLang は JIT 汚染。主値ではない。 |
| 2 | 73.38 | 46.16 | 108.89 | 1 回の sweep。 |
| 4 | 144.36 | 90.20 | 172.32 | 1 回の sweep。 |
| 8 | 253.73 | 159.29 | 280.51 | 1 回の sweep。 |
| 16 | 471.01 | 286.65 | 147.05 | 1 回の sweep。 |
| 32 | 848.70 | 529.14 | 144.75 | 1 回の sweep。 |
| 64 | 1,419.20 | 688.52 | 143.50 | 1 回の sweep。 |
| 128 | 2,526.96 | 1,158.50 | 140.09 | 1 回の sweep。 |

補助的な llama-bench は tg16 @ d1028 = 120.17 ± 7.61 tok/s だった。
HTTP、prefill、queue、同時実行を測らないため、主比較表には混ぜない。

## 6. MoE 結果

### 6.1 Qwen3-Coder-Next-FP8

モデルは Qwen3NextForCausalLM、48 layer、512 expert / 10 active、
hybrid attention、80.4 GB である。

- vLLM は model load には成功し（VRAM 175 GB）、request 時に失敗した。
  hybrid attention により attention block size=544 が選ばれ、Triton の
  tl.arange(0, BLOCK_SIZE) が power of 2 でなければならないためである。
  --block-size 16 と --no-enable-prefix-caching は回避にならなかった。
  [moe-c1.log](moe-c1.log) と [sweep-moe-vllm.log](sweep-moe-vllm.log) の
  全 request は HTTP 500 である。
- SGLang は動作した。C=1 clean は 52.17 tok/s
  （p50=0.3067 s、warmup 5 + measured 10）である。

| C | SGLang sweep output throughput (tok/s) |
| ---: | ---: |
| 1 | 42.03 |
| 2 | 64.00 |
| 4 | 36.53 |
| 8 | 73.05 |
| 16 | 186.85 |
| 32 | 275.17 |
| 64 | 247.14 |
| 128 | 526.84 |

この sweep も各 C 1 回であり、C=1 clean 値とは測定手順が異なる。

### 6.2 Qwen3-30B-A3B-FP8 と AITER

モデルは Qwen3MoeForCausalLM、48 layer、128 expert / 8 active、32.5 GB
であり、vLLM で推論できた。既定の selected backend は Rocm Attention
だった。

| measurement | default Rocm Attention | AITER Flash Attention |
| --- | ---: | ---: |
| C=1 clean p50 | 27.35 | 23.71 |

| C | default sweep (tok/s) | AITER sweep (tok/s) |
| ---: | ---: | ---: |
| 1 | 22.18 | 20.06 |
| 2 | 50.74 | 41.01 |
| 4 | 93.46 | 80.64 |
| 8 | 168.17 | 148.01 |
| 16 | 315.05 | 258.40 |
| 32 | 584.25 | 496.02 |
| 64 | 1,010.30 | 655.17 |
| 128 | 1,745.92 | 1,387.78 |

既定 environment では VLLM_ROCM_USE_AITER=True、AITER_MOE=True、
AITER_LINEAR=True だった一方、AITER_MHA=False、AITER_PAGED_ATTN=False、
FP8_MFMA_PAGE_ATTN=False だった。後者 3 つを有効にすると AITER Flash
Attention backend へ切り替わったが、全 C で遅く（約 -12% から -35%）、
ログには not found tuned config が繰り返し出た。既定 Rocm Attention の
値を reference とする。

### 6.3 Qwen3.6-35B-A3B-FP8

Qwen3_5MoeForConditionalGeneration、37.5 GB、mtp.safetensors 同梱の
モデルである。transformers 5.14.1 は認識したが、vLLM
0.11.2.dev は architecture 未対応だった。MTP は**未検証**である。

## 7. スコープ外と次の再現条件

- ATOM はこのレンタルで再評価していない。No-go の判断は
  commit a646804f6a8fcdbc02270337b71302536facb43a
  （docs: research AMD ATOM MI300X feasibility）を参照する。
- A′ はこの 1 台・NPS1/SPX・5 deterministic shape での fragment と
  A′ 対 CPU の物理検証を通過した。full-model logits、prefill/decode、
  B との正常な differential、actual occupancy/residency、partition 横断、
  thermal/clock/power、counter による HBM/L2 は未確認である。
- B control を ULLM_SMOKE_SKIP_B_CONTROL なしで
  k_or_v_tail_id1（期待 0.53125、観測 0.03125）から再現し、根因を修正して
  から 5 形状を再実行する。その後に初めて B を A′ の独立対照へ戻せる。
