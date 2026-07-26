# AMD ATOM の MI300X benchmark 実施可能性調査（2026-07-26）

Date: 2026-07-26

## 結論

**今回の残り数時間で、ATOM の `C=1..128` 本計測を新規に始める判断は no-go とする。**

- ATOM は AMD ROCm の公式リポジトリ
  [`ROCm/ATOM`](https://github.com/ROCm/ATOM) にある **AiTer Optimized
  Model** である。AITER を使う native 推論エンジンであり、別途 vLLM
  out-of-tree (OOT) plugin と SGLang model-implementation backend を持つ。
- `Qwen/Qwen3-14B-FP8` は `Qwen3ForCausalLM` なので、ATOM native の
  architecture registry と FP8 safetensors loader の対象である。しかし、AMD が
  「nightly CI で検証」と明記するモデル一覧、また性能 dashboard の追跡対象には
  **Qwen3-14B はない**。動作・性能とも実機未確認であり、「最適化済み」とは扱えない。
- `Qwen/Qwen3-Coder-Next-FP8` は `Qwen3NextForCausalLM`（48 層、512
  experts、top-10、`full_attention_interval=4`、FP8 128x128）なので native
  registry 上の architecture には一致する。しかし、AMD 公式資料にこの**正確な
  checkpoint**の recipe、CI 結果、MI300X×1 結果はない。公式 Qwen3-Next recipe は
  別モデルの `Qwen3-Next-80B-A3B-Instruct` を 8x MI308/MI355/MI350、TP=8
  で扱うものだけである。したがって Coder-Next は今回の比較対象から除外する。
- ROCm 7.2.4 / gfx942 自体は AMD の対応範囲で、同じ ROCm 7.2.4 を明示する
  ATOM v0.1.4 production image も存在する。OpenAI-compatible
  `/v1/completions` と `/v1/chat/completions` を提供するため、**動作が確認できた
  場合に限り**既存共通 HTTP client での同一条件計測は可能である。

この結論は「ATOM が遅い」と断定するものではない。今回の正確なモデルと
MI300X×1 について AMD の性能検証根拠がなく、初回導入コストが残り時間と CPU
critical path に見合わない、という判断である。

本調査は資料確認だけで完了した。GPU、`ullm-openai.service`、
`/etc/ullm/served-models/active.json`、`/opt/ullm`、activation/campaign に
一切触れていない。

## 1. ATOM の正体と配布形態

### 1.1 対象を確定したもの

対象は AMD ROCm organization の
[`ROCm/ATOM`](https://github.com/ROCm/ATOM)（MIT）である。README は ATOM を
**AiTer Optimized Model**、AITER に基づく `lightweight vLLM-like
implementation` と定義し、native engine は nano-vLLM から適応したものと記す。
ここでいう ATOM は、旧 GitHub Atom text editor 等の同名プロジェクトではない。

配布は次の三層である。

| 形態 | 確認できた内容 | 今回の位置付け |
| --- | --- | --- |
| Source / Python package | `git clone --recursive` + `pip install -e .`。AITER submodule を使う。 | 数時間の rental では build route を選ばない。 |
| GitHub release | stable `v0.1.4`、`v0.1.5` と prerelease `v0.1.6-rc0` が確認できた。 | 再現対象は image と同じ `v0.1.4`。 |
| Docker | AMD Verified Publisher の `rocm/atom`（native）と `rocm/atom-dev`（開発、vLLM/SGLang 向け tag を含む）。 | native image が Qwen3-14B の最小かつ最も再現しやすい経路。 |

`v0.1.4` release は paired AITER v0.1.15 として出荷され、production image
`rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4`
を明記している。アクセス時点の Docker Hub metadata はこの tag を Linux/amd64、
16,605,753,298 B、digest
`sha256:13d8564eeef3a267c1cc25410c78029097bd8f218b71ef37d32df971d73d25bb`
として返した。mutable な `latest` ではなく、この tag/digest を記録する。

### 1.2 vLLM / SGLang との関係

- **native ATOM:** vLLM fork ではない。独自 engine、scheduler/KV cache、AITER
  model/attention kernels、FastAPI/Uvicorn OpenAI server を持つ vLLM-like 実装で
  ある。
- **vLLM-ATOM:** ATOM package が vLLM の公式 plugin entry point に登録される
  OOT plugin である。vLLM 側が API server、CLI、scheduler、worker/cache 管理を
  継続し、ATOM 側が platform、model wrapper、attention backend、AITER kernels を
  差し替える。fork ではない。
- **SGLang-ATOM:** SGLang を fork するものではなく、SGLang framework の上で
  ATOM model implementation を選ぶ経路である。一般説明は upstream integration
  PR が未 merge とし、既知制約を「Qwen-Dense / Qwen-MoE のみ、TP/EP のみ」と
  している。この文書と後発 recipe には差異があるため、今回の Coder-Next を
  SGLang-ATOM で動かせるとは判定しない。

特に vLLM plugin は、現行 plugin guide の supported-model table に
`Qwen3NextForCausalLM` を列挙していない一方で、別に Qwen3-Next recipe がある。
この公式文書間の不整合も、Qwen3-Coder-Next の plugin 経路を採用しない理由である。

## 2. 「対応」と「最適化済み」を分離した確認

AMD の資料には少なくとも三つの強さの異なる表現がある。

| 区分 | AMD が実際に記載するもの | 判断に使う強さ |
| --- | --- | --- |
| Architecture support | native registry が HF `architectures` を照合して model class を選ぶ。README は Qwen3 / Qwen3-Next を support と列挙する。 | checkpoint を読む候補であること。性能保証ではない。 |
| Nightly CI validation | Model Run Guide は「下表の各 model recipe は nightly CI で検証」と明記する。 | 「最適化済み」と呼べる最小の公式根拠。 |
| Performance dashboard | README は DeepSeek-R1-0528、GLM-5-FP8、gpt-oss-120b の三つを nightly performance tracking 対象とする。 | 継続的な性能 regression 監視の根拠。 |

### 2.1 AMD が nightly CI 検証済みとして挙げる一覧

Model Run Guide の一覧は次である（precision / TP も同ガイドの表記）。

| Model | Precision | TP |
| --- | --- | ---: |
| DeepSeek-R1-0528 | FP8 / MXFP4 | 8 |
| GLM-5 | FP8 | 8 |
| GPT-OSS-120B | FP8 | 1 |
| Kimi-K2.5/K2.7 | MXFP4 | 4 |
| Kimi-K2-Thinking | FP8 | 8 |
| Qwen3-235B | FP8 | 8 |
| Qwen3-Next | FP8 | 8 |

Qwen3-14B はこの表にも dashboard にもない。Qwen3-Coder-Next は名前として
存在せず、generic な Qwen3-Next 行だけがある。release `v0.1.4` の joint
validation も DeepSeek-R1、MiniMax-M2.5、Qwen3-235B、GLM-5、Kimi-K2.5 の 5
モデルであり、今回の二つは含まれない。

### 2.2 今回の checkpoint ごとの判定

| 対象 | HF config / native registry | FP8 の直接 load | 最適化・性能の公式根拠 | 実行可否と性能の結論 |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-14B-FP8` | config revision `9a283b4…` は `Qwen3ForCausalLM`、40 層。v0.1.4 registry に同 architecture がある。 | config の `quant_method=fp8`、`weight_block_size=[128,128]` を ATOM の quant parser が読む設計。safetensors をそのまま走査・mmap load する。offline conversion は要求されない。 | exact model の recipe、nightly CI、dashboard、MI300X result は未確認。 | **コード上は対応候補、実機成功は未確認。** 「動くが遅い」も「動かない」も一次資料だけでは判定不可。比較用の最適化済み row にはしない。 |
| `Qwen/Qwen3-Coder-Next-FP8` | config revision `da6e2ed…` は `Qwen3NextForCausalLM`、48 層、512 experts、top-10、full-attention interval 4。v0.1.4 registry と native GDN/MoE implementation に architecture がある。 | 同じく `quant_method=fp8`、dynamic activation、128x128 block。architecture/loader の組合せ上は conversion-free load 候補。 | exact checkpoint の AMD recipe/CI/result は未確認。公式 Qwen3-Next recipe は **別 checkpoint** `Qwen3-Next-80B-A3B-Instruct` を 8x MI308/MI355/MI350、TP=8 とする。 | **単一 MI300X/gfx942 での起動・性能とも未確認。** 今回は試行しない。 |

ここでいう「直接 load」は ATOM documentation/source が示す loader/quantization
path に基づくものに限る。実際に safetensors shard 名、全 weight mapping、runtime
kernel を MI300X で通したわけではないため、成功保証ではない。Coder-Next の
`modules_to_not_convert`（BF16 のまま残る module 群）を含む実メモリ使用量も未確認で
ある。

## 3. MI300X / ROCm 7.2.4 と OpenAI API

### 3.1 要件との照合

- ATOM installation guide の一般要件は Python 3.10--3.12、ROCm 6.0 以降、ROCm
  PyTorch、MI200/MI300/MI350 series である。`GPU_ARCHS` の例には
  `gfx942 = MI300X` が明記される。
- AMD ROCm 7.2.4 compatibility matrix は `gfx942` を compute target として列挙し、
  ROCm 7.2.4 release notes も MI300X の firmware/driver 組合せを示す。従って
  **ROCm 7.2.4 + gfx942 という組合せ自体は AMD 対応範囲**である。
- レンタル機の OS、kernel driver、firmware/PLDM revision、Docker runtime の実値は
  本調査で確認していない。NPS1/SPX が ATOM の Qwen3 path で検証済みかも**未確認**。

### 3.2 将来、明示的に許可された場合だけ使う native 起動案

以下は調査で確認した API/flag を合わせた**参考コマンド**であり、この作業では実行
していない。`<MODEL_DIR>` は既存の Qwen3-14B-FP8 artifact だけを read-only mount
する場所、`18000` は空きを別途確認した非既存サービス用 port の例である。

```bash
ATOM_IMAGE=rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4

docker pull "$ATOM_IMAGE"

docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=16G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <MODEL_DIR>:/models/qwen3-14b-fp8:ro \
  "$ATOM_IMAGE" \
  python -m atom.entrypoints.openai_server \
    --model /models/qwen3-14b-fp8 \
    --kv_cache_dtype fp8 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --no-enable_prefix_caching \
    --host 127.0.0.1 --server-port 18000
```

`--kv_cache_dtype fp8` は KV cache の設定であり、weight conversion をする option
ではない。API server は `POST /v1/completions`、`POST /v1/chat/completions`、
`GET /v1/models`、`GET /health` を提供し、streaming は SSE である。このため既存の
共通 client が標準 OpenAI completion/chat API を使うなら、base URL を
`http://127.0.0.1:18000/v1` に替えるだけの同条件計測が可能である。

token-id 専用の拡張 request を既存 client が使う場合の互換性、実際の port 空き、
Docker daemon/権限は**未確認**である。`ullm-openai.service` を stop/restart する
必要はこの起動案に含めず、含めてはならない。

## 4. Hybrid attention と既知リスク

Qwen3-Coder-Next は hybrid full attention + Gated DeltaNet (GDN) である。ATOM
native source は `Qwen3NextForCausalLM` 用に、full-attention の paged KV と GDN
recurrent state を別の per-request cache pool として扱う。native `Config` の
`kv_cache_block_size` は既定 16 であり、調査した v0.1.4 / current source の
Qwen3-Next/GDN config path に、vLLM が今回設定した **544** の自動 block size は
見当たらなかった。

したがって、native ATOM は vLLM の hybrid-KV grouping が生んだ
`tl.arange(0, 544)` という**同一の設定経路を共有しない**。これは native route を
構造上は候補に残す根拠である。一方で、ATOM の GDN path 自体も Triton/AITER kernel
を使い、Qwen3-Next vLLM-plugin recipe は
`GATED_DELTA_RULE_TRITON_AUTOTUNE=1` を設定する。AMD source から
`arange's range must be a power of 2` の Coder-Next/gfx942 成功・失敗報告は
見つからなかった。

よって次は未確認のままである。

- native ATOM + Qwen3-Coder-Next-FP8 + gfx942 が同じ error を出さないこと。
- Coder-Next の first compile/autotune 時間と peak VRAM。
- vLLM-ATOM plugin が vLLM の hybrid cache 制約を回避すること。plugin は vLLM の
  framework-level cache/scheduler を使うため、この目的で選ぶ根拠はない。
- SGLang-ATOM integration が exact Coder-Next を stable release で扱えること。

## 5. 数時間の rental に対する実施現実性

### 5.1 時間・容量を消費する要素

| 要素 | 一次資料で確認できた値 | 今回への影響 |
| --- | --- | --- |
| native production image | 16,605,753,298 B（約 16.61 GB、15.47 GiB） | 新規 pull と layer 展開が必要。実際の download rate / disk headroom は未確認。 |
| vLLM plugin image | `rocm/atom-dev:vllm-latest` は 19,920,943,468 B（約 19.92 GB） | native より重く、Qwen3-Next plugin 根拠も不十分なので選ばない。 |
| Qwen3-14B-FP8 weight | official Hub API 上 4 safetensors、16,326,253,296 B。既存計画では local revision `9a283b4…` を保持済み。 | local artifact を read-only mount できれば再 download は不要。 |
| Qwen3-Coder-Next-FP8 weight | official Hub API 上 40 safetensors、80,381,394,600 B。local 保有は未確認。 | 未取得なら transfer だけで大きな risk。単一 GPU での resident overhead も未確認。 |
| first compile | ATOM README は初回 model compilation を約 10 分と記載。 | SGLang の AITER JIT と同様に、短い benchmark 窓を消費する。exact model の実時間は未確認。 |
| model load | v0.1.4 loader は safetensors mmap と `ThreadPoolExecutor` による並列 load を実装する。 | 現在の 64-core F32 corpus job と CPU/IO を競合し得る。worker 数・実負荷は実測していない。 |
| conversion | Qwen config の FP8 block metadata を読む path がある。 | offline conversion は不要な見込み。ただし初回 weight post-processing/pre-shuffle は残る。 |

image と Qwen3-14B weight だけでも source payload の単純和は約 32.93 GB、Coder-Next
なら約 96.99 GB である。Docker の展開済み layer、HF cache、compile cache、空き
disk の実効値は別なので、この値を必要容量保証には使わない。

### 5.2 判定

| 対象 | 判断 | 根拠 |
| --- | --- | --- |
| Qwen3-14B の full `C=1,8,16,128` benchmark | **No-go** | architecture/load path はあるが、AMD の optimized/CI/performance target ではない。新規 image pull、first compile、CPU-parallel load の後に得る row は experimental compatibility result に留まる。 |
| Qwen3-14B の将来の最小 smoke | **条件付き** | CPU corpus job 完了後、image が既に local cache にあり、別途 GPU 使用が明示許可された場合だけ、native engine を 30 分上限で health → single completion → C=1 の順に確認する価値はある。成功しても full sweep への自動昇格はしない。 |
| Qwen3-Coder-Next-FP8 | **No-go** | exact model / MI300X×1 の公式検証なし、80.38 GB transfer risk、hybrid GDN の gfx942 実行未確認、公式 recipe は別 checkpoint の 8 GPU。 |

この判断では既存の vLLM / SGLang / llama.cpp 結果を置換しない。ATOM を結果表に加える
なら、将来の最小 smoke 成功後も `experimental; exact model not AMD CI-validated` と
明記し、同一 client 条件・image digest・ATOM/AITER version・compile warmup を別途
記録する。

## 6. 未確認として残す事項

- レンタル機の OS/kernel driver/firmware、Docker 実行可否、空き disk、network
  bandwidth、actual NPS1/SPX support。
- stable v0.1.4 image が Qwen3-14B-FP8 の exact revision を end-to-end で load/
  generate すること、および同モデルの throughput/VRAM/accuracy。
- Qwen3-Coder-Next-FP8 の exact checkpoint が native ATOM と MI300X×1 で
  load/generate すること、hybrid Triton failure の有無、peak VRAM、性能。
- ATOM native の Qwen3-14B に vLLM/SGLang より速い kernel path が実際に選ばれる
  こと。AMD 一次資料に比較数値は見つからなかった。
- common client が token-id 固有拡張を必要としないこと、illustrative port `18000`
  が空いていること。

## 一次資料

すべて 2026-07-26 JST に確認した。`ROCm/ATOM` の current-source snapshot は
[`9d4bd543`](https://github.com/ROCm/ATOM/commit/9d4bd543bf88f26ae75944ebbfe20eff1dd788b0)
である。production image と同じリリースについては v0.1.4 の source/release を
優先した。

- [ROCm/ATOM README (v0.1.4)](https://github.com/ROCm/ATOM/blob/v0.1.4/README.md) — 名称、native engine、AITER、対応 architecture、first compile、API。
- [ATOM v0.1.4 release](https://github.com/ROCm/ATOM/releases/tag/v0.1.4) — paired AITER、production image、実際に joint validation した 5 model。
- [ATOM Model Run Guide (current snapshot)](https://github.com/ROCm/ATOM/blob/9d4bd543bf88f26ae75944ebbfe20eff1dd788b0/docs/model_run_guide.md) — nightly CI validation 対象一覧と grid。
- [ATOM model support guide (v0.1.4)](https://github.com/ROCm/ATOM/blob/v0.1.4/docs/model_support_guide.md) — architecture registry、safetensors/mmap/threadpool loader、Qwen3-Next GDN/MoE。
- [v0.1.4 Qwen3-Next native model class](https://github.com/ROCm/ATOM/blob/v0.1.4/atom/models/qwen3_next.py) と [v0.1.4 registry](https://github.com/ROCm/ATOM/blob/v0.1.4/atom/model_engine/model_runner.py) — two target architecture の source-level 照合。
- [Qwen3-Next recipe](https://github.com/ROCm/ATOM/blob/9d4bd543bf88f26ae75944ebbfe20eff1dd788b0/recipes/Qwen3-Next.md) — exact recipe checkpoint と 8 GPU hardware。
- [vLLM OOT plugin guide](https://github.com/ROCm/ATOM/blob/9d4bd543bf88f26ae75944ebbfe20eff1dd788b0/docs/vllm_plugin_backend_guide.md) と [SGLang model-impl guide](https://github.com/ROCm/ATOM/blob/9d4bd543bf88f26ae75944ebbfe20eff1dd788b0/recipes/SGLang-ATOM-Model-Impl-Backend.md) — framework との責務境界と制約。
- [native OpenAI serving guide](https://github.com/ROCm/ATOM/blob/v0.1.4/docs/serving_benchmarking_guide.md) — endpoint、SSE、`--server-port`。
- [Docker Hub native tag metadata](https://hub.docker.com/v2/repositories/rocm/atom/tags/rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4) と [vLLM dev tag metadata](https://hub.docker.com/v2/repositories/rocm/atom-dev/tags/vllm-latest) — tag、digest、image size。
- [AMD ROCm 7.2.4 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.4/compatibility/compatibility-matrix.html) と [ROCm 7.2.4 release notes](https://rocm.docs.amd.com/en/develop/release.html) — gfx942/MI300X と driver/firmware compatibility。
- [Qwen3-14B-FP8 config](https://huggingface.co/Qwen/Qwen3-14B-FP8/blob/9a283b4a5efbc09ce247e0ae5b02b744739e525a/config.json) と [Qwen3-Coder-Next-FP8 config](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8/blob/da6e2ed27304dd39abadd9c82ef50e8de67bdd4c/config.json) — exact architecture/FP8 metadata。各 Hub API の safetensors aggregate を容量欄に用いた。
