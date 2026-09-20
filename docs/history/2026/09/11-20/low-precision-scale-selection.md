# 低精度scale選択の修正: 作業履歴（2026-09-19〜）

[対応する計画](../../../../plans/archive/2026/09/11-20/low-precision-scale-selection.md) ／
先行調査 [MXFP8/vLLM FP8 KLD差の原因調査](qwen38-mxfp8-vllm-fp8-attribution.md) ／
[KLD再現手順](../../../../references/qwen38-kld-reproduction.md)

この文書は、計画の段階0〜4の実行結果・失敗試行・測定値・raw logits hashを段階ごとに記録する。
文脈が要約されても、この文書から再開できるようにする。段階5（既定への採用）は行わない。
既定の数値経路は変更せず、新規則は診断用opt-inとして実装・測定する。

## セッション条件（2026-09-19）

- `codex -p deepseek-flash`。名前付きluna/terra/solは起動せず、既定subagentのみ使用。
- GPU: R9700 `GPU-a8e9ddefa2d60f55`（gfx1201）とV620 `GPU-76a08c022586fed6`／`GPU-08b2ddcbd6e6b36c`（gfx1030）。
- V620を使う間はローカルQwenサービスを停止。R9700は `ci/tools/run_mtp_teacher_forced_r9700.py` の
  service lease手順（systemd user unitの停止、health/ready確認、performance_level記録・復元、service hash検証）に従う。
- commit・pushはしない。既存の未コミット変更 `docs/development/codex-environment.md` には触れない。

## 固定identity（段階0基準）

- KLD corpus: `$ROOT` = `/home/homelab1/datapool/qwen38-kld-20260918`
  - `cases-v1.json` sha256 `03f46dfd346b596814b6f4a247a1a51939d88058a7ac6057ae30e2135fd2baa9`
  - `vocab-v1.json` sha256 `b0776f4ee98950e8f3b13cbb4f9174ff66abcbad87a0aa6b398965bfc9cce884`
  - 8入力・2,632位置、248,077語彙、FP64集計、`KL(llama.cpp BF16 || 候補)`、温度1
- 参照: `$ROOT/results/llama-bf16-fp16/`（9 case、2,761行、全caseのSHA-256は下記）
- BF16原本: `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、
  lock `docs/models/locks/qwen3.8-27b-bf16.json`
- 段階0の基準source: 先行調査の診断opt-in commitをHEADとした `47b997e5`（sLLM build）。
  本セッションの作業開始時のHEADは `b0373a38`（この上で段階1以降を実装する）。
- 段階0用に再ビルドしたgfx1201 worker:
  - `sllm-qwen38-mx-kld-dump` sha256 `53b1672d14090f832f5822a648f0d61b84258e329ce2b9f752f8be29e1379368`
  - `sllm-qwen38-kld-dump` sha256 `8269401ec6f9136becf324453e63cebfef4ef12c527c04b323ccece4f159bca1`
  - 旧 `initial-runner-identities.json` のbinary hash記録は古い。段階0ではこのhashを使う。

### 参照raw logits hash（llama.cpp BF16、段階0・4共通の基準）

| case | sha256 |
| --- | --- |
| en_prose_129 | `8d93ce140edaa236940b5086160e12c5852737bef8e81b7af72c56f8796009c5` |
| ja_prose_129 | `9cc75d527992202610514775e26b1837eeac8dcc24a2df736c255dec175f430c` |
| zh_tech_257 | `2f6df3aee6114d8a7e686aa9555f2b7b8e5f2fcbab67c6daf1da7f4f2d6db2c8` |
| rust_code_257 | `d383acf5df613f4b9f7281728f8f3a49a04bebcea17e3beb893d7eee1d17fff5` |
| python_code_513 | `45c4b0b7be8cd2b6eb90055cc0c6d310f62a71ed4223242e99eb75d2be3347a1` |
| sql_code_1025 | `ded8b39785c35610255718f97aff2c0c07dd8e1dad45c53855f53f773905be08` |
| arithmetic_65 | `8a7da432c10c7628603b08aa46037a6576d1154b67faa9e5c390ede80ec1312c` |
| structured_257 | `a97db2d49531751c446e8199fa08f832b18191771011b2297ac0010074128add` |
| en_prose_129_repeat | `8d93ce140edaa236940b5086160e12c5852737bef8e81b7af72c56f8796009c5` |

（`$ROOT/results/capture-audit.json` の `files` から抽出。全case `nonfinite_count=0`。）

## 段階0: 基準測定

### 0.1 必要な5条件と既存captureの有無（R9700・chunk32）

| 条件 | 重み | KV | 既存capture | 対応 |
| --- | --- | --- | --- | --- |
| A | MXFP8（sLLM変換） | FP16 | 無 | 新規取得（下記） |
| B | MXFP6（sLLM変換） | FP16 | 無（gfx1030のみ） | 新規取得 |
| C | NVFP4（Unsloth） | FP16 | `sllm-nvfp4-fp16-gfx1201-chunk32` | hash再確認して再利用 |
| D | NVFP4（Unsloth） | MXFP8 E4 | `sllm-nvfp4-kv-mxfp8-e4-gfx1201-chunk32` | hash再確認して再利用 |
| E | MXFP8（sLLM変換） | MXFP8 E4 | 無（gfx1030のみ） | 新規取得 |

既存のNVFP4 capture（C・D）のper-case raw logits SHA-256（capture-audit.jsonより）:

| case | C: NVFP4/FP16 | D: NVFP4/MXFP8-E4 |
| --- | --- | --- |
| en_prose_129 | `012b396153d678187a369b60c1d8f2e9f68c0795fd4db0390dddb1b5fe1c1f08` | `abb95a5c925b27648f235abc92884678e604b69b2e782da824823835a5d0848b` |
| ja_prose_129 | `0c3e9005017228cfe3bf3205f2771b5e00f07f13178d7732442a59da55e31770` | `5d15bdb12f12df36db8880f49b2127635be308a8016ff94b78a9a8b416805f40` |
| zh_tech_257 | `b704e45945d95d83a96d3ba1ad134c0e23f7eba5b91eb4cbf19bf56f20b95c48` | `f6475fa2345dd471f9f0fa76940c578306943b69294cd860509b6c0c62ee177f` |
| rust_code_257 | `821d71b7a2981d8424f276520ad81518308c80e0722bede9c236d8bfdaaeca81` | `020e77a06d56a02f336c04de8b3b69465daef60f13caa524693eed4c99715a6d` |
| python_code_513 | `9b0ded07d23ea729df9dca79caf9a02d7a9e88849085e64c495f416eb24f91c3` | `e09506c7b733bcd8aa9da479d1cdc6ed58badb09861a4cf59633230434827721` |
| sql_code_1025 | `46228955186f40953854aa3f95ab9c3bb7a9a4e0fb679102b74041bab9e9c61b` | `62897047bd1f7b16df0f72d2e5ba029cce6d8255c8be314eaf06637787da5420` |
| arithmetic_65 | `233d0ab16a7e5185855f4fc71b6832f4f2880976bd04551637780f2c45976aba` | `6193e44e10dec18088bac86eb83db1cb9e1fa2416a6835cf72cff7baf335a5c1` |
| structured_257 | `569df9adac401933c2cb0c4fe14cb8a9194bb7704ce99f927b98a94a674eff22` | `c3cb035c36288114efb1ad0549b172518467fd055b24e0a440558846569c5581` |
| en_prose_129_repeat | `012b396153d678187a369b60c1d8f2e9f68c0795fd4db0390dddb1b5fe1c1f08` | `abb95a5c925b27648f235abc92884678e604b69b2e782da824823835a5d0848b` |

### 0.2 取得した測定（測定値はこの節に追記する）

（以下、各captureの実行コマンド・条件・平均KLD・raw logits hashを順に追記）

#### 0.2.0 測定値の要約（2026-09-19、R9700 gfx1201・chunk32）

KLD出力は `$ROOT/results/kld-lowp-baseline-*.json`。primaryはrepeat controlを除く2,632位置の平均。

| 条件 | 重み / KV | capture | primary mean KLD | all_rows mean | primary top1 |
| --- | --- | --- | --- | --- | --- |
| A | MXFP8 / FP16 | `sllm-mxfp8-fp16-gfx1201-chunk32-lowp-baseline` | `0.04590723672094489` | `0.050016673792619176` | `0.9145136778115501` |
| B | MXFP6 / FP16 | `sllm-mxfp6-fp16-gfx1201-chunk32-lowp-baseline` | `0.07144603166489717` | `0.08595953724276068` | `0.9114741641337386` |
| C | NVFP4 / FP16 | `sllm-nvfp4-fp16-gfx1201-chunk32` | `0.09221767965057509` | `0.10666350709823069` | `0.8974164133738601` |
| D | NVFP4 / MXFP8 E4 | `sllm-nvfp4-kv-mxfp8-e4-gfx1201-chunk32` | `0.10358649283915224` | `0.11717679092746038` | `0.8970364741641338` |
| E | MXFP8 / MXFP8 E4 | `sllm-mxfp8-kv-mxfp8-e4-gfx1201-chunk32-lowp-baseline` | `0.04656399363718403` | `0.050888858999522835` | `0.918693009118541` |

case別mean（A/C/D。repeatはen_prose_129と同値）:

| case | A | C | D |
| --- | --- | --- | --- |
| en_prose_129 | `0.133862` | `0.401403` | `0.394461` |
| ja_prose_129 | `0.075006` | `0.124747` | `0.145172` |
| zh_tech_257 | `0.034203` | `0.064800` | `0.064995` |
| rust_code_257 | `0.079342` | `0.117769` | `0.127522` |
| python_code_513 | `0.034481` | `0.061113` | `0.079250` |
| sql_code_1025 | `0.024212` | `0.052709` | `0.065545` |
| arithmetic_65 | `0.202985` | `0.364025` | `0.397654` |
| structured_257 | `0.035030` | `0.073481` | `0.077289` |

#### 0.2.1 条件A: MXFP8重み / FP16 KV（R9700, chunk32）

新規取得。同時に、既存の既定MXFP8 capture 2本とraw logitsがbyte一致することを確認した
（`sllm-mxfp8-default-newnative-fp16-gfx1201-chunk32`、`sllm-mxfp8-fp16-gfx1201-chunk32-attribution`）。

実行（2026-09-19、R9700 service `inactive` を確認した上で実行。workerはrebuild済みgfx1201）:

```
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 LD_LIBRARY_PATH=/opt/rocm/lib \
python3 ci/tools/qwen38_kld_sllm_mx.py \
  --source-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-BF16 \
  --manifest $ROOT/cases-v1.json \
  --worker $REPO/.local-artifacts/lowp-boundary/build/sLLM/gfx1201/release/sllm-qwen38-mx-kld-dump \
  --output-dir $ROOT/results/sllm-mxfp8-fp16-gfx1201-chunk32-lowp-baseline \
  --kind mxfp8 --gguf $ROOT/sllm-mxfp8/Qwen3.8-27B-MXFP8.gguf \
  --derived-lock $ROOT/sllm-mxfp8/Qwen3.8-27B-MXFP8.derived-lock.json \
  --model-lock docs/models/locks/qwen3.8-27b-bf16.json \
  --target gfx1201 --device-index 0 --kv fp16 --chunk-size 32
```

KLD: `python3 ci/tools/qwen38_kld_compare.py --reference $ROOT/results/llama-bf16-fp16 --candidate <capture> --inputs $ROOT/cases-v1.json --vocab $ROOT/vocab-v1.json --output $ROOT/results/kld-lowp-baseline-mxfp8-fp16.json`

per-case raw logits SHA-256（条件A・baseline）:

| case | sha256 |
| --- | --- |
| en_prose_129 | `a2959b371dfbaa9188f36299f5c8a6489d0f74c030a46b405c78346b373ca4f5` |
| ja_prose_129 | `3ea22a3c6df0009920708ca8d44c9844cdccd90bfeed82fb7f6974533cad4e7a` |
| zh_tech_257 | `055f9c1a45f491ce9fa8b4e15074ad672aa23df2d681a1d903ff99962efb0d0b` |
| rust_code_257 | `2f94ec4db4e09a138e170d8f6294a9be1c4bac06c7d1c9cc56d79ee3f1b8c09e` |
| python_code_513 | `1ae908ac17bed606c5a2572475b9ca8527688db117fbc7f512bf37df012d90e0` |
| sql_code_1025 | `224380f3cb7cb7329bcf0d654f7a93d4cc69e0b050e578e5b65976063838b90b` |
| arithmetic_65 | `ac7c8ecee92436fc91bb83d944dc303303678c87733fb0c3df9854f8df99a549` |
| structured_257 | `ec0cfeb4b2c33c27d04b099a1d0630a5c2cc89db957d53cb1683950e89b83684` |
| en_prose_129_repeat | `a2959b371dfbaa9188f36299f5c8a6489d0f74c030a46b405c78346b373ca4f5` |

#### 0.2.2 条件C: NVFP4重み / FP16 KV（R9700, chunk32）

既存capture `sllm-nvfp4-fp16-gfx1201-chunk32` を再利用。per-case SHA-256は0.1節の表と一致することを再確認した
（このcaptureはgfx1030の `sllm-nvfp4-fp16-gfx1030-chunk32` とmanifest sha256が同一）。
KLD出力: `$ROOT/results/kld-lowp-baseline-nvfp4-fp16.json`。

#### 0.2.3 条件D: NVFP4重み / MXFP8 E4 KV（R9700, chunk32）

既存capture `sllm-nvfp4-kv-mxfp8-e4-gfx1201-chunk32` を再利用。per-case SHA-256は0.1節の表と一致することを再確認した。
KLD出力: `$ROOT/results/kld-lowp-baseline-nvfp4-kv-mxfp8-e4.json`。

観察: 本番既定のNVFP4は、KVをMXFP8 E4にすると KLD が `0.0922`→`0.1036` と**悪化**する（FP16 KVの方が良い）。
これはKV MXFP8 E4のscale飽和の影響と整合する。段階4でKV新規則の効果を測る根拠になる。

#### 0.2.4 条件B: MXFP6重み / FP16 KV（R9700, chunk32）

新規取得。PASS（elapsed 135.8s）。実行は条件Aと同じ形で `--kind mxfp6`,
`--gguf $ROOT/sllm-mxfp6/Qwen3.8-27B-MXFP6.gguf`,
`--derived-lock $ROOT/sllm-mxfp6/Qwen3.8-27B-MXFP6.derived-lock.json`,
`--output-dir $ROOT/results/sllm-mxfp6-fp16-gfx1201-chunk32-lowp-baseline`。
KLD出力: `$ROOT/results/kld-lowp-baseline-mxfp6-fp16.json`。

case別mean: en_prose `0.382080`, ja_prose `0.074721`, zh_tech `0.044137`, rust_code `0.065037`,
python_code `0.056854`, sql_code `0.040952`, arithmetic `0.264354`, structured `0.049553`。

per-case raw logits SHA-256（条件B）:

| case | sha256 |
| --- | --- |
| en_prose_129 | `d50a55b0dd059b1a4e85bb852c0788da7b1eaa7aaa966be1500d6769d56d3e90` |
| ja_prose_129 | `32d58bdd38b3a9a600f0e9ddcb1b44959b09af3b41c82e224e81ad273bcd2a1f` |
| zh_tech_257 | `208e3cf24b658c21dd52e9449a0021164e3da019a31a0ad7eb3d811b2b31aa96` |
| rust_code_257 | `692bc9f9dcbfa1eb800201eeee61c73e7505b605c18907d79fff99201e28bcd7` |
| python_code_513 | `d32a3ab600fb4a0a75fe2494d1f0432ee5312e7e934d268194f902d19c88fad0` |
| sql_code_1025 | `77019599040d9daf5b8d8f94bf034af64724398ff26c408315516dfcd5a59bb4` |
| arithmetic_65 | `d280c2af424abe28991c391b6378dcbc81b23200d443ffe7f51c01755ba277ce` |
| structured_257 | `6a8f32b9a9584e6f1e9176c51801d852c9f55fbe7445433f05987b48c049ec93` |
| en_prose_129_repeat | `d50a55b0dd059b1a4e85bb852c0788da7b1eaa7aaa966be1500d6769d56d3e90` |

注: gfx1030・chunk32の既知値は `0.06707940031320425`（別target。段階4で両GPU比較に使う）。

#### 0.2.5 条件E: MXFP8重み / MXFP8 E4 KV（R9700, chunk32）

新規取得。PASS（elapsed 336.8s。KV MXFP8 E4経路のためBより遅い）。条件Aと同じ実行形で
`--kind mxfp8 --kv kv-mxfp8-e4`、`--output-dir $ROOT/results/sllm-mxfp8-kv-mxfp8-e4-gfx1201-chunk32-lowp-baseline`。
KLD出力: `$ROOT/results/kld-lowp-baseline-mxfp8-kv-mxfp8-e4.json`。

case別mean: en_prose `0.139130`, ja_prose `0.090158`, zh_tech `0.045173`, rust_code `0.074282`,
python_code `0.037837`, sql_code `0.023501`, arithmetic `0.180190`, structured `0.027496`。

per-case raw logits SHA-256（条件E）:

| case | sha256 |
| --- | --- |
| en_prose_129 | `47b0b3a5a1fe7bcc7c7070083fc5ba9e62ea23007e05842b35ea118f61473b73` |
| ja_prose_129 | `f84a99a7178ccecd84e0d0ebe807b86df7d4fdf31a21cbc745e2f13280d90f0e` |
| zh_tech_257 | `1c5f441d702328b9b0fb78e10032365822b0ea928ec3543aacaac86484e52795` |
| rust_code_257 | `95f7cbd79821ef68e4999acb948448155ad9a54a4f13271ba8625385c808f076` |
| python_code_513 | `dce4df05d291e543591316481872726ee50c8dbbd6808bfe47b33d0faf913c5a` |
| sql_code_1025 | `239e763767b52d313cecf485f419dbd990725257bd30a61d1237f274c76997eb` |
| arithmetic_65 | `964d6dbe4627bea5ac4172983c76d8bc24d4aec609d415cac81b252d180a62ba` |
| structured_257 | `88d3c9038f8f1f3cbc8000a9a68e4ce9007549b0b1b3dda282b33a2588a9b93a` |
| en_prose_129_repeat | `47b0b3a5a1fe7bcc7c7070083fc5ba9e62ea23007e05842b35ea118f61473b73` |

観察（段階0の5条件そろい踏み）:

- MXFP8重みはKV FP16とKV MXFP8 E4でほぼ同等（`0.04591` vs `0.04656`、差 `+0.00066`）。KV E4の飽和影響は小さい。
- 一方NVFP4重みでは同じKV E4が `0.09222`→`0.10359` と大きく悪化する。低精度重みほどKV誤差に敏感。
  段階4では両者を分けて評価する。

#### 0.2.6 段階0残件: NVFP4活性値のscale飽和カウンタ（R9700, chunk32）

計画の段階0「NVFP4 W4A4活性値で `amax/(6·global) > 448` となるブロックの割合を、上記コーパスで層ごとに集計」を
診断用counterで取得した。出力: `$ROOT/results/nvfp4-activation-saturation-gfx1201-chunk32.json`
（930,980 bytes、mtime 2026-09-19 05:44、15456レコード）。実行binaryは段階0の `sllm-qwen38-kld-dump`
（sha256 `8269401ec6f9136becf324453e63cebfef4ef12c527c04b323ccece4f159bca1`）。

クリーン集計（`total_blocks == m*k/16` かつ `saturated_blocks <= total_blocks`、15431レコード）:

| 代表shape (M×K) | 飽和ブロック / 全ブロック | 飽和率 |
| --- | --- | --- |
| M=1, K=5120 | 36 / 161,280 | 0.022321% |
| M=1, K=17408 | 27 / 548,352 | 0.004924% |
| M=32, K=5120 | 62 / 98,447,360 | 0.000063% |
| M=32, K=17408 | 0 / 167,430,144 | 0% |
| **全体** | **125 / 266,587,136** | **0.0000469%** |

観察: `amax/(6·global) > 448`（E4M3 block scale上限への頭打ち）となるブロックはコーパス全体で
0.00005%未満であり、M=1（decode）でやや高く、M=32（prefill）ではほぼゼロ。計画の「頻度が高い場合のみ
動的global scaleを別途提案」という条件には当たらない。既定のNVFP4活性量子化規則はこの計画では変更しない。

**既知の不具合（集計の際に記録）**: JSONは15456レコードだが、うち25レコードが破損している
（`total_blocks` が `m*k/16` と不一致、うち2件は `total_blocks=0`、残りは約10^18〜10^19の巨大値）。
`total_blocks`/`saturated_blocks` が16bit×16bitの積を64bitで受ける箇所の型不一致
（`unsigned long long*` と `unsigned long` のdevice→host読み出し）が疑われる。counter本体は
`native/lowp/src/lowp_kernel.hip.cpp` の `nvfp4_activation_stats_*` 付近。**集計は上記クリーンsubsetで行った**。
このcounterは診断用で、既定経路の数値には影響しない（下記byte一致で確認）。

**既定経路不変の確認**: 飽和測定runの全9 case raw logits SHA-256は、既存baseline条件C
（`sllm-nvfp4-fp16-gfx1201-chunk32`）の値（0.1節の表）と**完全一致（MATCH 9/9）**した。
これはcounter追加が既定のNVFP4 FP16 KV数値経路を変えていないことの確認である。

### 0.3 独立実装による重み誤差表

計画の予備測定（scratchの一回測定）を、変換対象の全行列について独立実装で取り直した。実装は
クリーンルームNumPy（`ci/tools/qwen38_lowp_weight_error_independent.py`。既存の
`ci/tools/qwen38_weight_error.py`は読まず・importせず）で、Qwen3.8-27B BF16原本の全量子化行列を対象に、
各ブロックの二乗誤差からBF16比の相対RMSを求めた。出力:
`$ROOT/results/lowp-weight-error-independent.json`（self_test `pass`、8 checks、mismatch 0、
elapsed 41.79s、3形式×11テンソル、各436,142,080要素、合計1,308,426,240要素走査）。

| 形式 | 規則 | 相対RMS | 飽和率 |
| --- | --- | --- | --- |
| MXFP8 E4M3 | 現行 floor | 0.029668 | 0.977% |
| MXFP8 E4M3 | 新 no-clip | **0.026613** | 0.194% |
| MXFP6 E3M2 | 現行 floor | 0.054014 | 1.264% |
| MXFP6 E3M2 | 新 no-clip | **0.052821** | 0.382% |
| MXFP4 E2M1 | floor | 0.115502 | 5.238% |
| MXFP4 E2M1 | ceil | 0.142173 | 0% |
| MXFP4 E2M1 | even（現行HIP規則） | 0.112813 | 3.188% |
| MXFP4 E2M1 | best-of-two（新規則） | **0.112246** | 3.644% |
| MXFP4 E2M1 | no-clip | 0.117711 | 1.418% |

NVFP4重みは配布物が外部量子化のため本表の対象外（計画の対象外節どおり）。

観察: MXFP8は飽和を避けるだけで相対RMSが2.97%→2.66%へ下がり、飽和率も0.98%→0.19%へ下がる。
MXFP6も5.40%→5.28%と小さく改善する。MXFP4は表現範囲が狭く、飽和を避けるno-clipはむしろ悪化
（11.77%）し、best-of-two（11.22%）が最小だった。計画の予備測定結論（E4M3/E3M2はno-clip、
E2M1はbest-of-two）を実データで再確認した。

`audit`節も健全: MXFP8/MXFP6 GGUFとも tensor_count 1695、recipe binding 496、量子化値i8 496、
scale tensor 496、missing/mistyped 0、unexpected type code 0。既存GGUFが破損なく読めることを
確認（受入条件3の一部）。

## 段階1: 共通の規則実装とhost oracle

### 1.1 実装（既定経路は不変、新規則はopt-in）

- Rust: `mxfp.rs`の`MxScalePolicy`を`Clipped`／`NoClipping`／`BestOfTwo`／`EvenMxfp4`へ一般化し、
  MXFP6・MXFP4にも適用した。`nvfp4.rs`に隣接2候補のbest-of-twoを追加。`kv_mxfp8.rs`の参照実装も
  同じ規則にした（`quantize_kv_mxfp8_with_policy`）。
- HIP: `low_precision_block_codec.hpp`に`ocp_mx_scale_code_no_clip`（0xffをNaN印として温存し
  254で頭打ち）と、`e2m1_scaled_squared_error`／`nvfp4_e4m3_scale_candidates`を追加した。
- 既定経路は旧規則のまま。新規則は環境変数でopt-inし、この段階では実装・検証のみ行った。

### 1.2 host oracle（Rust unit test）

`cargo test -p sllm-core --lib`で3モジュール。すべてPASS。

| module | 結果 | 新規則の主なtest |
| --- | --- | --- |
| `mxfp::` | 15 passed / 0 failed | `no_clipping_scale_is_the_smallest_block_max_representable_scale`、`mxfp4_even_scale_bumps_at_the_1_75_mantissa_threshold`、`mxfp4_best_of_two_never_exceeds_floor_or_even_block_error`、`special_blocks_keep_the_historical_handling` |
| `nvfp4::` | 7 passed / 0 failed | `best_of_two_scale_uses_one_of_the_two_adjacent_e4m3_codes`、`best_of_two_weight_rule_never_exceeds_nearest_block_error`、`nearest_weight_rule_matches_the_default_entry_point` |
| `kv_mxfp8::` | 5 passed / 0 failed | `no_clipping_kv_scale_is_the_smallest_non_saturating_e8m0_step`、`standard_scale_uses_floor_power_not_block16_overflow_avoidance` |

### 1.3 GPU oracle（両GPU、境界込み）

`native/lowp/tests/low_precision_block_codec_gpu_test.hip.cpp`に独立host再実装＋GPU実行の
`scale_rule_oracle_kernel`を追加（+715行）。GPU上の実ヘッダhelper出力と独立host実装のbyte一致、
およびE8M0のNaN印`0xff`をassertする。

| target | oracle cases | nan_marker_0xff | 結果 |
| --- | --- | --- | --- |
| gfx1030（V620） | 94 | true | PASS |
| gfx1201（R9700） | 94 | true | PASS |

JSONは`target`以外byte同一。境界ケース: 要素最大値（448／57344／28／6）、その直上のnextafter、
仮数1.5×／1.75×、厳密な2の冪、非正規域最大、全ゼロ、全NaN、全+Inf、混在、count=1（非整列ブロック）、
NVFP4の代表E4M3 code直下／一致／直上。`scale_rule_e4m3_power=8`、`scale_rule_e5m2_power=15`、
`scale_rule_mxfp6_power=4`。（注: E4M3のelement powerは8で、15はE5M2。E4M3コードに対して
E4M3 block規則を検証している。）

実行:

```
cmake --build .local-artifacts/lowp-boundary/standalone-gfx1201
LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=2 .local-artifacts/lowp-boundary/standalone-gfx1201/sllm_low_precision_codec_gpu_test
cmake --build .local-artifacts/lowp-boundary/standalone-gfx1030
LD_LIBRARY_PATH=/opt/rocm/lib .local-artifacts/lowp-boundary/standalone-gfx1030/sllm_low_precision_codec_gpu_test
```

（HIP device 0はgfx1030 V620、R9700 gfx1201はHIP device 2。KLD workerは`ROCR_VISIBLE_DEVICES`で
R9700をdevice 0へ再マップする。）

→ 受入条件1（新規則がCPU参照とGPUでbyte一致、両GPU、境界込み）の段階1分は充足。

## 段階2: 活性値とKVの量子化カーネル

### 2.1 opt-in環境変数

| 対象 | 変数 | 新規則 |
| --- | --- | --- |
| MXFP8活性値 | `SLLM_MXFP8_ACTIVATION_NO_CLIP_SCALE` | 飽和しない最小E8M0 scale |
| MXFP6活性値 | `SLLM_MXFP6_ACTIVATION_NO_CLIP_SCALE` | 同上（上限28） |
| MXFP4活性値 | `SLLM_MXFP4_ACTIVATION_BEST_OF_TWO_SCALE` | floor／floor+1の二乗誤差最小 |
| NVFP4活性値 | `SLLM_NVFP4_ACTIVATION_BEST_OF_TWO_SCALE` | 隣接2 E4M3 codeの二乗誤差最小 |
| MXFP8 E4／E5 KV | `SLLM_KV_MXFP8_NO_CLIP_SCALE` | 飽和しない最小E8M0 scale |

既定は旧規則のまま（未設定時は旧経路。段階4.1でbyte一致を確認）。

### 2.2 活性値量子化の追加コスト（V620 gfx1030、warmup 1＋measured 3のmedian）

新規`native/lowp/tests/lowp_quantize_cost_microbench.hip.cpp`（実行名
`sllm_lowp_quantize_cost_microbench`）。活性値量子化単体（GEMMなし）のBF16→低精度コスト。
ログ: `.local-artifacts/lowp-boundary/quantize-cost-v620.log`。

| 形式 | shape (M×K) | 旧規則 [ms] | 新規則 [ms] | 倍率 |
| --- | --- | --- | --- | --- |
| MXFP8 | 8192×5120 | 2.525 | 2.500 | ~1.0 |
| MXFP8 | 128×5120 | 0.0629 | 0.0637 | ~1.0 |
| MXFP8 | 8192×17408 | 1.973 | 2.100 | ~1.06 |
| MXFP8 | 128×17408 | 0.147 | 0.158 | ~1.07 |
| MXFP6 | 8192×5120 | 2.435 | 2.622 | ~1.08 |
| MXFP6 | 128×5120 | 0.0631 | 0.0670 | ~1.06 |
| MXFP6 | 8192×17408 | 2.046 | 2.151 | ~1.05 |
| MXFP6 | 128×17408 | 0.153 | 0.164 | ~1.07 |
| MXFP4 | 8192×5120 | 4.711 | 65.002 | ~13.8 |
| MXFP4 | 128×5120 | 0.343 | 4.470 | ~13.0 |
| MXFP4 | 8192×17408 | 17.890 | 267.978 | ~15.0 |
| MXFP4 | 128×17408 | 1.105 | 3.478 | ~3.1 |
| NVFP4 | 8192×5120 | 2.298 | 4.003 | ~1.7 |
| NVFP4 | 128×5120 | 0.190 | 0.319 | ~1.7 |
| NVFP4 | 8192×17408 | 7.678 | 13.628 | ~1.8 |
| NVFP4 | 128×17408 | 0.582 | 1.013 | ~1.7 |

観察: MXFP8／MXFP6のno-clipはほぼ無視できる追加コスト（≤8%）。NVFP4 best-of-twoは約1.7倍。
**MXFP4 best-of-twoは3〜15倍で、活性値量子化としては現実的でない**。MXFP4活性値は旧W4A4の
診断経路であり既定の数値経路ではないが、採用可否の材料として明示する。重み変換（実行時ではなく
一度だけ）なら同コストは問題にならない。測定は一回で、絶対値は±数%のばらつきを含む。

## 段階3: 重みの変換器

### 3.1 実装（既定は旧規則、新規則は診断opt-in）

- `crates/sllm-core/src/gguf_convert.rs`: `QwenMxScaleRule`（旧規則／`NoClipping`）と
  `semantic_model_id_with_scale_rule`、`build_/write_..._with_scale_rule`を追加。新規則GGUFは
  semantic IDに`:mx-scale=no-clipping`を付けて新旧を区別する。
- `crates/sllm-cli/src/bin/sllm-convert-qwen38-mx.rs`: `--mxfp8-no-clipping-scale`／
  `--mxfp6-no-clipping-scale`を追加。kindと一致しないフラグや相互排他、重複指定は拒否する。
  未指定時は旧規則（既定不変）。
- `crates/sllm-tools/src/artifact.rs`: MXFP4／NVFP4量子化に`QuantizeScaleRule`（best-of-two）を
  追加。既定recipeは`default_recipes_keep_the_historical_numeric_rule`のとおり旧規則のまま。

### 3.2 テスト

| 対象 | 結果 |
| --- | --- |
| `cargo test -p sllm-core --lib gguf_convert::` | 9 passed / 1 ignored（Gemma4 MoEはimmutable source cache要） |
| `cargo test -p sllm-cli --bin sllm-convert-qwen38-mx` | 3 passed（`mxfp6_no_clipping_scale_is_explicit`、`no_clipping_scale_is_explicit_and_composable_with_retention`、`no_clipping_scale_is_rejected_for_mxfp6`） |
| `cargo test -p sllm-tools --lib` | 9 passed（`scale_rule_parsing_and_effective_recipe_are_explicit`、`default_recipes_keep_the_historical_numeric_rule`、`best_of_two_recipes_reach_their_own_scale_rule`） |

### 3.3 新規則GGUF

| 形式 | 規則 | size [byte] | output sha256 | metadata sha256 | tensor_catalog sha256 |
| --- | --- | --- | --- | --- | --- |
| MXFP8 | 旧 | 31,997,388,928 | `f60f3a9f…` | `9680cfe3…` | `fb1c92c4…` |
| MXFP8 | no-clip | 31,997,388,960 | `d7401283…` | `a5bb9956…` | `fb1c92c4…` |
| MXFP6 | 旧 | 25,909,749,888 | `4fe9e947…` | `80aceb4c…` | `4dd220c7…` |
| MXFP6 | no-clip | 25,909,749,920 | `04818a9c…` | `edde09e8…` | `4dd220c7…` |

完全hash（derived-lock `output.sha256`）:

| 形式 | output sha256 |
| --- | --- |
| MXFP8 旧 | `sha256:f60f3a9f69fe853fa1f69f3e4e9ddbda921d9bbfac480607cb48469d9951b310` |
| MXFP8 no-clip | `sha256:d7401283116c6ce7d250c8a6f44629c95f310d119b2ec1b8b34196334e6bba92` |
| MXFP6 旧 | `sha256:4fe9e9470e5c7be439984646edce1cf438f4d66f00abcb93394dc787defd1847` |
| MXFP6 no-clip | `sha256:04818a9cf13045ddbe6b271c7f26b36519f0a8263ea9c6f236095cf5bdc7978d` |

- 各no-clip GGUFは旧GGUFより32 byte大きい（semantic ID文字列の分）。`tensor_catalog_sha256`は
  新旧で同一（tensor名・shape・offset不変）で、値・scaleのバイトだけが差し替わる。
- MXFP8-no-clipは先行調査（2026-09-18 09:14生成）の既存artifactを再利用した。MXFP6-no-clipは
  本セッションで2026-09-19 05:56に`--mxfp6-no-clipping-scale`で生成した。
- semantic IDはいずれも `qwen38:sha256:0498226db11f8c2446344aa23e185d934050cee7b8b8b90f34587f3283f42276:mx-scale=no-clipping`
  となり、旧規則（サフィックス無し）と区別される（受入条件3）。
- derived lockのconverter commitは`0000000000000000000000000000000000000000`（draft・未commit）。

## 段階4: 効果の測定

段階0と同じ条件（R9700 `GPU-a8e9ddefa2d60f55`／gfx1201、chunk32、`$ROOT`コーパス、KL(BF16||候補)）で、
形式ごとに「活性値だけ」「重みだけ」「両方」「KVも」を測る。V620（gfx1030）はMXFP6の代表1条件と、
後から補完したMXFP8 KV E5の1組のみ（4.2.4）。
R9700 service lease: 実行時 `sllm-qwen38-r9700.service` は inactive、`power_dpm_force_performance_level=auto`。
すべて `ci/tools/qwen38_kld_sllm_mx.py`／`qwen38_kld_sllm.py` + `qwen38_kld_compare.py` で取得。driverは
（Git管理外）`.local-artifacts/lowp-boundary/run_stage4_r9700.sh`。

### 4.0 新binary（段階4で使用）

段階1〜3の実装を含み、既定経路は旧規則のままのgfx1201 release worker:

- `sllm-qwen38-mx-kld-dump` sha256 `732cbaaa942904afbca04d7f73efe153766d4eaa2986a27cf79e7bd0edc00951`
- `sllm-qwen38-kld-dump` sha256 `463c0c600982f376e09db8cd9473b69871de7d4b1e457482ca66c4f68f44646e`

（段階0の `53b1672d…`／`8269401e…` とは別binary。段階0のcaptureとの比較は「同一条件でraw logitsが
byte一致するか」で行う。）

### 4.1 既定経路のbyte一致（受入条件2）

- MXFP8重み／FP16 KV（既定、`stage4-mx8-default`）: primary mean KLD `0.04590723672094489`、
  top1 `0.9145136778115501`。**段階0条件A（`0.04590723672094489`）と一致**。全9 caseのraw logits
  SHA-256も段階0条件Aと9/9完全一致（下表）。

| case | stage4-mx8-default sha256 | 段階0Aと一致 |
| --- | --- | --- |
| en_prose_129 | `a2959b371dfbaa9188f36299f5c8a6489d0f74c030a46b405c78346b373ca4f5` | yes |
| ja_prose_129 | `3ea22a3c6df0009920708ca8d44c9844cdccd90bfeed82fb7f6974533cad4e7a` | yes |
| zh_tech_257 | `055f9c1a45f491ce9fa8b4e15074ad672aa23df2d681a1d903ff99962efb0d0b` | yes |
| rust_code_257 | `2f94ec4db4e09a138e170d8f6294a9be1c4bac06c7d1c9cc56d79ee3f1b8c09e` | yes |
| python_code_513 | `1ae908ac17bed606c5a2572475b9ca8527688db117fbc7f512bf37df012d90e0` | yes |
| sql_code_1025 | `224380f3cb7cb7329bcf0d654f7a93d4cc69e0b050e578e5b65976063838b90b` | yes |
| arithmetic_65 | `ac7c8ecee92436fc91bb83d944dc303303678c87733fb0c3df9854f8df99a549` | yes |
| structured_257 | `ec0cfeb4b2c33c27d04b099a1d0630a5c2cc89db957d53cb1683950e89b83684` | yes |
| en_prose_129_repeat | `a2959b371dfbaa9188f36299f5c8a6489d0f74c030a46b405c78346b373ca4f5` | yes |

→ 段階1〜3の実装（診断opt-in含む）後も、既定のMXFP8数値経路はbyte不変。実行elapsed 355.7s。

### 4.2 KLD結果

| 条件 | run名 | primary mean KLD | top1 | 備考 |
| --- | --- | --- | --- | --- |
| MXFP8 既定（再取得） | `stage4-mx8-default` | `0.04590723672094489` | `0.9145136778115501` | 段階0Aと一致 |

#### 4.2.1 途中経過（2026-09-19 06:3x時点、binary A）

段階4のbinaryは2つある（下記4.3のMXFP6 reader修正の前後）。

- binary A（初回）: `sllm-qwen38-mx-kld-dump` sha256 `732cbaaa942904afbca04d7f73efe153766d4eaa2986a27cf79e7bd0edc00951`、
  `sllm-qwen38-kld-dump` sha256 `463c0c600982f376e09db8cd9473b69871de7d4b1e457482ca66c4f68f44646e`
- binary B（MXFP6 reader修正後）: `sllm-qwen38-mx-kld-dump` sha256
  `1fbb94d81237f3711dfbbecff0632cb4a52ea00ab00c11b7b5b2de8ef01cc4e3`、
  `sllm-qwen38-kld-dump` sha256 `4f431a25655f5ed0b49636b3d4fab4b22a82111e2f4ef60af0887a48f6920878`

R9700 `GPU-a8e9ddefa2d60f55`／gfx1201、chunk32、KV FP16、`$ROOT`コーパス:

| 条件 | run名 | primary mean KLD | top1 | elapsed | binary |
| --- | --- | --- | --- | --- | --- |
| MXFP8 既定（再取得） | `stage4-mx8-default` | `0.04590723672094489` | `0.9145136778115501` | 355.7s | A |
| MXFP8 重みのみ no-clip | `stage4-mx8-weight-noclip` | `0.037460179854181624` | `0.9209726443768997` | 622.0s | A |
| MXFP8 活性のみ no-clip | `stage4-mx8-act-noclip` | `0.02544621536047676` | `0.9422492401215805` | 349.5s | A |
| MXFP8 重み+活性 no-clip | `stage4-mx8-both-noclip` | `0.017764072077310768` | `0.9513677811550152` | 334.9s | A |
| MXFP8 KVのみ no-clip（KV E4） | `stage4-mx8-kv-noclip` | `0.04443971114789734` | `0.9190729483282675` | 314.7s | A |
| MXFP8 重み+活性+KV no-clip（KV E4） | `stage4-mx8-both-kv-noclip` | `0.02433971464722365` | `0.9460486322188449` | 320.6s | B |
| MXFP6 既定 | `stage4-mx6-default` | `0.07144603166489717` | `0.9114741641337386` | 130.0s | B |
| MXFP6 重みのみ no-clip | `stage4-mx6-weight-noclip` | `0.058688343210089505` | `0.916033434650456` | 123.7s | B |
| MXFP6 活性のみ no-clip | `stage4-mx6-act-noclip` | `0.10850084031841034` | `0.8905775075987842` | 122.5s | B |
| MXFP6 重み+活性 no-clip | `stage4-mx6-both-noclip` | `0.07451786358379653` | `0.90919452887538` | 122.6s | B |
| MXFP6 重み+活性 no-clip（KV E4） | `stage4-mx6-both-kv-noclip` | `0.10571579468571826` | `0.898176291793313` | 122.2s | B |
| NVFP4 既定 | `stage4-nvfp4-default` | `0.09221767965057509` | `0.8974164133738601` | 338.4s | B |
| NVFP4 活性 best-of-two | `stage4-nvfp4-act-best2` | `0.08831650219226525` | `0.9012158054711246` | 332.5s | B |
| NVFP4 KV no-clip（KV E4） | `stage4-nvfp4-kv-noclip` | `0.09609068319835838` | `0.8974164133738601` | 332.9s | B |
| NVFP4 活性 best-of-two + KV no-clip | `stage4-nvfp4-act-best2-kv-noclip` | `0.10623088197589965` | `0.8928571428571429` | 333.8s | B |

の`mx6-default` set_sha256 `8357064c0fcee629cd288f7677881f42e4e31251d4d2a1af2b59b2dce454806f`は、段階0条件Bの
per-case hashから再構成したset digest `8357064c…4806f`と**完全一致**する。`nvfp4-default`も段階0条件Cの
`0.09221767965057509`と一致する。binary A/Bや段階0 workerをまたいで既定経路の数値が保たれている
（正式なbridge byte一致は4.4で確認する）。

（MXFP8は重み・活性の両方を飽和なしscaleにすると 0.0459→0.0178 で、先行調査の参考値と一致。
MXFP6は逆で、**活性値のno-clipはKLDを大きく悪化させる**（`0.07145`→`0.10850`）。重みonlyは改善
（`0.07145`→`0.05869`）だが、重み+活性にすると既定よりわずかに悪い`0.07452`になる。ETA次第で
form別判断材料に反映する。`mx6-both-kv-noclip`はKVをFP16→E4に変える効果も混ざるため参考値。）

raw logits set_sha256（各captureの全9 case、`logits_digest.py`）:

| 条件 | set_sha256 |
| --- | --- |
| `stage4-mx8-default` | `773deda2b6a067b9619dc6141fbe4c052891618808cddd4c40de6a641a6ab38d` |
| `stage4-mx8-weight-noclip` | `e492797da635d85dc3f36cc36f7f40e1b78699ba278ed70335a42927eb2955bd` |
| `stage4-mx8-act-noclip` | `bd91db7447db22a25d22f9654f6fbb9fd9f78ab4f980a7cec832f5e01ff26c81` |
| `stage4-mx8-both-noclip` | `2d702fde392481aef58511597c82eea06b5a45e5756e2fd4b263b11c5f3cca8e` |
| `stage4-mx8-kv-noclip` | `6ee195216cddcef812b0cb00c8e2a2b175ee744ba26b0f4c0064417b87bb2e2c` |
| `stage4-mx8-both-kv-noclip` | `f06242d99fcc201180df3cbb25a567dc9c4f7e91ffa887967beb68df597212f2` |
| `stage4-mx6-default` | `8357064c0fcee629cd288f7677881f42e4e31251d4d2a1af2b59b2dce454806f` |
| `stage4-mx6-weight-noclip` | `ce2e5549c40315b52a124c7b40938ae619203d149d553e101eb38f4e6e7d6868` |
| `stage4-mx6-act-noclip` | `fa55afaa5f7ad991d3408fc94c45c95ccbab4c446062f13c21160221ecc3811c` |
| `stage4-mx6-both-noclip` | `3a41200af9416536d9c0ba34d3551306731303b43e89625acdc6510fc6755c41` |
| `stage4-mx6-both-kv-noclip` | `e53730dd3da2284f80f277558dcb170af9340b12023d1e6a2b0a3bd52ac7b8f2` |
| `stage4-nvfp4-default` | `0fe1f0c43df2a3d70d0be798eded00207901caf2edbb58b2fad2290992f57862` |
| `stage4-nvfp4-act-best2` | `9a53f1c0b5a941d8de558b3468b18d619c100bca5d7cddde31fb555844ff3b53` |
| `stage4-nvfp4-kv-noclip` | `9d3517a7c4903fcad113164001b897785114b01f0f19588b927348553b533bfe` |
| `stage4-nvfp4-act-best2-kv-noclip` | `667b20b95dafd16c0730469ceac5065061c5ba30ac32ec2e24dae79a7a6f5173` |
| `stage4-v620-mx6-default-b` | `9dca5587702ccda731c55250c8d2655bc9d412077260f7ee905bcc9d1d8e7643` |
| `stage4-v620-mx6-both-noclip` | `7e70166907aec12cf08926befdb32aecae1de907e3715c2a5e11b98d1cf69d2c` |

**binary帰属の訂正（2026-09-19 06:32）**: R9700 A driverのworker path
`.local-artifacts/lowp-boundary/build/sLLM/gfx1201/release/sllm-qwen38-mx-kld-dump` は、4.3の修正で
`build-fixed-gfx1201/`のbinary B（sha256 `1fbb94d8…`）に**ハードリンクで置き換えられた**（mtime 06:32:06）。
そのため06:32以降にlaunchされたR9700条件はbinary Bで走っている。上の表の`binary`列のように、
`mx8-default`／`mx8-weight`／`mx8-act`／`mx8-both`／`mx8-kv-only`はbinary A（06:30:27 launch分まで）、
`mx8-both-kv-noclip`以降はbinary Bである。binary A/Bの数値等価性は4.4のbridge測定
（`mx8-default-bridge`：binary B → binary Aの`stage4-mx8-default`とbyte一致を要求）で確認する。
追加のB driverはMXFP6行を再実行せず、bridge 2条件のみを実行する（A driverがすでにbinary Bで
`mx6-default`／`mx6-weight`／`mx6-act`／`mx6-both`／`mx6-both-kv`を取得するため）。

#### 4.2.2 V620（gfx1030）MXFP6代表2条件（同一binary B、`build-fixed-gfx1030/`）

`run_stage4_v620_b.sh all`。V620を使用する間、ローカルQwenサービスは`process: stopped`を確認。
R9700と同じく `$ROOT`コーパス、chunk32、KV FP16。

| 条件 | run名 | primary mean KLD | top1 | elapsed |
| --- | --- | --- | --- | --- |
| MXFP6 既定 | `v620-mx6-default-b` | `0.06707940031320425` | `0.9137537993920972` | 210.7s |
| MXFP6 重み+活性 no-clip | `v620-mx6-both-noclip` | `0.07877854428962727` | `0.9107142857142857` | 202.6s |

観察: gfx1030のMXFP6既定値`0.06707940031320425`は段階0の既知値（0.2.4の注）と一致し、
同一条件でのbinary同一性を確認した。ただしこのtargetでは重み+活性no-clipが既定より**悪化**する
（`0.06708`→`0.07878`）。R9700のbinary B結果（4.2.1）と合わせて形式別判断材料にする。

#### 4.2.3 binary A/B bridge と段階0同一性（R9700）

4.2.1のbinary帰属を踏まえ、binary B（`build-fixed-gfx1201/`）で既定条件だけを再取得し、
binary A／段階0とのraw logits byte一致を確認した（`run_stage4_r9700_b.sh`）。

| 条件 | run名 | primary mean KLD | set_sha256 | 比較対象 | byte一致 |
| --- | --- | --- | --- | --- | --- |
| MXFP8 既定 / FP16 KV | `stage4-mx8-default-bridge` | `0.04590723672094489` | `773deda2b6a067b9619dc6141fbe4c052891618808cddd4c40de6a641a6ab38d` | binary A `stage4-mx8-default` | yes |
| NVFP4 既定 / FP16 KV | `stage4-nvfp4-default-bridge` | `0.09221767965057509` | `0fe1f0c43df2a3d70d0be798eded00207901caf2edbb58b2fad2290992f57862` | binary B `stage4-nvfp4-default`／段階0条件C | yes |

段階0との同一性もper-case hashから独立に再構成して確認した:

- MXFP6既定（binary B `stage4-mx6-default`）のset_sha256
  `8357064c0fcee629cd288f7677881f42e4e31251d4d2a1af2b59b2dce454806f` は、段階0条件Bのper-case hashから
  再構成したset digestと一致する。
- NVFP4既定（binary B `stage4-nvfp4-default`／bridge）のset_sha256
  `0fe1f0c43df2a3d70d0be798eded00207901caf2edbb58b2fad2290992f57862` は、段階0条件Cの
  per-case hashから再構成したset digestと一致する。

→ 受理入力が変わらない限り、段階1〜3の実装とreader修正は既定のMXFP8／MXFP6／NVFP4数値経路を
byte不変に保つ（受入条件2）。binary Aで取った`mx8-*`行はbinary Bと同一視できる。
**MXFP6のR9700行とNVFP4全行、MXFP8の`both-kv`行はbinary Bで取得**である。

#### 4.2.4 MXFP8 KV E5M2 の補完測定（V620 gfx1030、同一binary B、2026-09-19 08:07〜08:20）

計画の修正方針表は MXFP8 **E5M2** KV を「飽和しない最小scale（上限57344）。段階0の測定で悪化しないことを
確認する」としているが、段階0の5条件にも段階4の条件にも E5 KV が含まれていなかった。選択器は
`Mxfp8E5` を gfx1030 専用としており gfx1201 では拒否するため、V620（gfx1030）で1組だけ補完した。
V620を使用する間、ローカルQwenサービスは `process: stopped` を確認（R9700は今回未使用）。

- worker: `.local-artifacts/lowp-boundary/build-fixed-gfx1030/sllm-qwen38-mx-kld-dump` sha256
  `275633bc76ea20da9cd9528f7539d283ca4c78dc621d5dd49711f9b99e02972a`（4.3のbinary Bと同一）。
- 重み: `$ROOT/sllm-mxfp8/Qwen3.8-27B-MXFP8.gguf` sha256
  `f60f3a9f69fe853fa1f69f3e4e9ddbda921d9bbfac480607cb48469d9951b310`、derived lock sha256
  `7697db20bfe75f0284e39d70de8f3eeb64b69a0dbd7d1d9592cef87243295f3a`（既定MXFP8重み、再量子化なし）。
- 条件: `$ROOT/cases-v1.json`（2,632位置）、chunk32、`--kv kv-mxfp8-e5`、`--target gfx1030`。E5 no-clip側は
  `SLLM_KV_MXFP8_NO_CLIP_SCALE=1` を付与。driverは `ci/tools/qwen38_kld_sllm_mx.py`（再現コマンドは下）。
- 実行（`execution.json`）: default `state=PASS`／elapsed 397.92s、no-clip `state=PASS`／elapsed 400.34s。
  両conditionのmanifest sha256は `a5c31b0fb1b76055b1711ec067bcc30058100a670b35413379ba6fe9ff1af2fc`。

| 条件 | output | primary mean KLD | all_rows mean KLD | top1 |
| --- | --- | --- | --- | --- |
| E5 default（旧規則） | `stage4-mx8-kv-e5-default-b` | `0.04710216493374535` | `0.052080956192894615` | `0.9217325227963525` |
| E5 KV no-clip | `stage4-mx8-kv-e5-noclip-b` | `0.046939477214239474` | `0.05211592466926194` | `0.9171732522796353` |
| no-clip vs default（paired） | `kld-stage4-mx8-kv-e5-noclip-vs-default-b` | `0.02684297133587814` | `0.028517004219124117` | `0.9433890577507599` |

KLD JSON: `$ROOT/results/kld-stage4-mx8-kv-e5-default-b.json`／`…-noclip-b.json`／
`…-noclip-vs-default-b.json`（primaryは`primary_excluding_repeat_control.mean`、all_rowsは`all_rows.mean`）。
default capture は既存の `sllm-mxfp8-kv-mxfp8-e5-gfx1030-chunk32-run2`（2026-09-18）と全9 caseの
raw logitsがbyte一致した（E5既定＝旧規則の再現。全9件の `.f32` を `sha256sum` で照合）。

raw logits SHA-256（`stage4-mx8-kv-e5-default-b`／`stage4-mx8-kv-e5-noclip-b`）:

| case | default | no-clip |
| --- | --- | --- |
| arithmetic_65 | `9bdcbb692fbe67a217e286aff539adfe0a6a094dbcb90a22b1811897de83652f` | `ff69a0aaf8c00c92013d2d32847d1ff8982f5d41c40ccc5bed95bad707dc09dc` |
| en_prose_129 | `0de3f42e729e118bf19c1fe7d606c39cbc140ad75d653028158dc9b9c7dc1968` | `e48d7900c120590a7d262daca8760ab0529abf2732f54171f7ca6d1ddaadb709` |
| ja_prose_129 | `113328623fc55f0933663d6b7696a019b6a113ca1d4d003cdf67a529a524fea4` | `7b7358b7d7dbe3466286d93befb6886b70b8b33d38147f59374e4ea567332e91` |
| python_code_513 | `8d7c3d1c4a1b762d343c252834339aa0a6e868b186ea6eed06a9d727369c8be3` | `b3607d245c6be04e5cd005cabda6dcadcd0806272277516035b77247affd67fe` |
| rust_code_257 | `a72754b009e02ceb053ae7cb6488314a35f34e6335a5f79e09741d62062886ad` | `8e8f5530bd46bf319584d9d9032e5d9db86a04f1d3752636c6c685719ce2fb0a` |
| sql_code_1025 | `d3b9b45bdb449d5c6773d3da855e0d3068056379f2a11590c552d4389192ea9c` | `e5c0922dca79251c2706559ca0b4839cbfbaa8086d562e375da7e24b7ffb0a7b` |
| structured_257 | `c63e8c20ecca932cb8fff7d423d36b5b2eac56f4860c0f537a1bf492d5afc119` | `9b76e00d2637c19a10d2575a2ed202775eab7d68718cf3c07837ec0186f7f96c` |
| zh_tech_257 | `107a371708707fe475f57de3807915f8002840b5d4de4f3d034b19e3549e7867` | `e7e3da418233526f6d3a8e7bbae8f3f0d156cea0b4a03ea52e0d2ad49f4a564d` |
| en_prose_129_repeat | `0de3f42e729e118bf19c1fe7d606c39cbc140ad75d653028158dc9b9c7dc1968` | `e48d7900c120590a7d262daca8760ab0529abf2732f54171f7ca6d1ddaadb709` |

観察:

- 参考（同一gfx1030・2026-09-18 capture・KL(BF16‖)）: MXFP8/FP16 KV `0.044441`、KV E4旧規則 `0.047018`、
  KV E5旧規則 `0.047102`（E5は本測定defaultと一致）。KVのE4/E5量子化はいずれもFP16 KVよりKLDを上げる。
- E5 **default（旧規則）** は既存run2 captureとraw logitsが9/9 byte一致。E5 KV選択時も既定規則は不変。
- E5 **no-clip** はBF16基準の primary mean KLD を `0.047102`→`0.046939` とわずかに改善するが、all_rows
  は `0.052081`→`0.052116` と僅かに悪化し、top1は `0.921732`→`0.917173` と**悪化**する。
- 同じgfx1030の既知値（2026-09-18 capture）では MXFP8/FP16 KV `0.044441` < KV E4（旧規則）`0.047018`
  < KV E5（旧規則）`0.047102`。E5 no-clip `0.046939` はE4 defaultを僅かに下回るが差は `0.00008` で、
  all_rows（`0.052116` > `0.051778`）とtop1（`0.9172` < `0.9195`）は逆に悪化する。E4 no-clip（R9700
  `0.044440`、4.2.1）やFP16 KVには届かない。→ E5 KVを既定へ上げる推奨材料にはならない。
- paired KL(no-clip‖default) は `0.026843`、top1一致 `0.9434` で、両variantは互いに近くない。BF16基準KLDの
  差（`0.00016`）は2規則間の僅差というより、どちらもBF16から同程度離れたまま別方向へ動いた差であり、
  no-clipが一貫して良いとは言えない。

再現コマンド（`$ROOT=/home/homelab1/datapool/qwen38-kld-20260918`、V620 drive時に
`LD_LIBRARY_PATH=/opt/rocm/lib`／`ROCR_VISIBLE_DEVICES=GPU-76a08c022586fed6` をexport）:

```bash
# E5 default（旧規則）
python3 ci/tools/qwen38_kld_sllm_mx.py \
  --source-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-BF16 \
  --manifest $ROOT/cases-v1.json \
  --worker .local-artifacts/lowp-boundary/build-fixed-gfx1030/sllm-qwen38-mx-kld-dump \
  --output-dir $ROOT/results/stage4-mx8-kv-e5-default-b \
  --kind mxfp8 --gguf $ROOT/sllm-mxfp8/Qwen3.8-27B-MXFP8.gguf \
  --derived-lock $ROOT/sllm-mxfp8/Qwen3.8-27B-MXFP8.derived-lock.json \
  --model-lock docs/models/locks/qwen3.8-27b-bf16.json \
  --target gfx1030 --device-index 0 --kv kv-mxfp8-e5 --chunk-size 32
# E5 no-clip: 同じコマンドに SLLM_KV_MXFP8_NO_CLIP_SCALE=1 を付与し、--output-dir を noclip 側へ変える
# KLD: qwen38_kld_compare.py --reference $ROOT/results/llama-bf16-fp16 --candidate <capture> \
#        --inputs $ROOT/cases-v1.json --vocab $ROOT/vocab-v1.json --output <kld.json>
# paired: 同じ qwen38_kld_compare.py の --reference を default capture、--candidate を noclip capture にする
```

### 4.3 失敗試行と修正: MXFP6 no-clip重みGGUFの読み込み（binary B）

- 2026-09-19 06:28、V620（gfx1030）の `v620-mx6-both-noclip` が33.4秒でFAILED:
  `build seed Qwen3.8 MX graph: invalid Qwen graph weight plan: GGUF MXFP6 recipe cannot declare MXFP8 no-clipping scale mode`
- 原因: 先行調査で入れたreader検証（`crates/sllm-core/src/qwen_graph.rs`）が `mx-scale=no-clipping` タグを
  MXFP8重み専用として扱い、MXFP6レシピを拒否していた。段階3でMXFP6 no-clipへ一般化した際の reader 側の
  追従漏れで、段階4のMXFP6重み測定（`mx6-weight`／`mx6-both`／`mx6-both-kv`）はR9700でも同様に失敗する状態だった。
- 修正: `GgufRecipeEncoding::Mxfp6E3m2Block32E8m0` アームの拒否分岐を削除し、`mx-scale=` タグを形式非依存の
  記録（そのGGUFの重みscaleを飽和なし規則で選んだこと）として扱う。MXFP6 no-clipは既定と同一の resident
  contract（`DType::U8`／`Encoding::Mxfp6W6A6`）へ写像する。拒否されていた入力が受理されるようになるだけで、
  受理済み入力の計画・数値は不変。
- 影響: 段階4のworkerを再ビルドし（binary B）、MXFP6の行は binary B、それ以外は binary Aで測定する。
  binary間の数値等価性は、既定MXFP8／既定NVFP4の橋渡し測定（4.4）でraw logitsのbyte一致を確認して示す。

- binary B のビルド（修正後、`.local-artifacts/lowp-boundary/`）:
  - gfx1201: `build-fixed-gfx1201/sllm-qwen38-mx-kld-dump` sha256
    `1fbb94d81237f3711dfbbecff0632cb4a52ea00ab00c11b7b5b2de8ef01cc4e3`、
    `build-fixed-gfx1201/sllm-qwen38-kld-dump` sha256
    `4f431a25655f5ed0b49636b3d4fab4b22a82111e2f4ef60af0887a48f6920878`
  - gfx1030: `build-fixed-gfx1030/sllm-qwen38-mx-kld-dump` sha256
    `275633bc76ea20da9cd9528f7539d283ca4c78dc621d5dd49711f9b99e02972a`、
    `build-fixed-gfx1030/sllm-qwen38-kld-dump` sha256
    `a29afc83089687ec4ecf74d511ccdb1ebbebcdb892f9e9d6d2a8ce06f147d915`
  - gfx1030の旧worker（`.local-artifacts/lowp-boundary/build/sLLM/gfx1030/release/`、binary A相当）は
    reader修正前なので、V620測定は `build-fixed-gfx1030/` のみを使う。

### 4.4 段階4拡張: Qwen3.5-4B品質fixtureとMTP M1

#### 4.4.1 Qwen3.5-4B no-clip GGUF生成

- 2026-09-19 06:25の初回試行は失敗: `sllm-convert-gguf: tool commit or executable SHA-256 is invalid`。
  `--converter-commit` に全ゼロを渡していたため `ToolIdentityV1::validate` が拒否した。
- 対処: `--converter-commit` に作業開始時点のHEAD `6116bef95e6eb38ea4b4f8460a2a2941561e8c75` を記録して再生成。
  （作業ツリーはdirtyなので、このcommitはsource系列の記録であり「そのcommitのビルド」を意味しない。）
- 生成結果（`.local-artifacts/lowp-boundary/build-cli-gfx1030/sllm-convert-gguf` sha256
  `d4116fc3b7acde4d9345ce1fa06849b6b610a9d126084536a821ad1082802ee9`）:
  - `/home/homelab1/.cache/sllm/derived/phase62-qwen35-mxfp8-noclip-bundle`（model.gguf 5,886,147,200 bytes、
    sha256 `305ae047edb8bf018a06983c7b9da1a289c752809cb018de2addd017bb7fba8a`、tensor 986）
  - `/home/homelab1/.cache/sllm/derived/phase62-qwen35-mxfp6-noclip-bundle`（model.gguf 4,993,874,560 bytes、
    sha256 `600fcda69e3967160859bc85c41e17f6d56489528b0e8366f6a5b4c94b4fe1bc`、tensor 986）
  - 既定規則の既存bundle `phase62-qwen35-mxfp8-bundle`／`phase62-qwen35-mxfp6-bundle` は変更していない。

#### 4.4.2 4B品質campaignのr1失敗とr2修正（2026-09-19 07:38）

- r1（`stage4-quality-manifest-gfx1201.json` → `stage4-quality-gfx1201-r1`）は
  `mxfp8-default`のreport解析でFAIL: `failed Expecting value: line 1 column 1 (char 0)`。
  原因は`run_lowp_boundary.py`の既定判定（`json=True`でreport.jsonをJSONとして読む）に対し、
  `sllm-qwen35-mx-weight-quality`のstdoutが`<capture path> sha256:<digest>`の平文1行だったこと。
  実作業自体は成功しており`stage4-quality-gfx1201-r1/mxfp8-default/capture.json`（102,463,983 bytes、
  sha256 `73c2b5959de0d6ca941adc8dfdc020f8fdc2f6c6f791e9b8a13edf2bddf5b196`）は生成済み。
- 対処: 各jobに`capture`キーを付け、runnerの`capture`分岐（`repeat.bitwise_identical`、
  `cleanup.final_cleanup_empty`、dispatch監査をassertしreport.jsonを読まない）を通す。出力先を
  `stage4-quality-gfx1201-r2`にした manifest `stage4-quality-manifest-gfx1201-r2.json` で再実行した。


#### 4.4.3 Qwen3.5-4B品質fixture（r2 campaign、2026-09-19 07:57〜07:59）

- campaign: manifest `stage4-quality-manifest-gfx1201-r2.json`、runner
  `python3 ci/tools/run_lowp_boundary.py <manifest>`、出力 `stage4-quality-gfx1201-r2/`、
  target gfx1201（`GPU-a8e9ddefa2d60f55`）、state `complete`、wall 146.2s
  （job: mxfp8-default 37.4s / mxfp8-noclip 40.1s / mxfp6-default 32.2s / mxfp6-noclip 31.0s）。
- worker: `.local-artifacts/lowp-boundary/build-quality-gfx1201/sllm-qwen35-mx-weight-quality`
  sha256 `ee196b30717d507593c0b2bd4c9954f7e78045ceeda6f1b378ee3cf303d3f6f0`。
  入力は4.4.1で生成した既定／no-clip GGUF（`phase62-qwen35-mxfp8[-noclip]-bundle`、
  `phase62-qwen35-mxfp6[-noclip]-bundle`）。各 `capture.json` sha256:
  - mxfp8-default `73c2b5959de0d6ca941adc8dfdc020f8fdc2f6c6f791e9b8a13edf2bddf5b196`
  - mxfp8-noclip  `43dc00a8f6f3d86be8b478fcb080c372b5b6cd1565052a2f9d97486a794a884f`
  - mxfp6-default `df9535c1b0ad92747128f4d09a3835e01c83554ce4af195c02f18ac9ecaddde0`
  - mxfp6-noclip  `ce33cf0cc4f76cb29d56ae01e2d04c6ea93ca37618033d862c4fcdc13ede4bfa`
- 比較: `.local-artifacts/lowp-boundary/compare_4b_quality.py --baseline <default capture>
  --candidate <noclip capture> --label <name>`。出力 `compare-mxfp8.json`（sha256 `a641a83d…`）／
  `compare-mxfp6.json`（sha256 `45095c61…`）。fixture sha256
  `a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d`（10ケース、decode/prefill各1行）。

MXFP8（既定 vs no-clip）:

- artifact sha256: baseline `f253d9f4…5d076f`（既定bundle）、candidate `305ae047…7fba8a`
  （4.4.1のno-clip GGUFと一致）。primary_logit_digest: baseline `0df04465…efa977`、
  candidate `01ce9055…3f0396`。
- primary top1不一致は20比較中3件: decode case1（b015、314→13）、decode case8（b512、278→13）、
  prefill case8（b512、265→926）。prefill case0（b001）はtop1一致。
- softmax KL: decode 0.0038〜0.0158、prefill 0.0055〜0.0545（最大はcase0 b001 prefill 0.054471、
  TV 0.127653、max_abs 1.78125）。mean_abs_logit_diffはdecode 0.059〜0.125、prefill 0.074〜0.308。
- 既定比で軽微な差。top1は3/20で変わるがKL・TVは小さい。

MXFP6（既定 vs no-clip）:

- artifact sha256: baseline `d0ff2e1d…a0264e`（既定bundle）、candidate `600fcda6…b4fe1bc`
  （4.4.1のno-clip GGUFと一致）。primary_logit_digest: baseline `c3db6d44…98b238`、
  candidate `343629e6…9c642f`。
- primary top1不一致は20比較中6件: decode case0（b001、370→198）、case1（b015、314→198）、
  case3（b017、1886→11）、case5（b256、14→7）、case7（b511、220→13）、prefill case8（b512、824→265）。
- softmax KLはMXFP8より明確に大きい: decode最大0.0920（case0）、prefill最大0.1267（case0 b001）、
  0.1023（case8 b512、TV 0.1821）。mean_abs_logit_diffはdecode 0.124〜0.297、prefill 0.135〜0.396。
- 既定比で4B品質を**悪化**させる。R9700 KLD（`mx6-both-noclip` 0.074518 > 既定 0.071446）と整合する。

観察: MXFP8 no-clipは4B品質で軽微（KL最大0.055）、MXFP6 no-clipは明確に悪化（decode top1が5/10で変化）。
重みのみの変更でこれだけ差が出るのは、E3M2の表現範囲が狭くno-clipが小さい値の分解能を落とすためと考えられる。

#### 4.4.4 Qwen3.8 MTP期待受理率（M1、gfx1201、2026-09-19 07:39〜07:52）

- 目的: MTP companion sidecarをno-clip規則で作り直したときの期待受理率への影響。weightのみの変更で、
  対象本体（Unsloth NVFP4モデル、model_sha256 `c473512c…afcc05`）とMTP sidecarのbase recipeは不変。
- companion（jobs.jsonの`SLLM_PHASE84_MTP_COMPANION_PATH`）:
  - MXFP8既定 `.local-artifacts/phase84/mxfp8`（encoding `mxfp8-w8a8-e4m3-block32-e8m0`、
    report `companion_digest` `sha256:6ef86986…e235858`。`docs/development/mtp-companion-quantization.md`の
    既定digestと一致）。
  - MXFP8 no-clip `.local-artifacts/lowp-boundary/mtp-sidecar/mxfp8-noclip`
    （encoding `…:mx-scale=no-clipping`、digest `sha256:8c3fec0b…2edb607`、payload 437,946,338 bytes）。
  - MXFP6既定 `.local-artifacts/phase84/mxfp6`（`mxfp6-w6a6-e3m2-block32-e8m0`、
    digest `sha256:f85ff503…c5722a`）。
  - MXFP6 no-clip `.local-artifacts/lowp-boundary/mtp-sidecar/mxfp6-noclip`
    （`…:mx-scale=no-clipping`、digest `sha256:64a139d7…de7f82`、payload 331,777,977 bytes）。
  - no-clip sidecarは段階3の`sllm-convert-qwen38-mtp --encoding mxfp8|mxfp6 --mxfp8-no-clipping-scale|--mxfp6-no-clipping-scale`
    で生成（source `unsloth/Qwen3.8-27B-NVFP4` rev `57926baca9a82b4d6906b43f2750d55315f5b10f`）。
- runner: `python3 ci/tools/run_phase86_mtp_catch_up.py --target gfx1201
  --binary .local-artifacts/lowp-boundary/mtp-sidecar/bench-gfx1201
  --jobs .local-artifacts/lowp-boundary/mtp-sidecar/jobs.json
  --output .local-artifacts/lowp-boundary/mtp-sidecar/m1-gfx1201-r1`。
  binary sha256 `939391189ea66434d68e8bf21bfa688123924d80501855f4b8c9cbe6b39be2a2`
  （dirty tree＝HEAD `6116bef95e6eb38ea4b4f8460a2a2941561e8c75`＋段階1〜3変更からビルド）。
  prefix `.local-artifacts/mtp-bench/claimb/gen2-bf16/prefixes.json`（sha256 `78d9cd07…1a4cfdd0`、
  8ケース、`suite-quick8.json`由来）。state `complete`、job 211.9／214.0／213.1／219.2s、
  jobs.json sha256 `402520d5…a987d152`。R9700 lease: 実行前`service_was_active=false`、終了後
  service hash不変・復元済み、performance level `auto`復元。
- 解析: `python3 ci/tools/mtp_expected_acceptance.py --baseline <default report>
  --candidate <noclip report> --top-k 20 --top-p 0.95 --bootstrap 20000 --seed 86 --out <path>`。
  M1 = temperature-1の期待受理率 `sum_t min(p'(t), q'(t))`、support transformはreference token selectorの
  top_k/top_p分岐。
  - MXFP8（`mxfp8-acceptance.json`）: mean `0.7579398998833502` → `0.756275812453996`
    （diff −0.166 pt、95%CI [−0.342, +0.009] pt、sign-flip p=0.1566、改善2／悪化6 prompt）。
    step1 `0.799183`→`0.796874`（−0.231 pt）、step2 `0.702017`→`0.701065`（−0.095 pt）。
  - MXFP6（`mxfp6-acceptance.json`）: mean `0.7533479287527172` → `0.751531597488952`
    （diff −0.182 pt、95%CI [−0.539, +0.161] pt、p=0.3976、改善3／悪化5 prompt）。
    step1 `0.795118`→`0.792049`（−0.307 pt）、step2 `0.696721`→`0.695895`（−0.083 pt）。
- canonical M1との関係: 既存M1（26プロンプト、gfx1201）の保存report
  `.local-artifacts/phase86-mtp-catch-up/final-gfx1201/P/report.json` を同じ解析器に通すと
  `baseline_mean=0.776222819973026` を再現する（`canonical-gfx1201-perprompt.json`）。
  今回の8ケースは、その26ケースのうち`gen2-bf16`にもともと入っていた8件とprompt/output prefixが
  sha256一致する。その8件だけのcanonical部分平均は`0.755378`で、今回の既定再測`0.757940`とは
  約0.26 pt差（MTP graph/binary側の差で、companion差ではない。model_sha256は同一、既定companionは
  既定規則のdigestと一致）。→ 今回のM1は**同一run内の既定 vs no-clipのpaired差**として読む。
- どちらの形式もno-clipはM1をわずかに下げるが、有意差はない（p=0.16／0.40、CIが0をまたぐ）。
  悪化幅はproposal step2よりstep1でやや大きい。

## 形式ごとの採否判断材料（段階5の入力、2026-09-19）

段階0〜4まで完了。以下は形式ごとの材料であり、**段階5（既定への採用）は未実施**。
既定の数値経路は旧規則のまま、新規則はすべて診断opt-in（環境変数・変換フラグ）である。
採否はユーザー判断で、ここでは推奨の向きだけを示す。

### KLD（R9700 gfx1201・chunk32・`$ROOT`コーパス・2,632位置、KL(BF16‖候補)）

| 対象 | 既定 | 新規則 | 備考 |
| --- | --- | --- | --- |
| MXFP8 重み | `0.045907` | `0.037460` | weight-noclip |
| MXFP8 活性 | `0.045907` | `0.025446` | act-noclip |
| MXFP8 重み+活性 | `0.045907` | `0.017764` | both-noclip（vLLM FP8 0.0145に近づく） |
| MXFP8 KV E4 | `0.046564` | `0.044440` | kv-noclip（重みは既定） |
| MXFP8 KV E5 | `0.047102` | `0.046939` | E5 no-clip。primary改善だがall_rows・top1悪化、FP16 KV `0.044441`に届かず（4.2.4、gfx1030のみ） |
| MXFP8 重み+活性+KV | `0.045907` | `0.024340` | both-kv-noclip |
| MXFP6 重み | `0.071446` | `0.058688` | weight-noclip |
| MXFP6 活性 | `0.071446` | `0.108501` | **悪化** |
| MXFP6 重み+活性 | `0.071446` | `0.074518` | **悪化** |
| MXFP6 重み+活性+KV | `0.071446` | `0.105716` | **条件の取り違え**。実際は既定重み＋活性no-clip＋KV no-clip（末尾の追記を参照） |
| NVFP4 活性 best-of-two | `0.092218` | `0.088317` | act-best2。**wave8の不具合版で測った値**。修正後は `0.105180`（段階5を参照） |
| NVFP4 KV E4 no-clip | `0.103586` | `0.096091` | kv-noclip（FP16 KV `0.092218`よりは悪い） |
| NVFP4 活性best2+KV no-clip | `0.103586` | `0.106231` | **wave8の不具合版で測った値**（段階5を参照） |

V620（gfx1030）MXFP6: 既定 `0.067079` → 重み+活性no-clip `0.078779`（**悪化**）。同gfx1030のMXFP8 KVは
FP16 `0.044441`／E4旧 `0.047018`／E5旧 `0.047102` で、E5 no-clip `0.046939`（primaryは僅かに改善、
all_rows・top1は悪化）。

### 誤差・飽和・コスト・品質・M1

| 形式／役割 | 重み誤差RMS | 飽和率 | 実行時コスト増 | 4B品質 | MTP M1 | 判断材料 |
| --- | --- | --- | --- | --- | --- | --- |
| MXFP8 重み | 2.97%→2.66% | 0.98%→0.19% | 変換時のみ | 軽微 | −0.17 pt（CI0含む） | 採用推奨寄り |
| MXFP8 活性 | — | — | ≤8% | 未測定（実行時経路） | — | 採用推奨寄り（KLD最大の改善） |
| MXFP8 KV E4 | — | — | 未測定（KV追記は別カーネル） | 未測定 | — | 保留（対照で改善・悪化の両方が出た。末尾の追記を参照） |
| MXFP8 KV E5 | — | — | 未測定（KV追記は別カーネル） | 未測定 | — | 非推奨（KVはFP16/E4が良く、no-clipの利得も小） |
| MXFP6 重み | 5.40%→5.28% | 1.26%→0.38% | 変換時のみ | decode top1 3/10変化 | −0.18 pt（CI0含む） | 条件付き・要判断 |
| MXFP6 活性 | — | — | ≤8% | — | — | 非推奨（KLD・品質とも悪化） |
| MXFP4 活性（旧W4A4診断） | 11.55%→11.22% | 5.24%→3.64% | 3〜15倍 | — | — | 非現実的（既定経路でない） |
| MXFP4 重み | 11.28%→11.22% | 3.19%→3.64% | 変換時のみ | — | — | 変換時の選択肢（改善小） |
| NVFP4 活性 best-of-two | — | 飽和<0.00005% | 約1.7倍 | — | — | 不採用（段階5で不具合を修正して測り直すと悪化） |
| NVFP4 KV E4 no-clip | — | — | 未測定 | — | — | 保留（既定活性では改善、best2併用では悪化。末尾の追記を参照） |
| NVFP4 重み | — | — | — | — | — | 対象外（外部量子化を再量子化しない） |

飽和頻度（NVFP4活性、段階0.2.6）: `amax/(6·global)>448` は全体で125/266,587,136＝0.0000469%。
M=1で最大0.022%、M=32でほぼゼロ。計画の「頻度が高い場合のみ動的global scaleを別途提案」という条件には
当たらない。

### 受入条件の状況（段階5を除く）

| # | 条件 | 状況 |
| --- | --- | --- |
| 1 | 新規則がCPU参照とGPUでbyte一致（両GPU、境界込み） | met（1.2 host oracle、1.3 GPU oracle 94ケース両GPU） |
| 2 | 旧規則のときのraw logitsが変更前とbyte一致 | met（4.1と4.2.3 bridge。MXFP8既定9/9、MXFP6・NVFP4既定もset digest一致） |
| 3 | 既存GGUF/sidecar/lockがそのまま読め、新旧identityが区別される | met（0.3 audit、3.3 semantic ID、mtp sidecar encoding名） |
| 4 | 段階0・4が同一条件でraw logits hash付き | met（4.2のset_sha256表、段階0との再構成一致） |
| 5 | 性能記録と活性値量子化コストの明示 | met（2.2 microbench。KV追記コストは未測定） |
| 6 | 既定切替はユーザー承認のみ | 段階5未実施のため未適用（既定は旧規則のまま） |

### 未測定・残件（段階5の判断材料として）

- **MXFP8 E5M2 KV**: 4.2.4で補完済み（gfx1030のみ、1組）。既定 `0.047102` → no-clip `0.046939`
  （primaryのみ僅かに改善、all_rows・top1は悪化）。同一gfx1030のMXFP8/FP16 KV `0.044441`・KV E4（旧）
  `0.047018`に劣り、E5 KVの既定採用は推奨材料にならない。gfx1201では`Mxfp8E5`が選択器で拒否されるため
  測定対象外。
- **KV追記の実行時コスト**: 活性値量子化のmicrobench（2.2）はKV追記を含まない。KVはattention経路で
  block単位に走るため、量は活性値より小さいが未測定（E4／E5のどちらの新規則でも同じく未測定）。
- **NVFP4活性 no-clipの実運用**: best-of-twoは約1.7倍のコストでKLD改善0.004。既定採用は性能と相談。
- **段階5の作業**: 採用形式を決めたあとに、既定の変換規則・実行時規則の切替、N2記録、main-plan、
  数値変更台帳、`docs/architecture/runtime.md`の更新を行う。旧規則は診断として残す。

## 追記: 対照測定と訂正（2026-09-19、オーケストレーター）

段階4の組合せ条件のうち、KVを新規則にしたものだけが単独変更より悪化していたため、対照を追加で取得した。

### 訂正: `stage4-mx6-both-kv-noclip` の条件の取り違え

`.local-artifacts/lowp-boundary/run_stage4_r9700.sh` の `mx6-both-kv` は、重みに新規則GGUF（`$MX6N`）ではなく
既定GGUF（`$MX6`）を渡していた。したがって `stage4-mx6-both-kv-noclip`（`0.105716`）は
「既定の重み＋活性値no-clip＋KV no-clip」の値であり、「重み+活性+KV」ではない。
活性値だけno-clipの `0.108501` に近いのはこのためである。上の表と判断材料の該当行は、この訂正で読み替える。

### 対照（R9700 gfx1201、chunk32、KV MXFP8 E4、同じ段階4 binary）

MXの実行binaryは段階4のbinary B（`1fbb94d8…`）、NVFP4は `4f431a25…` で、段階4と同一であることを確認して実行した。
driverは `.local-artifacts/scale-selection/run_controls.sh`（段階4 driverの関数をそのまま使用）。

| 重み | 活性値 | KV E4 旧規則 | KV E4 no-clip | FP16 KV（参考） |
| --- | --- | ---: | ---: | ---: |
| MXFP8 既定 | 既定 | `0.046564`（段階0 E） | `0.044440` | `0.045907` |
| MXFP8 no-clip | no-clip | `0.020563`（`ctl-mx8-both-kvold`） | `0.024340` | `0.017764` |
| NVFP4 | 既定 | `0.103586`（段階0 D） | `0.096091` | `0.092218` |
| NVFP4 | best-of-two | `0.097232`（`ctl-nvfp4-act-best2-kvold`） | `0.106231` | `0.088317` |
| MXFP6 既定 | 既定 | `0.062644`（`ctl-mx6-default-kvold`） | 未測定 | `0.071446` |
| MXFP6 no-clip | 既定 | `0.058164`（`ctl-mx6-weight-kvold`） | `0.055737`（`ctl-mx6-weight-kv-noclip`） | `0.058688` |

読み取り:

- **重み・活性値の新規則は、KVの形式によらず一貫して改善した。** MXFP8の重み＋活性値no-clipはKV旧規則でも
  `0.046564 → 0.020563`、NVFP4活性値best-of-twoは `0.103586 → 0.097232`、MXFP6重みno-clipは
  `0.062644 → 0.058164`。
- **KVのno-clipは、組み合わせる条件によって改善にも悪化にも振れた。** 既定の重み・活性値では改善
  （MXFP8 `−0.0021`、NVFP4 `−0.0075`、MXFP6重みno-clip `−0.0024`）だが、MXFP8重み＋活性値no-clipでは `+0.0038`、
  NVFP4 best-of-twoでは `+0.0090` と悪化した。
- KVを量子化した方がFP16 KVより小さいKLDになる条件もある（MXFP6既定 `0.062644` < `0.071446`）。
  このコーパスでは、KV形式による差がこの規模（0.002〜0.01）だと、誤差どうしの打ち消し合いに左右されており、
  KV no-clipの効果は一貫した改善とは言えない。KV no-clipの既定採用は、この測定だけでは推奨しない。

top1一致率は `ctl-mx8-both-kvold` 0.9476、`ctl-nvfp4-act-best2-kvold` 0.8929、`ctl-mx6-default-kvold` 0.9107、
`ctl-mx6-weight-kvold` 0.9236、`ctl-mx6-weight-kv-noclip` 0.9149。

raw logits SHA-256（9ケース）:

- `stage4-ctl-mx8-both-kvold`: en_prose_129 `ad6c17a33d9b0eefacc9c9b04255bb3070f30c2ea37bc0519121b6a1e5dd7977`、ja_prose_129 `d0204788ce8e4a632668b503a0c3c7163adc3bd8f995a1f7a64aacc03081d572`、zh_tech_257 `6050d8dcd3a465cbf14eba451346bd31c8d2d712a031245d874ad5ef99c656d7`、rust_code_257 `91309c97a9d8df7e98d5f2df9d23b5b33423d8d983c70d98ba7280749c7ad8a5`、python_code_513 `589a7295df851dff2e7c4848d6bb0b47b19d69eff1354eea3644669d46fb3466`、sql_code_1025 `e7d50d4146ba2a7b6732860282e7820feeb1dda014d0311c916f141f764d2c42`、arithmetic_65 `ecf84ab5dadb2e3d17e18f3400fca2412dd8af3c9c85141e2fd92056e887e326`、structured_257 `33d69f1decdd1b2feb5585b0aa61d4fbec558100e2705e0e3dddcd8451830737`、en_prose_129_repeat `ad6c17a33d9b0eefacc9c9b04255bb3070f30c2ea37bc0519121b6a1e5dd7977`
- `stage4-ctl-nvfp4-act-best2-kvold`: en_prose_129 `68b439ad7644beaac5f5837c5d3a09110509a02e00a717e6fb71c55a9b9b2fb1`、ja_prose_129 `7dfbf9e31a17a31587ebdbdf26d50e3f31901825e89b8fa2648710bab0ae856e`、zh_tech_257 `1c5c2af50464e21ddcb00a0d738125d1f9d3436f0a7119852eaccd5a1411b42e`、rust_code_257 `0686647ff89c7bb724d50f4a384a9dc2c19b71795a777220b32d57d25dd93458`、python_code_513 `0714029e4dc938013837b87cd66ab1723dc8f87b8ad0086bfff400b2fc5b64be`、sql_code_1025 `57056adb39f50a9a13e044eacb1e72de4aeea476f86018bdf02c64ecec394fad`、arithmetic_65 `bbba17ee824dcf2c201a6d3c727ee938af0f38a219ba40cc832d89800ee304a1`、structured_257 `22087725c8d68816ed747a56337ebe19b1cc48e3f06490c48936a0f319a7affc`、en_prose_129_repeat `68b439ad7644beaac5f5837c5d3a09110509a02e00a717e6fb71c55a9b9b2fb1`
- `stage4-ctl-mx6-default-kvold`: en_prose_129 `96e36d3e48985548aa52e97db2d1c9cddb0447faf86a8641b09f707eaf20b594`、ja_prose_129 `59a53bcb01288a086f1021c63b368151fbd5e737ce4500507755a22b512ebdfe`、zh_tech_257 `f44078f1da1adf997842bc16f964bd1d4857ca53d448c78e4b7de9fb77fc13db`、rust_code_257 `8e0b5a9aed1b328a7d86e96ac96d6c2dfaf11e515eebd2e6ebca4fd86f499e72`、python_code_513 `5f72a5976348744970b27cbf84fb0fbfaa60996b467c503a4910a8e38317c2f6`、sql_code_1025 `e20ebe869f1b58bf719534e52596789dbb22d969552b128426763618c1977ff6`、arithmetic_65 `717461edc87d0532e25e1cde9cc09e5c4500e18af0a9b37579df620da33b112d`、structured_257 `b6ef629bb84b8f222ced15b3896aa5da1ee20bfd34a930cebfeb8127d7465889`、en_prose_129_repeat `96e36d3e48985548aa52e97db2d1c9cddb0447faf86a8641b09f707eaf20b594`
- `stage4-ctl-mx6-weight-kvold`: en_prose_129 `d5a442af23af8617c44e172555a073bebc0164ce4ac919cd2da59590a65a26a7`、ja_prose_129 `3f7a478a0ba8b907ddfee1d4895ba0733f4079e71430b385ff7e43215ecdcfc8`、zh_tech_257 `3b5064775d449307d32b59bcbe84793830c4f6b217c82f599e8dbb89421fae34`、rust_code_257 `0121ead3a2bf2bbc8b9e57c266e0659c4a9a4538fad3d9cd2d3209b91be572cc`、python_code_513 `4c51801d19c0ef42df8664479e0df97e8065e8fcf67f888b00d0e227b1eb3545`、sql_code_1025 `ff3d6a4374d33b907eb87c12cbad6d7acba1433fc0e73de8e800648abacf05b8`、arithmetic_65 `94aba29b8287b89d62ab1523f6c5a72a004de852d445f50bf31623584d68fdbd`、structured_257 `aa535e56cfdace0f8694f9127f33f8667d7db6e20335b4eab66744ae434852e5`、en_prose_129_repeat `d5a442af23af8617c44e172555a073bebc0164ce4ac919cd2da59590a65a26a7`
- `stage4-ctl-mx6-weight-kv-noclip`: en_prose_129 `7d52bf93ca9b995d0813866dd0839a752b4ad159df67db3afbfeab47434f3f2c`、ja_prose_129 `21985f4a99e070b031767cce59399b2e3c5dd4b1fac54a8658150901f37bd748`、zh_tech_257 `87db6d54a9bd5d135c85d3faf738979094d388d21709bb27a296b739c81b2c49`、rust_code_257 `bcb9d9def332795972511977584f428c7a4034c8fc2e25b418634d3235d505ae`、python_code_513 `ff037a6a297ba004911a8b11bea475f5914a8ce1706288f43ce0b7d536d4e22b`、sql_code_1025 `925f59f0430dc2fcaf36449a1b2d1e6289d061f6bad811b55878c1b9a418a4c4`、arithmetic_65 `21ed0cf5dd20d6562afcac706031ca2feb0d1a17c9d51e1d42e1727ed385b41d`、structured_257 `d6f5ea7c900400a64bf71855b28a1bc740047c63b47c00a38945a65ca57a8d3e`、en_prose_129_repeat `7d52bf93ca9b995d0813866dd0839a752b4ad159df67db3afbfeab47434f3f2c`

R9700 serviceは実行前後ともinactive、performance levelは3枚ともautoのままだった。

### 測定後のソース整形と検査（2026-09-19）

GPU測定の後に、`native/hip/src/kv_state_kernel.hip.cpp`、`native/lowp/include/lowp/detail/low_precision_block_codec.hpp`、
`native/lowp/src/lowp_kernel.hip.cpp`、`native/lowp/tests/lowp_quantize_cost_microbench.hip.cpp` へclang-format 18、
Rustへrustfmtを適用した（空白・改行のみ）。また、テストからだけ使われていた参照実装
`quantize_mxfp4_e2m1_even_scale` をclippyの未使用エラー解消のため `sllm-core` から公開した。いずれも演算は変えていない。
段階4のbinaryは整形前のsourceから作ったものである。CI台帳のhashを更新し、`cargo test`（sllm-core／sllm-tools／sllm-cli）、
clippy、MSRV、clang-format、lowp境界検査、CIのPythonテスト1,005件、matrix検査、リンク検査がPASSした。

## 段階5: 既定への採用（2026-09-19）

2026-09-19のユーザー指示は「採用推奨＋NVFP4＋MXFP6重みを既定にする」だった。実装と検証の結果、次のとおりにした。

| 対象 | 既定 | 旧規則の選び方 |
| --- | --- | --- |
| MXFP8 重み（Qwen3.5／3.8変換器、MTP sidecar） | 飽和しない最小E8M0 scale | 変換時に `--legacy-floor-scale` |
| MXFP6 重み（同上） | 飽和しない最小E8M0 scale | 変換時に `--legacy-floor-scale` |
| MXFP8 活性値 | 飽和しない最小E8M0 scale | 診断用 `SLLM_MXFP8_ACTIVATION_LEGACY_FLOOR_SCALE=1` |
| NVFP4 活性値 | **旧規則（最近接）のまま** | 2候補選択は診断用 `SLLM_NVFP4_ACTIVATION_BEST_OF_TWO_SCALE=1` |
| KV、MXFP6活性値、MXFP4 | 旧規則のまま | 新規則は診断opt-inのまま |

- 変換器の既存の明示フラグ（`--mxfp8-no-clipping-scale`／`--mxfp6-no-clipping-scale`）は受け付けたまま、既定と同じ意味になる。
  既定で作ったGGUF／sidecarはsemantic IDとderived lockに `mx-scale=no-clipping`／`scale_mode` を記録する。既存のGGUFはそのまま読める。
- 旧規則の環境変数は本番の切り戻し手段ではなく診断用である。切り戻しはbinary単位で行う。

### NVFP4を既定にしなかった理由（wave8の不具合）

段階5の演算子evidenceで、NVFP4の2候補選択を既定にしたGPU出力がCPU参照と一致しなかった（gfx1030、m=7・k=17・n=15、row 5）。
原因は既定で使われるwave8量子化カーネルの2候補選択の不具合だった。`__shfl_down` によるブロック最大値の縮約後、
正しい値を持つのはlane 0だけだが、各laneが自分の部分最大値から2つのscale候補を計算していたため、
laneごとに異なるscaleで誤差を足し合わせ、ブロックによって誤った候補を選んでいた。v1カーネルとCPU参照は正しかった。
lane 0の最大値を全laneへ配る修正を入れた。あわせて、v1カーネルとCPU参照の誤差の合計順をwave8と同じツリー順にそろえた。

修正後の2候補選択で測り直すと（R9700、chunk32、段階4と同じ入力）:

| 条件 | 平均KLD | top1 |
| --- | ---: | ---: |
| NVFP4 旧規則、FP16 KV（段階4既定とbyte一致） | `0.092218` | `0.8974` |
| NVFP4 2候補選択（修正後）、FP16 KV | `0.105180` | `0.8864` |
| NVFP4 旧規則、KV MXFP8 E4（段階0 D） | `0.103586` | — |
| NVFP4 2候補選択（修正後）、KV MXFP8 E4 | `0.101350` | `0.8997` |

段階4の `stage4-nvfp4-act-best2`（`0.088317`）とその組合せ行は、不具合版で測った値であり、2候補選択の効果を表さない。
正しい2候補選択はFP16 KVで大きく悪化し、KV E4でもわずかな改善にとどまるため、既定にしなかった。
ブロック単位で二乗誤差を最小にするscaleが、活性値ではKLDを下げるとは限らないことを示している。

### 検証

- 両GPUの演算子evidence（最終binary）: NVFP4 W4A4（既定、2候補選択opt-in、2候補選択opt-in＋FORCE_BASELINEのv1）、
  MXFP8/MXFP6 W/A（既定、旧規則）がCPU参照とPASS。不具合版binaryでは同じNVFP4 evidenceがFAILしていた。
- R9700 KLD（修正前の段階5 binary、MXFP8経路は同一）: MXFP8新既定は段階4 `stage4-mx8-act-noclip` と、
  MXFP8旧規則は段階4既定と、NVFP4旧規則は段階4 NVFP4既定と、いずれも9ケースのraw logitsがbyte一致した。
- R9700 KLD（最終binary、MX worker `0cc5bc0f…`、NVFP4 worker `66ee3788…`）: 既定のMXFP8（既存GGUF）は `0.025446` で
  段階4 `stage4-mx8-act-noclip` と、既定のNVFP4は `0.092218` で段階4 NVFP4既定と、9ケースのraw logitsがbyte一致した
  （`stage4-s5c-mx8-default`、`stage4-s5c-nvfp4-default`）。R9700 serviceはinactiveのまま、performance levelはautoのまま。
- host: 変換器CLIの既定・旧規則・競合フラグのテストを追加し、`cargo test`（sllm-core／sllm-tools／sllm-cli）、
  clippy、clang-format、lowp境界検査、CIのPythonテスト1,005件がPASS。

### 既存MX成果物の再変換と旧成果物の削除（2026-09-19）

ユーザー指示により、sLLMが変換したMXFP8／MXFP6の成果物を新しい既定（飽和しないscale）で同じパスへ再変換し、旧規則のものを削除した。
変換器はworking treeのstage 5 source（commit未作成）からbuildし、`--converter-commit` にはHEAD `6116bef9…` を記録した。

| 成果物 | 新しいsemantic ID／encoding |
| --- | --- |
| Qwen3.5-4B MXFP8／MXFP6（`~/.cache/sllm/derived/phase62-qwen35-mxfp{8,6}-bundle`） | `…:mx-scale=no-clipping` |
| Qwen3.5-9B MXFP8（`phase63-qwen35-9b-mxfp8-bundle`） | 同上 |
| Qwen3.5-27B MXFP6（`phase71-qwen35-27b-mxfp6-bundle`） | 同上。source fingerprintは旧版と同じ `a4a0a619…` |
| Qwen3.8-27B MXFP8／MXFP6（`/home/homelab1/datapool/qwen38-kld-20260918/sllm-mxfp{8,6}`） | 同上 |
| Qwen3.8 MTP sidecar MXFP8／MXFP6（`.local-artifacts/phase84/mxfp{8,6}`） | `mxfp{8,6}-…-e8m0:mx-scale=no-clipping` |

- Qwen3.8とQwen3.5-4BのMXFP8／MXFP6の新GGUFは、段階4で測定に使った診断用no-clip GGUFとファイル全体がbyte一致した。
  したがって段階4のKLD（MXFP8 `0.017764`、MXFP6 `0.058688`、R9700・FP16 KV）が新しい既定の値である。
- Qwen3.5-27Bの元ディレクトリには、lock作成後にHugging Faceのdownload metadata（`.cache/huggingface/…`、`.gitattributes`）が
  追加されていたため、lockの23ファイルだけを一時ディレクトリへコピーして変換し、変換後にその一時コピーを削除した。
  変換器は多重linkのfileを拒否するため、hard linkではなく通常のコピーを使った。元ディレクトリは変更していない。
- 削除: 旧規則の成果物（上表の旧版）、旧規則の診断GGUF（`sllm-mxfp8-retain-gdn-ab`）、旧規則の
  `phase62-qwen35-mxfp{8,6}-bundle.invalid-unknown-extension`、新既定とbyte一致して重複となった診断用no-clip版
  （`sllm-mxfp{8,6}-no-clip`、`phase62-qwen35-mxfp{8,6}-noclip-bundle`）。合計約205GB。
  過去の履歴・証拠が参照するこれらのpathは、以後は存在しない。
- Qwen3.5-9B／27BとMTP sidecarの新成果物は、変換のPASSとidentityの確認だけで、GPU実行による確認はしていない。
