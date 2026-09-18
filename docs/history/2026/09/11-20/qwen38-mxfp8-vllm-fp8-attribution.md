# Qwen3.8-27B MXFP8 / vLLM FP8 KLD差の原因調査（2026-09-18）

## 対象と結論

先行する[全語彙KLD比較](qwen38-cross-engine-kld.md)では、sLLM MXFP8の平均KLDがvLLM公式FP8より高かった。この調査では、同じR9700 `gfx1201`でsLLMを再実行し、重み・活性値それぞれのE8M0 scale選択を独立に変えた。共通の8入力・2,632位置、248,077有効語彙、温度1、`KL(llama.cpp BF16 || 候補)`、raw FP32 logitsからFP64で正規化・集計する条件を固定した。

主因は、sLLMのMXFP8量子化が32要素ブロックのscaleを `2^(floor(log2(max_abs))-8)` とし、E4M3FNの有限最大値448を超える値を飽和させることだった。重みと活性値の両方で、**同じE4M3/E8M0・block32・W8A8のまま**有限値が飽和しない最小scaleを選ぶと、sLLMの平均KLDは `0.04590724 → 0.01776407`。vLLM公式FP8の `0.01451873` との差は `0.03138850 → 0.00324534` へ約89.66%縮んだ。既定方式は変更せず、診断用の明示opt-inだけを追加した。

## 固定identityと実行条件

- BF16原本: `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、[Qwen3.8 lock](../../../../models/locks/qwen3.8-27b-bf16.json)。公式FP8は別の `Qwen/Qwen3.8-27B-FP8` checkpointを固定file hashで検査した。
- 参照: llama.cpp BF16重み、V620×2、FP16 KV。sLLMの4つのfactorial条件は全て同じR9700 `gfx1201`、同じFP16 KV、chunk32、同じsource binaryと固定token-ID入力。vLLM FP8も同じR9700だがBF16 KV、vLLM `0.21.0+rocm722`、CPU weight offload指定8 GiB、chunk32である。
- 計測値には参照GPU、KV dtype、engine/kernelの差が残る。以下のsLLM内4条件では、それらを固定してscale選択だけを変えた。

| 候補 | 重みscale | 活性値scale | 平均KLD | p95 | top-1一致率 |
| --- | --- | --- | ---: | ---: | ---: |
| vLLM公式FP8 | 128×128 BF16 inverse-scale | per-token K128実数scale | 0.01451873 | 0.05046123 | 95.2508% |
| sLLM MXFP8既定 | E8M0切り捨て | E8M0切り捨て | 0.04590724 | 0.17935156 | 91.4514% |
| sLLM・重みだけ飽和回避 | E8M0飽和回避 | E8M0切り捨て | 0.03746018 | 0.14287136 | 92.0973% |
| sLLM・活性値だけ飽和回避 | E8M0切り捨て | E8M0飽和回避 | 0.02544622 | 0.09444631 | 94.2249% |
| sLLM・両方飽和回避 | E8M0飽和回避 | E8M0飽和回避 | 0.01776407 | 0.05703398 | 95.1368% |

重みだけの変更で平均KLDが18.40%、活性値だけでは44.57%、両方では61.30%下がった。この百分率は各条件と既定との**観測差**であり、KLDを加法的な誤差寄与に分解した値ではない。両方変更後も最大位置KLDは `2.53231`（vLLM `1.88859`）。算術入力では両方変更後 `0.12307`、vLLM `0.03004` で、全入力が同等になったわけではない。各入力の平均と全位置の集計は、集約JSON（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/attribution-summary.json`）に保持する。

## 形式、範囲、BF16比重み誤差

実際のtensor catalogで、両者が共通して量子化するテキスト2次元行列は400個、24,326,963,200要素だった。公式FP8は128×128ごとにBF16 inverse-scaleを置き、sLLM MXFP8は各行のK方向32要素ごとにE8M0の2の累乗scaleを置く。sLLMのみ、GDN `linear_attn.in_proj_a/b` の96行列（48層×2、23,592,960要素）もMXFP8化している。`lm_head.weight` とembeddingは両者ともBF16である。

独立した[NumPy重み比較器](../../../../../ci/tools/qwen38_weight_error.py)は、BF16/F8 safetensorsとMX GGUFのpayloadを直接読み、E4M3FN・E8M0を独立に復元した。共通400行列全要素のBF16相対RMS誤差はMXFP8 `3.00997%`、公式FP8 `2.65053%`。MXFP8では全要素の `0.80618%` が最大有限値に飽和し、その位置がMXFP8の二乗誤差の `23.8807%` を占めた。公式FP8での飽和率は `0.002652%`。同じ12行列・同じ1,572,864サンプル要素を使う反実仮想では、scaleを飽和回避へ変えるだけで相対RMSが `2.97393% → 2.66302%`。実際に変換した診断GGUFの代表16行列では `2.97244% → 2.66037%`、飽和要素は0だった。全496行列レポート（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/weight-error-qwen38-full-enriched.json`）、同一サンプルとhead identity（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/weight-error-head-paired-summary.json`）、診断GGUF照合（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/weight-error-no-clip-representative16.json`）を分けて保存した。

量子化対象外の公式FP8テキスト重み449個はBF16原本とpayload byte一致し、別に確認した巨大な`lm_head`とembeddingもBF16原本・公式FP8・sLLM元GGUFの3者でSHA-256一致した。非量子化テキストの全照合（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/official-fp8-unquantized-text-all-identity.json`）を保持する。元モデルや出力ヘッドの取り違えで上表を説明することはできない。

## 活性値と実行経路の対照

[HIP活性値quantizer](../../../../../native/lowp/src/lowp_kernel.hip.cpp)も重み変換と同じ切り捨てE8M0 scaleを使用する。`SLLM_MXFP8_ACTIVATION_NO_CLIP_SCALE=1` はGPUで有限値が448を超えるblockだけscaleを一段階上げる。デフォルトのカーネルはそのまま残した。model-freeのR9700数値oracleではM=1/K96/N9とM=32/K2048/N4096の2ケースが成功し、両条件でmatmul kernel IDは既定と同じ18/22、出力は変更され、2反復はそれぞれbyte一致、fallbackなし。演算レポート（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/mxfp8-activation-no-clip-operator-gfx1201.json`）。

別のM=1対照では、同じMXFP8重み・同じR9700・FP16 KVで活性値量子化を行うW8A8の平均KLDが `0.04530611`、活性値をBF16のまま使うW8A16が `0.01714540`。top-1一致率は `92.0213% → 95.4407%`。ただしM=1対chunk32の同一重み内の直接KLDも `0.01670664` あるので、M=1の改善幅をchunk32へそのまま移してはいけない。chunk32の同一経路でscaleだけを変えた主表が、この問題の直接的な証拠である。

同じMXFP8 GGUFのchunk32をV620 `gfx1030`とR9700 `gfx1201`で走らせると、BF16基準の平均KLDは `0.04444137` と `0.04590724`。target間の直接KLDは `0.01379207` で出力差はあるが、vLLM FP8との差がR9700でも残ることを確認した。vLLMでBF16原本をR9700で動かした英語65位置の対照では `KL(llama BF16 || vLLM BF16)=0.00822808`、同じvLLM BF16を参照に公式FP8が `0.11939078`、sLLM元MXFP8が `0.17734913`。これは**65位置だけ**のengine/KV対照であり、全2,632位置のBF16 engine差ではない。

GDN A/BだけをBF16へ戻した別GGUFでは、全96 retained tensorがBF16原本とpayload byte一致した。全位置の平均KLDは `0.04590724 → 0.05234552` と悪化した。retained時はBF16 HIPBLAS、元のMX時はW8A8 row8へ変わるため純粋な重み誤差のA/Bではないが、追加96行列のMX量子化を除くことは元の高KLDの解消策にならなかった。sLLMはHIPのGDN再帰・attention、vLLMはROCm/Triton・FLA等を通る。残差にこれらの経路、KV dtype、scaleの粒度が含まれる可能性は残る。

## 実装、検証、範囲

- BF16原本から生成するconverterへ`--mxfp8-no-clipping-scale`と`--retain-gdn-in-proj-a-b`を、厳密なrecipe/derived-lock identity付きの診断opt-inとして追加した。[Rust量子化](../../../../../crates/sllm-core/src/mxfp.rs)、[GGUF converter](../../../../../crates/sllm-core/src/gguf_convert.rs)、[graph検証](../../../../../crates/sllm-core/src/qwen_graph.rs)。旧Qwen3.8 GGUFの`qwen35:<fingerprint>`名前空間は既定coverage専用として読み込みを維持する。
- 元のGGUFは、Rust graph変更前後の全9ケースのraw logits SHA-256が一致した。native quantizer変更後も、環境変数を設定しない65位置のraw logitsは旧バイナリとbyte一致した。デフォルト数値経路は変更していない。旧・新対照（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/ablation-control-identity.json`）。
- 4条件のうち既定MXFP8だけが診断kernel追加前のバイナリで全位置を取得していたため、同じ新バイナリで既定MXFP8の全9ケースを取り直した。raw logitsは既存の2回の全位置取得と9ケースすべてでbyte一致し、主表の既定値はそのまま比較条件をそろえた値として扱える（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/sllm-mxfp8-default-newnative-fp16-gfx1201-chunk32/`）。
- `cargo test -p sllm-core --lib`: 630成功、24件は既存のGPU artifact等を必要としてignored。converter CLIの2 focused tests、gfx1201 release build/GPU numerical oracleとgfx1030 release compile-only、Python構文検査が成功した。gfx1030に新診断kernelのGPU正しさを主張しない。
- 7つのsLLM full-corpus captureを現在のraw fileに対して再SHA-256検査した。全19,327行／19,197,122,560 bytes、入力ID・shape・非有限0・HIP dispatch・fallbackなし・英語repeat byte一致を確認した。capture audit（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/attribution-capture-audit.json`）。vLLM FP8の旧captureは[先行調査](qwen38-cross-engine-kld.md)のaudit対象である。
- KLDの改善はこの固定Qwen3.8モデルと8入力、R9700、FP16 KVに限定する。別モデル・別GPU・長文・性能・タスク品質を認定せず、診断opt-inを本番既定へ昇格していない。

[対応する計画](../../../../plans/archive/2026/09/11-20/qwen38-mxfp8-vllm-fp8-attribution.md) ／ [再現手順](../../../../references/qwen38-kld-reproduction.md) ／ 機械可読集計（Git管理外: `/home/homelab1/datapool/qwen38-kld-20260918/results/attribution-summary.json`）
