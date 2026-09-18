# 低精度形式のscale選択の修正（MXFP8／MXFP6／MXFP4／NVFP4）

## 状態

- 計画済み・次（2026-09-18作成）。フェーズ番号は割り当てていない。実装はCodexで行う。
- 2026-09-18のユーザー決定により、**次の作業としてすぐに実施し、Phase 87より先に行う**。
  Phase 87は同じ量子化カーネルを変更するため、先にscale規則を確定させて最適化の基準をそろえる。
- 前提: [MXFP8／vLLM FP8のKLD差の調査](../../../../archive/2026/09/11-20/qwen38-mxfp8-vllm-fp8-attribution.md)の成果
  （診断opt-inの実装を含む）は2026-09-18にcommit済み。そのcommitを段階0の基準sourceとする。

## 背景

Qwen3.8-27BのKLD調査で、sLLM MXFP8が高いKLDを示す主因は、32要素ブロックのE8M0 scaleを
`2^(floor(log2(max)) - emax)`（OCP MX仕様の参照アルゴリズム）で決めているため、ブロック最大値付近が
要素形式の最大有限値で飽和することだと分かった。重みと活性値の両方を飽和しないscaleにすると、
R9700・chunk32・2,632位置の平均KLDは0.0459から0.0178へ下がった（vLLM FP8は0.0145）。
同じ種類の問題がMXFP6、MXFP4、NVFP4、KV cacheにもあるかを調べた結果が以下である。

## 実装の棚卸し（2026-09-18時点）

| 形式 | 役割 | 実装箇所 | 現在の規則 | 飽和 |
| --- | --- | --- | --- | --- |
| MXFP8 E4M3 | 重み（Qwen3.5/3.8変換、MTP sidecar） | `crates/sllm-core/src/mxfp.rs` `quantize_mxfp8_e4m3` | floor（診断用に`--mxfp8-no-clipping-scale`あり） | あり（最大値の仮数が1.75以上のブロック） |
| MXFP8 E4M3 | 活性値（W8A8） | `native/lowp/src/lowp_kernel.hip.cpp` `sllm_matmul_bf16_to_mxfp8_e4m3_block32_v1` | floor（診断用に`SLLM_MXFP8_ACTIVATION_NO_CLIP_SCALE=1`あり） | あり |
| MXFP8 E4M3／E5M2 | **KV cache**（Qwen3.8とreviewed Qwen3.5-4Bの既定はE4） | `native/hip/src/kv_state_kernel.hip.cpp`（`ocp_mxfp8_e8m0_scale`と`BlockCodec::scale_code`の2か所）、Rust参照`crates/sllm-core/src/kv_mxfp8.rs` `standard_mx_scale` | floor | あり |
| MXFP6 E3M2 | 重み | `mxfp.rs` `quantize_mxfp6_e3m2` | floor | あり（上限28） |
| MXFP6 E3M2 | 活性値（W6A6） | `lowp_kernel.hip.cpp`（`BlockCodec<Mxfp6E3Block32>::scale_code`） | floor | あり |
| MXFP4 E2M1 | 重み（sLLM自身の量子化tool） | `crates/sllm-tools/src/artifact.rs` `quantize_mxfp4` | ceil（飽和なし） | なし（ただし下記のとおり誤差は最小でない） |
| MXFP4 E2M1 | 活性値（旧W4A4） | `lowp_kernel.hip.cpp`／codec `mxfp4_even_scale_code` | 最大値の仮数が1.75以上なら指数を繰り上げ | 一部あり（仮数1.5〜1.75で上限6へ飽和） |
| NVFP4 | 重み（sLLM自身の量子化tool、evidence） | `crates/sllm-core/src/nvfp4.rs` `quantize_nvfp4_weights` | block scale `amax/6/global`をE4M3へ最近接丸め | 小（丸めが下向きのとき最大約6%） |
| NVFP4 | 活性値（W4A4） | `lowp_kernel.hip.cpp` `sllm_matmul_bf16_to_nvfp4_block16_v1`／`_wave8_v1` | 同上。globalは配布物の較正済み`input_scale`（固定値） | 小。加えて、活性値が較正値を超えるとblock scaleがE4M3上限448で頭打ちになり大きく飽和しうる（頻度は未測定） |

共通codecは`native/lowp/include/lowp/detail/low_precision_block_codec.hpp`の`ocp_mx_scale_code`／
`mxfp4_even_scale_code`。attention側はMX KVを復号するだけで量子化しない。FP8 outer（行ごとの実数scale）は飽和しない。

外部で量子化済みの重み（Unsloth／NVIDIAのNVFP4、AMDのMXFP4）はscaleが配布物で固定されており、この計画では再量子化しない。
これらのモデルでも、活性値とKVの修正は効く。

## 形式ごとの最適な規則（実重みでの予備測定）

Qwen3.8-27B BF16原本の4行列（layer20 gate／down／GDN qkv、layer43 o_proj、各先頭512行）について、
独立したNumPy実装でBF16比の相対RMS誤差を比べた（scratchの一回測定。本測定は段階0で行う）。

| 形式 | 現行 | 飽和しないscale | 2候補から誤差最小を選ぶ |
| --- | --- | --- | --- |
| MXFP8 E4M3 | 2.95〜3.06% | 2.66% | 2.66%（飽和しないscaleと同じ） |
| MXFP6 E3M2 | 5.40〜5.43% | 5.27〜5.29% | 同左 |
| MXFP4 E2M1 | floor 11.6%、even 11.3%、ceil 11.8% | 11.8%（ceilと同じ） | 11.2% |
| NVFP4 | 最近接 9.47〜9.50% | 切り上げ 10.0% | 8.82〜8.85% |

結論:

- 仮数3ビットのE4M3では、飽和を避けるだけで最適になる。正規数の範囲では相対誤差が一定なので、
  scaleを1段上げても損はほぼない。
- E3M2も飽和を避ければほぼ最適だが、改善幅は小さい（約3%）。
- E2M1（MXFP4・NVFP4）は表現できる範囲が狭いため、飽和を避けると小さい値の分解能が落ち、かえって悪化する。
  **ブロックごとに隣接する2つのscale候補を試し、二乗誤差の小さい方を選ぶ**のが最良だった。
  MXFP4の現行「even」丸めは最良に近く、sllm-toolsのceilはむしろ悪い。

## 修正方針

| 対象 | 新しい規則 |
| --- | --- |
| MXFP8 E4M3（重み・活性値・KV） | 飽和しない最小のE8M0 scale（`ceil(log2(max/448))`相当。既存の診断実装を正式化） |
| MXFP8 E5M2（KV比較形式） | 飽和しない最小scale（上限57344）。段階0の測定で悪化しないことを確認する |
| MXFP6 E3M2（重み・活性値） | 飽和しない最小scale（上限28） |
| MXFP4 E2M1（重み・活性値） | floorとfloor+1の2候補から、ブロック二乗誤差が小さい方 |
| NVFP4（重み・活性値） | `amax/6/global`を挟む隣接2つのE4M3 codeから、ブロック二乗誤差が小さい方。global scaleの扱いは変えない |

- 同点時は小さいscale（従来に近い方）を選び、決定的にする。
- NaN／Inf／ゼロ／非正規数の既存の扱いは変えない。E8M0の0xff（NaN印）を有限scaleに使わない。
- NVFP4活性値の較正値超過は、まず発生頻度を測る（段階0）。動的global scaleへの変更は、
  配布物の較正と異なる形式になるため、この計画では行わず、頻度が高い場合に別途提案する。

## 識別子と互換性

- **重み**: 新規則で作ったGGUF／sidecarはrecipeのsemantic IDに規則を記録する（KLD調査で入れた
  `mx-scale=no-clipping`と同じ方式を、`mx-scale=`／`nvfp4-scale=`として形式ごとに一般化する）。
  既存のGGUF・sidecar・lockはそのまま読み込め、数値も変わらない。既定の変換規則を新規則へ切り替える。
- **活性値・KV**: 実行時の量子化なので、既存モデルの出力も変わる。数値・出力影響変更台帳へ
  形式ごとにN2として記録する（誤差の期待値は下がるが、要素ごとの非増加は保証されないため）。
  既定への採用は、段階4の測定結果をもとにユーザーが形式ごとに判断する。
- 比較用に旧規則を選べる明示的な診断用環境変数を残す（本番の切り戻し手段ではない。切り戻しはbinary単位）。

## 作業段階

### 段階0: 基準測定

- KLD調査のコーパス（8入力・2,632位置、全語彙、`KL(llama.cpp BF16 || 候補)`）とR9700・chunk32で、
  次の現行値を固定する。既存の測定がある条件はraw logitsのhash一致を確かめて再利用する。
  - Qwen3.8 MXFP8（重みsLLM変換）、MXFP6（同）、KV FP16
  - Qwen3.8 NVFP4（Unsloth）、KV FP16と既定のMXFP8 E4の両方（**本番の既定構成**）
  - Qwen3.8 MXFP8、KV MXFP8 E4
- NVFP4 W4A4活性値で`amax/(6·global) > 448`となるブロックの割合を、上記コーパスで層ごとに集計する（診断用counter）。
- 予備測定の重み誤差表を、変換対象の全行列について独立実装で取り直す。

### 段階1: 共通の規則実装とhost oracle

- Rust: `mxfp.rs`の`MxScalePolicy`に`NoClipping`（既存）と`BestOfTwo`をそろえ、MXFP6・MXFP4にも適用する。
  `nvfp4.rs`に隣接2候補の選択を追加する。`kv_mxfp8.rs`の参照実装も同じ規則にする。
- HIP: codecに`ocp_mx_scale_code_no_clip`と、2候補の誤差比較の補助を追加する。
  既存の診断用`mxfp8_e4m3_activation_scale_code<true>`はこれへ置き換える。
- host oracle: 各形式で、最大値がちょうど上限、上限の直上、仮数1.5／1.75付近、ゼロ・NaN・Inf・非正規数のブロック、
  端数長（K非整列）を含むテストを追加する。CPU参照とGPU量子化結果のbyte一致をGPU oracleで確認する。

### 段階2: 活性値とKVの量子化カーネル

- MXFP8／MXFP6／MXFP4／NVFP4（v1とwave8の両方）の活性値量子化、MXFP8 E4／E5のKV追記（2か所）を新規則へ移す。
  最初は新規則を診断用環境変数でopt-inにし、既定は旧規則のまま実装・検証する。
- 活性値量子化の追加コストを、8192/128の代表条件（1 warmup＋3 measured）で記録する。

### 段階3: 重みの変換器

- Qwen3.5／3.8のMX変換器、MTP sidecar、sllm-toolsの量子化（MXFP4／NVFP4）に新規則を入れ、semantic IDへ記録する。
- 測定用に、Qwen3.8 MXFP8／MXFP6の新規則GGUFを生成する（約30GBずつ。`/home/homelab1/datapool`に置く）。

### 段階4: 効果の測定

- 段階0と同じ条件で、形式ごとに「活性値だけ」「重みだけ」「両方」「KVも」の組を測る。
- 追加で、Qwen3.5-4B MXFP8／MXFP6の既存品質fixture（10ケース）とQwen3.8 MTPの期待受理率（`mtp-bench-v1`のM1）への影響を記録する。
- 両GPU（gfx1030／gfx1201）で演算子oracleと代表の1条件を確認する。

### 段階5: 既定への採用（ユーザー判断）

- 測定結果、N2の記録、性能への影響をまとめ、形式ごとに既定切り替えの可否をユーザーへ提示する。
- 採用した形式は既定を切り替え、旧規則は診断用として残す。main-plan、数値変更台帳、`docs/architecture/runtime.md`を更新する。

## 受入条件

1. 各形式の新規則が、CPU参照とGPUでbyte一致する（両GPU、境界ケースを含む）。
2. 旧規則を選んだ場合、既存モデルのraw logitsが変更前とbyte一致する。
3. 既存のGGUF・sidecar・lockがそのまま読み込め、recipe identityが新旧で区別される。
4. 段階0・4の測定が同一条件（入力、GPU、chunk、KV、binary）でそろい、raw logitsのhashとともに記録されている。
5. 性能を記録し、活性値量子化のコスト増を明示する。新しい速度下限は設けない。
6. 既定の切り替えは、ユーザーが形式ごとに承認したものだけとする。

## 対象外

- 外部で量子化済みの重みの再量子化。
- NVFP4の動的global scale化（頻度測定のみ）。
- vLLM FP8との残差（KV型、実行経路）の追加調査。
- scaleを実数にするなど、形式そのものの変更。

## 検証コスト

- GPU: 段階0・4でR9700のKLD取得（1条件あたり約5〜6分）を十数条件、両GPUの演算子oracle、代表性能1条件。
  V620を使う間はローカルQwenサービスを止め、R9700はservice lease手順に従う。
- ディスク: 新規則GGUF 2本で約60GB。
