# 低精度カーネルのライブラリ境界化（native/lowp）

## 状態

- 計画済み・未着手（2026-09-17作成）。フェーズ番号は割り当てていない。
- 2026-09-17のユーザー決定:
  - **Phase 87より先に実施する**。Phase 87（他精度残差最適化）の変更は境界化後のライブラリへ入れる。
  - 下記「検証」の両GPU再検証を承認済みとする。
  - 依存境界の自動検査（host CI）は受入条件に加える。単体ビルドのHIP compile-only CI行は追加しない。
  - MXFP4はW4A8（活性化MXFP8）を前提にする（下記「MXFP4の扱い」）。

## 目的

RDNA向けに最適化したMXFP8／MXFP6／NVFP4／MXFP4の行列積カーネルを、sLLMのランタイムから独立した
ライブラリとしてリポジトリ内に切り出す。将来の別リポジトリ化（`git subtree`／`git filter-repo`）を
機械的な作業にすることが狙いであり、今回はリポジトリを分けない。

- sLLMはこのライブラリの利用者の1つになる。カーネルの採否の根拠となる実モデル測定は引き続きsLLMで取る。
- 演算の意味・数値・選択結果は変えない。純粋な構造変更（数値台帳上N0）として扱う。

## 現状（2026-09-17時点の調査）

- 低精度カーネルは`native/hip/src/matmul_kernel.hip.cpp`（8,510行）にBF16・FP8・hipBLAS経路と混在し、
  `gfx`分岐は269箇所ある。本体に`#include`される`nvfp4_decode_scale_lut.inc`、`nvfp4_prefill_wmma_compensated.inc`、
  `nvfp4_small_m_vgpr_reuse.inc`、`fp8_prefill_short_m32.inc`も同じTUに入る。
- 再利用を意図した部品は既にある。`low_precision_block_codec.hpp`（scalar／block codec、typed view）と
  `low_precision_matmul_provider.hpp`（`sllm_lowp`名前空間の形式契約、`prepare_provider_plan`）は
  sLLMのヘッダに依存しない。
- 形式ごとのvariant選択（`select_mxfp8_variant`等）と`KernelVariant`は`matmul_kernel_internal.hpp`（4,003行）にあり、
  BF16／FP8の選択と同居している。`launch_mxfp8_w8a8`等の起動関数は`sllm_matmul_kernel`名前空間にある。
- 呼び出し元は`matmul_runtime.inc`（18箇所）、`qwen38_projection_pack_runtime.inc`（5箇所）、
  `moe_expert_kernel.hip.cpp`（2箇所）。codecは`kv_state_kernel.hip.cpp`と`causal_attention_kernel.hip.cpp`からも使われる。
- ビルドは`crates/sllm-hip-sys/build.rs`が`native/hip`をCMakeで構成し、`matmul_kernel.hip.cpp`はtargetごとに
  HIPコンパイラで直接コンパイルされる。build.rsはG2 source inventory（`ci/matrix/rmsnorm-g2-build-inputs-v1.json`）の
  順序digestを検査する。
- パスまたはhashでこれらのファイルを列挙するCI資産: `ci/matrix/hip-runtime-compile-v1.json`、
  `ci/matrix/rmsnorm-g2-build-inputs-v1.json`、`ci/matrix/rmsnorm-p0-public-path-inputs-v1.json`、
  `ci/matrix/path-to-suite-v1.json`、`ci/schema/hip-runtime-artifact-v1.schema.json`、
  `ci/tools/run_h3_public_runtime_compile.py`、`ci/tools/validate_rmsnorm_p0_contracts.py`。
- `matmul_kernel.hip.cpp`の一部はllama.cpp由来で、`THIRD_PARTY_NOTICES.md`がこのパスを参照している。
- 計画作成時点の作業ツリーには、上記ファイルを含む未コミット変更（native／crates計20ファイル、約5,100行追加）があった。
  2026-09-17にcommitした（push未実施）。

## 範囲

### 対象

- 行列積: MXFP8 E4M3 W8A8／W8A16、MXFP6 E3M2 W6A6／W6A16、NVFP4 W4A16／W4A4、MXFP4 W4A4（内部のみ。下記「MXFP4の扱い」）。
- 活性化量子化器（BF16→各形式）。
- `sllm_lowp`の形式契約・provider plan、上記形式のvariant選択とshape条件、起動関数。
- codec・typed view（`low_precision_block_codec.hpp`）。
- FP8 outer（per-tensor FP8 W8A8のgfx1030 software経路）。MXFPとingress補助関数を共有しており、
  `sllm_lowp::MatmulFormat`にも既に含まれるため同時に移す。
- 対象target: exact `gfx1030`、`gfx1201`。`gfx942`の経路は同じソースに残るが、今回は実機検証しない。

### 対象外

- BF16行列積、hipBLAS／hipBLASLt経路、FP8 native経路。
- MXFP8 KV append／attentionのカーネル本体（codecの利用者としてライブラリへ依存させるだけにする）。
- Qwen3.8のprojection pack、MoE expert、graph span、公開ランタイムのdescriptor、evidence ABI。
- 別リポジトリ化、パッケージ配布・版管理方針、kernel symbolの改名。
- GGUF／ModelOpt／compressed-tensors配置からの変換器（後続候補。下記参照）。
- 数値・性能を変える最適化。

## 設計

### ディレクトリ構成

```text
native/lowp/
  CMakeLists.txt          # 単体でconfigureできる。sLLMからはadd_subdirectoryで使う
  README.md               # 対応形式、target、API、配置、来歴
  include/lowp/lowp.h     # 公開C API
  src/                    # カーネル、起動関数、選択、plan（HIP TU）
  src/internal/           # codec、形式契約、共有補助
  tests/                  # host契約テスト、GPU oracle、単体example／microbench
```

- 名前空間は既存の`sllm_lowp`を維持する（改名は別リポジトリ化の時点で判断する）。
- 依存してよいもの: HIP runtime、`hip_fp8.h`、rocWMMA、C++標準ライブラリ。
  `include/sllm/*`、`native/hip/src/*`、evidence ABIには依存しない。
- 依存の向きは`native/hip` → `native/lowp`の一方向とする。BF16側とlowp側が共有する小さな補助
  （`float_to_bf16_rne_bits`、`dim3`、workgroup定数など）は`native/lowp/src/internal/`へ置き、sLLM側がそれを使う。

### MXFP4の扱い

2026-09-03のユーザー決定（MXFP4はW4A8だけを対応方針とする）を、2026-09-17にライブラリの前提として再確認した。

- 公開APIのMXFP4は**W4A8だけ**とする。重みはOCP MXFP4（E2M1、block 32、E8M0）、活性化はOCP MXFP8（E4M3、K方向block 32、E8M0）。
  この組合せは新しいversioned contractとして定義する。
- W4A8カーネルの実装はPhase 87の範囲である。境界化の時点では、公開APIに形式だけを定義し、planは未対応として拒否する。
- 既存のMXFP4 W4A4（reviewed Qwen3.5-35B-A3B MXFP4のrouted expertと、同モデルの`ocp-mxfp4-w4a4-mixed`経路で使用中）は、
  純粋な移動としてライブラリ内部へ移すが、公開C APIには出さない。sLLMは内部API経由で従来どおり使う。
  W4A4をW4A8として報告・自動選択しない。
- 既存W4A4経路をW4A8へ置き換えるか、廃止するかは、Phase 87でW4A8が実装された後にユーザーが判断する。

### 公開C API（`lowp.h`）

最小限の3系統にする。

1. **版と能力**: `lowp_version()`、`lowp_target_from_name()`、対応形式・targetの問い合わせ。
2. **plan**: `lowp_matmul_plan(const lowp_matmul_request *, lowp_matmul_plan *)`。
   要求は形式、重み・活性化の配置、exact target、`m`/`n`/`k`。結果はprovider、variant、tile、workspace footprint、
   不採用理由。現行`prepare_provider_plan`と`select_*_variant`をここへ集約する。
3. **launch**: `lowp_matmul_launch(plan, buffers, stream)`と`lowp_quantize_activation(...)`。
   device bufferとHIP streamは呼び出し側が所有する。ライブラリはメモリを確保しない（workspaceは呼び出し側がfootprintに従って渡す）。

配置はsLLMの現行内部配置（行優先、block scaleプレーン分離）をライブラリの正規配置として文書化する。

- sLLMは本番でもこのC APIを通してplanとlaunchを行う。本番経路でAPIを使うことで、外部利用に耐えるAPIであることを継続的に確認する。
- 監査用の数値ID（`KernelVariant`、`ProviderKind`、`TilePolicy`、`InnerProduct`の値）は変えない。
  ライブラリのvariant値は現行`KernelVariant`の値をそのまま採用し、既存evidenceとの対応を保つ。

### ビルド

- `native/lowp/CMakeLists.txt`は、sLLMと同じHIPフラグ（`-mcode-object-version=6`、wavefront、`-ffp-contract=off`）で
  targetごとのHIP objectを作る。単体configure時はフラグを自前で設定し、sLLMから使う時は親の設定を受け取る。
- `native/hip/CMakeLists.txt`は`add_subdirectory(../lowp ...)`でリンクする。
- `crates/sllm-hip-sys/build.rs`の`rerun-if-changed`とG2 source inventoryへ`native/lowp`を追加する。

## 作業段階

### 段階0: 前提の確定

- 作業ツリーにある未コミットの変更（Phase 85追加実験〜Phase 86、FORCE_BASELINE修正）を先にcommitし、
  移動をそれらと混ぜない（2026-09-17に実施済み）。
- 移動前の基準を記録する（両GPU）。
  - HIP kernel objectのsymbol一覧と、低精度kernelごとの命令列digest（可能な範囲）。
  - 選択表: 形式×target×`m`/`n`/`k`の境界表（各閾値の`B-1/B/B+1`と非整列値を含む）に対する
    `prepare_provider_plan`／`select_*_variant`の出力。host上で生成してfixture化する。
  - 下記「検証」の実モデル行の基準出力。

### 段階1: ヘッダの移動

- `low_precision_block_codec.hpp`、`low_precision_matmul_provider.hpp`を`native/lowp/src/internal/`へ移し、
  include pathを変更する（`kv_state_kernel`、`causal_attention_kernel`、テスト、phase78 probeを含む）。コード変更なし。
- CMake、build.rs、CI資産のパスを更新し、host CIとHIP compile-onlyを通す。

### 段階2: カーネル本体と選択の分離

- 対象形式のカーネル、`.inc`、起動関数、variant選択を`native/lowp/src/`へ移す。関数本体は変えない。
- `matmul_kernel.hip.cpp`／`matmul_kernel_internal.hpp`には、BF16・FP8 native・hipBLASとlowpへの委譲だけを残す。
- `qwen38_projection_pack_runtime.inc`、`moe_expert_kernel.hip.cpp`の呼び出しをlowp経由に替える。
- 段階0の選択表fixtureと一致することをhostテストで確認する。

### 段階3: 公開C APIと本番接続

- `lowp.h`を実装し、`matmul_runtime.inc`の低精度経路をC API経由に切り替える。
- C APIのhost契約テスト（不正引数、未対応target、`k`非整列、空次元、footprint）を追加する。

### 段階4: 単体ビルドとテスト

- `cmake -S native/lowp`でsLLMなしにconfigure・buildできるようにする（`gfx1030`／`gfx1201`）。
- codec GPU oracle（既存`low_precision_block_codec_gpu_test`）をlowpへ移し、行列積の独立FP32 oracleテストとmicrobenchを追加する。
- 単体exampleを1つ用意する（BF16活性化を量子化し、MXFP8重みと掛け、FP32 oracleと照合する）。

### 段階5: 検証と文書

- 下記の検証を実施する。
- `native/lowp/README.md`、`docs/architecture/runtime.md`（境界の追記）、`THIRD_PARTY_NOTICES.md`（llama.cpp由来部分の新パス）、
  main-planを更新し、履歴を`docs/history/2026/09/11-20/`へ置く。

## 受入条件

1. `native/lowp`がsLLMのヘッダなしで単体configure・buildできる（exact `gfx1030`、`gfx1201`）。
2. 対象形式の行列積カーネル・起動関数・variant選択が`native/hip/src`に残っていない。sLLMはC API経由で使う。
3. 選択結果が不変である: 段階0の境界表fixtureとprovider・variant・tile・footprintがすべて一致する。
4. 監査IDの数値が不変である。
5. 数値が不変である（両GPU、HIP-only、fallback 0、cleanup 0）:
   - 既存の低精度演算子oracleとcodec oracleがPASSする。
   - 内部へ移すMXFP4 W4A4も、既存の演算子evidence（`sllm-mxfp4-w4a4-evidence`）で移動前と一致する。
   - 実モデルの固定入力で、移動前後のlogits hashと生成token列が一致する。対象はQwen3.8-27B NVFP4（代表8,192/128）と、
     Qwen3.5-4BのMXFP8 W8A8／MXFP6 W6A6（KV MXFP8 E4）の短い行。
6. 性能は記録する。代表8,192/128の1 warmup＋3 measuredで、移動前との差が測定のばらつきを超える場合は原因を調べて記録する。
   新しい速度下限は設けない。
7. host CI、HIP compile-only CIが通る。CI資産のパスとhashが更新されている。
8. 来歴記録が新パスを指している。
9. `native/lowp`内の`#include`に`sllm/`と`native/hip/src`が現れないことを、host CIの検査で確認する。
10. 公開C APIがMXFP4をW4A8としてだけ定義し、W4A4を公開しない。

## 検証

- host: lowp契約テスト、選択表fixture照合、既存`public_runtime_host_test`、markdownリンク検査。
- HIP compile-only: 両target。
- GPU（両GPU、直列、R9700はservice lease手順に従う）: 低精度演算子oracle、codec oracle、
  上記の実モデル一致確認、代表条件の性能記録。
- 移動によりsourceの意味上のidentityが変わるため、既存のGPU証拠はhashでは再利用できない。上記GPU検証は計画上必要な再実行である。

## 採用しなかった提案

- **単体ビルドのCI行**（`cmake -S native/lowp`のHIP compile-onlyを既存H3 workflowへ追加）: 2026-09-17のユーザー判断で追加しない。
  単体ビルドは段階4で手動確認する。

## 後続候補（今回は実施しない）

- 標準配置からの変換（GGUF MXFP4 block、ModelOpt／compressed-tensors NVFP4、OCP MXの要素・scale配置）。
- MXFP8 KV append／attentionカーネルのライブラリへの移動。
- `gfx942`の実機検証。
- symbol・名前空間の改名、版管理、別リポジトリ化。
- llama.cpp HIPバックエンドへの提案（ggmlのblock形式に合わせた再実装が必要）。

## リスクと停止条件

- **TU分割によるコード生成の変化**: kernelは同じ本体でも、TUを分けるとhost側のinline化などが変わりうる。
  段階2で命令列digestまたはsymbol一覧に差が出た場合は、数値一致と性能記録で判断し、差の原因を記録する。
- **Phase 87との衝突**: 境界化の途中でPhase 87を並行しない。
- **CI資産の検査が厳しい**: G2 inventoryの順序digestなど、パス変更で壊れる検査が多い。段階1で先に通してから本体を移す。
- AGENTS.mdの停止条件（同じ単位の2回reject、1時間超の停滞、検証・文書が30%超、見積り1.5倍超、受入条件の変更）に従い、
  該当したら作業を止めて再計画する。
