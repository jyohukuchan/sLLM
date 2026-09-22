# Phase 87 段階7: 対象を取り違えた試行（破棄）

2026-09-22。段階7として実施された試行が計画の対象と異なっていたため、実装を破棄した。
**段階7本来の対象（活性値量子化を前段producerへ融合）は未着手である。**
この文書は、同じ取り違えを繰り返さないために試行と破棄の理由を残す。

## 計画の対象と、実際に行われたこと

[計画の段階7](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#今後の順序2026-09-22整理)は、
`rmsnorm`／`residual_rmsnorm`／`elementwise`（SiLU×up）／GDN gated normといった**前段のproducer kernel**へ
活性値量子化を畳み込み、producerが`(codes, scale)`をconsumerへ直接渡す契約にすることである。
1 tokenあたり185回の量子化kernelとその起動をnodeごと減らすのが目的で、
[vllm-mxfp4の分析](vllm-mxfp4-optimization-analysis.md)のP0-1と同じ設計を指す。

実際に実装されたのは逆方向で、**consumer側**であるQwen3.8 gate/up projection packのkernel内で
BF16→NVFP4 block16量子化を行い、gate/upの両出力を1 dispatchで生成する候補だった
（exact `K=5120,N=17408,M=1..4`）。これは、

- WU2の候補C2（量子化をID103のGEMVへ融合、0.18944 ms/tokenで1%基準未達）と同種であり、
  [vllm-mxfp4分析](vllm-mxfp4-optimization-analysis.md)でも「再提案しない」と整理済み、
- かつ段階8（gate/upとGDN qkv/zのdual-output bundle）の範囲

である。段階7の対象families（FP8 per-row、NVFP4 block16＋tensor scale、MXFP8 KV前処理）は一つも評価していない。

## 破棄の理由

1. 対象が計画と異なり、段階7の成果にならない。
2. 計測が存在しない。exact `gfx1030`の最初のsmokeで`execution resource is busy`となって終了しており、
   単体ベンチ、独立oracle、families別の上限算出、打切り線の判定のいずれも行われていない。
   この失敗はcompletion/lifecycleの実装不良であり、producer融合の可否を示す証拠ではない。
3. 不採用候補のコードがproduction sourceに残っていた。公開ヘッダへkernel ID
   `SLLM_HIP_QWEN38_PROJECTION_PACK2_KERNEL_ID_NVFP4_FUSED_PRODUCER_V1`を追加し、
   `lowp_kernel.hip.cpp`へkernel本体、`public_runtime.hip.cpp`へstub、
   `qwen38_projection_pack_runtime.inc`へ常に`false`のselector分岐を残す形で、
   WU-C1で整理した方針（不要な切替・実験経路を残さない）と逆行していた。
4. [AGENTS.mdのkernel融合方針](../../../../../AGENTS.md)を満たしていない。
   共有device helperの合成、作業単位冒頭での資源（VGPR／LDS／occupancy）確認、
   ビット一致版を先に作って対照にすること、のいずれも実施されていない。
5. CIのclang-formatとJSON manifest hashが未更新で、h0がFAILする状態だった。

## 破棄の内容と現在の状態

`include/sllm/hip.h`、`native/hip/src/public_runtime.hip.cpp`、
`native/hip/src/qwen38_projection_pack_runtime.inc`、
`native/lowp/include/lowp/detail/lowp_kernel_internal.hpp`、`native/lowp/src/lowp_kernel.hip.cpp`
を`37809dc2`時点へ戻した。破棄したdraftは
`.local-artifacts/phase87/stage7-discarded-draft/`（source copyと`stage7-consumer-fusion-draft.patch`）に保持し、Gitへは追加しない。
既定の実行経路、WU1／WU1.1／WU2／段階5／段階6の採用結果、数値・出力影響変更台帳に変更はない。
revert後にclang-formatとJSON manifest検査がPASSすることを確認した。

## やり直す際の条件

- 対象は前段producerへの融合に限る。consumer側（matmul kernel内）への量子化取り込みは
  WU2で1%基準未達と判定済みであり、段階7の候補にしない。
- AGENTS.mdのkernel融合方針に従う。特に、ビット一致版を先に作って分解版を対照にすること、
  代表1変種の資源を冒頭で確認することを満たす。
- 着手時にfamiliesごとの上限（削減node数×1 nodeあたりのgap＋量子化kernel自体の時間）を算出し、半分を打切り線とする。
- commit前に`validate_cpp.py --mode format`、`cargo fmt`、clippy、CI hash連鎖の更新を通す。

計画: [Phase 87計画](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
