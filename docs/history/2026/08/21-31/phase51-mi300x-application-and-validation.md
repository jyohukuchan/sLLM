# Phase 51 MI300X適用・実機検証履歴

## 2026-08-25: exact gfx942実機検証とtarget分離候補による完了

### 判定

- Phase 51は`completed-target-separated`として完了した。Hot AisleのMI300X VF x1、BDF `0000:ff:00.0`、
  ROCr UUID `GPU-6104e2a75685060a`、exact `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14.0で実行した。
- base HEAD `184ef043b281e9dda839b78bfbb0e565d3047612`のPhase 51 baseline sLLMと固定llama.cppは、通常5行
  3 warmup＋10 measured、長時間2行1 warmup＋3 measuredを全てPASSした。baseline sLLM binary SHA-256は
  `69c2971ea9655b8a29a3656dd0787f5c24dd3bad24536cc801eda71493516a80`である。
  sLLMは全行HIP-only、fallbackなし、反復一致、cleanup 0、HBM/GTT baseline復帰を満たした。llama.cppも全行full GPU
  offload、cleanup failure 0、要求memory resetとprocess終了後のHBM/GTT復帰を満たした。
- llama.cpp性能同等gateは7行全て未達だったが、これは事前に性能目標と残差報告へ限定しており、正しさ・資源PASSを
  FAILへ読み替えない。差は特にGDN長prefillと長decodeに残った。

### MI300X最終7行（sLLM対llama.cpp、E2E中央値）

| input/output | sLLM (ms) | llama.cpp (ms) | 比率 | 正しさ・資源 |
| --- | ---: | ---: | ---: | --- |
| 17/17 | 262.078418 | 118.669010 | 2.20848x | PASS |
| 32/32 | 489.354945 | 213.199725 | 2.29529x | PASS |
| 1,024/128 | 5,284.468480 | 899.280708 | 5.87633x | PASS |
| 32/256 | 3,826.382054 | 1,655.594779 | 2.31118x | PASS |
| 10,001/2 | 22,922.828037 | 764.818424 | 29.97159x | PASS |
| 100,000/2 | 772,743.055927 | 11,605.687579 | 66.58313x | PASS |
| 32/20,000 | 3,288,825.108588 | 133,470.349834 | 24.64087x | PASS |

追跡済みaggregateは`ci/matrix/phase51-mi300x-summary-v1.json`（SHA-256
`d9830f4903c420e085eaf9f0826ab445164497c12706efd1f3809aeebd7bd4b1`）である。raw evidenceは
`/home/homelab1/.local/share/sllm-evidence/phase51/final-current-20260825`へ退避した。sLLM producer要約は
SHA-256 `ba64065bc185c1da7bd3f9a257e64cea27b16db90368d83d3d1ef3fce4edfbd0`、llama.cpp producer要約は
`54f551839c9f506ec985e5038396b3feeef4dc1e524bcfe30ffc87b803813a67`である。固定peerはllama.cpp tag `b10453`、
commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`、wrapper SHA-256
`b1e6890b504366f9e3e7f80b41868356a1b58d4d977c4ccdc169939c29de029d`、GGUF SHA-256
`636158bd8a217374134cc2455aa40603f7579366fda0f0f5efcbf8bcba37c045`である。

### profileとGDN wave64候補

- fresh `10,001/2` profileは1 warmup＋3 measuredをPASSし、device time内訳はGDN `72.724689629%`、
  full attention `26.364577745%`、projection `0.684758506%`、other `0.225974119%`だった。profile要約SHA-256は
  `a9728434f2603ac1149da64b8700205b4721737e6412b7a240105faa68670f8d`である。MTPはこのBF16固定peerに
  unavailable／not-emittedであり、MTP性能の検証主張はしない。
- Amdahl上限に従い、exact gfx942 wave64向けGDN column-state v3を実装した。選択条件はexact suffix、M `>=128`、
  明示opt-in `SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE=1`であり、既定は従来providerのまま、
  `SLLM_GDN_FORCE_BASELINE=1`が常に優先する。gfx1030/gfx1201ではv3 symbolもselectorも非選択である。
- operator 7 shape（1, 3, 17, 32, 127, 128, 129）と同一state上のv3 128-token→forced-baseline 128-token継続はPASSした。
  scalar sequential oracleに対する最大絶対誤差は`0.00390625`、最大相対誤差は`0.014705882`でbaselineと同一、
  fallbackなし、cleanup 0だった。
- model-level `10,001/2` A/Bは出力`[23066,23066]`完全一致、fallbackなし、cleanup 0で、prefill中央値を
  `22.718162442`秒から`6.410255551`秒へ短縮した（3.54403x）。ただし残り時間内に候補入り全7行を再取得できないため、
  v3を既定採用せず`target-separated`の明示opt-in候補として残した。Full Attention以下の候補は新しい残差順へ延期した。

### 3 target追跡と既知制約

- `ci/matrix/three-target-gpu-summary-v1.json`（SHA-256
  `a5c468b83053f9e0d641bed618f9e7bbf4001b4b4cc434ffe15b13d2d61203fb`）へgfx1030/RDNA2、gfx1201/RDNA4、
  gfx942/CDNA3の7行、exact target selector、正しさ、資源、既知制約を固定した。fallback可能なdefault targetは置かない。
- gfx942は7/7で正しさ・資源PASS。gfx1201は既存Phase 50の`100,000/2` OOMを資源FAILとして保持する。
  gfx1030は最終採用identityが通常5行だけであり、長時間2行を同一binaryの結果として合成しない。`100,000/2`は不採用
  long-prefill-v2 binary、`32/20,000`はbinary SHA未記録のdirect evidenceとして行別source/hashを保持し、両行を
  non-comparable、target全体を`per-row-mixed` identityとして明示した。
- MI300A、MI325X、bare metal、複数GPU、FNUZ FP8のllama.cpp比較、他モデルへ結果を一般化しない。

[対応する計画](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
