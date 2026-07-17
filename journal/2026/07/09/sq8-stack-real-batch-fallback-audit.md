# SQ8_0 stack real-batch fallback audit

## 前回の要点

- SQ8_0の計画後半には、vLLM + FP8との比較をM10として含めている。
- layer3のfull projection component smokeでは、direct SQ8_0 batch matvecが `14/14` まで確認できている。
- M10の最終比較には、component rowではなくfull-package real-batchまたはserver-style uLLM rowが必要。

## 今回の変更点

- 既存のAQ4 self-attn stack real-batch runnerを、layer3 full-projection SQ8_0 sidecar artifactで診断した。
- 実行結果は `batching_mode=real`、`prefill_real_batch=true`、`decode_real_batch=true`、request parallelism `4` だった。
- ただしSQ8_0側は `sq_execution_mode=materialized_f32_fallback`、`sq_projection_boundary=none`、direct SQ8_0 matvec counterはすべて `0` だった。
- 原因は、`Qwen3PackageModelRuntime::load_with_sq_overlay` から入るstack/model-loop loaderが `materialize_package_projection_matrix` でSQ8_0 projectionをF32 runtime weightへ展開するため。
- 計画文書に、この経路はscheduler-connectivity診断であってM10 serving比較用rowではないことを追記した。

## 次の行動

- stack/model-loop layer runtimeを、materialized F32 weightではなくresident SQ8_0 projection抽象に接続する。
- 候補は既存の `PackageAq4ResidentMatvec` の再利用、またはstack real-batch用のresident layer abstraction追加。
- q/k/v/o/gate/up/downがfull-package real-batch pathで `matvec_batch` を呼べるようになってから、vLLM + FP8 serving比較へ進める。
