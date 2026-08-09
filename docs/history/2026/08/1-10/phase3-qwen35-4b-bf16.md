# Phase 3 Qwen3.5-4B BF16 text生成履歴

## 2026-08-04

- 正本の開発順序とGit外`sLLM.md`を照合し、Phase 3の完了点がQwen3.5-4B BF16のtext-only CLI生成とG3までを含むことを再確認した。
- model lock・RMSNorm・G2・P0までの既存案をPhase 3全体の完了点にせず、public runtimeとmodel-bound最小数値経路を作るStage A子計画へ位置付けた。
- Phase 3全体をStage A model-bound最小経路、Stage B model I/O/frontend、Stage C baseline operator、Stage D model graph/state、Stage E CLI/G3へ分割した。
- exact `gfx1030`/`gfx1201`の同一immutable candidateでfull model G3をPASSするまでPhase 3を完了扱いにしないgateを追加した。
- vision、MTP、Qwen3.5-2B/9B、OpenAI API、最適化、quantizationをPhase 3から除外し、正本の開発順序を維持した。
- 固定llama.cpp/vLLMのfull-model reader調査を行い、hybrid layer schedule、full-attention KV、linear recurrent/conv state、tensor分類、tokenizer/CLI、G3 evidence順序を[reader記録](../../../../references/qwen3.5-phase3-full-model-reader.md)へ固定した。main agentが両local checkoutの完全SHAを再確認した。
- 固定cacheと固定vLLM/llama.cppを再照合し、full-attention Q/gateのhead-wise packing、text-only MRoPE、GDN projection・convolution・recurrent update、BF16入力/weight・FP32 accumulationの契約を確定した。
- Phase 3 text-onlyのstateをconvolution BF16 `[3, 8192]`、recurrent F32 `[32, 128, 128]`、full-attention KV FP16 `[4, T, 256]`へ固定し、request-local lifetimeとprefill/decode共通transitionを要求した。
- config EOS 248044とchat-template EOS 248046の差異は、停止集合`[248046, 248044]`、生成tokenだけの判定、stop tokenのvisible output除外、reportへの停止identity保持として解決した。GPU toleranceとG3 goldenは実装後の独立evidence gateとして残した。
- B1 tokenizer依存readerでlocal crate cacheと固定tokenizer metadataを監査し、`tokenizers =0.21.4`のdefault featureを無効化して`onig`だけを使い、任意Jinjaではなくtyped Qwen3.5 text-only rendererを実装する方針を固定した。停止policyのversioned lock/schema/API化と、全依存のroot lock・license・MSRV offline evidenceをB1前提とした。

## 2026-08-10

- A5で手書きlocal commandと現行workflow contractがずれた運用負債に対し、実行機能を持たないtracked host-only Phase 3 Stage A evidence planner、closed JSON schema、H0 matrix/path登録、focused回帰を追加した。plannerは既存workflow、matrix、G1/G2/P0総合validator、G1 builderのpure layout helperをauthorityとして、exact `gfx1030`→`gfx1201`のH3/G1/G2/P0 path・environment・output ownershipをcanonical JSONへ導出する。
- CLI経路はreviewed/tested/workflow SHAとtree OIDがclean checkoutへ一致することを必須とし、authority file hash、repo containment、全path componentのsymlink、短い未作成run root、target順序、AF_UNIX projectionをfail-closedに検証する。API-only identity seamはtest専用でCLIから選択できない。
- focused planner 11/11、fail-closed 46/46、matrix/JSON/G1/G2/P0 validator、Python compile、diff check、dirty-local H0 316/316が`PASS`した。fresh独立reviewは過去の`common.py` authority漏れ、symlink component、schema順序、CLI bypass回帰、総合validator/workflow検証、H0件数の6指摘が全て修正済みでHigh/Medium 0件と判定し、01:26 JSTに`PASS`した。GPU、model cache、container、build、networkは実行せず、canonical V620/R9700 evidence identityを更新していない。
- 既存P0 builderのsame-UID/trusted-solo output symlink安全負債は、数週間の単独trusted developmentというユーザー承認済み境界に従い延期を維持する。sLLMのcanonical V620は`0000:03:00.0` / `GPU-76a08c022586fed6`のままとし、spare V620は他開発に使用できる。

[対応する計画](../../../../plans/active/2026/08/1-10/phase3-qwen35-4b-bf16.md)
