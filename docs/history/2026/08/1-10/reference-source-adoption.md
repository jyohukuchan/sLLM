# 参照source採用履歴

## 2026-08-02

追加調査対象8件から、LMDeployとKTransformersだけを正式なlocal参照sourceとして採用した。MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用とし、localへcloneせず、今後の採用予定にも置かない。過去の [参照source取得計画](../../../../plans/archive/2026/08/1-10/reference-source-acquisition.md) と対応履歴は、取得時点の事実として変更していない。

## 固定した取得事実

- LMDeploy:
  - official origin: `https://github.com/InternLM/lmdeploy.git`
  - local path: `reference/LMDeploy`
  - release: `v0.15.0`、公開時刻 `2026-07-31T13:00:46Z`
  - lightweight tag/commit時刻: `2026-07-31T12:51:11Z`
  - lock revision: `f4b8140ba19cd823c541241cbb113cc32f854e6a`
  - Apache-2.0、shallow、detached、clean、recursive submodule status空、LFS pointer 0件
  - tracked file 1,646件、checkout全体 `15210661` apparent bytes、`.git` を除くworktree `12208451` bytes
- KTransformers:
  - official origin: `https://github.com/kvcache-ai/ktransformers.git`
  - local path: `reference/KTransformers`
  - release: `v0.6.4`、公開時刻 `2026-07-23T14:32:53Z`
  - lightweight tag/commit時刻: `2026-07-23T13:23:34Z`
  - lock revision: `924754a00bd8e5c6a2ad97929065c113f35782cf`
  - Apache-2.0、shallow、detached、clean、LFS pointer 0件
  - tracked file 1,415件、checkout全体 `124228926` apparent bytes、`.git` を除くworktree `77580448` bytes
  - upstream treeの `third_party/custom_flashinfer`、`third_party/llama.cpp`、`third_party/pybind11`、`third_party/sglang` はgitlinkである。recursive submodule statusは4行全て未初期化を示す `-` で、各submodule worktreeは空である。

## 文書への反映

- [source-lock manifest](../../../../references/source-lock.md) を5件から7件へ更新し、両sourceのcloneには `GIT_LFS_SKIP_SMUDGE=1` を指定した。40桁SHA、shallow、detached、clean、LFS pointer 0件、tracked file数を検査し、KTransformersだけはgitlink 4件のpath、`-` status、空worktreeをfail-closedに検査する。
- [推論engine参照](../../../../references/inference-engines.md) の固定一次参照へLMDeployとKTransformersを移し、それぞれscheduler・blocked KV・quantization・native kernelと、CPU-GPU協調・MoE offload・weight/KV配置・実行計画を読む対象にした。
- 未採用6件は過去に想定した調査範囲だけを残し、順位と将来優先区分を削除した。
- [CI・テスト方針策定計画](../../../../plans/active/2026/08/1-10/ci-test-strategy.md) のlocal source factとexact-revision再調査対象を7件へ同期した。
- [main plan](../../../../plans/main-plan.md) に採用判断を重要決定として記録し、完了状態と次のCI・test再調査対象を7件へ同期した。
- 参照はuLLMの対応実績、性能、正しさの証拠ではなく、直接reuse許可でもない。llama.cpp以外をreader-onlyとする既存のprovenance境界を維持した。

## 検証

- `git diff --check` で担当文書のwhitespace errorがないことを確認した。
- 担当文書内のrelative Markdown linkが全て解決することを確認した。
- 旧件数表現、採用済み2件の将来候補扱い、KTransformersのrecursive submodule statusを空とする誤記が残っていないことを確認した。

[対応する計画](../../../../plans/archive/2026/08/1-10/reference-source-adoption.md)
