# Codex 開発環境

確認日: 2026-09-05。Codex CLI／app-server 0.153.3で設定の読込みと下記接続を確認した。
開発方針は `AGENTS.md`、toolchainは[環境手順](environment.md)を正本とする。

## Subagent

[プロジェクト設定](../../.codex/config.toml)で既定を `gpt-5.6-luna`、reasoningを `xhigh` に固定する。
個別agentがmodelを指定した場合は、その指定が優先される。メインモデルは変更しない。

このhostの `~/.codex/config.toml` にも同じ既定値を設定した。
`~/.codex/agents/{luna,terra,sol}.toml` は、Lunaを通常作業、Terraを横断調査・統合、
Solを反復失敗後や特に深い専門推論が必要な場合に使う役割へ同期した。
全履歴の複製を一律に要求せず、独立した委譲には対象file・目的・期待結果を簡潔に渡す。

## Skills

- [sllm-validation](../../.agents/skills/sllm-validation/SKILL.md): 変更範囲に応じて既存のhost、
  compile-only、GPU correctness／performanceの入口を選ぶ。Phase 7用の行と現行modelの検証を区別する。
- `~/.codex/skills/magpie-kernel-evaluator`: AMD公式のHIP kernel評価・比較skill。
  [amd/skills](https://github.com/amd/skills/tree/e867fa4ae4516f644221cb04dcdf24008a43cb99/skills/magpie-kernel-evaluator)
  のcommit `e867fa4ae4516f644221cb04dcdf24008a43cb99` から導入した。
  upstreamの10ファイルをそのまま配置し、repository rootのMIT LICENSEを同梱した。
  取得元と各fileのSHA-256は、このhostの `~/.codex/skill-sources/amd-skills.json` に保存している。
- `push`、`update`、globalの `qwen38-subagent` は既存のものを利用する。

MagpieのCLI／Python package本体は未導入。skillの導入は、Magpie runtime、vLLM、PyTorch、Triton、
別ROCmの導入や、それらを使ったGPU測定の実施を意味しない。
Magpieを実際に使用するときはHIP workflowと既存の数値testcaseが適合するかを確認し、
sLLMのNumPy oracle、固定toolchain、実測済みtargetの扱いを維持する。
AMDの他のskillは主にLemonade、vLLM、PyTorch trace向けのため今回の対象外とした。
`rocm-doctor` はupstreamで `staging` にある開発中のskillであり、導入していない。

## MCP

| 接続 | 設定場所 | 確認内容 |
| --- | --- | --- |
| OpenAI Docs | global config | 公式文書の検索・取得 |
| Firecrawl | global config | 既存localhost API経由のMarkdown取得 |
| Hugging Face | project config | 公開Qwen configの取得、トークン不要 |
| GitHub | 既存plugin | 接続・認証。`gh` CLIの認証も確認済み |

Hugging Faceは `https://huggingface.co/mcp` へ接続し、`hf_fs` で公開model、paper、文書を調べる。
公開情報用の設定にlocal tokenをコピーしない。private／gated資源の認証は別の利用条件として扱う。

Firecrawlはこのhostの `~/firecrawl/docker-compose.yaml` と既存volumeを利用する。
2026-09-05にFoundationDB初期化の既存DB判定を修正し、APIと依存serviceを再開した。
`/v0/health/liveness`、`/v0/health/readiness`、Codex MCPからの `https://example.com` のscrapeを確認した。
globalのMCP launcherは動作確認した `firecrawl-mcp@3.24.0` に固定している。

## 確認と再読込み

```bash
scripts/dev/check-environment.sh host
codex doctor --summary
codex mcp list
```

`check-environment.sh` はCIと同じ `clang-format-18` を検査する。
`codex doctor` のMCP設定検査は実際のtool成功を保証しないため、接続を変更した場合は対象toolも1回実行する。
非対話shellの `TERM=dumb` 診断と、API／設定の不具合を区別する。
新しいskillは次のターンから利用できる。既定modelやMCPの再読込みが必要な場合は新しいtaskで確認する。

今回の確認は設定、skill認識、host環境、MCP接続に限定し、sLLMのGPU測定は実行していない。
参考として実行した全体C++ format検査には、今回の担当外の既存C++／HIP差分に違反が残っていた。
これはhost環境検査の成功や、GPU correctnessの結果とは別に扱う。
