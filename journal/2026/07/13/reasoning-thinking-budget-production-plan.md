# Reasoningとthinking budgetの本番計画

## 前回の要点

- Qwen3.5 9B AQ4ではreasoningが配信設定、Gateway、worker protocol、出力分離の各境界で
  未接続だった。
- OpenWebUI 0.9.4は`delta.reasoning_content`を表示できるため、UI patchは不要だった。

## 今回の変更点

- `docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md`へ、モデル非依存の
  reasoning dialect、厳密なbudget、OpenWebUI E2E、性能・品質benchmark、release evidence、
  rollbackまでの計画を保存した。
- 見積もりは試作2〜4人日、AQ4 beta累計8〜12人日、production候補累計15〜25人日とした。
- 他AIへの引き継ぎ用として、repository/runtime snapshot、関連実装境界、現行AQ4/llama.cpp/
  OpenWebUI配備、性能証跡、network未確認事項、test資産、開始時checklist、secret禁止事項を
  計画のSection 13へ追記した。
- repository HEADとactive workerのpromotion source commitが異なること、firewallの8001 ruleは
  live照合が必要なことを既知の不一致として明示した。
- この作業では実装、配信設定、サービス状態を変更していない。

## 次の行動

実装指示を受けたら、現行AQ4のAPI、token列、prefill/decode性能をPhase 0の基準証跡として
保存し、versioned API/manifest/worker仕様を固定する。

## 2026-07-13 継続作業

### 前回の要点

- v2 manifestをactiveにしたQwen3.5 9B AQ4で、HTTP/SSE、OpenWebUI reasoning、soak、
  restart、Stopの実測ゲートは成功していた。
- worker failureゲートだけは、準備プローブの実行方式とreasoning時の証跡上限が未整合だった。

### 今回の変更点

- `run-openwebui-failure-gate.py`のDocker readiness probeを、固定probe imageのcurl専用仕様に
  合わせた。HTTP 200を`{"ready":true,"status":200}`へ変換し、read-only、capability drop、
  retry、timeoutの制約を維持した。
- failure gateのSocket.IO証跡上限を、browser側の2048件と一致させた。実測の最終証跡は539件
  だった。
- raw service journalに含まれる公開モデルIDを秘密情報として誤検出しないようにした。一方で
  API token、URL、prompt、復旧マーカーは引き続きcleartext検査対象とした。
- reasoning UIのDOMに合わせて、reasoning blockを除いた回答本文をfailure/stop/soakの検証にも
  適用した。reasoning browser smokeは通常チャットURLを使い、検証後に生成チャットを削除する。
- 実機worker failure gateが成功した。worker停止、systemd一回復旧、Docker network内ready=200、
  OpenWebUI失敗表示、入力復帰、復旧チャット、Socket.IO証跡、secret scanを確認した。
- 検証中にsystemd restart rate limitへ到達したため、ゲートを停止して`reset-failed`とstartで
  uLLMを復旧した。最終状態は`active/running`、R9700はuLLM workerのみ、ready=200である。

### 次の行動

関連Python回帰71件、Ruff、Node構文、`git diff --check`を再確認した。次は既存のrelease
evidenceを最終source commitへ更新し、active manifest・promotion receipt・campaign・browser
証跡のidentityを揃えてrelease validatorを通す。
