# Qwen3.8 27B NVFP4 R9700 サーバー統合

現在の配置は後続のユーザー指示によるFP16 KV＋Phase78高速設定へ更新済み。
以下は初回MXFP8配置時の記録であり、[高速経路適用履歴](qwen38-r9700-server-fastpath.md)を最新状態とする。

日付: 2026-09-06。ユーザーがR9700・固定モデル・同時実行1要求の限定実装を承認し、
当初のMXFP6 KV指定をMXFP8 KVへ変更したことを記録する。

## 変更

- `sllm-server --qwen38-nvfp4 ABSOLUTE_DIRECTORY`を追加。
  exact Unsloth成果物の検証、既存混合NVFP4/FP8 resident model、MXFP8 E4 KV、
  既存のChat・SSE・サンプリング・キャンセル処理を接続した。
- tokenizerとchat templateを実ファイルのsize/SHA-256で検証する専用frontendを追加。
  tokenizerの248077 ID spanとモデルの248320語彙行を区別する。
- gfx1201・logical device 0の専用profileとし、GGUF／dynamic manifest／MTP入力との混在を拒否する。
  prefix cache、checkpoint、context shifting、draftは無効。埋め込みとtool protocolを公開しない。
- 既存スケジューラーの単一workerを使用する。配置時は待ち行列1件とした。
  KV kernelおよび既定のPhase78性能selectorは変更していない。
- MXFP8数値検証runnerに`--q-heads`を追加し、従来16 headを既定のまま、24 headを検証可能にした。
  JSON schemaの`query_heads`は旧証拠との互換性のため省略可能、今回の証拠では24を明記する。

## 配置

- GPU: Radeon AI PRO R9700、gfx1201、PCI `0000:07:00.0`、UUID `GPU-a8e9ddefa2d60f55`。
  `ROCR_VISIBLE_DEVICES`にUUIDを指定して単一GPU可視化する。
- ROCm 7.14.0、LLVM 23、code object v6、wave32。
- モデル: `/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4`。
- コンテキスト: 配置時に16384を明示。省略時のモデル推奨値262144と区別する。
- API: `http://127.0.0.1:8000/v1`、モデル名`qwen3.8-27b-nvfp4`。
- OpenWebUI: 既存containerを再開し、port 3000で提供。既存の有効な
  `http://172.20.0.1:8000/v1`接続先を利用する。
- 常駐: user systemd `sllm-qwen38-r9700.service`。
  起動script・binary・raw reportはGit対象外の`.local-artifacts/qwen38-r9700-server/`へ配置。
  停止signalは既存サーバーのgraceful shutdownに合わせてSIGINTを指定する。

## 検証

- frontend: 96 passed、1 ignored。専用templateのthinking／空system／assistant historyを含む。
- server library: 125 passed、1 ignored。CLI: 29 passed。
- MXFP8 runner host: 5 passed。既存Phase53 CI/schema契約: 15 passed。
- exact gfx1201のrelease build成功。
- MXFP8 GPU oracle: query heads24、KV heads4、head dimension256のattentionが一致。
  KV appendは31/32/33/255/256/257の6寸法で格納値・scale・paddingを照合。
  HIP append6、attention1、fallbackなし、cleanup retryable/durable0、terminal zero。
  attention oracleは既存の1 token・zero-query条件であり、長文品質評価ではない。
- 実モデル: greedy「2+2」→「4」、日本語SSE「こんにちは、お元気ですか？」、
  sampling「Blue」、生成token「1」受信後の切断と次要求「9−5」→「4」を確認。
  2要求同時送信は順次処理し、それぞれ「3」「5」と回答した。
- 終了監査: resident 21,650,549,568 bytesから最終GPU current/request/workspace bytesすべて0、
  retryable cleanup／durable quarantineとも0。要求のHIP dispatchを確認しfallbackなし。
- OpenWebUI containerからモデル列挙とChat「OK」を確認。
  ブラウザーのログイン操作を自動実行したという証拠ではない。

## 統合レビュー

1件の指摘（apply-template APIが旧Qwen3.5のkind/hash/sizeを返す）を修正した。
実Qwen3.8のkind/hash/sizeへ分岐し、旧Qwen3.5との取り違えをhost testで確認した。

## 範囲と残件

この配置はテキストChat専用。MXFP6 KV、モデルrecipe指定のstatic FP8 KV、MTP、vision、
複数要求バッチ、他GPU／他モデルを対応済みとは扱わない。Phase79全体の完了ではない。
MXFP8による長文品質の包括評価や新しい速度達成基準は本作業に含まない。

数値oracle・build identity・終了監査: [compact evidence](qwen38-nvfp4-r9700-server-evidence.json)。
終了監査は最終template identityだけの修正前に取得し、backend/core/HIP/model/descriptor/build条件が
不変の範囲へ再利用する。最終binary自体の終了監査とは区別する。

最終binaryでapply-templateのQwen3.8 identity、埋め込みの400拒否、Chat「4」を再確認した。
サービスは自動起動を有効化して稼働中。

計画: [完了計画](../../../../plans/archive/2026/09/1-10/qwen38-nvfp4-r9700-server.md)。
