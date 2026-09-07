# Qwen3.8 27B NVFP4 R9700 サーバー統合

状態: 完了（2026-09-06）。

## 承認済み範囲

ユーザーの明示指示に基づき、sLLMでQwen3.8 27B NVFP4をR9700に常駐させ、
MXFP8 E4 KV、同時実行1要求でOpenWebUIに接続する。
既存の検証済みUnsloth safetensors成果物を専用起動引数から扱う限定経路とし、
一般モデルのGGUF入力方針、MXFP6 KV、複数要求バッチ、他GPUへの対応は拡張しない。

## 実装前の受入条件

- 成果物の既存検証を通し、実際のtokenizerとchat templateを利用する。
- gfx1201で既存NVFP4 resident modelとMXFP8 E4 KVを使用する。
- HTTPモデル列挙、通常Chat、SSE、繰り返し要求とキャンセル後の再利用を確認する。
- 推論は同時1要求とし、既存の待ち行列・キャンセル処理を再利用する。
- 変更範囲のhost testと実機の数値検証、サービス疎通を記録する。
- 既存OpenWebUIからモデルが見え、Chat APIに接続できることを確認する。

実機の初回配置はcontext 16384を明示指定する。性能目標の追加や他モデル最適化は行わない。

## 作業

- [x] 専用モデル読み込み、frontend、CLIの統合
- [x] focused host検証、gfx1201 buildとGPU検証
- [x] 常駐サービスとOpenWebUI接続
- [x] 統合レビューと履歴・互換性記録

履歴: [実装・配置・検証記録](../../../../../history/2026/09/1-10/qwen38-nvfp4-r9700-server.md)。
