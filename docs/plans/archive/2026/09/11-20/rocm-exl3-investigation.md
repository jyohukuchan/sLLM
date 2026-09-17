# rocm_exl3 実機調査

2026-09-17ユーザー依頼。referenceへのclone、動作・実装調査、必要なモデル取得・量子化を許可。追加指示で軽微なバグ修正も許可。

## 受入範囲

- 固定revision、ライセンス、実行環境、再現手順を残す。
- R9700 gfx1201でビルドと独立数値参照付き演算子検査を試す。実行可能なら小型実モデル生成とEXL3量子化を確認する。
- 未対応・失敗は成功と区別し、原因と修正の有無を残す。既存サービスは停止しない。
- EXL3形式、HIP移植、attention、量子化フロー、sLLMへの参考点を調べる。productionへの外部コード移植は行わない。
- 外部コードは専用container、モデル・buildはcheckout外へ置く。既存作業を保持しcommit/pushしない。

## 状態

完了。固定HEAD `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5` をcloneし、5ファイルの局所修正と
host ROCr preloadによりR9700上で114件の数値・構造検査、Qwen3.5-2BのEXL3 4bpw量子化、
英日生成、EOS停止、通常終了を確認した。性能はdecode改善・prefill低下。校正はsmoke用16×256。
失敗した候補、bundled-only終了時segfault、修正patch、再現手順を履歴へ記録した。
モデルとbuildはcheckout外、containerは停止して保持。commit/pushなし。

履歴: [調査結果](../../../../../history/2026/09/11-20/rocm-exl3-investigation.md)
