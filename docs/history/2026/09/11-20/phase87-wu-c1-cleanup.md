# Phase87 WU-C1: 不要な切替と実験経路の削除

## 状態と受入条件

完了（2026-09-20）。WU-C1計画の8環境変数と変換CLIの旧scaleフラグを、参照元・専用分岐・診断統計・CPU参照まで整理する。
既定の値・演算順・selector scopeは変更しない（N0）。既存の床関数を使うKV／MXFP6活性値／MXFP4、
`SLLM_PHASE87_PROFILE`、指定外の従来opt-in／FORCE_BASELINEは維持する。

- 両target release build（ffp-contract=off）。
- Rust kv_state、変換CLI、evidence toolと影響するcore test。
- native host selector、lowp host/GPU codec、両GPUのattention公開API（指定7 context×M1〜4）。
- WU1.1最終binaryを保存し、削除前後で両GPU MTPなし8192/128、1 warmup＋1 measuredの生成token列を照合する。
- CI source/aggregate hashとmanifest間参照を収束させ、H3/public runtime/RMSNorm contract validatorsで確認する。
- 現行文書の切替説明を除去し、数値台帳の差し戻し方法はGitの対応変更を戻す方法へ訂正する。
  既存のdocs/historyは当時の記録として変更しない。

## 削除した経路と維持したもの

- attentionのGQA共有／split128の2環境変数をruntime、Rust validator、host/publicテストから除去。
  V620 GQA共有、両target KV長8192以上・M1〜3のsplit128という採用scopeは維持した。
- MXFP8 activationの旧floor切替、MXFP6 activationとMXFP8 KVのno-clip実験、NVFP4／MXFP4 activationの
  best-of-two選択、NVFP4 saturation統計と専用kernel／CPU oracleを削除した。
- 3 converterの旧scale flagsを削除し、通常MXFP8/MXFP6変換は採用済みno-clipping recipeへ固定した。
  MTP converterの未知引数判定を値の取得より前へ置き、旧flagだけを渡してもunknown argumentとなるようにした。
- opt-in比較専用の`lowp_quantize_cost_microbench`とCMake登録を削除した。
  WU0の`phase87_copy_bandwidth`は対象外。一度の過剰削除をレビューで検出し、開始時SHA一致で復元した。
- 床関数、MXFP8既定activationが使うno-clip helper、weight artifact側のbest-of-two recipe、
  既存artifactのmetadata reader、従来opt-in／HOLD／FORCE_BASELINEは保持した。
  `SLLM_PHASE87_PROFILE`を持つbenchmark sourceも開始時SHAと一致する。
- 8個の有効な環境変数に加え、既に無効果だった旧名`SLLM_MXFP8_ACTIVATION_NO_CLIP_SCALE`のコメントを整理した。
  9個目の動作する切替を削除したものではない。

## 検証結果

| 検証 | 結果 |
| --- | --- |
| 両target release build（benchmark＋2 evidence binaries） | PASS、ffp-contract=off |
| Rust kv_state | 30 tests PASS |
| 変換CLI | GGUF 7、Qwen3.8 MX 2、MTP 2 tests PASS |
| Rust MXFP evidence | 24 tests PASS |
| native host selector／lowp host API・selection | 1／2 tests PASS |
| codec GPU（両target） | 各1104 decode codes、5 encode boundary sets、format/provider契約PASS |
| lowp matmul GPU oracle（両target） | 各8形状PASS |
| MXFP8／MXFP6 evidence GPU（両target） | 各6形状PASS、CPU oracle／fallbackなし |
| NVFP4 evidence GPU（両target） | 各21形状PASS、CPU oracle／fallbackなし |
| attention公開API（両target） | 各28ケースPASS、独立oracle／repeat／cleanup |
| H3／public-runtime H3／RMSNorm H3 validators | 3種PASS |

NVFP4 evidenceのRust harnessは0 testsであり、compile成功だけをnumerical PASSとは扱わない。
更新したCPU参照を実際に通すGPU evidence各21形状で補った。MTP converterも初期harnessが0 testsだったため、
旧flag拒否と既定recipe保持の2テストを追加した。

CI hashはHIPのsource/aggregate、HIP manifestのraw hash、RMSNormのci_contract aggregateを順に更新し、
2 iterationで固定点へ達した。3 validatorはこの最終manifestでPASSした。
sourceレビューは1回、correctness blockerなし。既存の履歴156ファイルは開始時hashとすべて一致する。

## N0のモデル確認

WU1.1最終binary対削除後binary、両GPU・MTPなし・8192/128、各1 warmup＋1 measured。
4組の128-token配列（target×warmup/measured）は全要素一致し、text hashも一致した。
すべてHIP実行、fallbackなし、terminal logits finite、same-provider repeat、cleanup zeroを確認した。
これは今回の削除前後のN0確認であり、WU1.1のN1変更前の生成列へ戻したという意味ではない。
性能再評価を目的としない単回測定なので、速度差を改善・退行として判定しない。
GPU性能設定とR9700 serviceの元の状態も保持した。

[集約結果JSON](phase87-wu-c1-results.json)にsource/binary identity、test/log SHA、hash更新連鎖とtoken一致結果を保存した。
raw report、生成token配列、binaryは`.local-artifacts/phase87/wu-c1/`に保持する。

開始時のsourceとdirty状態を`.local-artifacts/phase87/wu-c1/initial-state.json`、`before/`へ保存し、
control binaryのSHAがWU1.1最終identityと一致することを確認した。生成物・raw log・binaryはGitへ追加しない。

計画: [Phase87 WU-C1](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-c1-直近作業の不要な切替と実験経路の削除wu11の後wu2の前)
