# Qwen3.8 R9700 server高速経路の適用

2026-09-06、ユーザーがFP16 KVへの変更を受容し、既存の高速経路の適用を指示した。
初回のMXFP8 E4配置ではPhase78 opt-in環境変数を起動scriptに設定していなかった。

## 変更

- 専用backendのKV指定にFP16を追加し、graph／request／起動・終了auditへ指定値を渡す。
  CLI省略時のMXFP8 E4は維持し、今回の配置では`--kv-cache-encoding fp16`を明示する。
- 常駐processにPhase78 R9700設定の12個のruntime opt-inを渡す。
  HIP Graph spans、deferred completion、FP16 KV append/attention chain、NVFP4/FP8 GDN
  projection pack2、NVFP4 activation wave8、decode DP4A wave4/LDS LUT、GQA6 P32/P64、
  GQA6 rocBLAS F32 prefill、NVFP4 F16 staging prefillを含む。
  最後のprefill stagingは従来どおりopt-inであり、generic default採用へ昇格しない。
- Qwen3.8だけ乱数サンプリングでdevice selectorを注入せず、解決済みseedを既存host samplerへ渡す。
  明示device selectorで無効化されていたHIP Graph／KV chainを通常サンプリングでも使用可能にする。
  他モデルのサンプリング経路は変更しない。温度付き要求はlogits readbackの費用が残る。

モデル名`qwen3.8-27b-nvfp4`、R9700 single-visible、context16384、同時1要求、待ち行列1件、
OpenWebUI接続とuser systemd自動起動は維持する。

## 検証

変更前後のHTTP SSEで同一prompt 36 tokens、出力上限128 tokens、temperature0を使用し、
1 warmup＋1 measuredを取得する。最初から最後のcontent eventまでの時間に対する127 token分の
速度推定であり、GPU kernel-only速度や厳密な長文benchmarkとは区別する。
FP16 KVとopt-inを同時に変える総合比較で、個々の寄与を分離する測定ではない。

| 条件 | decode推定tok/s | TTFT秒（warm後） |
| --- | ---: | ---: |
| 変更前 MXFP8 E4／opt-inなし／temperature0 | 7.881 | 2.842 |
| 変更後 FP16／高速opt-in／temperature0 | 19.922 | 0.475 |
| 変更後 FP16／高速opt-in／temperature0.7・seed12345 | 12.822 | 0.334 |

同条件greedyでは約2.53倍。出力文字列とusageが一致した。
温度付き要求はhost側logits処理の費用が残り、Phase78 greedy性能と同一ではない。

- server host: library125 passed/1 ignored、Qwen3.8専用production/CLI各2 passed。
- 最終release build成功。稼働processの12環境変数とFP16 ready reportを照合。
- 完了要求10件はHIP-only/fallbackなし。生成途中キャンセルもHIP実行後のcleanupを確認した。
  生成前キャンセル1件はGPU未選択のためGPU PASSの件数へ含めない。
- 128 token要求のaudit kernel dispatchは176768から163656へ減少。
- SSE、日本語、sampling、2要求直列実行、実token受信後の切断と次要求の回答を確認。
- 最終binaryの終了監査でcurrent/request/workspace bytesすべて0、retryable cleanup/durable quarantine0。
- 統合レビューで、host samplerへの切替後もseedが渡り、通常decodeのGraph/chain条件を満たすことを確認。

数値・起動設定・binary identity・終了監査は[compact evidence](qwen38-r9700-server-fastpath-evidence.json)へ保存した。
公開前の`production.rs`のrustfmt整形は同JSONのbefore/after SHA256対応で記録し、元ソースへのrustfmt適用結果との一致とfrontend/server host testの再実行を確認した。稼働binaryと元のGPU測定結果は変更していない。

最終再起動後もOpenWebUI containerからモデル列挙・Chat「OK」が成功し、自動起動設定のserviceは稼働中。

計画: [完了計画](../../../../plans/archive/2026/09/1-10/qwen38-r9700-server-fastpath.md)。
