# Qwen3.8-27B: BF16基準の全語彙KLD比較

2026-09-18ユーザーgoal。EXL3 3/4/5bpw、sLLM NVFP4/MXFP6/MXFP8、vLLM FP8、
llama.cpp BF16（複数GPU）を比較し、第一巡完了後にKV量子化等の組合せを拡張する。
動作しない構成も原因を調べ、修正・別の実行経路・段階的な測定で可能な限りデータを取得する。

## 測定契約

- BF16 source: local Qwen3.8-27B、HF download metadata revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- 共通token manifestを固定。全engineで同じteacher-forced入力位置の全語彙logitsをFP32で保存し、samplingを介さない。
- 基準p=llama.cpp BF16、候補q。temperature1でFP64 logsumexpにより `KL(p||q)` を計算。
- 正規化対象は公式tokenizerで定義された共通token IDs。padding/engine固有virtual IDは除外し、ID対応を照合。
- 非finite、位置/語彙不一致、ゼロcase、CPU fallback、crashを成功扱いしない。平均/中央値/p95/p99/max、top1、位置別とcase別を残す。
- まず自然言語/日本語/code/math/structured textと短長boundaryの第一巡。単一case成功を全scope完了としない。
- 第一巡はMTPなし、基本KV FP16/BF16を優先。第二巡でengineのKV量子化・長context・chunk条件を広げ、重み差とKV差を分ける。

## 計画範囲

1. 現在のmodel/source/binary/imageを識別し、必要なEXL3 plain 3/4/5bpw公開artifactを固定revisionで取得。
2. BF16 baselineをV620×2で実行し、repeat/position/resetを確認。R9700は候補側へ使用。
3. 各engineのraw full-logit adapterを作り、同一manifestでdump/KLD計算。未対応や失敗は診断を継続する。
4. 初期8構成（baseline＋候補7: EXL3×3、sLLM×3、vLLM×1）を網羅してから、利用可能なKV設定の比較へ進む。
5. 総合表、失敗と修正、再現手順、model/runner/hashを保存。完了時のみplanをarchive。

## 第二巡の具体的な比較条件

第一巡を終えた後、同じcase-v1で以下を試す。未対応条件はエラー原因と実際に試した修正・代替を記録し、
GPU実行できた条件を成功数に数える。重み差とは別に、同じengine／重みの非量子化KVに対するKLDも計算する。

- EXL3 3/4/5bpwそれぞれ: K/V 8/8、4/4bit。4bpwのみ6/6、4/8も追加。
- sLLM NVFP4/MXFP6/MXFP8それぞれ: standard OCP MXFP8 E4とE5 KV。
- MXFP6/MXFP8はchunk1に加えて32／64の教師強制block処理を比較し、M=1と複数行で変わる活性値・演算経路を記録する。
- vLLM FP8: FP8 E4M3、E5M2 KV。非量子化KVは実際の解決dtypeを記録する。
- llama.cpp BF16: Q8_0、Q4_0 KV。BF16重みを維持し、基準FP16 KVとの差を測る。
- 長さ4,097の自然文／codeを追加し、末尾257位置を取得する。baseline、EXL3 4bpw、sLLM MXFP6、vLLM FP8を
  非量子化KVと代表的な量子化KVで比較し、短いcontextだけの結果に限定しない。

この列挙はユーザーの「色々なパターン」を具体化した実験範囲であり、未対応のKV形式をproduction対応へ昇格する約束ではない。

## 資源と境界

- artifacts/models/raw logits: `/home/homelab1/datapool/qwen38-kld-20260918/`（checkout外）。
- 外部runtimeはGPU/readonly model/sourceと専用workだけをmountしたcontainer。host credentialsを渡さない。
- 外部sourceの新規copy/adaptは同じ作業でimport logへ記録。既存checkout再使用や依存package利用は新規importではない。
- productionへの採用やcommit/pushは本goalに含めない。既存main-plan決定を黙って変更しない。

## 完了（2026-09-18）

初期8構成、KV形式・chunk32/64・長さ4,097の追加条件を取得し、40条件のcaptureと64組のKLD比較を完了した。
全条件の入力／位置／shape／raw finite／hashと、実行終了・GPU使用の証拠を確認した。
R9700で拒否されたNVFP4 E5はV620へ移し、同GPUのFP16 controlとともに取得した。
vLLM FP8 checkpoint＋E5 KVはwriter単体修正・GPU oracleの成功後もruntime guardが拒否するため、
full-model成功数に含めず理由を保存した。未検証の条件を成功へ読み替えていない。

再現: [再現手順](../../../../../references/qwen38-kld-reproduction.md)

履歴: [測定履歴](../../../../../history/2026/09/11-20/qwen38-cross-engine-kld.md)
