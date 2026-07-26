# SQ8_0 prefill adaptive chunk selection

## 前回の要点

BY の full-model 実測では固定 M=128 が N=128 で 883.091 tok/s、M=2048 が
N=4095 で 126.686 tok/s だった。一方で固定 M=2048 は N=128 で M=1 fallback
となり 33.474 tok/s まで落ちるため、単一の固定幅を既定にできなかった。M=256
は M=128 を上回らず、M=4096 は N=4095 の実トークンだけでは合法な行を作れない。

## 今回の変更点

- `Adaptive` を SQ8_0 serving の既定にし、N=1..511→M=128、512..1023→M=512、
  1024..2047→M=1024、2048..4096→M=2048 を選ぶようにした。固定幅の環境変数
  override は維持した。
- lower CK/layer/stack/F32 paged-KV admission を測定済みの M=256..4096 へ広げた。
  serving scheduler は M=4096 を拒否し、real-token tail rewind を保つ。
- Adaptive は M=128 で起動し、Ready/reset baseline で選択 M の workspace・resident
  hidden・prompt buffer だけを transactional に差し替える。短い prompt が大きい
  M の常駐バッファを持たない。
- R9700 実測（warm-up を除く 5 sample median）は N=128/512/1024/2048/4095 で
  887.490 / 566.458 / 388.173 / 232.671 / 126.791 tok/s。N=128 は維持され、
  N=4095 の llama.cpp 差は 9.610x から 7.955x へ縮小した。
- 5 prompt の final hidden/logits/greedy token は F32 byte-exact。N=4000 長文は
  83 token IDs・467 文字が一致し、10 ケース生成も 10/10 text/token-ID 一致。
  decode は 27.592484 tok/s。
- GPU isolation は失敗した CLI 入力検証 1 回と成功した実測 1 回の計 2 窓。最終的に
  `ullm-openai.service` は active、`NRestarts=0`、llama service は inactive/disabled。
  active AQ4_0 manifest は変更せず、SQ8_0 のための served-model promotion は行わなかった。

詳細な raw evidence と手順は
[`prefill-adaptive-chunk/README.md`](../../../benchmarks/results/2026-07-27/prefill-adaptive-chunk/README.md)
に保存した。

## 次の行動

- Flash2 本体の再設計はこの範囲外。M=2048 でも attention が支配的という BY の
  handoff を次のカーネル作業で使う。
- 新しい prompt-length benchmark を得た場合は、M=256 を含む選択境界を実測で
  再評価する。最大幅を機械的に選ぶ規則には戻さない。
