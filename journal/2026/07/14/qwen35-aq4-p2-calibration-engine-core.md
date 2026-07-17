# Qwen3.5 AQ4 P2 calibration engine core

## 前回の要点

P2 full-modelの通常sessionは、prepared tokenを生成してからpublish/commitするtransactionを
持つ。一方、source oracleとのfull-vector calibrationには、post-final-RMSNorm hiddenと
既存LM-head logitsをprepared境界で観測し、source token列をdecode historyへreplayする専用
経路が必要だった。

## 今回の変更点

- resident LM headの既存logits host stagingを再利用し、token ID昇順で1024 f32ずつ渡す
  borrowed visitorを追加した。LM headを再実行せず、語彙長の追加vectorを確保しない。
- post-final-RMSNorm hiddenは固定1024 f32 scratchでchunk D2Hし、hidden長のhost vectorを
  確保しない。これらのD2H/synchronizeはcalibration APIを明示的に呼んだ場合だけ発生する。
- sessionへhash-bound calibration replayを追加した。token列のcanonical SHA-256はdomain tag、
  u64 token count、u64 little-endian token IDsから再計算し、宣言hashとの一致を開始前に要求する。
- observerは現在のpending prepared handle、nonce、generated indexと一致する場合だけ一度実行
  できる。before-prepare、重複、stale、after-publish、cancel、callback failureはfail closedとなり、
  abort/resetでtransactionをdiscardして再利用できる。
- calibration publishは`predicted_token_id`を診断結果として保持しつつ、hash-bound
  `committed_replay_token_id`を次のdecode入力へcommitする。通常のproduction request、wire、
  session configは変更せず、通常pathにD2Hや追加synchronizeはない。
- CPU mock testsでchunk shape/order、予測とreplayの分岐後もreplay tokenが次stateへ入ること、
  nonce/pending境界、source sequence validation、observer/publisher failure、lifecycle
  commit/discard/reset、通常pathがobserverを呼ばないことを確認した。

独立QAで、AQ4 direct-top1はfull logits bufferを生成せず、load-time prewarmまたは以前のstepの
bufferをobserverが読む可能性が見つかった。修正としてLM headへ`generation_epoch`と
`full_logits_epoch`を追加した。prewarmはepoch 0で無効、head実行開始時に以前のfull-row
validityを消し、成功時だけgeneration epochを進める。AQ4 direct-top1成功はfull epochを持たず、
通常のAQ4/F32 full-row生成だけがcurrent epochを持つ。prepared tokenは生成epochを保持し、
calibration startとobserveはtop-1 full-row capability、current epoch、full epochの一致をD2H前に
要求する。direct-top1 productionのkernel、top1 result、wire pathは変更せず、calibration用の
LM head再実行も追加していない。

検証:

- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine`
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine calibration_ -- --test-threads=1`
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine qwen35_aq4_session::tests --lib -- --test-threads=1`
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine qwen35_aq4_head_runtime::tests --lib -- --test-threads=1`
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --lib -- --test-threads=1`

QA修正後のlib testは717 passed、1 ignored（isolated HIP device必須test）だった。GPU/live inferenceは本作業の
対象外なので実行していない。

## 次の行動

別担当のcalibration binaryが、このsession APIを手動driveしてsource/path streamとのf64 metricを
逐次集計する。binaryは各stepでprepare、observe、calibration publishの順序を守り、resultへ
source sequence hash、predicted token、committed replay tokenを別fieldとして保存する。
