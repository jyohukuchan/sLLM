# Phase 87 段階5: decode実行制御

2026-09-21、Qwen3.8 NVFP4の固定device samplingについて段階5を完了した。
MTPなしの1 tokenと、MTPありのdraft・verify・採用状態選択を、それぞれ一つのHIP graphとして再生する。
両GPUで変更前と生成tokenが一致し、停止・予算・context末尾、正常解放、実際の非同期起動を確認した。
Phase 87全体のW×A16廃止や他モデルの受入完了を意味しない。

## 実装

- device上の制御recordに位置、KV長、sampling counter、pending token、採用行数と停止状態を保持する。
  次のembeddingへtokenを渡し、MTPのhiddenもdevice bufferで引き継ぐ。
- targetは最初のeager decode 1 token、MTPは最初のeager blockで必要なworkspaceを準備する。
  以後の各段は同じgraphを再生し、段ごとの再instantiateを行わない。起動コストは通常decode時間に含める。
- MTPはverify各prefixのGDN checkpointを保持し、採用prefixを固定したstate bufferへコピーする方式を選んだ。
  hostへ戻ってrestore／replayする方式を避け、受理判定・状態選択・次の入力をgraph内で完結させる。
  この方式には追加のGDN state copyがある。未採用scratchを公開しないため、whole graphで進めたGDN stateを
  legacyの1-step rewind対象にはしない。
- graph→fence→D2Hをqueueへ入れ、現在のreadbackを待つ前に次のgraphをenqueueする。
  D2Hはcapture前に確保した2-slot pinned poolを使う。使用中slotの再利用やpageable memoryへのfallbackはしない。
  result recordは192 bytes。通常のeager selectorの16-byte recordと区別する。
- EOS／予算で切ったprefixだけを採用し、先行した余分な1段はdrainして破棄する。
  全replayの完了を確認してから公開KV／GDN metadataを更新する。
- configured MTP width 1〜8に対し、device側のactive widthを予算・残容量へ合わせる。幅0ではtarget selectorを使う。
  inactive行は位置処理・状態更新を行わない。captureの固定形状が末尾を跨ぐ場合も、active行の物理境界を検査する。
- Coreのadmission、Rust HIP wrapper、native API、device kernelの全層を対応させた。
  固定形状の一時的な範囲拡張は、実際に収録中の同一queueに限定する。通常実行と公開済み状態の容量検査は維持した。

通常のeligibleなfresh requestで自動有効化する。固定temperature=1／top-k=20／top-p=.95、empty mask／additive、
textの検証済みartifact、HIP exact target、MXFP8 E4 KV、adapterなしが対象である。
penalty・grammar・logprobs・stop string等の制約、prefix hit／publication、context shiftは既存経路を使う。
CLIのsampling defaultは変更していない。serverのfresh prefix publicationはVMM共有を作るため、whole captureを設定しない。

## 通常速度とN0

変更前binaryを保存し、同じ8192入力／128出力、seed123、chunk2048、capacity10240、MXFP8 E4 KV、
MTP width2、1 warmup＋3 measuredで比較した。各試行は新しいrequestで、両GPUとも単一要求である。
数値は非profile計測の中央値。速度下限は設けていない。

| GPU | MTP | 変更前 tok/s | 段階5 tok/s | 差 |
| --- | --- | ---: | ---: | ---: |
| V620 gfx1030 | なし | 15.772662 | 15.676187 | -0.61% |
| V620 gfx1030 | あり | 28.846991 | 29.424955 | +2.00% |
| R9700 gfx1201 | なし | 21.409412 | 20.862236 | -2.56% |
| R9700 gfx1201 | あり | 34.034502 | 34.610392 | +1.69% |

MTPありは両GPUで速くなり、MTPなしは遅くなった。全4条件・全warmup/measuredで128 tokenが変更前と完全一致した。
全行でHIP-only、fallbackなし、cleanup zero。中央値／MAD、E2E、binary／report hashは
[結果JSON](phase87-stage5-results.json)に記録する。

## profileと非同期性

通常速度とは別に、同じ8192/128で1 warmup＋1 measuredを取得した。
`rocprofv3 --kernel-trace --hip-trace --marker-trace`、`SLLM_PHASE87_PROFILE=1`を使用し、
最後のROCTX decode区間を既存`phase87_decode_profile.py`と`phase87_profile_breakdown.py`で集計した。
profile wallを通常TPOTへ混ぜない。単位はms/token、capture起動だけms/request。

| GPU・MTP | GPU kernel busy | HIP API外の空白 | 同期/poll＋transfer内の空白 | capture起動 |
| --- | ---: | ---: | ---: | ---: |
| V620・なし | 59.8718 | 7.9972 | 0.0271 | 18.8577 |
| V620・あり | 32.0577 | 3.5524 | 0.0187 | 20.6244 |
| R9700・なし | 49.1697 | 6.6776 | 0.0231 | 17.9568 |
| R9700・あり | 27.8419 | 2.7600 | 0.0202 | 21.7757 |

HIP API外の空白にはkernel間の隙間、host処理、profiler負荷等が含まれ、原因を一つに断定しない。
その他API・launch内の空白も結果JSONへ分けて記録した。capture起動はassembly／instantiateの範囲であり、
最初のeager warm block等とは区別する。
[段階0](phase87-stage0.md)の同期/poll＋transfer内空白は同じ列順で0.550／0.460／0.669／0.504 ms/tokenだった。
段階0からの全体時間差には先行WUの変更も含むため、段階5単独の速度差には直前baselineの表を使う。

非同期性はAPI呼び出し順だけでなく、実traceで確認した。`hipGraphLaunch`のcorrelation IDでGPU kernelを対応付け、
各launchの間にreadback enqueueがあり、次launchが前graphのGPU処理完了より先に返ることを確認した。
stream上で前graphの後にあるD2Hが完了するまで、CPUが待っていたという解釈は成り立たない。

| GPU・MTP | 確認した連続replay組 | 次launch完了の最小先行時間 ms |
| --- | ---: | ---: |
| V620・なし | 126/126 | 56.156 |
| V620・あり | 50/50 | 81.611 |
| R9700・なし | 126/126 | 41.345 |
| R9700・あり | 52/52 | 67.525 |

再生区間中の再instantiate、`hipHostMalloc`／`hipHostRegister`は0。
target graphは1,220 node／1,218 kernel、MTP width2は1,325 node／1,308 kernel。
残るmemcpy／memset nodeを、traceに現れるruntime copy／fill kernelと区別して照合した。
graph内の隙間とgraph間隔も別集計した。graph間隔にはD2H／fence等を含み、host待ちだけの値とは呼ばない。

## 停止・境界・所有関係

- 最終候補のEOS相当のstop-ID試験では、両GPUのtargetが3 token、V620 MTPが8 token、R9700 MTPが7 tokenで停止した。
  変更前token列の同じprefixと一致し、公開KV／GDN長は8194／8199／8198だった。全ケースで余分なreplay 1回を破棄した。
- 短contextの17/17・capacity34は、両GPUのwidth2とR9700のwidth1／3でlegacyとN0一致した。
  EOSで境界到達前に終わらないよう、診断時だけ`SLLM_PHASE87_STOP_TOKEN_ID=none`を指定した。
- 収録開始位置の境界は、17入力でwidth2／出力5／capacity22、width3／出力6／capacity23を両GPUで比較した。
  legacyと同じ5／6 token、公開長21／22、有効replay 1＋破棄1、cleanup zeroを確認した。
  V620の対照はresident専用のgrow修正前だが、実測memory kindはVMMで、修正前後とも同じphysical capacityを使う。
- 両GPUのnative制御226 checks、selector/PQ 182 checksがPASS。width1〜8の容量44条件・予算44条件、
  width0のtarget selector、非zero prefixのeager P/Q byte一致、align token／hidden 0〜8を含む。
- static9行のKV graphをremaining1〜9で再生し、eager bytesと照合した。FP16／MXFP8、attentionの既存数値oracleもPASS。
  短contextのstaged32 M1〜9は従来packed providerとbyte一致した。
- preprocessのcapacity22境界はactive行1／2／3／9でeagerとbyte一致、inactive suffixのcanary不変、
  invalid positionの無書込みを両GPUで確認した。
- R9700の実public C APIで、resident／VMM両方の末尾を跨ぐ固定形状のcaptureを確認した。
  V620では2件のpinned D2H、3件目のbusy拒否、slot再利用と世代2／1の内容を確認した。
- nativeが出力した7種の192-byte ResultV1をRust側の生成器を使わずdecodeし、token・位置・counter・flags・logprobを照合した。
  fence Failure／Pending／wait errorでもbackend finalizationと状態公開が起きない回帰testを追加した。
- Rust KV tests 32件、capture scope／位置境界、frontend routing、server production 38件（既存ignored 1件）を確認した。
  警告緩和なしのpublic runtime host testと、両exact HIP buildもPASS。

## 実装中に修正した問題と検証範囲

checkpoint不要時のNULL受渡し、device phaseのhost側参照、target／MTPのqueue不一致、wait後に残るcopy owner、
VMMの固定backing、artifact fingerprintの取り違えによる自動有効化漏れを修正した。
pageable readbackをpinned poolへ置き換え、allocation失敗・rollback時のslot／queue所有関係も整理した。

小さい残容量では、短context provider選択、preprocess位置、Core／Rust HIP／nativeの固定形状検査、
resident planeのgrow量に段階的な不備が見つかった。失敗をPASS扱いせず、対応する層の検査と実モデルを追加した。
正式profileの前にはmarker指定漏れとruntime copy／fillの集計も修正した。

一度の統合確認で指摘されたfence finalizationは、下位の`fence.token()`がSuccess以外を拒否しており、不具合ではなかった。
追加した3条件の回帰testでも保護を確認した。prefix publication時の共有VMMとwhole captureの衝突はserver routingで修正した。

## 再現先

- 最終通常計測・EOS・profile・N0／async照合: `.local-artifacts/phase87/stage5/final-gfx{1030,1201}-r1/`。
- 各workerの`status.json`に実行argv、binary hash、子processと終了結果を保存した。
  `run_final_validation.py`はbinaryの差替えを拒否し、全チェック成功後だけPASSを返す。
- 変更前通常測定: `.local-artifacts/phase87/stage5/fullmodel-baseline-gfx{1030,1201}/`。
- 境界: `tail-w2-gfx1030-r2`、`tail-w2-gfx1201-r5`、`remaining-tail-gfx1201-r1`、
  `entry-gfx1030-r2`、`entry-gfx1201-r3`。
- native境界と短context: `active-width-native-gfx{1030,1201}-r1/`。
- モデル、実行binary、raw traceはGitへ含めない。結果JSONは集約値とhashだけを保持する。

[対応する計画](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
