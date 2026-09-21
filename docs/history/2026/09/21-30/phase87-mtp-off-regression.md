# Phase 87: 段階5後のMTPなし速度低下の解消

## 調査対象

2026-09-21のユーザー依頼により、段階5のMTPなし速度低下を調べた。
MTPありの高速化と不可避なトレードオフなら経路を分ける、という条件付き依頼である。
前段の測定値は[段階5履歴](../11-20/phase87-stage5.md)に保持する。

## 原因の切り分け

R9700で保存済みの段階5前バイナリを再実行すると、MTPなしは21.396535 token/sで、
以前の21.409412 token/sを再現した。一方、段階5最終バイナリはwhole graph無効で
20.654639、whole graph有効で20.869463 token/sだった。単なるgraph無効化では解消しない。

旧／現行legacyの同条件profileでは、演算選択、grid、dispatch数は一致していたが、
attention stage1は2.010806→3.219204 ms/token、mergeは0.152746→0.620605 ms/tokenへ増えた。
合計の増分1.676258 ms/tokenは、通常測定の増分1.678734 ms/tokenとほぼ一致した。
graph対応でsplit数を実行時の値へ変えたため、legacyにも定数除算・定数loopの最適化喪失が及んでいた。
profile付き実行のtoken/sを通常性能として扱っていない。

別の費用として、whole graphではGDNの状態を毎回固定バッファへコピーしていた。
48層×127 replayの選択kernelはV620で0.591242、R9700で0.699651 ms/tokenを占めた。
1層の状態はconv 61,440 bytes＋recurrent 3,145,728 bytesで、48層の論理read＋writeは
307,888,128 bytes/token。ただしこれは実測DRAM転送量ではない。
この選択kernelはlegacy経路にはなく、legacyの低下原因と混同しない。
graph作成費用は通常測定で約0.09 ms/token相当だった。

## 共通修正

- GDN状態の入出力を有効generationの奇偶で交互に使う。全prefix採用時は出力がそのまま次の状態となる。
  MTPの部分採用時だけ、選択checkpointを正しい出力バッファへ反映する。
- MTPなしのM1では状態選択kernelを除き、1 replayあたり48 nodeを削減する。
  MTPありは選択nodeを維持し、全採用の場合はコピーせず終了する。
- 公開するactive slotは初期slotと成功generationの奇偶で決定する。
  余分なNoop replayをgenerationへ数えず、停止後に状態を書き換えない。
- attentionは、動的制御でP32／P128を選んだ後、定数の除算・loop上限を持つdevice関数へ分岐する。
  capture時のworkspace strideと有効split数を分離し、8192境界を跨ぐreplayにも対応する。
  eager側はtemplateでdevice control処理を除く。
- host側で残り1出力と確定している場合だけ、現在のreplayの後続を予約しない。
  それ以外は先行予約を維持し、EOS等の予測できない停止では予約済みreplayを安全に破棄する。
  終端予測と異なる結果は公開前に拒否する。破棄数は実際に予約・破棄した数となる。

MTPなし／ありの意味的な経路分割は追加していない。変更は同じ計算の実装上の費用を除くものとした。

コピー削減とattention修正だけの中間測定では、MTPなしはV620 16.074373、R9700 21.327123 token/s。
MTPありは29.720601／35.076311 token/sとなった。R9700のMTPなしには約0.3%の差が残り、
最終破棄graphがprofile上で43.261933 ms、うちkernel実行38.071180 msを消費していたため、
上記の終端予約削減を追加した。これはprofile上の費用であり、通常実行の短縮量と同一視しない。

attention制御値のwave内共通読込も小さな比較probeで試したが、M1・8192 tokensの
4 warmup＋20測定でcontrolled中央値0.215801→0.215080 msと明確な改善を示さず、不採用とした。
残るcontrolled stage1の費用について、不可避であるとも全面解消したとも主張しない。
今回の完了判断は同条件の全モデル性能と正しさによる。

## 通常性能

Qwen3.8-27B-NVFP4（revision `57926baca9a82b4d6906b43f2750d55315f5b10f`）、
8192入力／128出力、batch 1、MXFP8-E4 KV、容量10240、prefill chunk 2048、
固定sampling（temperature 1、top-k 20、top-p 0.95、seed 123）。
MTPは幅2・BF16 companion。各条件1 warmup＋3測定の中央値で、profileは無効。

| GPU | MTP | 段階5前の再測定 | 修正前（段階5） | 修正後 |
| --- | --- | ---: | ---: | ---: |
| V620 / gfx1030 | なし | 15.838435 | 15.719190 | 16.186784 |
| V620 / gfx1030 | あり | — | 29.450342 | 30.117398 |
| R9700 / gfx1201 | なし | 21.396535 | 20.869463 | 21.490851 |
| R9700 / gfx1201 | あり | — | 34.622534 | 35.638975 |

単位はtoken/s。MTPなしは両GPUで段階5前の再測定値と元の履歴値を上回った。
修正後のMADは表の順に0.004018、0.008208、0.001312、0.005501 token/s。
通常4条件の生成token列は修正前と完全一致し、最終の不要な破棄replayはいずれも0回。
MTPありの速度改善も維持できたため、MTP有無による全体経路の分割は不要だった。

## 検証状況

通常性能、EOS、容量境界の全12ケースがPASS。
[測定値・証拠digest](phase87-mtp-off-regression-results.json)へ、各段階の比較、最終binary／source、
レポートと検証logのSHA-256を保存した。commit／pushは行っていない。

- 両targetの実HIP統合buildを確認。初期のstub-only cargo checkはGPU buildの証拠から除外した。
- 状態選択120 case／GPUと公開C API lifecycleがPASS。3有効generation＋budget後Noop、
  奇偶の状態planeとeager結果のbyte一致、公開metadata、pinned readbackを確認した。
- launch関数の引数追加をhost stubへ同期し、strict host runtime build／fault testがPASS。
- 終端予約削減のfocused host testは5件PASS。通常の先行予約、予測可能なbudget終了、
  不正な非終端応答、失敗したfence、破棄時の失敗と非公開を確認した。
- 両GPUのattention／KV device-control testがPASS。P32／P128境界、M1〜9の短い入力、
  capacity境界とinactive領域、独立attention oracle（max ULP 0）を確認した。
- parityとattentionを対象に一度の累積integration reviewを行い、correctness blockerなし。
- EOSはMTPなしが1有効＋1破棄、MTPありが3有効＋1破棄で両GPUともPASS。
  17入力・出力5／6・容量22／23のMTP幅2／3は、それぞれ1有効・破棄0で生成列が一致し、
  公開状態長21／22と正常解放を確認した。
- 最終R9700・MTPなしの実traceは126 replay、125組すべてで次graphの起動完了が前graphの
  GPU終了より先行した（最小45.872201 ms）。replay区間の再instantiate／pinned新規確保はなく、
  状態選択コピーkernelも消えている。profile付き速度は通常性能表へ混ぜていない。

作業証拠は`.local-artifacts/phase87/stage5-off-regression/`、native状態テストは
`.local-artifacts/phase87/stage5/parity/`。モデル、binary、raw profileはGitへ追加しない。

計画: [MTPなし速度低下](../../../../plans/archive/2026/09/21-30/phase87-mtp-off-regression.md)。
