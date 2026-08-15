# Phase 14→15: Qwen/Gemma共通RDNA性能bridge

> 状態: completed
> 作成日: 2026-08-15

## 目的

Phase 14完了時点のQwen3.5とGemma 4 production pathを同じexact RDNA targetで再計測し、Phase 15 Weight NVFP4へ
入る前にmodel/dtype非依存または両modelへ共通する上位の構造的性能残差を最大二つだけ改善する。異なるmodel、tokenizer、
token条件のthroughputをparity比較せず、kernel category、host launch/completion、transfer、submission/boundaryの実測から
採否を決める。

本bridgeは新しい製品Phase番号を挿入しないが、独立した受入条件、history、commit、push境界を持つ。

## 開始条件

- Phase 13のmodel-neutral prepared execution制御とPhase 14のGemma 4 production pathが完了している。
- Qwen3.5-2BとGemma 4-12Bのreviewed lock、official cache、R9700 exact `gfx1201`、V620 exact `gfx1030`が利用できる。
- Phase 9 baselineと現行runnerのmetric、fallback、cleanup、build/model identityを区別できる。
- profiler raw trace、model、binary、生成token列を追跡せず、local artifactとbounded summaryだけを使う。

## スコープ

- Qwen3.5-2Bを最小共通model、Gemma 4-12Bを二つ目のproduction architectureとする。
- R9700 `gfx1201`をprimary、V620 `gfx1030`をbounded secondaryとする。
- O1はshort-oddと32/32相当を基本とし、candidate境界だけB-1/B/B+1を追加する。
- kernel dispatch、HIP runtime/transfer、M=1 matvec、MLP、attention、normalization/elementwise、model固有opをboundedに集計する。
- candidateはprepared command/graph replay、launch/completion削減、共通matvec/MLP/fusion、共通layout/provider tuningの順に
  最大二つを選ぶ。

次は含めない。

- RDNA4 FA3-like。ただしattentionが代表wall timeの支配要因へ移った事実の記録は許容する。
- model/dtype固有の大規模rewrite、NVFP4実装、long service、全model/全shapeの総当たり。
- llama.cppと条件が一致しないGemma値のparity claim。
- CPU fallback、timeout、crash、zero selectionをGPU PASSとして扱うこと。

## 受入条件

1. fresh candidate identityがsource/build/model/toolchain/exact GPUを固定し、測定前後のGPU healthと他process不在を確認する。
2. Qwen3.5-2BとGemma 4のR9700代表pathを取得し、V620は収容可能な同じQwen caseとGemma bounded caseへ範囲を明記する。
3. wall timeを少なくともhost/launch、matvec、MLP、attention、normalization/elementwise、transfer/otherへ分け、kernel countと
   durationをbounded summaryへ残す。
4. candidateをprofile上位から最大二つに限定し、出自、対象model/GPU、期待効果、実装cost、expiryを記録する。
5. 各candidateに非整列値と境界両側の数値oracle、fallback/cleanup、対象O1、別canonical caseの短い回帰を持たせる。
6. default採用は反復medianで対象caseが改善し、別modelまたは別canonical caseへ明確な退行がない場合に限定する。
7. Qwen/Gemmaのprepared cache、transaction、sampling、service semantics、GPU fail-closed境界を変えない。
8. affected host checks、focused GPU、1回のintegration review、findingだけのre-review、plan/history/main plan/Phase 15 baseline同期を
   完了する。
9. bridge単位の必要最小限commitを作成し、current GitHub branchへ通常pushする。

## 実装順序

### B0: fresh identityとprofile runner

- clean pushed Phase 14 identityからexact target binaryをbuildし、Qwen/Gemma lock/cacheとtoolchainを固定する。
- Qwen/Gemmaのshort-odd、32/32相当を同じmetric vocabularyへ正規化する。
- profiler出力をrepository外へ保存し、kernel/runtime/memory-copyを名前とsemantic categoryへbounded集計する。

### B1: profile取得と候補選定

- R9700で両modelのO1を取得し、V620でQwen O1とGemma bounded representativeを取得する。
- count、total/median duration、host API/transfer、submission、boundary、TTFT/TPOT/E2Eを比較する。
- 最大二つのcandidateを優先順どおり選び、attention非支配ならFA3-likeを除外する。

### B2: candidate 1 bounded実装

- micro/oracleと対象外境界を先に固定し、共通execution/providerへ最小変更を入れる。
- O1 repeated medianと別model/canonical caseを比較し、採用またはrevertせず無効化できる形でrejectを記録する。

### B3: candidate 2 bounded実装

- B1に二つ目の明確な共通candidateがある場合だけ実施する。
- B2と同じ数値、fallback、cleanup、repeated median、非退行条件で採否を決める。

### B4: closeout

- bounded summary、採否、残差、Phase 15開始baselineをhistory/main planへ同期する。
- affected checksとintegration reviewを完了し、本planをarchiveしてbridgeだけをcommit/pushする。

## 完了結果

- B0/B1でQwen/Gemmaのfresh R9700 profileとV620 bounded caseを取得し、attention非支配を確認した。
- candidate 1のGemma workspace/prepared semantic再利用を採用し、malloc/freeを`92.2%`削減した。
- candidate 2のM=1 BF16 matvec streaming loadを採用し、両exact targetの17/17 numerical oracleを通した。
- R9700 fresh baseline比でGemma `3/17` `+3.07%`、`32/32` `+3.89%`、Qwen3.5-2B
  short-odd `+1.62%`となり、V620にも明確な退行はなかった。
- H0 `513/513`、H1 `421/421`、H2 `36/36`とfocused host/GPU checksを完了した。clean identityの
  H3 exact target直接コンパイルは本bridge commit後・push前に実行する。

## 計測lane

| lane | 内容 | 使用範囲 |
| --- | --- | --- |
| BR-H | host contract、schema、aggregator | 各work unit |
| BR-P | rocprof kernel/runtime/transfer bounded aggregate | B0/B1と採用後 |
| BR-O0 | 変更target、short-odd、方向性 | candidate iteration |
| BR-O1 | short-odd + 32/32、反復median | 採否 |
| BR-X | 別modelまたは別canonical case | 非退行確認 |

## 停止・再計画条件

- profiler overheadで代表pathを分類できない場合はraw trace量を増やさず、実行auditとkernel microbenchへ分ける。
- candidateが二回reject、検証/docsがwork unitの30%超、実装見積りが1.5倍超、または機能進捗が1時間止まった場合は
  同じcandidateを続けず次順位またはPhase 15へ進む。
- 明確な共通candidateがない場合は無理に二件実装せず、fresh profileと残差をPhase 15 baselineへ渡してcloseoutする。

[対応する履歴](../../../../../history/2026/08/11-20/cross-model-rdna-performance-bridge.md)
