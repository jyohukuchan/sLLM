# FORCE_BASELINEを診断用参照経路（oracle）として再定義する

> 状態: 完了（2026-09-17。受入基準1〜6、Stage 0〜6の実施範囲を確認）
> 起点: `e25bcc07` ＋ 未commitの作業ツリー
> 2026-09-17ユーザー指示: 「oracle的な用途に限定して計画して」
> 2026-09-17追加指示: 本計画を完了する。受入基準1〜6と非対象を維持して着手。

## 背景

`SLLM_NVFP4_W4A4_FORCE_BASELINE=1` はQwen3.8-27B NVFP4のprefillを実行できない。
両GPUで `layer.0.mlp_gate_matmul` が `backend status 260: matmul kernel launch: invalid configuration argument` で失敗する。
`SLLM_QWEN38_NVFP4_PROJECTION_PACK2=0` で非パック経路にしても同じノードで失敗するため、
**packingとの相互作用ではなくNVFP4 W4A4 baseline matmulそのもの**の問題である
（[欠陥記録](../../../../../../ci/matrix/nvfp4-force-baseline-defect-v1.json)）。
baselineは `TilePolicy::Elementwise` だが、実kernelは出力1要素につき256スレッドの1ブロックを使う
（着手時のソース確認で「1スレッド」という計画記述を訂正）。
MLP gate prefillは1回あたり `m=2048 × n=17408 = 35,651,584` 出力になる。
既定経路は正常で、失敗はlayer 0で明示的に発生し、cleanupは両GPUで全ゼロだった（fail-closedとして正しい）。

一方、[Phase82の採用範囲台帳](../../../../../history/2026/09/1-10/phase82-default-adoption-scope.md)は
`FORCE_BASELINE=1` を「既定より優先するrollback」と定義し、
NVFP4 activation quantizer wave8、decode ID67、Phase78のprefill ID59／decode ID58の
**ロールバック手段として名指ししている**。Phase82の検証は演算子fixture（26件・5モード）であり、
Qwen3.8のフルモデルprefill形状ではなかった。

## この計画の立場: ロールバックではなくoracle

`FORCE_BASELINE` を**本番ロールバックとしては直さない**。

- elementwise baselineはprefill規模で特殊化カーネルより桁違いに遅いと見込まれ、
  本番障害時に実際に選ばれる復旧手段にならない。
- 実際に機能する本番ロールバックは**前のバイナリ／commitへ戻すこと**であり、
  [数値出力変更台帳](../../../../../compatibility/numerical-output-changes.md)が既にrollback commitを記録している。

一方、診断用の参照経路としての価値は高い。この一連の数値調査
（Phase84.5、復号経路の対照、MXFP8 GPU差のStage C 3本、WMMA確認）は、
実モデル上で参照実装と比較する手段が無かったため、**すべて手作りの診断buildを要した**。
毎回build・source復元・hash照合が発生している。

### 2層のoracle（役割を混同しない）

llama.cppがGPU backendをCPU backendの参照実装モード（`ggml_backend_cpu_set_use_ref`）と照合し、
vLLMが各CustomOpに `forward_native` を持つのと同じく、役割を2層に分ける。

| 層 | 実体 | 役割 | 独立性 | スケール |
| --- | --- | --- | --- | --- |
| **T1: 真値** | 既存のhost独立FP32／NumPy oracle | 「この演算の正しい答えは何か」 | **独立**（別言語・別実装） | 演算子単位 |
| **T2: 帰属** | GPU上の `FORCE_BASELINE` 参照kernel | 「どのsubsystemが差を生んでいるか」 | **非独立**（codec等を共有しうる） | フルモデル |

**T2は真値ではない。** 特殊化kernelとcodecやscale算出を共有していれば、同じ欠陥を共有する。
T2の仕事は、実モデルで**1つのsubsystemだけを差し替え、他を固定したまま**差が消えるかを見ることに限る。
真値の判定は常にT1で行う。T1は既に機能しており、本計画で変更しない。

## 非対象

本番ロールバックの実装・保証、baselineの高速化、既定選択の変更、
vLLM型の「コンパイラで参照経路を本番品質にする」方式の導入、
MXFP8のGPU差調査の再開（ユーザー指示で終了済み）、commit／push、公開CI。
llama.cpp／vLLMは設計の参照にとどめ、コードは取り込まない。
取り込む場合は同じ作業中に[import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log)へ記録する。

## 受入基準（実装前に凍結）

1. **T2の約束範囲を明文化し、その範囲だけを保証する。**
   約束外の形状では、`invalid configuration argument` のような汎用エラーではなく、
   **「この形状はbaseline参照経路の対象外」と名指しするエラー**で失敗する。
2. T2を有効にしても、**差し替え対象subsystem以外の演算・chunk・graph構造を変えない。**
   帰属実験で2つ以上を同時に変えないためである。
   特にprefill chunkを縮める回避策は、演算順を変えて帰属を汚すので採らない。
3. T2 kernelの数値は、**実モデル形状（prefill境界を含む）でT1と照合してPASS**する。
   2の冪だけでなく非整列値と各境界の両側を含める。
4. GPU証拠はexact target・UUID、HIP-only、非zero dispatch、fallbackなし、cleanup 0で判定する。
   CPU emulation、timeout、crash、zero selectionはPASSにしない。
5. 既定経路の出力・選択・性能を変えない。変更したsourceは既定経路の数値に影響しないことを
   既存の既定経路テストとdigestで確認する。
6. 本計画の結論として、台帳の「rollback」記述を誤解のない形へ直す。

## 段階

### Stage 0: 棚卸し（実装なし、最初に必ず行う）

11種の `FORCE_BASELINE` について、Qwen3.8-27B NVFP4で実際に動くかを表にする。

```
SLLM_MX_WA_PREFILL_FORCE_BASELINE     SLLM_GDN_FORCE_BASELINE
SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE SLLM_NVFP4_W4A4_FORCE_BASELINE
SLLM_MATMUL_FORCE_BASELINE            SLLM_CAUSAL_ATTENTION_FORCE_BASELINE
SLLM_NVFP4_FORCE_BASELINE             SLLM_FP8_OUTER_DECODE_FORCE_BASELINE
SLLM_MX_WA_M1_FORCE_BASELINE          SLLM_FP8_QUANT_FORCE_BASELINE
SLLM_FP8_OUTER_FORCE_BASELINE
```

各フラグ × {prefill到達、decode到達} × {gfx1030、gfx1201} について、
成否・失敗ノード・エラー文言・対象kernelが実際に切り替わったか（kernel traceでsymbol確認）を記録する。
**フラグを立てても対象経路へ到達していない（Qwen3.8では使われない）ものは「無効果」と区別して記録する。**

- 入力は既存の単一条件prefix `.local-artifacts/mtp-bench/wmma-check/prefix-one.json`
  （`code-rust-bugfix-zh`、8,284 prompt token）を再利用する。
- decodeだけを見る条件は、prefillが短い既存fixtureで補う。
  prefillで落ちるフラグのdecode可否を確かめるためである。
- 各runは1条件・1回でよい。性能値は取らない。

Stage 0だけで「NVFP4 W4A4以外は動く」「多数が同じ理由で落ちる」などの全体像が決まる。
以降の段階の対象はStage 0で失敗したフラグに限る。

### Stage 1: T2の約束範囲を定義する

Stage 0の結果から、各フラグが保証する範囲（format、target、M／K／Nの範囲、prefill／decode）を決め、
`docs/development/` に1文書としてまとめる。
**約束範囲は「実際に使う帰属実験で必要な形状」から決め、全形状網羅を目標にしない。**
最低限、Qwen3.8のdecode（M=1）と、実モデルのprefill chunk（M=2048を含む）を対象とする。

### Stage 2: 大きなM×Nでのlaunchを成立させる

Stage 0で「形状が大きすぎてlaunchできない」と判明したフラグだけを対象にする。

- まず `invalid configuration argument` の**実際の原因パラメータを特定する**
  （grid次元、block次元、共有メモリ量、1 launch当たりのthread数上限のどれか）。
- 参照kernel内部で**launchを複数へ分割**して上限内に収める。
  加算順・FP32累積・BF16 RNE・scale算出は変えない。
- graph側のchunk変更では回避しない（受入基準2）。
- 速度は目標にしない。帰属実験で数条件を流せれば足りる。

### Stage 3: 約束外で明示的に失敗させる

Stage 1の範囲外では、prepare時に**対象外であることを名指しするエラー**を返す。
host contract testで、範囲の内側・外側の両境界を検査する。

### Stage 4: 検証

- **T1照合**: 各T2 kernelを実モデル形状（prefill境界、非整列K/N、M=1とM=2048の両側）で
  既存host独立oracleと照合する。
- **フルモデル到達**: Qwen3.8で各フラグを単独で立て、両GPUで完走、有限値、HIP-only、cleanup 0を確認する。
  kernel traceで対象symbolへの切替を確認する。
- **帰属の実用確認**: 既に測った現象を1つ使い、T2で差し替えて期待どおり動くかを見る。
  例: WMMA確認で判明したgfx1201のNVFP4 MLP prefill（`Nvfp4W4A4PrefillGfx1201WmmaKahan`）を
  T2へ差し替え、`target_hidden_sha256` の変化を記録する。
  **これはMXFP8 GPU差調査の再開ではない。** T2が帰属に使えることの動作確認に限り、原因の結論は出さない。

### Stage 5: 台帳の記述を直す

[Phase82台帳](../../../../../history/2026/09/1-10/phase82-default-adoption-scope.md)と
[Phase78記録](../../../../../history/2026/09/1-10/phase78-nvfp4-cross-model-measurement.md)の
「rollback」記述を、次のように書き分ける。

- 本番ロールバック: 「commit `<id>` へのバイナリロールバック」
- `FORCE_BASELINE`: 「診断用参照経路（T2）、対象範囲: <Stage 1の文書>」

過去の履歴文書は当時の記録として残し、訂正は追記で行う。
[数値出力変更台帳](../../../../../compatibility/numerical-output-changes.md)にも同じ区別を1節で記す。

### Stage 6: 再発防止

- **範囲内（本計画で実施）**: Stage 3のhost contract test。
  CPU CIはhost contract・小さなoracle・compile-onlyに限るという方針に沿う。
- **提案（ユーザー承認が必要、本計画では実施しない）**:
  既定採用kernelを持つ各モデルについて、Stage 1の約束範囲を実モデルで確認するGPU smokeを1本ずつ置く。
  GPUの新しい検査は承認なしに追加できないため、提案として記録するにとどめる。
  今回の欠陥は、このsmokeがあればPhase78の時点で検出できた。

## 実行と記録

- laneはDraft。kernel launch分割はkernel変更なので、Stage 2の完了時に統合reviewを1回行う。
- 再計画条件（AGENTS.md準拠）: 同一work unitが2回却下、review時間が実装時間を超過、
  機能的進捗が1時間以上停止、検証/文書が作業の30%超、見積り1.5倍超、または受入基準の変更。
  **Stage 0で失敗フラグが多数に及んだ場合は、Stage 2以降の範囲をユーザーと再確認する。**
- GPU運用: exact UUID（V620 `GPU-76a08c022586fed6`、R9700 `GPU-a8e9ddefa2d60f55`）を確認する。
  V620使用時はローカルQwen serviceを停止して利用不可として扱う。
  R9700は `sllm-qwen38-r9700.service`（user unit）を測定時だけ停止し、
  unit／run.sh／binaryのSHA256一致、healthz／readyz 200、performance level復元を確認する。
  既存の `ci/tools/check_nvfp4_baseline_convergence.py` のlease処理を流用してよい。
- 本番sourceを変更する段階（Stage 2・3）では、既定経路の既存テストとdigestで非退行を確認する。
- raw出力・trace・buildは `.local-artifacts/force-baseline-oracle/` へ置き、Gitへ追加しない。
  集約は `ci/matrix/force-baseline-oracle-v1.json`。完了時に本計画を
  `docs/plans/archive/2026/09/11-20/` へ移し、対応する
  `docs/history/2026/09/11-20/force-baseline-reference-oracle.md` を作って相互リンクする。

## 実行メモ（2026-09-17、各時点の記録）

- 変更前sourceのSHA256一覧、両targetの固定binary、toolchain情報を
  `.local-artifacts/force-baseline-oracle/` に保存した。
- Stage 0は`stage0-gfx1030`／`stage0-gfx1201`へ実測中。既存の8,284-token／256位置のlong fixtureと、
  既存pilotの64-token／4位置のshort fixtureを使い、chunkは2,048のまま維持する。
  実行を開始したrunnerでは各フラグのshortも取得する。更新後runnerはlong失敗時とdefault controlにshortを絞る。
- gfx1201のStage 0は24 run（defaultと11フラグ、各long/short）が終了した。longの失敗は
  `SLLM_NVFP4_W4A4_FORCE_BASELINE`だけで、layer 0の既知launchエラーとcleanup 0を再現した。
  同フラグのshortは参照kernelを840回実行して完走した。他の完走には無効果フラグも含むため、
  完走数をT2の検証済みフラグ数へ読み替えない。R9700 serviceはhash不変、health/ready 200、
  performance level一致で復帰済み。gfx1030はFP8 prefill参照経路のlong runを継続している。
- 最小HIP probeでは両exact targetで、256-thread blockのgrid 16,777,215は成功、16,777,216以上は
  `hipErrorInvalidConfiguration`だった。deviceの`maxGridSize[0]`だけでは検出できない、
  X方向総スレッド数の32-bit上限を原因として特定した。
- [T2範囲文書](../../../../../development/force-baseline-reference-oracle.md)と大形状用の独立FP32 sampled oracleを
  準備し、旧Phase78／82台帳にはT2と本番binary rollbackの区別を追記した。GPU数値検証の完了は未主張。
- 検証時間割合の再計画: gfx1030のFP8 prefill参照実装が長時間を要し、Stage 0の待機が作業時間の30%を
  超えた。進行している既存runは停止せず監視を継続する。新たな広範な比較suiteは追加せず、残りの検証は
  計画済みの失敗フラグのT1照合、フルモデル到達、既定経路digestとhost境界検査に限定する。
  受入基準と11フラグの棚卸しは維持し、修復対象を未到達の別formatへ広げない。
- 順序の再計画: gfx1201の全11フラグとNVFP4のlong失敗／short成功、両GPUのlaunch境界probeが揃ったため、
  観測済み失敗フラグであるNVFP4 W4A4の修復を残りの棚卸しと並行する。棚卸しは固定済み変更前binaryを
  使い続けるのでsource編集で条件は変わらない。Stage 0の残りを省略せず、多数の別フラグで失敗が判明した
  場合の範囲再確認も維持する。未確認のフラグを修復対象へ追加しない。
- Stage 2/3の実装、最終host検査、統合reviewと指摘箇所の再確認を完了した。両targetの独立FP32演算子
  検証（各37 case、既定controlを含む）と、最終M=1 packの全出力／grid／cleanup検証はPASS。
  gfx1030の期限付きwaitによる失敗試行は保持し、診断測定器を非破壊queryへ直した新試行の成功と区別する。
- gfx1201の修復後longはHIP-only、非finite 0、fallbackなし、cleanup 0でPASSした。
  既定longのhidden/logit hashは変更前と一致し、T2切替でhashが変わることを確認した。
  r1実モデル証拠とr3のmetadata訂正の対応は`r1-r3-evidence-mapping.json`へ記録し、
  r3のfull-model traceを取得したとは主張しない。詳細は下記履歴を正本とする。
- gfx1030のFP8 prefill longは5,336.647秒で完走した。続く棚卸しで
  `SLLM_FP8_OUTER_DECODE_FORCE_BASELINE=1`がlong/shortともtarget decode row 0の
  Graph span admissionで拒否されることを確認した（cleanup 0）。失敗フラグはこれとNVFP4 W4A4の2種。
  計画の「失敗したフラグだけを対象」に従い、FP8 decodeのM=1参照kernelを同じstateless Graph経路で
  実行できるよう整合を調べる。Graph無効化やchunk変更で回避しない。
- r4でgfx1030の既存M=1 FP8 GDN pack Graph許可リストへ`Fp8Emulation`だけを追加した。
  shape/workspace/lifetime条件とkernelは不変。両GPUで既存の実GDN pair oracle、入力変更、
  3-node Graph capture/replay、全出力、cleanup 0を確認した。gfx1201はNativeのままで無効果。
  11フラグの棚卸しは両GPUで終了し、gfx1030の最終default／NVFP4／FP8 decodeのlongを実行中。

## 完了確認（2026-09-17）

- 初期棚卸しは両targetの11フラグを完了。無効果を区別し、NVFP4 W4A4の両GPU long失敗と
  gfx1030 FP8 decodeのlong/short失敗を保持した。
- 修復後は両GPUのNVFP4 long、gfx1030のFP8 decode longがHIP-only、非finite 0、fallbackなし、
  cleanup 0で完走した。参照kernelの実dispatchをtraceで確認した。
- 両GPUの演算子37 case、NVFP4 M1 pack、FP8実GDN pair／3-node Graph、host境界検査はPASS。
  既定演算子digest、実モデルhidden/logit hash、全kernel／launch geometryは変更前と一致した。
- rollback訂正を3文書へ追記し、T1/T2と本番binary復旧を区別した。常設GPU smokeは提案のまま残す。
- [集約台帳](../../../../../../ci/matrix/force-baseline-oracle-v1.json)と
  [検証履歴の受入基準表](../../../../../history/2026/09/11-20/force-baseline-reference-oracle.md#受入基準と証拠)へ証拠を対応付けた。
  raw、失敗試行、source/build identityは`.local-artifacts/force-baseline-oracle/`へ保持する。
- commit／push、公開CI、既定最適化、MXFP8 GPU差調査の再開は行わない。

## 関連資料

- [完了した実装・検証履歴](../../../../../history/2026/09/11-20/force-baseline-reference-oracle.md)
- [NVFP4 FORCE_BASELINE欠陥記録](../../../../../../ci/matrix/nvfp4-force-baseline-defect-v1.json)
- [WMMA帰属](../../../../../../ci/matrix/gfx1201-wmma-attribution-v1.json)
- [MXFP8 GPU差の調査記録](../../../../../history/2026/09/11-20/mxfp8-gpu-divergence.md)
- [Phase82採用範囲台帳](../../../../../history/2026/09/1-10/phase82-default-adoption-scope.md)
- [メイン計画](../../../../main-plan.md)
