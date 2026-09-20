# Phase87 WU1: MXFP8 E4 decode attention

## 状態・固定した対象

完了（2026-09-20）。計画とWU0再計測を読み直し、V620 stage1短縮の打ち切り線を
6.2072 ms/token、R9700の参照を1.3935 ms/tokenとする。
対象はMXFP8 E4、q_heads=24、kv_heads=4、head_dim=256、M=1〜3。
本番のID93 staged32をcontrolとし、C1 GQA共有、C2 複数key処理、C3 split/occupancyの3候補を比較する。

## 仮説と候補

- C1: 6 query headが共有するK/Vをtileで一度だけ復号し、6 waveがLDSを共有する。
  online softmaxのkey順と32 splitのworkspaceを維持する。最初の8-key tile（LDS 16 KiB）に加え、
  barrier回数を半減する16-key tile（LDS 32 KiB）を同じ候補内で一度比較する。
- C2: C1の共有tileを使い、wave内を4 key×8 laneへ分ける。dotは4本の8-term累積をbalanced treeで結合し、
  4-key単位でonline softmaxを更新する。単純な32-term直列dotを避ける。
- C3: controlのsplit数を64／128へ増やす。gfx1030のfused E4 decodeを維持し、
  mergeは全partial valueをLDSへ置かずglobal workspaceを読む。64／128 splitがLDS上限を超えないようにする。

llama.cppの`fattn-vec.cuh`／`fattn-tile.cuh`を参照した。vecはquery headからKV headへのmappingと
keyごとの処理が中心であり、6 query head間のMXFP8復号共有を直接提供しない。
候補は既存sLLMのcodecとstaged workspace契約から実装した。外部コードのコピー・移植はない。

## 数値演算と資源

C1は計画時にはN1相当を予想したが、実装ではdotの8-term逐次積和、32-lane還元、key順のonline softmax、
FP32の部分和、stage2とBF16 RNEを維持した。MXFP8からFP32へ復号した値の置き場所だけをLDSへ移し、
演算順を変えていないため、N0としてcontrolとのbitwise比較も記録する。
C2は4-key単位のsoftmax更新とtree加算、C3はpartition数を変えるため、controlとのbitwise一致を前提にしない。

8-key版C1のcode objectはgfx1030でLDS 16,384 B／VGPR 44／SGPR 27、gfx1201でVGPR 41／SGPR 25、
spillなし。control stage1はgfx1030でLDS 0 B／VGPR 48／SGPR 28、spillなし。

## 検証条件

- exact gfx1030／gfx1201、context 1023／1024／1025／8192／8193、M=1〜3。
  打ち切り線の比較にはWU0と同じ平均context 8256、M=1も実測する。境界8192／8193の結果を
  context 8256の固定参照と直接混ぜず、同一processのcontrolとの差×16層を使う。
- 既存Phase83の独立host FP32 oracleとfixtureを再利用する。公開テストの基準
  max BF16 ULP≤4、max absolute error≤0.03125、max relative error≤0.04を採用する。
- 未来tokenのK/Vが支配的になるcausal-tail fixtureも使い、maskとMごとの終端を検査する。
- finite、repeat、output/workspace canary、allocation cleanupを検査する。
- 単体性能は同一processのAB／BA（3 round、各9 sample）。各variantに300 msの継続warmupを入れる。
  stage1とstage2を分け、両stage合計も記録する。C3のmerge増加を隠さない。
- 採用判断後、両GPUの通常8192/128、MTPなし／あり、1 warmup＋3 measuredを測る。
  production選択へ接続する場合は関連host/compile検査と、数値変更の記録を行う。

## 単体結果と採否（300 ms連続warmup）

両targetそれぞれ24ケース×5実装（controlを含む）、計240 oracle行がPASSした。
指定15形状、参照平均context 8256/M1、causal-tail 6形状、強いQ/Kの2形状を含む。
最大BF16 ULPは1、最大absolute errorは0.00195312、finite／repeat／outputとworkspaceのguard／cleanupは全PASS。
C1は両GPU全24ケースでcontrolとbitwise一致した。C2／C3にはbitwise差がある。

次表はcontext 8256、M1、同一process AB-BA-ABの27サンプル中央値。
削減は`(control stage1 − candidate stage1) × 16層`で、stage2は別途記録した。

| GPU | 候補 | control stage1 (ms) | candidate stage1 (ms) | 削減 (ms/token) | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| V620 | C1 GQA共有・tile8 | 0.729974 | 0.336087 | **6.3022** | 6.2072を超え、採用 |
| V620 | C2 4-key grouped | 0.729134 | 0.394608 | 5.3524 | 打ち切り線未達 |
| V620 | C3 split64 | 0.730694 | 0.446649 | 4.5447 | 打ち切り線未達 |
| V620 | C3 split128 | 0.728454 | 0.415048 | 5.0145 | 打ち切り線未達 |
| R9700 | C1 GQA共有・tile8 | 0.214843 | 0.216043 | −0.0192 | 改善なし |
| R9700 | C2 4-key grouped | 0.213643 | 0.251924 | −0.6125 | 退行 |
| R9700 | C3 split64 | 0.213923 | 0.154682 | 0.9479 | 打ち切り線未達 |
| R9700 | C3 split128 | 0.215403 | 0.142122 | 1.1725 | 1.3935に未達 |

V620 C1のstage1+stage2合計は0.743494→0.349406 ms/層。
8192／8193境界でもstage1の16層換算短縮は6.3156／6.2920 ms/tokenだった。
初期screenの8193では6.1953 ms/tokenで固定線の近傍だったため、その単発値だけで採否を決めず、
予定した全境界と参照と同じ8256を計測した。打ち切り線は変更していない。

C1内のtile16比較は、V620/8256でcontrol 0.726974→candidate 0.607852 ms（短縮1.9060 ms/token）と退行した。
LDSだけが16→32 KiBへ増え、VGPR44／SGPR27とspillなしは同じ。LDSによるresident block数の制限が
原因として考えられるが、occupancy counterを実測した結論ではない。tile8へ戻し、追加の探索は行わない。
C2は共有に加えて4-key処理を行ってもC1を超えず、C3は並列性を増やすだけでは復号の重複が残る。
R9700ではGQA共有のLDSと同期費用を回収できず、既存ID93を維持する。

## 実装への接続

- exact gfx1030、既存staged32が有効なMXFP8 E4、Q/KV heads=24/4、D256、M1〜3にC1 tile8を適用する。
  M4、gfx1201、既存staged32対象外は従来経路を維持する。
- `SLLM_CAUSAL_ATTENTION_DECODE_GQA_SHARED=0`で元のID93 stage1へ戻せる。
  未設定または`1`のみ共有版を有効にし、その他の値はcontrolとする。
- kernel ID93、2 dispatch、workspace `[M,24,32,258]`、stage2を維持する。
  公開workgroupは従来通り最大のstage2の256、gridはstage1の実数`M*4*32`へ更新する。
  起動数の整合性を保つため、計画のnative 3ファイルに加え、Rust側のmetadata validatorと対応する
  host/public GPUテスト・fake HIP宣言を最小限同期した。新たな数値形式やABI structは追加していない。
- 測定したscratch C1とproduction C1の関数bodyは、空白・行commentを除去した比較で一致する。
  引数pointer型だけが既存native launcherに合わせて`uint8_t*`から`void*`となる。

## 通常モデル比較（8192/128）

同じQwen3.8-27B NVFP4 model、MXFP8 E4 KV、固定device sampling、MTP width2、1 warmup＋3 measured。
比較前binaryを保存し、controlと最終buildを別processで測る。decode tok/sは127 decode transitionの値。
単体stage1の削減量とend-to-end TPOTの短縮は、cache・他operator・host待ちを含む条件が異なるため同一視しない。

| GPU | MTP | control tok/s | candidate tok/s | 変化 |
| --- | --- | ---: | ---: | ---: |
| V620 | なし | 14.2953 | 15.4512 | +8.09% |
| V620 | あり | 25.4302 | 27.1806 | +6.88% |
| R9700 | なし | 19.0793 | 19.1923 | +0.59% |
| R9700 | あり | 34.9447 | 35.1302 | +0.53% |

V620のTPOTはMTPなし69.9532→64.7197 ms/token、MTPあり39.3234→36.7909 ms/token。
両GPUの全4構成で生成token列・visible/text hashは前後一致、最初の分岐位置はなし。
MTPの受理数はV620 74/106、R9700 78/100でそれぞれ前後一致した。
R9700は実行kernelを変更しておらず、表の約0.5%の差をC1の改善とは扱わない。これは固定したcoding8192入力の比較であり、新たなBF16比品質の認定ではない。

## 統合検証

- gfx1030／gfx1201のproduction release build: PASS（本番の`-ffp-contract=off`）。
- native host selector: PASS（1/1）。Rust staged32 metadata検査: PASS（1 test）。
  M1〜3/M4、1023/1024/1025、対象GPU、既定選択とcontrol指定を確認した。
- 公開API GPU検証: 両GPUでlength 1023/1024/1025、M1〜4、独立oracle、repeat、state解放がPASS。
  V620 M1〜3の共有gridとM4/R9700の従来gridを検査する。
- 1回の統合レビューと、その後のproduction接続差分の確認でcorrectness blockerなし。
  CPUのhost検査やcompile成功をGPU PASSへ読み替えていない。
- 最終ビルドの両GPU/MTPなし・ありの所定モデル比較を完了した。全runがHIP、fallbackなし、
  finite、ID93実行あり、repeatあり、cleanup zero。GPU性能設定とR9700 service状態を復元した。
  WU2には未着手。

開始時のsource/hashと既存変更は`.local-artifacts/phase87/wu1/initial-state.json`に保存した。
既存の未コミット変更を維持し、この作業ではWU2へ進まない。

## 結果と再現入口

[集約結果JSON](phase87-wu1-attention-results.json)に全形状の中央値/MAD、oracle集約、
source/binary/report SHA、公開API結果、モデル前後の速度・生成列比較を保存した。
raw JSONL、27個の時間sample、生成token列、binary、source snapshotは
`.local-artifacts/phase87/wu1/`に保持し、Gitへ追加しない。

- probe build: `native/hip/tests/phase87_wu1_probe.hip.cpp`をROCm 7.14の`amdclang++`で
  `-O3 -ffp-contract=off -std=c++17 -Wall -Wextra -Werror -x hip --offload-arch=<gfx1030|gfx1201>`、
  `-I include -I native/hip/src -I native/lowp/include`を指定してcompileする。
- operator: `python3 ci/tools/run_phase87_wu1.py --target <target> --binary <probe> --output <new-directory>`。
- fullmodel: 既存`ci/tools/run_phase86_mtp_catch_up.py`を再利用する。各execution JSONにmodel、
  protocol、環境変数、binaryとjob listのhashを保存した。新しい比較では共有版のcontrolに
  `SLLM_CAUSAL_ATTENTION_DECODE_GQA_SHARED=0`を使える。
- 集約: `python3 ci/tools/summarize_phase87_wu1.py --root .local-artifacts/phase87/wu1 --output <summary.json>`。

計画: [Phase87 WU1](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu1-v620-mxfp8-e4-decode-attention-stage1)
