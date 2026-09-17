# FORCE_BASELINE診断用参照経路の整備

状態: 完了（2026-09-17）。11フラグの棚卸し、NVFP4修復、FP8 decode Graph整合、host検査、
両GPUの演算子検証と修復後の実モデル確認を完了した。受入基準1〜6を満たし、失敗試行も保持する。

## 目的と範囲

`FORCE_BASELINE`を本番rollbackではなく、GPU上でsubsystemを差し替えるT2診断経路として扱う。
独立host FP32 oracleをT1とし、codec等を共有しうるT2を真値とは呼ばない。
本番復旧は既知のbinary／commitへ戻す。Phase78／82台帳と数値出力変更台帳へ訂正を追記した。

対象はQwen3.8-27B NVFP4本体、BF16 MTP、MXFP8 E4 KV、exact gfx1030／gfx1201。
固定long入力は8,284 prompt tokenと256 teacher-forced位置、chunkは2,048のまま。
shortは既存pilotの64 prompt token／4位置を使う。
KVやsampler、モデルの量子化既定、graph構築は変更しない。

## 原因と修復

NVFP4 W4A4 baselineは出力1要素につき**256-thread blockを1個**起動する。
当初の「1要素1スレッド」という記述は誤りだった。
両GPUの最小HIP probeで、grid 16,777,215 blockは成功し、16,777,216以上は
`hipErrorInvalidConfiguration`となった。deviceの`maxGridSize[0]`は2,147,483,647だが、
X方向の総thread数が32-bitに収まる制約が別に存在する。
QwenのM=2,048、N=17,408は35,651,584 block／9,126,805,504 threadとなり、この制約を超える。

- baselineのlauncherを1 launch最大16,777,215出力に分割した。
- kernelへflat output offsetを渡し、元のFP32積和、wave／block reduction、BF16丸めを維持した。
- T2範囲をM=1..4,096、K=1..17,408、N=1..65,536、exact gfx1030／gfx1201に固定した。
  standaloneは非整列K/Nを扱い、projection-packの既存K16／row契約は維持する。
- workspace queryとprepareで、zero shape、K=17,409、範囲外target等を参照経路の対象外と明示する。
- standaloneのdispatch数はquantizer 1回＋分割数、grid情報は最大main launchを示す。
- 既存のM>=64 baseline pack拒否とmember matmulへのfallbackは維持した。
  直接admitされるM=1 packには範囲検査を加えた。

統合レビューで、M=1 packのquantizer gridがwave8のまま報告される既存の不整合を確認した。
実launchと同じ環境判定へ直し、baselineのscalar quantizerではK/16=320 blockを報告する。
実形状のaggregate gridは320＋17,408＋17,408＝35,136 blockとなる。
演算、kernel選択、pack admissionはこの訂正では変えていない。指摘箇所の再確認は完了し、追加blockerなし。

## 検証済みの結果

| 検査 | gfx1030 | gfx1201 |
| --- | --- | --- |
| 最終HIP build | PASS | PASS |
| 既存小形状の独立FP32 oracle 21 case | PASS（r1） | PASS（r1） |
| 実モデルM=1／2,048のgate/down、分割境界 7 case | PASS（r3） | PASS（r1） |
| 非整列値、M/K/N境界、最大単一launch境界 8 case | PASS（r3） | PASS（r1） |
| 既定M=2,048/K=5,120/N=17,408の出力digest | 変更前と一致 | 変更前と一致 |
| 最終M=1 pack public C ABI oracle | PASS（r3） | PASS（r3） |
| NVFP4修復後の固定long実モデル | PASS（r4） | PASS（r1、r4までの意味差分mapping確認済み） |
| FP8 decode実GDN pair／Graph | PASS（r4、ID6） | PASS（r4、既存ID5のまま） |

大形状T1は独立FP32の固定座標を照合し、各launch境界の前後も含める。
出力全体のfinite検査とSHA256を併用する。小形状は既存の全出力照合を維持する。
実モデルshapeと境界を検証した結果であり、admission範囲内の全M×K×Nを総当たりしたものではない。
baselineのdispatch数は、実形状と分割境界の7 caseで`[2,2,4,2,2,3,3]`だった。

最終pack probeは両GPUとも2 repeat、全出力のBF16 ULP差0、deterministic、fallbackなし、cleanup 0。
rocprof traceはscalar quantizer 2回とbaseline matmul 4回を記録した。
rocprofのglobal thread数はquantizer 81,920、各member 4,456,448で、256で割ったblock数の和が
C ABIの35,136と一致した。待機や結果解析だけをGPU成功と見なしていない。

host検査は`production public runtime host fault test`がPASS。
範囲内外、zero、K=17,409、unsupported target、workspace/prepareのエラー文言、
割当／cleanup、分割数、既定経路維持、pack gridを確認した。
CPUのhost検査はGPU数値証拠に含めない。

## 失敗した測定器の試行

gfx1030の大形状検証r1は`resource is busy`、r2は
`public completion wait timed out`とpost-error cleanupの`Busy`で失敗した。
この2試行はPASSへ書き換えず、raw結果を保持する。
当初の測定器は30秒の期限付きwaitを使い、cleanup errorが元のerrorを隠していた。
期限付きwaitは`Pending`ではなくerrorを返すため、Pendingの再待機だけでは解消しなかった。

r3のlong oracleは非破壊`query()`で同じlive handleを保持し、10ms間隔で状態を確認する。
30秒ごとに進捗を記録し、実際のFailure/errorは失敗のまま扱う。
gfx1030では大形状のPendingを経て成功と数値照合に到達し、最後にcleanup 0を確認した。
従来の小形状モードの期限は変更していない。
通常runtimeへの待機policy変更ではなく、診断用測定器の修正である。

## 実モデルと棚卸しの結果

gfx1030のFP8 prefill参照経路は5,336.647秒でlongを完走した。これは実行に要した時間の記録であり、
性能比較ではない。FP8 decode参照経路はlong/shortともtargetの最初のdecodeで
`graph span plan is outside the exact gfx1030/gfx1201 M1 stateless contract`となり、cleanup 0で失敗した。
この失敗記録を保持し、Graph構造を保ったadmission整合を修復した。

原因はgfx1030 FP8 GDN packのGraph許可リストにID6 `Fp8Emulation`が無かったことだった。
r4では既存のM=1/K=5,120/N=10,240・6,144、workspace 5,124 bytes、dtype、同一variant、
queue/lifetime条件を維持して、このvariantだけを追加した。kernel、選択、Graph構造、chunkは変更していない。
既存のFP8 GDN GPU oracleを使い、独立FP32解析式と全出力の一致、入力変更、3-node capture、
2回replay、fence/finalize、cleanup 0を確認した。gfx1030 traceは参照kernelを20回記録し、
gfx1201はID5 Nativeのままで参照kernelのdispatchは0だった。
同じtest binaryが続けて実行した既存NVFP4検査は追加の完了条件とせず、出力された時間値も性能claimに使わない。

gfx1201はdefaultと11フラグのlong/short計24 runを終了した。
gfx1030は11フラグのlong/short 22 runとdefault longを終了したが、**default shortの対照runは欠落している**（集約の該当行は`status: MISSING`、`same_gpu_default_missing: true`）。このためgfx1030の各フラグshort runには同一GPUの既定short対照が無い。gfx1030の既定経路の不変性は、存在するdefault longの対照とdispatch集計（314,642）で確認しており、short側の対照欠落はこの結論に影響しない。欠落行を補完済みとは扱わない。
gfx1201の変更前longの失敗はNVFP4 W4A4だけで、同フラグのshortはbaselineを840回dispatchして完走した。
無効果のフラグをT2の検証済み経路へ昇格させない。BF16 matmulフラグはMTPだけでなく、
本体GDNのBF16 `in_proj_a`／`in_proj_b`にも効くことをsourceと実測で確認した。
attentionフラグは全kernelをgenericに戻す指定ではなく、専用wave経路を残す部分適用である。

修復後gfx1201のlongは315,820実dispatch、HIP-only、非finite 0、fallbackなし、cleanup 0で完走した。
同じcandidateの既定longは変更前のhidden/logit hashと一致した。
T2への切替ではhidden/logit hashが変わり、subsystem差し替えが帰属に使えることを確認した。
これはGPU間差の原因特定やMXFP8調査の再開ではなく、T2の動作確認に限る。
各R9700 service leaseはunit/run.sh/binary hash不変、health/ready 200、performance level一致で復帰した。

## 証拠の対応

raw、source hash、binary、build log、traceは`.local-artifacts/force-baseline-oracle/`に保存する。
`initial-bin`が変更前、`candidate-r1-bin`がlaunch修復、`final-r3-bin`がmetadata／測定器の修正、
`final-r4-bin`がFP8 Graph admission整合を含む最終版である。
R1/R2の失敗試行とR3の成功試行を別directoryで保持する。
検証の一部は同じGPUの既存測定と併走したため、取得時間を性能比較の証拠に使わない。

R1→R3でproduction実行に関係する変更はpack quantizerのgrid報告／preflight計算だけである。
kernel、算術、graph／chunk構築、direct dispatch数は不変。固定モデルのM<=2,048/K<=17,408では旧／新gridは
ともに非zeroかつUINT32内で、large-M baseline packのmember fallbackも同じである。
decode graphはpack経路の対象外で、Rust側の`grid_size_x`もdispatch evidenceへコピーされるだけである。
この対応を確認してR1のgfx1201 full-modelと演算子数値証拠を再利用し、変更された報告値はR3の
両GPU pack probeとhost検査で別途証明した。R3そのもののgfx1201 full-model traceを取得したとは主張しない。
対応記録は`r1-r3-evidence-mapping.json`に保存した。

gfx1201 full-modelのbaseline kernelは44,744回実dispatchされ、最大global X thread数は4,294,967,040。
全launchで`0 < global X < 2^32`をtraceから再計算した。C ABIのgridはblock数、rocprofのgridはglobal thread数
であるため、両者を混同しない。

gfx1030の修復後NVFP4 longも315,818実dispatch、HIP-only、非finite 0、fallbackなし、cleanup 0で完走した。
参照kernelは44,744回、最大global Xは4,294,967,040で、全launchが上限内だった。
両targetの既定longは変更前とhidden/logit hashが一致し、kernel名・workgroup・global gridの
全dispatch集計も一致した（gfx1030: 314,642、gfx1201: 314,644）。GPUの選択と仕事量を維持している証拠であり、
同時実行条件が揃わないwall timeの同等性は主張しない。`default-launch-invariance.json`に記録した。

R3→R4はgfx1030のFP8 GDN packの既存参照variantを許可する1行だけのproduction差分である。
NVFP4はその分岐より前にreturnし、gfx1201 Nativeの分岐も不変なため、既存証拠を対応付けた。
`r3-r4-evidence-mapping.json`と両GPUの新しいFP8 pair／Graph probeを併せて確認した。

gfx1030のFP8 decode修復後longも314,642実dispatchで完走した。参照emulation kernelは59,905回、
HIP-only、非finite 0、fallbackなし、cleanup 0を確認した。raw証拠監査は
`verification-audit.json`、受入条件と文書を含む最終監査は`final-audit.json`に記録する。
集約台帳は[force-baseline-oracle-v1.json](../../../../../ci/matrix/force-baseline-oracle-v1.json)。
常設GPU smokeの追加、commit／push、公開CI、既定経路の性能最適化は本作業の対象外。

## 受入基準と証拠

| 計画の受入基準 | 確認内容と証拠 |
| --- | --- |
| 1. T2の約束範囲と範囲外エラー | [範囲文書](../../../../development/force-baseline-reference-oracle.md)、NVFP4 workspace/prepare host境界検査、FP8既存M1 stateless Graph契約。全shapeの数値検証とは区別する。 |
| 2. 対象subsystem以外を固定 | launchのoutput offset分割、同じFP32積和と丸め。chunk 2,048・Graph構築は不変。FP8は既存3-node Graphへ同一variantを許可するだけ。source snapshotとr1→r3→r4の差分対応を保存。 |
| 3. 実形状と非整列／境界のT1照合 | 両GPU各37 case（既定control 1、小形状21、実形状／分割境界7、範囲境界8）。NVFP4 packとFP8実GDN pairは全出力照合。 |
| 4. exact GPUでの完走とcleanup | UUID固定、HIP-only、dispatch非zero、非finite 0、fallbackなし、cleanup 0。初期のlaunch／Graph失敗と測定器r1/r2失敗は保持。修復後のNVFP4両GPUとFP8 decode gfx1030 longもPASS。gfx1201 FP8 decodeは初期棚卸しで無効果の完走を確認。 |
| 5. 既定経路の維持 | 既定演算子digestと両GPUの実モデルhidden/logit hashが変更前と一致。全kernel／launch geometry集計も一致。source差分に既定の算術・kernel選択変更なし。wall timeの性能比較は実施しない。 |
| 6. rollback表記の訂正 | Phase78、Phase82、数値出力変更台帳へ追記。本番は既知commitのbinary、FORCE_BASELINEは範囲を限定したT2と分類。 |

Stage 6の再発防止はhost contract testで実施した。常設GPU smokeはAI起点の未承認提案として残す。
範囲はモデルごとに約束範囲の実機runを1本、費用はGPU時間と保守、期限は将来のユーザー判断時であり、
本作業の完了条件や公開CIには追加しない。


[完了計画](../../../../plans/archive/2026/09/11-20/force-baseline-reference-oracle.md) ·
[T2約束範囲](../../../../development/force-baseline-reference-oracle.md)
