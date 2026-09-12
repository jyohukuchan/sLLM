# Phase84.5: MTP経路の限定診断

状態: 限定診断完了（2026-09-12）。通常経路・診断対照・過去証拠の範囲を区別する。公開commitのCIは最終公開手順で確認する。

## 固定した診断範囲

Qwen3.8 27B NVFP4 target、BF16 MTP専用重み（共有FP8 head）、MXFP8 E4 KV、T1/P.95/K20、幅2、V620 gfx1030／R9700 gfx1201。短いcoding／日本語の固定履歴で、接続仕様、通常実行と逐次参照、採用0/1/2個の次状態を調べる。採用率や速度の改善は条件にしない。

| 項目 | 状態 | 証拠 |
| --- | --- | --- |
| MTP hidden/token/position/norm/head契約 | 静的照合で明白な不整合なし | 下記固定artifactと参照実装。GPU数値証拠とは別 |
| 固定履歴のMTP hidden／同一token q | 両GPUで一致 | r3、hidden exact、q最大誤差5.12e-8 |
| target M3／M1 | V620 exact。R9700はattention加算順の数値差 | r3既定のscreen超過を保持。r7のprefillを維持した同一演算対照でexact |
| 採用0/1/2個の復元と次proposal／target | 同じ演算条件で一致 | V620 r3、R9700 r7、通常R9700のcompanionも一致 |
| 固定p/q・補正・境界 | 既存証拠を対応付け、不足support検査を追加 | 両GPU p/q、checkpoint、R9700 support 6/6 |
| host／build | PASS | 診断host 2/2、Clippy、fmt、両target release build |

作業計画: [Phase84.5](../../../../plans/archive/2026/09/11-20/phase84-5-mtp-path-correctness.md)。

## 固定artifactと接続仕様の照合

対象は`unsloth/Qwen3.8-27B-NVFP4` revision `57926baca9a82b4d6906b43f2750d55315f5b10f`。configは`Qwen3_5ForConditionalGeneration`／`qwen3_5_text`、hidden5120、64層、MTP1層である。最初の調査でQwen3.5-27B lockをQwen3.8固有証拠として扱いかけたため、実artifact metadataへ訂正した。

vLLM revision `756794a9a7f08900c00fbfaa6d8332631503f528`の[設定対応](https://raw.githubusercontent.com/vllm-project/vllm/756794a9a7f08900c00fbfaa6d8332631503f528/vllm/config/speculative.py)と[MTP構造](https://raw.githubusercontent.com/vllm-project/vllm/756794a9a7f08900c00fbfaa6d8332631503f528/vllm/model_executor/models/qwen3_5_mtp.py)から、当該architectureのMTP対応を確認した。embeddingとhiddenを各RMSNorm（1＋weight）に通し、embedding→hiddenの順でconcat、FC、MTP decoder、最終norm、共有headへ進む。MTPへ渡すtarget hiddenは最終norm前である。外部実装のcodeは移植せず、契約の比較だけに使用した。

llama.cpp revision `718f7b4175bf8b6af6f5eac09fee10754b3ecddd`の[投機制御](https://raw.githubusercontent.com/ggml-org/llama.cpp/718f7b4175bf8b6af6f5eac09fee10754b3ecddd/common/speculative.cpp)も参照した。prefix先頭tokenにはzero hidden、以後のtoken x[i]にはtarget hidden h[i−1]を対応させる。prefix長nの次はpending token x[n]とh[n−1]からdraftを開始し、同一proposal内ではMTP出力hiddenを次のdraftへ再帰入力する。target検証はpending＋draft2個の3行で、次サイクルのseedは確定したtarget hiddenへ戻す。保持するdraft KVをすべてtarget hiddenから作り直す別方式を参照実行と混同しない。

幅2では確定target入力1/2/3行が採用draft0/1/2個に対応する。MTPは0個採用なら未確定行をrewind、1個なら既存の確定prefixを保持、2個なら最後のdraftをstate-onlyで追加する。token位置は各requestのcommitted lengthと対応する。静的な構造・indexの照合結果に限定し、GPU上の数値・状態が一致した証拠にはしない。

## 再利用する既存native証拠

[p/qとcheckpointのsource・binary対応](phase84-5-native-evidence-reuse.json)を記録した。両GPUの固定K20 p/qは独立long-double oracleで幅1/2/3/8、採否・補正・異常supportを確認済み。linear checkpointはM3の確定prefix1/2/3行、次M1、rewindを確認済みである。sourceの整形差はPhase83.5 closeout対応を参照し、各probeの現在のbinary hashと実行ログhashも確認した。過去のnative演算証拠を再利用するもので、今回の実モデル経路を再実行した証拠ではない。

R9700では未実行だったsupport構築oracleを、保存済みgfx1201 binaryで追加実行して6/6 PASSした。kernel/test/support headerのsource hashは保存buildと現行で一致する。top20→top-p .95、非整列語彙長、mask/additive/ties、全mask／nonfinite拒否を含む。これは小さい数値fixtureの実GPU検証であり、実モデルの候補分布照合は別途行う。

## 初回診断 r1（未通過）

両targetの診断binaryは同じsource before/afterでbuild成功。V620はtarget block照合を通過後、旧private KV readbackがFP16だけを受け付けるためstatus272で停止した。これはMXFP8の状態破損を示す結果ではなく、診断入口の非対応である。既存state image exportの4planeを利用する診断へ変更する。

R9700はtargetのM3 blockとM1逐次実行でargmaxは一致したが、hidden最大BF16距離32816／logits32791で初期screenを超過し停止した。近zeroの符号変化を含み得るULPだけでは差の大きさを判断できず、絶対差・相対差・境界値の記録を追加する。3ULPはGDN小fixtureに由来する暫定screenで、full-modelの正当化済み許容差ではない。この結果を正しさPASS、モデル能力由来、またはMTP接続不良の確定とは扱わない。

R9700 serviceは元unit/binary/configで復元し、health/ready200、既存hash一致を確認した。raw結果は`.local-artifacts/phase84-5/diagnostic-gfx1030-r1/`と`diagnostic-gfx1201-r1/`へ保持する。

## 4plane診断 r2

V620は2入力、draft2行、target M3対M1、採用0/1/2個のtarget/companion状態と次計算をPASSした。MTP hiddenと同一tokenのlogprobはexact、KV value/scaleも一致した。

R9700はMTP hidden・同一token q・companion KV/次hiddenが一致した一方、target M3対M1はcoding hidden最大絶対差5.1875／logits5.5625、Japanese26.5／3.45819だった。targetの採用0/1/2でもKV/hidden差が残り、r2はFAILのまま。prefix checkpointはM3中間状態を保持するので、M3とM1の数値差がある場合は復元後にも同じ差が残る。状態管理の欠陥とはまだ断定できない。

r2入力はchat templateのprefixを17/129tokenへ切り出しており、最初のdraftは両入力でnewline等へ確率が集中した。MTP分布の一般的な検証として弱いため、以後は実際のcode／日本語本文を含む固定入力へ修正する。先行結果は削除せず条件と限界を保持する。

## R9700の小行数matmul対照 r2

同じr2 binary／入力で`SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201=1`を指定した。target hidden／logitsの最大差とlogits hashは既定経路と一致し、FAILは解消しなかった。この切り替えだけでは原因を説明できず、実際のprepared経路への適用と他演算を調査する。結果は`.local-artifacts/phase84-5/diagnostic-gfx1201-rowgrid-r2/`に保持した。serviceは元unit／binary／configへ復元し、health／ready200とhash一致を確認した。

codingの3行目は通常softmaxで逐次参照→blockのKL約0.152、TV約0.211だった。これは固定K20/P.95とは異なる補助観測であり、合否閾値やモデル品質評価ではない。argmax一致だけでは分布差を無視できないため、数値差を未解決として残す。

## 本文fixture r3

入力をraw Rust code 127tokenと日本語本文129tokenへ変更し、128token chunk境界の両側を含めた。各requestのprefill hidden hashも記録して、比較開始時の一致を確認する。

V620は全ケースPASS。target M3/M1のhidden／logitsはbit一致し、採用0/1/2個のKV value/scaleと次計算も一致した。MTP hiddenはexact、非退化したcoding draftの同一token logprob誤差は最大4.77e-8。独立CPU乱数器の抽選token一致は合否条件ではない。hostのBF16差分／固定分布oracle検査は2/2 PASS。結果は`.local-artifacts/phase84-5/diagnostic-gfx1030-r3a/`に保持した（`r3`ディレクトリはrunner引数誤りで作られた空ディレクトリであり、GPU実行証拠ではない）。

R9700のr3ではprefill hiddenが全requestで一致し、MTP hiddenもexact、同一token q誤差は最大5.12e-8だった。一方targetはrow1以降に差があり、coding hidden最大8／logits0.75390625、日本語3.125／1.00390625。r3既定とshared-activation無効化対照は同じtarget logits hashで、両方FAILのまま。MTP接続／samplingの不一致は検出していないが、target数値差の原因をまだ確定していない。

追加のsource照合では、旧Qwen3.5-4BのGDN bundleはK2560限定のため本artifactに適用されないと確認した。短prefixではstaged32 attentionのKV長1024以上という条件も満たさず、実際はR9700のM1でwave、M3でgenericのattention演算になる。適用されない候補を追加実測の対象にしない。

attentionの演算順を切り分けるr4は、診断用buildだけでgfx1201 wave選択を無効化してM1/M3ともgenericにした。元sourceはbuildのbefore/after hash一致を確認後に復元した。この一時変更は本番修正・既定採用ではなく、通常経路のGPU PASSへ読み替えない。patchとbuild identityは`.local-artifacts/phase84-5/attention-control/`と`candidate-gfx1201-attention-control-r4/`へ保持した。

## 再実行入口

通常の`sllm-phase78-qwen38-benchmark`に`SLLM_PHASE84_5_DIAGNOSTIC=1`、`SLLM_PHASE78_TARGET`、`SLLM_PHASE78_DEVICE`、`SLLM_PHASE78_MODEL_PATH`を指定する。対象GPUのrelease buildを使い、`ROCR_VISIBLE_DEVICES`でGPU UUIDを固定する。KV既定はこの診断ではMXFP8 E4、MTPはBF16限定で、量子化companion指定は拒否する。任意の`SLLM_PHASE84_5_DUMP_DIR`はraw差分の保存先であり、生成物はtrackしない。通常R9700では本履歴の数値screen超過により非zero終了するため、これを新しい普遍的CI gateとして使わない。

独立attention probeは`native/hip/tests/phase83_mxfp8_prefill_gpu_test.cpp`を対象targetの通常native archiveとlinkし、`--decode-block-parity`で8ケースだけを選択する。完全なbuild引数とbinary／archive hashはcompact reportに含める。引数なしの従来prefill suiteは維持した。

## 最終判定

**限定範囲でMTP接続・固定sampling・状態復元の不整合を検出せず、診断を完了する。** 本番kernel、provider既定、MTP重み、sampling profileは変更しない。

r4はnativeだけ演算を変更したためRust側のexact-provider検査でstatus287となった。数値証拠ではなく、失敗として保持する。r5ではnative選択とRust側の想定providerをともにgenericへ揃え、target／symbol／fallback検査自体は維持した。R9700の2入力でtarget M3/M1のhidden・logits、採用0/1/2個後のKV有効payload・次target計算がすべてbit一致した。MTP側の復元・次proposalも一致した。これによりr3の差をattentionのM1 wave／M3 genericという演算順の違いへ切り分けた。一時patchは復元済みで、本番の演算選択変更として採用していない。

r5はprefillのwave選択も無効にしていたため、それだけを最終の因果対照とはしなかった。r7ではM2〜M4のattentionだけwaveへ変更し、prefillとM1は通常providerのまま保持した。両入力でr3とprefill hidden、逐次target logits、draft tokenがexactのまま、M3のhidden／logitsと採用0/1/2後のKV・次計算まで差0となった。このprefillを維持した対照を最終の切り分け根拠とする。r7のnative／Rust双方の一時patchも復元済みである。

通常nativeを再buildした独立attention probeでは、prefix127/129それぞれのM3と対応するM1×3、計8ケースを実行した。同一query／KVで各演算を既存host oracleへ照合し、全ケースでBF16差0、fallbackなし、正常終了を確認した。M1 wave／M3 genericのdispatch metadataも確認した。小fixtureで丸め差が現れなかった事実を、実モデルでも差がないという主張へ拡大しない。

最初の3ULP full-model screenはGDN小fixtureを流用した診断上の仮置きで、計画が要求する適用可能な演算oracleの許容差ではなかった。閾値は緩めず、r2/r3のFAIL記録も変更しない。最終判断は接続仕様、通常MTPの同一履歴照合、固定p/q oracle、通常attentionの独立oracle、同一演算での状態復元対照を組み合わせたものとする。r3の数値screen超過をGPU PASSへ読み替えない。

確認したのは短い2入力・幅2・single GPU・対象artifactに限る。R9700の既定MTP on/offで同じlogits／出力になること、full-model BF16比で品質劣化が同程度であること、採用率がモデル能力だけで決まることは証明していない。数値差の品質・採用率への影響は未評価であり、今回の診断を速度改善や品質改善の根拠にしない。

証拠: [実モデル・対照・attention oracleのcompact report](phase84-5-full-model-path-evidence.json)、[再利用native証拠](phase84-5-native-evidence-reuse.json)。最終sourceとr3の差は診断コードのlifetime省略・局所lint属性と独立attention test追加だけで、通常推論のsourceは同一である。GPU実行後はR9700 serviceの元unit／binary／configへの復元、health／ready200、hash一致を各leaseで確認した。累積reviewでは重大な阻害事項なし。M1/M3差分は観測値として残し、各演算の独立oracleと区別する。

計画: [Phase84.5](../../../../plans/archive/2026/09/11-20/phase84-5-mtp-path-correctness.md)。

公開前H0は627件を実行し、C++ testの整形1件だけ失敗した。clang-format-18で修正し、同じC++整形validatorを再実行してPASSした。probe sourceの変更は空白のみであることとhash対応を記録し、GPU数値証拠を再利用した。Rust fmt／Clippy、診断host 2/2、Markdown local linksもPASS。公開後のclean commitに対するH0/H1/H2およびcompile-only CI結果は当該commitのchecksを正とする。
