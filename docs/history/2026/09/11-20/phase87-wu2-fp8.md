# Phase87 WU2: R9700 FP8 W8A8 decode projection

## 状態・受入条件

完了（2026-09-20）。開始時HEADは`.local-artifacts/phase87/wu2/initial-state.json`へ保存。
R9700の現行hipBLASLtに対し、native packed FP8変換を使う専用GEMV（C1）と、
同じactivation quantizerを融合するGEMV（C2）を比較する。過去のID92移植とrank再選択は行わない。

- Qwen3.8のFP8 8形状、M1〜3、非整列K/N、独立FP32 oracle、finite/repeat/guard/cleanupを確認する。
- 既存quantizerをcontrolとして実呼出しし、C2のscale／encoded activationも照合する。
- 300 ms継続warmup、同一process AB-BA-AB、各round9 samples。quantizer、matmul、合計を分ける。
  重みを512 MiB以上のpoolで巡回し、大形状では1行列だけでもpool条件を満たす。小さい境界のnumerical専用caseは除く。
- 探索打ち切り線は2.0406 ms/token（weight read余地4.0813の半分）。採否はこれと分け、
  全roundの改善方向、通常TPOT比1%以上の短縮、他M/shapeの非退行、数値N0/N1を確認する。
- 233論理matmulに対しactivation quantizerは185回/token。GDN qkv/zの48組はquantizerを共有するので、
  C2の集約で量子化削減を二重計上しない。片側だけ採用したpairでは残る量子化を差し引かない。
- 両GPU単体と、通常8192/128のMTPなし／あり（1 warmup＋3 measured）を記録する。
  数値変更・最初のtoken分岐を数値台帳へ残す。WU2以降の別作業単位へ範囲を広げない。

現行sourceのhipBLASLt controlはM1で形状ごとのrank7/8/9（lm_head等はrank0）、M2〜3で
pinned libraryのalgorithm indexを使う。古い履歴のrank7固定という説明へ戻さず、現在の選択をそのまま測る。

計画: [Phase87 WU2](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu2-r9700-fp8-w8a8-decode-projectionwu-c1の後に渡す)

## 単体結果と採用範囲

両GPUで本番8形状×M1〜3、境界27件、相殺6件、zero 2件の各59ケースがPASS。
独立FP32 oracle、全出力finite／repeat、出力canary、quantizer bitwise、cleanupを確認した。
詳細値とsource／binary／raw digestは[結果JSON](phase87-wu2-fp8-results.json)に保持する。

C1（native変換＋dword GEMV）はR9700のM1全形状で現行Ltより遅い。
C2（量子化融合）はN1024だけ短縮したが0.18944 ms/tokenに留まり、通常TPOT 51.05918 msの1%に届かない。
C2の同形状は共有qkv/z pairではないので、ここには共有quantizerの二重計上はない。
C3としてC2のnative FP8 dot4 coreを量子化から分離し、16-byte読み出しと4独立accumulatorを用いた。
これは3番目の候補familyであり、Lt rank再選択や既棄却ID92の再試行ではない。
初期のscalar dword読み出しからvector16へ一度改良した後、sourceを固定して全matrixを測定した。
初期screenの値を最終採否へ混ぜていない。

探索版の合計短縮は3.361940 ms/token。以下はproduction kernelを直接リンクした最終確認の値。

| K | N | 回数/token | control dot ms | C3 dot ms | 短縮 ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5120 | 10240 | 48 | 0.112041 | 0.104681 | 0.353280 |
| 6144 | 5120 | 64 | 0.067880 | 0.058640 | 0.591360 |
| 5120 | 17408 | 16 | 0.213083 | 0.148802 | 1.028496 |
| 5120 | 6144 | 48 | 0.068801 | 0.068520 | 0.013488 |
| 5120 | 12288 | 16 | 0.136521 | 0.108481 | 0.448640 |
| 17408 | 5120 | 8 | 0.165642 | 0.148841 | 0.134408 |
| 5120 | 1024 | 32 | 0.029040 | 0.017760 | 0.360960 |
| 5120 | 248320 | 1 | 2.342748 | 2.003623 | 0.339125 |

全8形状でAB／BAの3roundすべて改善方向。合計**3.269757 ms/token（通常TPOT比6.4039%）**で、
採用基準1%と探索打ち切り線2.0406 ms/tokenを上回る。対象をexact gfx1201、OCP E4M3 FN、上記M1形状へ限定する。
M2〜3は多くの形状で回帰したためLtを維持し、gfx1030、gfx942、他形状、FNUZは変更しない。
K5120/N6144の効果は小さいが全roundが正であり、共有qkv/zとも同じproviderを選択する。
C3はquantizerを削除しない。個別probeのquantizer込み合計差は3.410020 ms/tokenだが、採否には
共通quantizerの計測揺れを全て除き、233演算のdot部分だけを加重集計した3.269757 ms/tokenを使う。
実モデルでは185回の共有を維持し、48 pairのquantizer削減を二重計上しない。

## 数値分類 N1

実数式は `BF16RNE((Σ_j A_fp8[j] W_fp8[j]) s_a s_w)`。
activationのamax、scale、FP8丸めは現行quantizerをそのまま使い、weight／KV／sampling recipeを変更しない。
各laneが4つの独立accumulatorへnative DOT4を加え、4 subtotalのbalanced和、wave32の5段shuffle和、
FP32 outer scale 2乗算、BF16 RNEの順に計算する。

[AMD RDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
の`V_DOT4_F32_FP8_FP8`はFP32 accumulatorへ4積を加え、round-to-nearest-evenを用いる。
有限E4M3FNの値と積はFP32で正確に表現できる。最大の非scale和も`448²×17408=3,493,855,232`でoverflowしない。
非zero積は最小2^-18の整数倍なので、このdotの入力積・相殺でFP32 subnormalは生じない。

`u=2^-24`、`γ_d=du/(1-du)`、`B=ceil(K/512)`とする。
DOT4あたり保守的に4加算丸め、subtotal合成3加算、wave合成5加算と数えて、
scaleを含むforward absolute error boundは`γ_(4B+10) |s_a s_w| Σ|A_j W_j|`。
最終BF16丸めは共通項として別に加える。hipBLASLtのFP32 compute contractに対する標準の
K項dot＋2 scale bound `γ_(K+2)`と比べ、対象K=5120/6144/17408ではそれぞれ
`γ_50 / γ_58 / γ_146`対`γ_5122 / γ_6146 / γ_17410`となり非増加である。
これは標準RNE、有限入力・scale、scale適用時にoverflowがない領域での解析であり、
未知のLt内部加算順、出力ごとの精度改善、token一致を主張しない。scale演算のunderflow等は共通の絶対誤差項として扱う。
量子化の丸めstage・入力項を欠落させず、dotの加算順のみを変更するためN1とする。

## Production統合・モデル計測

ID103 `matmul.fp8.outer.gfx1201.dot4.v1`として、共有qkv/z量子化、graph eligibility、dispatch auditを含めて統合した。
公開APIはgfx1201の49ケース、gfx1030の25ケースが全て0 ULPでPASS。
混合符号・振幅の独立oracleは正525／負499／zero 0の非自明な出力を確認した。
共有projectionのdirect／shared／graph再実行、変更activation、repeat、cleanupも両GPUでPASS。
両target release build、host 2 tests、Rust format、CIのH3／public-runtime／RMSNorm 3 validatorsがPASS。

通常8192/128、各1 warmup＋3 measuredの中央値は次のとおり。全構成がHIP、fallbackなし、finite、cleanup zero。

| GPU | MTP | control tok/s | C3統合後 tok/s | 変化 | 最初のtoken分岐（0始まり） |
| --- | --- | ---: | ---: | ---: | --- |
| V620 | なし | 15.8974 | 15.8925 | −0.03% | なし |
| V620 | あり | 28.8898 | 28.9344 | +0.15% | なし |
| R9700 | なし | 19.5851 | 21.4268 | +9.40% | 15 |
| R9700 | あり | 33.9536 | 34.4633 | +1.50% | なし |

同provider内の全反復の生成列は一致した。MTP受理数はV620 77/102、R9700 75/106で前後一致。
R9700のgraph span数は128を維持し、新ID103のdirect dispatchを監査で確認した。
V620は新IDを選択せず、両MTP設定の生成列・text hashは前後一致する。
R9700のMTPなしは生成trajectoryが異なるため、モデル全体の+9.40%をkernel単体短縮そのものとは扱わない。
採用判断は固定入力の単体3.269757 ms/token（TPOT比6.4039%）に基づく。
品質controlは独立oracleと固定model/input/samplingの比較であり、perplexity／全model高精度品質評価は未実施。

C1/C2は不採用、C3は上記M1範囲にN1として採用し、WU2を完了する。新しいrollback環境変数は追加しない。
比較前HEAD `7090673be96f8921f11f5e26f9c00d4d12ef0ba1`と保存binaryを基準とし、Gitで導入差分を戻せる。


## 検証中に修正したテストと証拠対応

混合符号の新CPU oracleは当初scaleを各積へ掛けてから加算し、相殺で微小残差を作っていた。
FP8積のFP32和へ最後にouter scaleを掛ける契約に合わせて修正した。許容誤差は変更していない。
同時にhost oracleのE4M3FN subnormal単位を2^-9へ訂正した。
周期的な入力で全出力が相殺しないよう混合fixtureを整数hashで生成し、正負両方の非zero参照値を必須とした。
lm_headの約1.27 GB uploadは公開APIの単転送上限を超えたため、テスト転送を64 MiBずつに分割し、最終lm_headもPASSした。
これらをkernel数値不正として隠さず、初回ログをローカルに保持する。

探索probeは変更前sourceを`probe-source/`へ保存した。production単体は`production-probe-build-identity.json`で
`SLLM_PHASE87_WU2_PRODUCTION=1`、実kernel source、binary hashを結び、probeにリンクしていないruntimeの証拠と分ける。
公開API／modelにリンクしたruntime、qkv/z、graph、public header、benchmarkとbinaryの対応は
`production-build-identity.json`へ保持する。レビューは1回行い、correctness/security blockerなし。
証拠の対応範囲を追記し、別々のbuildを同じidentityとして扱わない。
