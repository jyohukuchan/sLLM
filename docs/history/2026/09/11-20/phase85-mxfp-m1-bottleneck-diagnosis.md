# Phase85 M1 MXFP bottleneck diagnosis

## 2026-09-13: V620／R9700 counter and clock diagnosis

M1、K=5,120、N=17,408を対象に、exact `gfx1030`（V620）とexact `gfx1201`（R9700）のraw rocprofv3 PMCを、BF16／MXFP8／MXFP6ごとにfetch、cache、instruction、wait groupへ分けて採取した。各groupはwarmup 100、measured 5、kernel iteration 101..105で、clock probeは各profile 100反復である。集計はcounter analysis（Git管理外: `.local-artifacts/phase85-m1-bottleneck/counter-notes/gate-counter-analysis.json`）とmarkdown view（Git管理外: `.local-artifacts/phase85-m1-bottleneck/counter-notes/gate-counter-analysis.md`）に置いた。counter catalogとtarget別の単位・式はRDNA catalog note（Git管理外: `.local-artifacts/phase85-m1-bottleneck/counter-notes/rdna-counter-catalog.md`）に分離している。

V620のraw VALU周辺では、SALU instruction medianはBF16 6.74M、MXFP6 23.71M、MXFP8 40.02Mで、量子化形式の命令量はBF16より約3.5倍／5.9倍だった。wave32 instruction medianはBF16 44.76M、MXFP6 155.59M、MXFP8 160.03Mだった。これらは命令数でありFLOPsまたはVALU throughput saturationへ読み替えない。集計のMACはM×K×N、FLOPsは2×M×K×Nとして、出力列数・per-output・per-useful-MACを区別した。wave数はBF16のN×8、col2のceil(N/2)×8を個別に照合した。

V620のFETCH_SIZEはBF16 178,268,288 bytes、MXFP6 69,636,352 bytes、MXFP8 91,922,304 bytesの中央値で、catalog定義のcache effects込み外部read proxyはBF16約501 GB/s、MXFP6約161 GB/s、MXFP8約131 GB/sだった。raw 64B read requestが大半で、値はpacked payloadとscale／encoding overheadを含む転送量に近い。これはDRAM/TCCの直接測定ではなく、GL2C/EA proxyである。

clock perturbationでは、peakからmin_sclk（MCLK条件を合わせた比較）へのmatmul中央値比がBF16 2.03、MXFP6 4.15、MXFP8 3.76だった。V620のセンサー実測はpeak SCLK約2,433 MHz、min_sclk約496 MHz、min_mclk約494 MHzで、peak/min_sclkのMCLKは約1,000 MHz、min_mclkのMCLKは約96 MHzだった。min_sclkからmin_mclkへの比はBF16 6.26、MXFP6 1.006、MXFP8 1.004で、SCLK自体はほぼ同じ低値の条件として扱える。この比較は低SCLK条件でのmemory sensitivityであり、通常SCLKでのmemory sensitivityへ外挿しない。BF16ではread proxyが高い一方、MX形式ではSCLK条件で量子化kernelの時間が大きく変わるため、観測は単純な外部帯域律速だけでは説明しにくく、shader-side instruction／issue／waitとclock条件の影響を優先して切り分ける。

R9700のroot PMC probeはGRBM_COUNTとSQ_WAVES以外の選択counterがzeroまたは欠落し、`usable=false`として保持した。R9700についてDRAM、TCP、VALU、VMEM、occupancyの律速をこのPMCから断定しない。counter unavailableはこの方法の制約であり、全体の実行結果を一律FAILへ変換しない。

MTPの解釈では、shared FP8 headを固定した既存MTP条件を維持する。M1 bodyのcounterからMTP companionやMTP全体の律速を帰属させず、MTP headの分離測定を別証拠として扱う。

## Branchless ablation

V620の追加ablationでは、full output SHAがlegacy／col2／branchlessの全比較で一致し、codecはE4M3全256 code、E3M2全64 code、E8M0全256 codeをGPU上で照合し、有限値bitwise一致・NaN class一致で両targetともPASSだった。V620 O geometryではpeak条件でlegacy約219 us、col2約170 usとなり、以前の退行はこの定常条件で再現しなかった。clock・反復条件への感度があり、今回だけでは既存selector guardを変更しない。V620のM1実測では、MXFP8のSALUは約40.028Mからbranchless約6.508M、VALUは約86.719Mから約113.252M、FETCH_SIZEは約89,769.75から約89,766 KiB、kernelは約705.891 usから約354.645 usへ変化した。MXFP6はSALU約23.710Mから約6.929M、VALU約102.667Mから約114.506M、FETCH_SIZE約68,004.25から約68,004.125 KiB、kernel約438.807 usから約369.566 usだった。これはdecode control overheadの仮説を支持するが、SALU pipe saturationや普遍的なshader限界の証明ではない。

R9700のdirect-scale ablationは追加測定済みで、lane0 scale＋shuffleだけをall-lane same-address loadへ置換した比較である。full BF16 output SHAは各pairで一致し、oracleは全件PASSだった。gateは706.426 usから601.265 us（-14.9%）、downは727.486 usから626.185 us（-13.9%）、Oは219.282 usから184.881 us（-15.7%）へ変化した。生成ISAではds_bpermuteが23から20、waitが65から57、scalarが154から133となり、native E4 decode／reductionは不変だった。この結果はscale read/broadcast＋wait chainの部分的コストを示すが、R9700の残るMXFP8／MXFP6差を個別kernelへ帰属させるPMCは使えないため、残るload依存・命令発行・変換の寄与率は未確定である。

## 同条件での演算子時間と適用範囲

実MTPのgate重み `[17408,5120]` と固定synthetic activationを用い、M=1、stock `profile_peak`、
warmup 100＋measured 100のGPU event中央値を比較した。重み量子化は計時外、activation quantizerとmatmulは別計時。
以下はmatmulのみであり、実MTP推論の速度ではない。

| GPU | 形式 | 現行Columns2（BF16は既存M1） | 診断用branchless | branchless＋direct-scale |
| --- | --- | ---: | ---: | ---: |
| V620 | BF16 | 367.08 µs | — | — |
| V620 | MXFP8 | 721.11 µs | 364.71 µs | — |
| V620 | MXFP6 | 441.29 µs | 377.55 µs | — |
| R9700 | BF16 | 325.20 µs | — | — |
| R9700 | MXFP8 | 713.64 µs | 706.11 µs | 601.27 µs |
| R9700 | MXFP6 | 544.56 µs | 514.34 µs | — |

direct-scaleの対照branchlessは同じ測定回で706.43 µs。V620 MXFP8はBF16とほぼ同等で、
0.6%差を有意な優位とは主張しない。MXFP6はまだ約2.9%遅い。gate以外にdown `[5120,17408]`、
O `[5120,6144]` のsynthetic inputでも確認した。44 matmul行、4 codec行、8 ablation PMC行、
6 direct-scale行のprobeはすべて数値検証PASS。同形式のvariant間は全出力BF16 hashが一致した。
有限性は全出力、独立CPU dequantized FP32 oracleは各行7出力点でabs≤0.5またはrelative≤0.02を確認した。

R9700もpeak→min_sclkではSCLK約2324→478 MHz、MCLK約1124 MHz一定で、MXFP8が約3.73倍、
MXFP6が約4.58倍遅くなる一方、512 MiB連続読出し時間は約0.988→1.009 msにとどまった。
V620の同対照も約1.106→1.120 ms。低いlogical GB/sだけで律速を判断せず、このclock対照とablationを根拠にする。

既存public harnessのrepeat eventにはactivation quantizer＋matmulとlaunch間のstream idleが含まれるが、
重みupload／prepare、出力D2H／hashは含まれない。kernel traceでもV620 Oの大差は主にmatmul本体に出ていた。
従ってhost転送だけを短縮して解消する問題ではない。MTP全体ではcompanion 8行列とは別に固定のshared FP8 head、
本体verification、採用数が効く。今回の診断変更を本番へ採用したMTP／KV cache推論の再測定は行っていない。

次の最適化対象はV620の復号制御、R9700のscale共有・待ち経路である。R9700に残るMXFP6 unpack／load依存や
MXFP8の変換・発行の内訳は別の対照実験が必要で、今回の証拠から積和FLOPsの飽和とは断定しない。

## 再現情報と復帰

V620 UUID `GPU-76a08c022586fed6`、R9700 UUID `GPU-a8e9ddefa2d60f55`。
production比較対象は既存 `.local-artifacts/phase85-m1/build-{gfx1030,gfx1201}-c/matmul_kernel.hip.cpp.o`。
診断probe／source／compile recipe／hashは `.local-artifacts/phase85-m1-bottleneck/probe/` と
`probe-identities.json`、job command・環境・stock clock遷移は `prep/` と各 `execution.json`、
数値結果は `probe-reports/`、集約は `ablation-summary.json`、counterは `pmc-ablation/` に保存した。
実MTP gate sliceのSHA256は `e2fa26989b3103cc595990f53c13a295f95507d0a480276e642a084e6b1f62cc`。
raw model slice／生成binary／traceはignoredのまま保持する。

今回の追加変更は診断probeと文書に限定した。本番kernel／selectorの挙動は変更していない。
両GPUのperformance levelは元の`auto`へ戻し、R9700 serviceは元のunit／binary／run.sh hash一致、
healthz／readyz HTTP 200へ復帰した。commit／pushは行っていない。

[診断計画](../../../../plans/archive/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md) /
[メイン計画](../../../../plans/main-plan.md)
