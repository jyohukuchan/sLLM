# Phase 87 WU-2V: V620 FP8 decode残件

## 着手時の範囲と根拠（2026-09-25）

WU2はR9700 `gfx1201` のM=1 native FP8を採用して完了した。段階2の残件はV620 `gfx1030` の
software FP8 W8A8 decodeである。現行M=1の本番projectionはID82、M=2〜3のverifyはID92、
lm_headはID68を使う。token毎のactivation quantizerはprojectionと分けて測る。

WU0のwarm read計測に基づくV620 FP8 8形状の正の余地は1.05206 ms/token、探索継続線は
0.52603 ms/token。最大の余地は`(M,K,N)=(1,6144,5120)`の0.68118 ms/tokenで、
現行は5.55197 ms/token、read参照は4.87079 ms/tokenである。反対に`(1,5120,6144)`の
余地は0.08785 ms/tokenにすぎない。このため、未専用化の後者へwrapperを追加するより、
前者のweight address生成を調べる。

最初のC1はwave内で同一のweight column baseを明示的にscalar化するtest-only候補とする。
現行production ID82をcontrolにし、生成ISA／resourceと同一process AB-BA-ABを比較する。
候補が数値一致しても、同等のISAが既に生成されている場合または0.34059 ms/tokenの
形状別探索線に届かない場合はC1を打ち切る。この線は採否条件ではない。
本番採用の前には公開dispatchとQwen3.8通常8192/128、MTPなし／ありを確認する。

現行Stage3の`gfx1030` lowp code objectで、対象production symbol
`sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k6144n5120_v1`は
VGPR 170、SGPR 26、LDS 544 B、spill 0、wave32である。既存の`hipFuncGetAttributes` probeは
active blocks 2を報告した。ISAではweight引数がSGPR `s[12:13]`にある一方、
block／wave／columnとlaneのoffsetをVGPRで合成し、最終`global_load_dwordx2`は
VGPR addressを参照している。従ってwave共通baseの明示的なscalar化は構造上の差を作れるが、
速度改善は実測で判定する。対象objectは追跡対象外の
`.local-artifacts/phase87/stage3/cargo-gfx1030/release/build/sllm-hip-sys-0f6f9ebc80964eda/out/native-hip-build-gfx1030/lowp/CMakeFiles/sllm_lowp.dir/src/lowp_kernel.hip.cpp.o`。

既存probeの差分調査では、後続C2にactivation-shared wave4／4-columnを選んだ。
現行と同じ32 columns/workgroupでactivationを一度FP16へ展開して8 waveへ共有するが、
K6144では12,288 Bのdynamic LDSとbarrierが増える。現行ID82のLUT decodeへ揃えてから比較し、
probe既往値を採用根拠へ転用しない。direct-wave8はID68対照のprobeであり、現在のID82に対する
数値・性能のidentityがないためC2へ先行させない。

前回のWU2と現在の[計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-2v-段階2のv620-fp8-decode残件2026-09-25着手)を比較対照とする。

## 実施結果

### C1: wave共通weight baseの明示scalar化

test-onlyの[専用probe](../../../../../native/hip/tests/phase87_stage2_v620_probe.hip.cpp)と
[runner](../../../../../ci/tools/run_phase87_stage2_v620.py)を作り、現行production ID82の
`launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32`を直接対照とした。
production quantizerと同じactivationを使い、C1は64-bit weight column baseを
`__builtin_amdgcn_readfirstlane`でwave一様にした。FP8 LUT、読み出し順、4列の累積順、
外側scale、BF16 RNEを現行と同じにした。これは本体selectorへは接続していない。

canonical V620 `GPU-76a08c022586fed6`に限定し、pattern 0（混合符号）、1（相殺）、
2（zero）を各々独立FP32 oracle、quantizer bitwise、全5120出力のcontrol bitwise、finite、
repeat、output guard、cleanup 0で確認した。3条件すべてPASS、最大0 ULP。
weight poolは566,231,040 Bで、300 ms継続warmup後、同一process AB-BA-AB各9 samplesを取得した。
候補のcode objectはVGPR 169／SGPR 26／LDS 544 B／spill 0、現行は170／26／544 B／0。
`hipOccupancyMaxActiveBlocksPerMultiprocessor`の256 thread queryでは候補2 active blocks。

| pattern | control dot ms/call | C1 dot ms/call | 64回換算の短縮 ms/token | control込み総時間 ms/call | C1総時間 ms/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.089001 | 0.088681 | +0.02048 | 0.107481 | 0.106881 |
| 1 | 0.089040 | 0.088881 | +0.01018 | 0.107761 | 0.107561 |
| 2 | 0.089961 | 0.089601 | +0.02304 | 0.107961 | 0.107881 |

pattern 0のdot各roundの64回換算短縮は+0.02048、+0.04096、−0.04608 ms/tokenで、
全round改善ではない。C1の改善は形状別探索線0.34059 ms/tokenから大きく離れ、
量子化込み総時間も測定揺れの範囲なので**C1を打ち切り、本番へ採用しない**。
これはC1という仮説の終了であり、V620 FP8段階2全体の終了ではない。

最初のr1は計測後にbinaryが上書きされ、`execution.json`記載SHAとの対応を再確認できなくなった。
同じsourceでr2を再実行し、計測binaryの現在SHAとreport記載SHAの一致を確認した。
上表の実測原票は追跡対象外の`.local-artifacts/phase87/wu-2v/c1-gfx1030-r2/`に保存した。
`execution.json` SHA-256は
`2945735e4960d71575443026f7f09d966d41a51915d9f5916778df0e9290c0b2`、
probe binaryは`3370733417d1b8acdeb68a6962925c36a49124a9a97393579f985e5bed6e7e05`。
runnerはsource／lowp archiveの前後SHA一致、exact UUID、GPU実行、fallbackなし、全run exit 0を確認した。
GPU性能levelはrunnerが変更していない。Qwen serviceは実行前に停止を確認した。

### C2: activation-shared wave4／4-column

[専用probe](../../../../../native/hip/tests/phase87_stage2_v620_c2_probe.hip.cpp)で、
M=1 K6144のFP8 activationをID82と同じLUTからworkgroupあたり一度だけFP16へ展開し、
8 waveで共有した。weightの2-chunk先読み、4-column、FP32 dot順、BF16 RNEは維持した。
候補resourceはVGPR 144、SGPR 26、総static LDS 12,832 B（activation配列12,288＋LUT 544）、
runtime dynamic LDS 0、spill 0、256 thread queryで3 active blocks。
raw identityの`shared_bytes=12288`とrunnerの同値検査はactivation配列の論理byteだけを表す。
canonical V620のpattern 0〜2でproduction quantizer、現行ID82との全出力bitwise、
独立FP32 oracle最大0 ULP、finite、repeat、guard、cleanup 0をPASSした。

| pattern | control dot ms/call | C2 dot ms/call | 64回換算の短縮 ms/token | control込み総時間 ms/call | C2総時間 ms/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.089361 | 0.096320 | −0.44538 | 0.107722 | 0.114921 |
| 1 | 0.088881 | 0.095281 | −0.40960 | 0.107481 | 0.113961 |
| 2 | 0.090201 | 0.096361 | −0.39424 | 0.108881 | 0.115081 |

AB-BA-ABの全roundで遅く、activation共有のLDS stagingと追加barrierを相殺できなかったため
**C2は不採用**。runnerはexact UUID、566,231,040 B weight pool、300 ms継続warmup、
各round 9 samples、source前後hash一致、GPU実行、fallbackなし、cleanupを確認した。
実測原票を`.local-artifacts/phase87/wu-2v/c2-gfx1030-r1/`へ保存した。
`execution.json` SHA-256は
`87c14f2629d9a88d5430915919c497cfd1ae438ef305150de6d386ff5a04ff13`、
binary SHA-256は`57f0afe4bdd5d406a097da187248484171f54bfd24faaa9948cce7cdc98e69d2`。
GPU性能levelは実行前後とも`auto`である。

### C3: ID82 LUTを保ったdirect-wave4 pair2

2-chunk先読みを1組ずつのdirect-wave処理へ変え、ID82のLUT、4-column、各列のK順／4 dot順、
外側scaleとBF16 RNEを維持した[専用probe](../../../../../native/hip/tests/phase87_stage2_v620_c3_probe.hip.cpp)を作った。
候補resourceはVGPR 91、SGPR 26、LDS 544 B、spill 0、256 thread queryで5 active blocks。
C1/C2と同じ現行ID82対照と
production quantizerで、canonical V620のpattern 0〜2が全出力bitwise、
独立FP32 oracle最大0 ULP、finite、repeat、guard、cleanup 0をPASSした。

| pattern | control dot ms/call | C3 dot ms/call | 64回換算の短縮 ms/token | control込み総時間 ms/call | C3総時間 ms/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.089361 | 0.120201 | −1.97376 | 0.108041 | 0.140641 |
| 1 | 0.088881 | 0.120801 | −2.04288 | 0.107761 | 0.139802 |
| 2 | 0.090041 | 0.122561 | −2.08128 | 0.108481 | 0.141521 |

AB-BA-AB全roundで約30〜37%遅い。VGPR削減の代わりに先読みを失い、weightの待ちを隠せなくなった
可能性が高いが、原因はcounterで確定していない。**C3も不採用**とし、V620 M1
`(6144,5120)`のこの3候補の探索は終了する。本番kernel／selectorは変更しない。
候補sourceのC1から残った誤解を招くコメントだけを修正し、r1のbinary上書きも検出したため、
現在sourceでr2を再実行した。上表の実測原票は`.local-artifacts/phase87/wu-2v/c3-gfx1030-r2/`、
`execution.json` SHA-256は
`91c31f1082e9c2e7d62f687791fde84d20417e3a53442322913cc871b0a5ff8c`、
binary SHA-256は`e7466c1fcbda2549709897acbf8d5c2e6705a869d233db4e610dcf803152c2c0`。
occupancy queryのrawは`.local-artifacts/phase87/wu-2v/resource-query/occupancy.json`
（SHA-256 `06eb943af179927e0e26a2c24514edb8ba7d8670d1834fc347af4966cb8a15ad`）。

### 段階2の残りの監査と終了判断

3候補とも数値契約は満たしたが速度を改善しなかった。WU0の同条件warm read参照で、
対象`K6144,N5120`以外のV620 FP8 7形状の正の余地は合計**0.370885 ms/token**である。
全8形状の探索継続線0.526032 ms/tokenを下回る。lm_head `K5120,N248320`は
現行2.613156／read参照2.519058 ms/tokenで、余地は**0.094098 ms/token**。
これらは物理帯域の厳密な上限ではなく、この段階で採用した探索停止の参照値である。
V620のM=2〜3は既存ID92を維持する。WU2では両GPUの8形状×M=1〜3と非整列境界を
数値検査済みで、R9700のM=2〜3は候補が回帰したため採用していない。

token毎のFP8 activation量子化は、段階0時点で185回／1.64 ms/tokenだった。
段階7で88 nodeのFP8 producer融合を両GPUに採用し、MTPなしgraphは1170→1082 nodeとなった。
残りはLinearAttention `input_rmsnorm` 48、GDN out 48、最終lm_head前1の計97 node。
この3群はBF16/FP8の2出力binding、head間row amax、最終行scale aliasという具体的な阻害条件があり、
段階7完了時に[backlog P3](../../../../plans/backlog.md)へ移された。段階2の残作業として
再び新規producer／ABIを実装しない。現行の量子化kernelとFP8 M=2〜3 providerは維持する。

以上により、**段階2のV620残件は新規採用なしで終了**する。C1〜C3はtest-onlyで、
production kernel、selector、graph、ABIを変更していない。通常8192/128のMTPなし／ありを
未変更sourceで再実行して改善を主張することはしない。現在のMTPなし／ありの既定経路は
[段階7の通常計測](phase87-stage7-c1.md)と[WU-3Pの通常計測](phase87-mtp-nvfp4-prefix-prefill.md)に記録済み。
段階2全体は、R9700のWU2採用とこのV620探索終了を合わせて完了とし、次は段階1へ進む。

対応する計画: [Phase 87 WU-2V](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-2v-段階2のv620-fp8-decode残件2026-09-25着手)。
