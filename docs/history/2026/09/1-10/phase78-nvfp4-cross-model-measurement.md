# Phase 78後の他モデルNVFP4測定

2026-09-05、ユーザー指示「NVFP4重みの場合は？」により、既存Gemma 4 12BのNVFP4／FP8混合GGUFを測定した。
他モデルでも共通経路によるprefill高速化は観測した。一方、旧版との生成token差があり、品質維持を満たした汎用採用とは扱わない。
新しい実装や既定selectorの変更は行っていない。Phase78の完了判断・目標緩和も維持する。

## 条件と再計画

- locked `phase20-final-gemma4-nvfp4.gguf`、FP16 KV、単一要求、greedy、EOS無視。token 23066の17個／512個を入力し17／32個出力。
- 旧版はPhase75 V620／Phase74 R9700、現行はcommit `40ab582b049cff7effadbca75fe951d6cef5bd96`のCLI。
  [前回測定](phase78-cross-model-measurement.md)と同じbinary hashを再利用し、今回のモデルを新たに実行した。
- 通常は旧版→現行版→旧版、各1 warmup＋3 measured。旧版は前後6測定、現行は3測定の中央値。
- V620 512/32の最初の旧版runが640秒かかったため、同条件は旧版3測定・現行3測定で比較し、2回目の旧版を省略した。
  スケジューラだけを一時停止し、進行中のGPU計算がexit0・model cleanup0で終了してからスケジューラを終了した。
  GPU計算のtimeout／強制終了によるPASSではない。rawの`v620-replan.json`と`finish-v620.log`へ経緯を保存した。
- 主V620は03:00.0、R9700は07:00.0をUUIDで単独可視化した。追加V620-B（43:00.0）の切り分けは同じB上の条件間だけで比較した。
- clocksは変更せず、別GPU間では同時実行。Gemma CLIはprefill chunk指定を受け付けないため指定していない。

## 旧版全体と現行既定の比較

速度単位はtok/s。倍率はwhole-model観測値であり、NVFP4演算だけの倍率ではない。

| GPU | 入力／出力 | prefill 旧→現行 | decode 旧→現行 | token一致 |
| --- | --- | --- | --- | --- |
| V620 | 17/17 | 2.703→13.098 (+384.61%) | 1.621→1.627 (+0.37%) | 一致 |
| V620 | 512/32 | 3.824→18.945 (+395.50%) | 1.543→1.540 (-0.21%) | 不一致 |
| R9700 | 17/17 | 20.516→18.179 (-11.39%) | 10.922→10.921 (-0.00%) | 不一致 |
| R9700 | 512/32 | 20.572→24.460 (+18.90%) | 8.718→8.764 (+0.53%) | 不一致 |

全主比較でstopと記録されたauditは一致したが、token列はV620短文だけ一致し、残る3条件は不一致だった。
旧版の前後runがある条件では旧版同士のtoken列は一致した。audit一致は記録された件数／backend等の範囲で、全kernel symbol traceの一致ではない。

## 共通経路とNVFP4だけの切り分け

GemmaのNVFP4はW4A4であり、prepare時に共通`select_nvfp4_w4a4_variant`を使う。旧版のID11から、
現行既定ではdecode ID58／prefill ID59へ変わる。`SLLM_NVFP4_W4A4_FORCE_BASELINE=1`で旧ID11へ戻せる。
この既定の共通NVFP4経路は、先の「共通改善は主に4系統」という説明で挙げ漏らしていた。

GemmaのFP8重みもouter-dimensionのE4M3FNであり、V620のprefillは旧emulationから現行ID71へ変わる。
従ってV620の約4.85～4.96倍はFP8側の改善込みである。R9700のFP8は両版ともnative providerである。
この経路説明は公開ソースのselectorに基づき、今回per-kernel profilerを追加したものではない。

V620-Bの同一GPU・現行binary・17/17比較では、NVFP4だけ旧経路へ戻すとprefillは11.812 tok/s、
既定は12.719 tok/s（約7.69%増）、decodeは1.585／1.570 tok/sだった。両条件の生成tokenは一致した。
R9700短文では同じswitchによりprefillが20.439 tok/sへ戻り、生成tokenも旧版と完全一致した。
R9700のtoken差はNVFP4経路変更に起因すると切り分けられるが、差の許容性・品質は別途未検証である。

## 汎用DP4A opt-inの追加観測

`SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A=1`はモデル名に依存せず、M>1、Kが16の倍数で選択できる。
Gemmaの主要NVFP4 MLP K=3840／15360は条件を満たす。コード変更なしで次を観測した。

| GPU・同一GPU内比較 | 入力／出力 | 現行既定prefill | DP4A prefill | 倍率 |
| --- | --- | --- | --- | --- |
| V620-B | 17/17 | 12.719 | 79.902 | 6.28x |
| R9700 | 17/17 | 18.179 | 119.110 | 6.55x |
| R9700 | 512/32 | 24.460 | 423.371 | 17.31x |

DP4Aのtoken列は既定／旧版と不一致だった。decodeはV620-B約1.57、R9700短文約11.24、512入力約8.81 tok/sで、
prefillの倍率をdecode改善へ一般化しない。既存のopt-inを測っただけで、既定採用・品質承認・モデル固有最適化を追加していない。

## 検証の限界と証拠

計17 GPU run、17 warmup＋51 measured要求はexit0で完了し、run内の反復token一致、HIP-only、fallback false、
dispatch数>0、request／workspace／model cleanup0を確認した。各CLIの`state=PASS`はこのrun内確認の範囲であり、
cross-version token一致や独立数値oracle／品質評価のPASSへ昇格しない。新しいlogits oracle、品質corpus、
物理DRAM bandwidth／offloadの検証は行っていない。旧／現行の累積順序差はコード上の説明であり、品質無影響の証明ではない。

raw・argv・stderr・hashは`.local-artifacts/phase78-nvfp4-cross-model/`および
`.local-artifacts/phase78-nvfp4-cross-model-v620b/`へ保存した。中央値、min/max/MAD、モデル／binary／raw hash、
各armのtoken一致範囲を[集約記録](phase78-nvfp4-cross-model-evidence.json)に残す。

[完了計画](../../../../plans/archive/2026/09/1-10/phase78-nvfp4-cross-model-measurement.md) ·
[先行MXFP8測定](phase78-cross-model-measurement.md)
