# 最新llama.cppのMTP量子化比較

> 状態: 計測完了（2026-09-14）
> 2026-09-14ユーザー指示: 最新llama.cpp、V620／R9700を使い、様々な条件でMTP量子化がBF16 MTPより速いか測定する。

開始時のupstream masterを `bc52a12b38941b0a690ade65fbc5749715224e30` に固定し、
変更しないsourceからROCm7.14のexact gfx1030／gfx1201 Releaseをbuildした。
本体は固定Qwen3.8-27B UD-Q5_K_XL GGUF、batch1、単GPU、FA on、KV f16、context9216、batch2048／ubatch512。
最新版にはMXFP8／MXFP6形式がないため、BF16、Q8_0、Q6_K、Q4_0のMTPを比較する。

## 比較範囲

1. **専用8行列の比較**: original BF16 GGUFからblk.64の8行列と7 F32 normを抽出し、
   shared embedding／head／output normは固定本体のpayloadを全形式に同じまま入れる。
   8行列だけを量子化する。入力128／2048／8192 × MTP幅1／2／4、出力128の9条件に、
   2048入力／512出力／幅2、2048入力の日本語推論・英語文章／256出力／幅2を加えた12条件。
   4形式48行＋一意prompt／出力条件のMTP off6行で、GPUごとに54行。
2. **MTPファイル全体の比較**: original BF16のembedding／headも含む10行列を同時に量子化し、
   norms8個はF32を維持する。128／8192入力、128出力、幅2の2条件×4形式＋off2行でGPUごとに10行。
   8行列比較と混ぜず、共有headまで量子化する効果を別に示す。

各条件1 warmup＋3 measured、target temperature1／top_k20／top_p0.95／min_p0／seed123、
反復penalty1。固定長比較のためignore_eos=true、prefix cache無効・cache RAM0。
MTPはlatest upstreamのgreedy draft経路、n_min0／p_min0／n_max1/2/4とし、幅はserver再起動で変更する。
server起動やprofile時間を通常decode性能へ加算しない。offは各lane内で形式間に重複させない。

## 計測と解釈

- native timingsのprompt／predicted token数・時間・tok/s、HTTP wall、draft提案／採用数、
  Prometheusのdraft回数・位置別採用数を保持する。draft専用walltimeのAPI項目はないため推定値を作らない。
- prompt/cache countと出力長、MTPの実提案数を確認し、不一致・missing・timeout・crashは有効な比較へ使わない。
- warmupを除く全反復の中央値とMAD、各条件のBF16比、採用率を比較する。小差を普遍的な改善と主張しない。
- 量子化形式間で出力履歴が変わるためtoken一致は必須にせず、hashと反復内の変動を記録する。
- HIP GPUの小規模MUL_MATをBF16／Q8_0／Q6_K／Q4_0でCPU参照値と照合する。実モデルは全layerGPU指定で、
  exact UUIDとGPU telemetry、終了時の資源回収を記録する。CPU emulationをGPU成功へ読み替えない。
- sLLMの既存結果は本体NVFP4／MXFP8 KV／異なるMTP samplerである。絶対速度の直接順位ではなく、
  各engine内のBF16 MTPに対する量子化効果を比較する。

## 実行と証拠

2GPUを独立processで並行測定する。Qwenローカルserviceは停止中。
R9700の既存serviceはactive接続なしを確認して一時停止し、測定後に元のunit／binary／設定とhealthへ復帰する。
`.local-artifacts/llama-mtp-20260914/` にsource／build／model／raw測定を保存し、Gitへ追加しない。
上流copyは[import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log)へ記録した。sLLM runtimeの変更・commit・pushは行わない。

[メイン計画](../../../../main-plan.md)

## 完了結果

両GPU合計128構成・512 requestを完走した。量子化対BF16 MTPの84比較はすべて中央値でBF16を上回り、
同条件の生成token列も一致した。専用8行列のdecode幾何平均改善はV620 Q8_0/Q6_K/Q4_0で
3.17%／4.02%／4.44%、R9700で2.46%／3.28%／4.38%。MTPファイル全体では
V620 10.06%／12.22%／16.94%、R9700 10.93%／10.75%／13.94%だった。
R9700の元serviceと両GPUのauto設定を復帰し、全server processを正常終了した。
生成GGUFは検証後にcheckout外cacheへ同一filesystemで移し、logical pathをsymlinkで維持した。

[結果・条件・解釈](../../../../../history/2026/09/11-20/llama-mtp-quantization-benchmark.md) /
[全128行CSV](../../../../../history/2026/09/11-20/llama-mtp-quantization-results.csv)
