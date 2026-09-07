# Phase 78後の他モデル測定

2026-09-05、ユーザー指示「他モデルでも計測して」により、Qwen3.5-4B／9Bを両RDNA GPUで測定した。
要求準備の短縮は全条件で観測したが、prefill／decode全体の大幅な高速化は確認しなかった。
Phase 78の完了判断・緩和済み目標は変更せず、追加最適化は実装していない。

## 条件

- 同じlocked GGUF（OCP MXFP8 E4M3 W8A8 block32 E8M0）、FP16 KV、単一要求、greedy、EOS無視。
- token 23066を17個／512個入力し、17個／32個出力。chunkは入力長と同じ。render／tokenizeは含まない。
- 旧版→現行版→旧版。各armは1 warmup＋3 measured。旧版値は前後6測定の中央値、現行版は3測定の中央値。
- V620はUUID `GPU-76a08c022586fed6`（03:00.0）、R9700は`GPU-a8e9ddefa2d60f55`（07:00.0）。
  別GPUで同時実行し、各processはUUIDで単独可視化、device index 0。SLLM opt-in環境変数は解除した。
- 現行CLIはcommit `40ab582b049cff7effadbca75fe951d6cef5bd96`から両targetをrelease build。
  旧版は保存済みPhase75 V620／Phase74 R9700 CLIを今回再測定した。過去の性能値との比較ではない。

## 実測結果

速度の単位はtok/s。括弧は旧版中央値に対する変化率で、統計的有意差の判定ではない。

| GPU | モデル | 入力／出力 | prefill 旧→現行 | decode 旧→現行 | 要求準備 ms 旧→現行 |
| --- | --- | --- | --- | --- | --- |
| V620 | Qwen3.5-4B | 17/17 | 50.093→51.085 (+1.98%) | 15.086→15.114 (+0.19%) | 6.566→4.195 (-36.11%) |
| V620 | Qwen3.5-4B | 512/32 | 1009.687→1009.579 (-0.01%) | 14.637→14.442 (-1.33%) | 8.228→5.964 (-27.51%) |
| V620 | Qwen3.5-9B | 17/17 | 24.638→24.819 (+0.74%) | 8.600→8.582 (-0.21%) | 6.347→4.941 (-22.16%) |
| V620 | Qwen3.5-9B | 512/32 | 545.692→544.439 (-0.23%) | 8.498→8.427 (-0.84%) | 8.012→6.126 (-23.54%) |
| R9700 | Qwen3.5-4B | 17/17 | 73.317→73.361 (+0.06%) | 22.140→22.035 (-0.47%) | 6.773→5.132 (-24.23%) |
| R9700 | Qwen3.5-4B | 512/32 | 3779.081→3859.420 (+2.13%) | 20.538→20.350 (-0.91%) | 8.104→6.241 (-22.99%) |
| R9700 | Qwen3.5-9B | 17/17 | 40.430→40.745 (+0.78%) | 13.493→13.486 (-0.05%) | 6.449→4.650 (-27.89%) |
| R9700 | Qwen3.5-9B | 512/32 | 2000.813→2019.521 (+0.93%) | 12.802→12.874 (+0.56%) | 8.571→6.280 (-26.73%) |

要求準備（request start→prefill submit）は22.16～36.11%、約1.41～2.37 ms短縮した。
prefillは−0.23～＋2.13%、decodeは−1.33～＋0.56%。decodeの小さな低下も観測結果として残す。
旧版の前後差・各armのmin/max/MADは集約JSONへ保存した。例えばV620 9B 512入力では旧版同士のdecode差も
約1.64%あり、小差を一律に改善／退行と確定しない。全条件・全形式への速度保証は導かない。

## 検証と適用範囲

24 run、24 warmup＋72 measured要求が成功し、全比較で入力／生成／visible／decode入力token、停止理由、
記録されたauditが一致した。各sampleでHIP-only、fallback false、dispatch数>0、要求cleanupを確認し、
各run終了後のmodel allocationも0だった。これは既存runnerの数値・出力確認の範囲であり、
新しいlogits oracle、全kernel trace、物理DRAM帯域／offload計測を実施したという意味ではない。

今回の結果はQwen3.5系の共通host／attention経路を含む全体比較である。旧版はhashで固定したhistorical draft
binaryであり、Phase78の個別変更だけを無効化した比較ではない。clone省略・plan共有等の寄与を個別に分解しない。
動的clock、各arm 1+3、別GPU間で同時実行するhost条件も含む探索観測である。

MXFP8とFP8 outer-vectorは別形式なので、ID71の汎用性能は今回測定していない。現行の通常CLIはgfx1030の
Qwen embedded FP8 GGUFを拒否する。共通kernelとして存在することと、他モデルの公開入力経路で使用可能なことは区別する。
非Qwenアーキテクチャ、より長いcontext、BF16／NVFP4等の他形式も今回の測定対象外である。

## 証拠と再実行

raw結果・実行argv・stderr・binary・集計scriptはGit管理外の`.local-artifacts/phase78-cross-model/`に保存した。
通常CLIの`benchmark --lane direct --gguf … --derived-lock … --model-size 4B|9B --target gfx1030|gfx1201`を使い、
上記条件を`--input-token-ids`、`--max-new-tokens`、`--prefill-chunk-tokens`、`--kv-cache-encoding fp16`、
`--greedy --ignore-eos --warmups 1 --measured 3 --device-index 0`で明示した。
モデル・binary・rawのSHA256、全8条件の中央値／min／max／MAD、旧版の出典を
[集約記録](phase78-cross-model-evidence.json)に保存する。

[完了計画](../../../../plans/archive/2026/09/1-10/phase78-cross-model-measurement.md) ·
[Phase78の目標変更を含む完了記録](phase76-78-qwen38-nvfp4.md)

後続の[Gemma NVFP4測定](phase78-nvfp4-cross-model-measurement.md)では、Gemmaの混合GGUF内のFP8投影にもID71が適用されることを確認した。
上記のQwen FP8入力制限をGemma混合GGUFの制限へ一般化しない。
