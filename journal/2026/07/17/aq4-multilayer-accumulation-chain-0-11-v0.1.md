# AQ4 multilayer accumulation chain 0--11 v0.1

## 前回の要点

- Phase 2のcommit 'de72dbd8' / 'bc697800' は、既存の3 context・9 output record fixtureでlayer 0--3をCPU-only chain比較した。
- layer 0--3の相対L2は0.042451、0.075076、0.092594、0.106254と増加したが、増分は+0.032624、+0.017518、+0.013660へ縮小していた。
- 前回は4点のzero-origin線形外挿0.850029を採用してH8を'explains'とした。ただし、4点だけでは劣線形曲線の収束や途中の非単調性を判別できなかった。

## 今回の変更点

### 安全なCPU測定

- 既存の'--chain-layer-range'機構と同じfixtureを再利用し、'0:11'を一度だけ実行した。実際のトポロジーはlinear-attention 9層とself-attention 3層（index 3、7、11）である。依頼文が挙げた3と7に加え、inclusive endpoint 11も必然的に含まれる。
- 0:3の'[linear, linear, linear, self]'一ブロックに対し、0:11は同じブロック3回なので計算量を3倍と見積もった。層ごとのstreaming破棄によりピークメモリは範囲長に比例しないことを確認し、45分timeoutを付けた。
- 実測はwall 2:37.13、最大RSS 332008 KiB、swap 0、exit status 0だった。AQ4 chain reportは'cpu:0'、source comparisonは'cpu-only'であり、GPU、active production service、systemd unit、active manifest、P3 harnessには触れていない。

### 成長曲線とH8再評価

| layer | kind | relative L2 | delta |
| ---: | --- | ---: | ---: |
| 0 | linear_attention | 0.042451 | — |
| 1 | linear_attention | 0.075076 | +0.032624 |
| 2 | linear_attention | 0.092594 | +0.017518 |
| 3 | self_attention | 0.106254 | +0.013660 |
| 4 | linear_attention | 0.119419 | +0.013165 |
| 5 | linear_attention | 0.125536 | +0.006117 |
| 6 | linear_attention | 0.077143 | -0.048393 |
| 7 | self_attention | 0.094488 | +0.017345 |
| 8 | linear_attention | 0.094775 | +0.000287 |
| 9 | linear_attention | 0.092623 | -0.002152 |
| 10 | linear_attention | 0.074961 | -0.017662 |
| 11 | self_attention | 0.080827 | +0.005866 |

- curveはlayer 5の0.125536を頂点に非単調となった。layer 11の0.080827はproduction final 0.615の13.1%に留まる。
- 既存toolのzero-origin線形外挿は0.215539（productionの35.0%）となり、raw reportの分類は'partially_explains'へ変わった。
- 12点を使う全期間平均delta外挿は0.150601（24.5%）、直近4遷移の符号付き平均は0.012522（2.0%）、下落を0に丸めた上方寄り直近平均でも0.111591（18.1%）である。
- 初期の正の増分だけが等比減衰すると仮定した収束値は0.137306（22.3%）であり、実際のlayer 5->6負deltaによりその前提自体は棄却される。self-attention block end（layer 3/7/11）の比を使う補助的な等比level外挿は0.040793（6.6%）である。
- 0.615に達するには残り20遷移で平均+0.026709/層が必要で、全期間平均の7.66倍かつlayer 0以後の全正deltaを上回る。したがって、**H8は部分的に説明できるが、CPU chain上のH8単独ではproduction 0.615を説明できない**と再分類する。前回より根拠は強くなり、前回の'explains'判断は弱まった。
- self-attention layer 3では前後のdeltaが+0.017518、+0.013660、+0.013165と滑らかに縮小した。layer 7では直前のlinear layerで-0.048393となった後に+0.017345へ戻るが、次は+0.000287であり、持続的なattention jumpではない。

### 証跡と検証

- 成果物は
  'benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-multilayer-accumulation-v0.1/chain-0-11-v0.1/'
  に置いた。raw AQ4/BF16 report、resource estimate、time output、delta CSV、複数モデルの分析、SHA256SUMSを含む。
- 実行コマンド:

    /usr/bin/time -v -o .../time-v.txt timeout --signal=TERM --kill-after=60s 45m \
      python3 tools/compare-aq4-multilayer-accumulation.py \
      --chain-binary target/debug/ullm-aq4-layer0-family-isolation \
      --package /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package \
      --hybrid-input benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/input/hybrid-input.jsonl \
      --source-model /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
      --chain-layer-range 0:11 --aq4-output .../aq4-chain --output .../compare \
      --chunk-bytes 16777216

- 検証:
  - 'python3 -m py_compile tools/compare-aq4-layer0-hybrid.py tools/compare-aq4-multilayer-accumulation.py' は成功。
  - raw comparison / delta CSV / analysis JSONの12層・hash・H8 verdict整合性チェックは成功。
  - 'pytest -q tests/test_aq4_layer0_family_isolation.py tests/test_aq4_multilayer_accumulation.py' は13 passed。

## 次の行動

- このスコープではPhase 3のGPU kernel差分、production構成監査、fix実装には進まない。
- H8の残る寄与はCPU-only chainで定量化した。GPU pathまたはproduction構成を調べるかは、別途明示的な承認とPhase 3の手順が必要である。
