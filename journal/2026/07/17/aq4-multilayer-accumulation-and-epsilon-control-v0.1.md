# AQ4 multi-layer accumulation and epsilon control v0.1

## 前回の要点

- Phase 1 (`aq4-layer0-hybrid-diagnostic-v0.1`) はCPUのBF16 source比較でlayer 0 output hiddenの相対L2を`0.042451`と測定した。これはfamily単体の量子化誤差と同じ桁であり、07/14に得た最終出力`0.615`をlayer 0単体では説明しない。
- したがって、優先仮説H8（小さなlayer誤差が深さ方向に累積・増幅する）を、GPU・resident service・active manifestに触れないCPU-only診断として検証することにした。

## 今回の変更点

### 実装と範囲

- `ullm-aq4-layer0-family-isolation`に`--chain-layer-range START:END`を追加し、manifestから連続層の型を解決して、AQ4 f32出力を次層入力へ渡すchain診断を実装した。current layerのinput/output sequenceとlayer-local stateだけを保持し、各出力は比較用にstreamした後、全層collectionを作らず破棄する。
- recurrent/Conv stateはモデルの層ごとのstateであり、layer 0のstateをlayer 1へ物理的に流用するとモデル式を変えてしまう。そのため各linear-attention layerはfull temporal contextをreplayして自身のcold stateから進め、chain hiddenは前層出力を渡した。case間・層出力間でsource embeddingへ戻るresetはしていない。
- `qwen35_package_contract`のmanifest-derived topologyとsource `config.json`の双方から、32層の並びが`[linear, linear, linear, self] x 8`、self-attention indexが`3, 7, 11, 15, 19, 23, 27, 31`であることを確認した。最初のself-attentionを含む最小連続範囲`0:3`を測定対象にした。
- source comparatorは同一safetensors shardのidentity SHA-256を一度だけ記録するようにした。比較対象・hash・出力内容は変えず、chain時の不要な再読込みを除去した。
- Phase 2b用に既存harnessの診断専用`--post-norm-epsilon-source-control`を使い、AQ4 post-norm epsilonを一時的にsource相当の`1e-6`へ切り替えられるようにした。通常時は既存のAQ4 runtime値`1e-5`のままで、production設定は変更していない。

### Phase 2: 0--3層のCPU測定

fixtureはPhase 1と同じ3 context / 9 output recordsである。AQ4 chain reportのdeviceは`cpu:0`、source compare reportのdeviceは`cpu-only`であり、GPUやresident serviceは使用していない。

| layer | kind | relative L2 | cosine | max abs |
| ---: | --- | ---: | ---: | ---: |
| 0 | linear_attention | 0.042451384 | 0.999106949 | 0.069626808 |
| 1 | linear_attention | 0.075075875 | 0.997374924 | 0.174329758 |
| 2 | linear_attention | 0.092594143 | 0.995868575 | 0.253928185 |
| 3 | self_attention | 0.106253646 | 0.994378165 | 0.202241421 |

- 成長曲線は単調増加だが、遷移増分は`+0.032624491`、`+0.017518268`、`+0.013659503`と縮小する。従って分類は「おおむね線形または劣線形」であり、self-attention layer 3でのjumpや超線形増幅は観測しなかった。
- layer 0開始前を0とする線形外挿を採用すると、layer 31で`0.106253646 * 32 / 4 = 0.850029167`となる。これは既知のproduction最終相対L2 `0.615`の`138.2%`（`+0.235029167`、38.2%過大）である。
- 初期比率に基づく幾何外挿は`556.197559`になるが、観測増分が縮小している事実と矛盾するため採用しない。結論は **H8で概ね説明できる**（diagnostic verdict: `explains`）。ただし4層からの単純外挿であり、原因確定やproduction fixの根拠にはしない。

### Phase 2b: post-norm epsilon control

| stage | AQ4 1e-5 | AQ4 1e-6 control | control - default |
| --- | ---: | ---: | ---: |
| post_norm relative L2 | 0.178437633 | 0.178529637 | +0.000092004 |
| mlp_output relative L2 | 0.104565046 | 0.103788976 | -0.000776070 |
| layer_output relative L2 | 0.042451384 | 0.042349396 | -0.000101987 |
| diagnostic lm-head readout relative L2 | 0.026798801 | 0.026853115 | +0.000054314 |

- source epsilonに揃えるとlayer outputの相対L2は`0.2402%`、二乗誤差は`0.4799%`低下した。一方post-norm直後とdiagnostic readoutは僅かに悪化しており、後段MLPを経た影響は非単調である。
- layer output改善`0.000101987`を32層へ意図的に大きめに線形反復しても`0.003263596`、最終`0.615`の`0.5307%`に過ぎない。epsilon差の寄与はH8結論に対して無視できる規模であり、このスコープでproduction epsilonを修正する理由にはならない。

### Evidence / verification

- Artifact: `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-multilayer-accumulation-v0.1/`
  - `compare/growth-curve.csv` / `compare/growth-curve.md`: 成長曲線
  - `compare/comparison.json`: source/package/input/hashに束縛した層別比較
  - `epsilon-control-summary.md`: Phase 2bの集計
- 実行した主な検証:
  - `cargo build --package ullm-engine --bin ullm-aq4-layer0-family-isolation` — 成功（既存runtime C++のsubobject-linkage warningのみ）
  - `python3 -m py_compile tools/compare-aq4-layer0-hybrid.py tools/compare-aq4-multilayer-accumulation.py` — 成功
  - `pytest -q tests/test_aq4_layer0_family_isolation.py tests/test_aq4_multilayer_accumulation.py` — `12 passed`
  - `python3 tools/compare-aq4-multilayer-accumulation.py ... --chain-layer-range 0:3 ...` — `status: valid`
  - `python3 tools/compare-aq4-layer0-hybrid.py ...`（AQ4 `1e-5`） — `status: valid`
  - 同コマンドに`--post-norm-epsilon-source-control`を追加（AQ4 `1e-6` control） — `status: valid`
- 07/16のP3一時停止harness / GPU lock / resident service / systemd / active manifestには参照・変更をしていない。Phase 3以降のGPU kernel差分、構成差監査、fix実装にも進んでいない。

## 次の行動

- 今回のスコープでは、self-attentionを含む最小範囲0--3でexit基準を満たした。layer 4--31の実測chain、32層実測、GPU kernel差分、構成監査、production修正は未実施であり、別途明示的な承認が必要である。
- 次の判断は、本evidenceの線形/劣線形成長とH8 `explains`判定を入力にして行う。epsilon controlは小さく、単独のfix候補としては進めない。
