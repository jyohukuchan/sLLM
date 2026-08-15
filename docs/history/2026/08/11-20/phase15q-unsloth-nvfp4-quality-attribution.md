# Phase 15Q Unsloth NVFP4品質要因切り分け履歴

## 2026-08-16: FP4公式入力と内部状態の再整理

- ユーザー決定により、提供元NVFP4 PTQ/QAT checkpointとMXFP4/MXFP8 QAT/native modelをfirst-class model inputとして扱う。
  Phase 15Qの`correctness-only opt-in`はS0/U0/O0というsLLM製W4A16 PTQ converter candidateの不採用を表した過去の
  evidence labelへscopeし、NVFP4/MXFP4 encoding、提供元artifact、同一artifactを正しく実行するproviderへ一般化しない。
- BF16 sourceから作るsLLM PTQには従来KLD budgetを維持する。提供元PTQ/QATは同じquantized artifactのreference実行と
  task評価、BF16正本を持たないnative low-bit modelはartifact fidelity、reference実行、task評価で判定する。
- 最終GGUFではdtype/encodingにかかわらず同じload/generate/serve操作を使い、providerを自動選択する。量子化artifactの選択を
  十分なユーザー意思とし、低bitを理由とする許可flag、確認、通常警告を追加しない。内部ではruntime成熟度、provider優先順位、
  converter品質、model evidenceを独立管理する。
- この時点では製品・受入・interface方針だけを更新し、新しいPhase、詳細計画、source implementationは作成していない。

## 2026-08-15: 実装・実機検証・closeout

### Artifactとsource identity

- `google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`をBF16 controlへ固定した。
  `model.safetensors`は`23,919,549,408` byte、SHA-256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`、model lock fingerprintは
  `sha256:381c94bcb48a26d8ef83d1c3d7c5a3513ef8fac4a638752731b85c119385f09d`である。
- `unsloth/gemma-4-12b-it-NVFP4` revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`を量子化artifactへ固定した。
  `model.safetensors`は`9,304,966,064` byte、SHA-256
  `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`である。
- 両artifactの同名BF16 source tensor 349個をrange hashで比較し、すべてbyte-identicalだった。identity digestは
  `7f64136cfe41dd6880205dff182808cc6d94fdbb58aebd81a12621e890e2f0dc`、mismatchは0件だった。Unsloth側のBF16 entry総数629にはattention/input scale等の
  量子化metadataが含まれるため、source tensor数へ混ぜていない。
- modelと生成sidecarはmodel cacheおよびrepository外の`~/.cache/sllm/evidence/phase15q`だけに置き、Gitへmodel、slice、
  raw logits、reportを追加していない。
- fixed prompt manifestは32 case・96位置、SHA-256
  `827d0b2cb0d972016ae9e3b66a168966146fdd5f1f251dc3897fbdef1d0f4107`である。最終S0/U0/O0 sidecar fingerprintは順に
  `sha256:0b76a4b495f794aefe306acfa2436bb51f4d10529592cb6e5b5c160f2f584459`、
  `sha256:34ffc0b6867721d1ac1140313bcf67130c351cfc9cde948c24b19fa106236775`、
  `sha256:81b66903b7061cd23d877d38a7c273e2f279a94fece1b9492e860faf45736409`である。

### Decoder、importer、runtime

- repository外safetensorsをpositional readする独立decoder/importerを追加した。low-nibble-first E2M1、K-axis block 16、
  OCP E4M3FN block scaleをbyte-preservingで取り込み、compressed-tensorsのreciprocal `weight_global_scale`を
  sLLMのmultiplicative FP32 tensor scaleへ変換する。W4A16 primary laneでは`input_global_scale`を適用しない。
- independent testsはE2M1全16 code、nearest-even tie、E4M3値、zero、non-aligned境界を確認した。production/non-aligned
  K/N `15/16/17`・`31/32/33`をR9700 exact `gfx1201`とV620 exact `gfx1030`で照合し、最大relative error
  `0.0036375308`、fallbackなしでPASSした。
- Gemma 4 IT model lock、weight plan、graph、resident uploadへverified partial/full NVFP4 bindingを追加した。通常の
  full-model runnerはexact 144 tensorを要求し、layer診断だけが明示`--allow-partial`でsubsetを受理する。

### Matched attribution

比較は同じBF16 source、MLP gate/up/down 144 tensor、BF16 activation/attention、FP16 KV、packed-dequant provider、
32 fixed prompt、96 teacher-forced位置を固定した。S0は既存min-max、U0はUnsloth `imatrix_mse` payload、O0は
sampled weight MSEを目的としたbounded per-tensor scale searchである。

- U0のsampled weight MSEは144/144 tensorでS0より悪く、U0/S0比medianは`1.3933`だった。
- O0は120/144 tensorでS0よりweight MSEを改善したが、比medianは`0.99735`で改善幅は小さかった。
- R9700 layer単独差し替えではU0のmedian KLDがlayer 0で`0.01646→0.00991`、layer 1で
  `0.00911→0.00811`、layer 47で`0.01795→0.00276`へ改善した。一方、選択6 layer累積U0のmax KLDは
  `12.5620`であり、局所改善は一様に累積しなかった。

| target | variant | KLD median / p90 / max | top-1一致 | S0より低KLDの位置 |
| --- | --- | --- | ---: | ---: |
| R9700 `gfx1201` | S0 | `0.3315 / 3.4727 / 11.7972` | `59/96` | control |
| R9700 `gfx1201` | U0 | `0.1619 / 2.3621 / 9.1781` | `76/96` | `66/96` |
| R9700 `gfx1201` | O0 | `0.2880 / 2.1219 / 14.4025` | `63/96` | `54/96` |
| V620 `gfx1030` | S0 | `0.3715 / 3.5324 / 5.1655` | `60/96` | control |
| V620 `gfx1030` | U0 | `0.1736 / 1.9045 / 7.5777` | `73/96` | `61/96` |
| V620 `gfx1030` | O0 | `0.3433 / 2.4327 / 6.4180` | `67/96` | `49/96` |

R9700/V620とも全dispatchはHIP、fallback false、nonfinite 0、cleanup 0だった。B0 model residentは
`23,814,729,316` byte、candidate residentは`11,605,373,092` byte、peakはそれぞれ
`24,147,052,468`/`11,937,696,244` byteだった。
3 fixed greedy caseの最初のdivergence位置は、S0が`[なし, 7, なし]`、U0/O0がともに`[0, 1, なし]`で、
median logit改善だけでは生成trajectoryの一貫改善を示さなかった。

### 判定と採否

- U0はweight MSEが悪いにもかかわらず両targetでmedian KLDとtop-1をmaterialに改善したため、activation-aware calibrationは
  同一E2M1/block-16 format内の重要な品質要因である。weight-only MSEはこのmodelの十分なconverter目的関数ではない。
- U0の改善は全位置・全layerで一貫せず、max KLD `9.1781`/`7.5777`は既存budget `0.05`を大幅に超えた。O0も
  worst caseを救済しない。したがって原因をconverterだけ、または数学的なFP4型限界だけへ帰属せず、algorithm余地と
  model/format/configuration ceilingの双方がある`mixed`と判定した。
- S0/U0/O0のいずれもdefaultまたはproductionへ採用しない。R9700/V620 NVFP4は`correctness-only opt-in`を維持する。
  sensitive tensorをBF16/FP8へ残すmixed precisionと、再現可能なactivation-aware converterは将来のbounded follow-upである。
- Unsloth公開checkpointのW4A4 MLP、attention W8A8、KV FP8を含むM0は、未実装要因をweight algorithm比較へ混ぜるため
  primary evidenceとして実行していない。本結果をmixed checkpoint、native FP4、NVIDIA、CDNA3へ一般化しない。

### 統合検証

- local report SHA-256はanalysis `88638d49485929e273d42470e72050aaa13d0c0c4a3869522238dea07bc3fc44`、
  R9700 full `c3194cb246f6a0bd67f4cd2a551c51bf00679a387b289307acd862617cf01010`、V620 full
  `e9acc0ee72d65cc10f72db4d796501ff5c966eb10aefb7a09d70167154bff9ab`、R9700 layer sensitivity
  `865bbc174fb93f1f51f2a981d836045306ab926b48ed193ed882ccef83570fb6`である。report自体はGitへ追跡しない。
- Python Phase 15Q tests、Rust workspace全target tests、dependency closure、JSON manifest/matrix、exact Gemma lock、
  target別release build、両GPU operator/layer/full-model、format、diff checkをPASSした。
- integration reviewでCI Rust toolchainが`if let` chainを受理しない互換性findingを検出した。nested `if`へ修正し、
  dependency closureとaffected build/testをfocused re-reviewした。
- main plan、runtime、model lock、NVFP4仕様、GPU/software compatibilityを同期し、本planをarchiveした。
  Phase 16 KV cache FP8/NVFP4を次の作業として開始できる。

## 2026-08-15: 詳細計画作成

- ユーザーの明示指示により、Phase 16より前の次タスクとして、NVFP4の高いKLDが量子化algorithmと数値formatの
  どちらに主に由来するかを調べるPhase 15Qを追加した。
- `unsloth/gemma-4-12b-it-NVFP4` revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`を候補に固定した。
  artifactは9,304,966,064 byte、SHA-256 `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`である。
- remote header/configをbounded readし、MLP 144 tensorがU8 packed E2M1、E4M3 block scale、F32 global scaleを持ち、
  weight observerが`imatrix_mse`であることを確認した。一方、公開checkpoint全体はMLP W4A4、attention W8A8、KV FP8の
  mixed-precisionであり、sLLMのweight-only NVFP4と直接比較できない。
- primary比較を、exact Gemma 4 12B-it BF16 source上でMLP 144 tensorだけ入れ替える`B0/S0/U0/O0`とした。
  activation、attention、KV、runtimeを固定し、Unsloth mixed checkpoint直接実行はsecondary laneへ分ける。
- artifact/source lock、independent decoder、tensor/layer sensitivity、複数logit位置のKLD分布、generation/service回帰、
  algorithm/format/runtime/mixedの判定規則をplanへ固定した。この時点ではmodel payloadの取得、source実装、GPU実行、
  provider状態変更を行っていない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase15q-unsloth-nvfp4-quality-attribution.md)
