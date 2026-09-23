# Phase 87 段階9: MTP draft専用の縮小語彙lm_head

## 2026-09-24 受入完了

Qwen3.8 NVFP4のMTP draftだけに、ID昇順の98,304語のFP8 output headを別residentとして追加した。
targetの全248,320行headとp/q受理契約は維持する。モデル側artifactが両ファイルともない場合は
全語彙draftを維持し、片方だけある場合やhash・語彙順が異なる場合はfail-closedする。

### 語彙集合と再現手順

- 生成tool: `ci/tools/qwen38_draft_vocab.py`。入力のrevision、各fileのSHA-256、tokenizer SHA-256を
  照合し、domain内相対頻度を等重みで合算する。同点は小さいtoken IDを優先し、33個の追加tokenを必ず含める。
  最終出力は正確に98,304個の昇順`u32` little-endian IDとする。
- 追跡するコンパクトな入力lock: `ci/matrix/qwen38-draft-vocab-source-lock-v1.json`。
  `ci/tools/qwen38_draft_vocab_sources.py`で、6つのcleanな参照repositoryの固定commitと4つの
  parquet入力から各source fileのSHAを列挙したmanifestを再構成する。source本文・生成IDはGitへ追加しない。
- corpusはen_web 60 MB、SWE-chatのagent会話・code 60 MB、ja/zh Wikipedia各60 MBと、
  code_py 40 MB、code_cpp 50 MB、code_other 15 MBの7 domain。SWE-chatの205 repositoryを確認し、
  sLLM repositoryに一致する項目は0件。mtp-bench-v1のrepo素材・corpusは使用しない。
- manifest canonical SHA-256は
  `3c6fc65c10eee31d86ccd7a2e54e58eaa9282dfea68d7861fe171bad28329ab4`。
  語彙payload 393,216 byteのSHA-256は
  `24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0`。
  2回の生成でpayloadとmetadataがbyte一致し、コンパクトlockから再構成したmanifestとも構造一致した。
  実生成物はモデルrootの`.sllm/mtp-draft-vocab-98304.u32`と`.metadata.json`へ置いた。

初回の4 domain順位は、保存済みTier A 26条件の反実仮想でM1がV620 −1.47 pt、R9700 −1.57 ptと
低下したため使わない。code domainを加えた上記候補は同じ反実仮想でV620 −0.229 pt、
R9700 −0.231 ptだった。この比較の全語彙対照はPhase 86の保存reportであり、
段階9の最終実runは同じ現行sourceから全語彙Aを取得し直して照合する。

### 実装と途中検証

- draft graphのoutput weight/logitをN行にし、FP8 value rowとBF16 row scaleをID順にgatherして
  MTP有効時だけ別residentへ置いた。追加resident memoryは段階7の全語彙MTP比
  504,102,912 byte（約504 MB、両GPU同値）。
- fixed K20 selectorのmapped v3は、workspace tailのsorted global-ID mapを使い、
  chosen tokenとsupport recordのIDをGPU内で変換する。公開C ABI probeは両GPUで
  非整列語彙数98,303／98,304／98,305、tie、support、無効descriptorをPASSした。
- 通常8192/128のMTPあり、0 warmup＋1 measuredの縮小head smokeは両GPUでPASS。
  V620は受理77/101、R9700は75/105、両方で段階7と生成token SHA-256が一致し、
  HIP-only、fallbackなし、cleanup zeroを確認した。
- hostはsllm-core 678件、sllm-hip 163件、Stage9語彙tool／loader／M1計算器のfocused test、
  core/hip/CLI/serverを含むclippy all-targetsがPASS。

### Tier A 26条件の実M1（2026-09-24）

最初の「全語彙」診断runはStage0入口で縮小語彙artifactを読んでいたため、対照として使用しない。
Phase86 P診断だけに限定した全語彙指定をStage0入口にも接続し、同じ現行sourceから全語彙Aを取り直した。
両GPUの全26条件でrunnerと数値reportはPASSした。語彙を戻した後のdraft raw logit幅は248,320行、縮小版Bは98,304行である。

| GPU | 全語彙A M1 | 縮小版B M1 | 差 | 探索時の反実仮想差 |
| --- | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | 0.7770 | 0.7745 | −0.254 pt | −0.229 pt |
| R9700 `gfx1201` | 0.7761 | 0.7733 | −0.279 pt | −0.231 pt |

全語彙Aのdraft top-1がS外だった行はV620で29、R9700で27。両GPUとも26条件中9条件でblock進行が変わった。
探索時との差が0.03 ptを超えた条件はV620で9件、R9700で8件で、すべてblock進行が変わった条件だった。
進行が同じ17条件では、両GPUとも全語彙Aと縮小版Bのtarget logits SHA-256が17/17一致した。
各条件の最初のdraft行についてAのS行とBは両GPUとも26/26でbitwise一致した。
実M1の差は、S外top-1とそれに伴うblock進行の変化で説明できる範囲にあり、探索時の候補を維持する。
raw証拠は`.local-artifacts/phase87/stage9/m1-gfx{1030,1201}-{full-fixed,reduced}/P/`、
比較reportは同じディレクトリ階層の`m1-gfx{1030,1201}-full-fixed-vs-reduced.json`へ置いた。

### 通常8192/128の速度とMTPなし対照（2026-09-24）

同一processでtarget residentと全語彙／縮小語彙の両MTP residentを保持し、両順序AB／BAを各1 round、
variantごとに1 warmup＋3 measuredで実行した。両GPUで全roundが1%基準を超えてPASS。

| GPU | 順序 | 全語彙A TPOT ms | 縮小版B TPOT ms | 短縮 |
| --- | --- | ---: | ---: | ---: |
| V620 `gfx1030` | AB | 31.931 | 30.739 | 3.73% |
| V620 `gfx1030` | BA | 31.999 | 30.772 | 3.83% |
| R9700 `gfx1201` | AB | 27.515 | 26.490 | 3.72% |
| R9700 `gfx1201` | BA | 27.497 | 26.504 | 3.61% |

全16 run/targetでHIP-only、fallbackなし、request cleanup zero、最終session cleanup zero。
各roundの全語彙／縮小版の生成token列は一致し、受理数はV620が77/101、R9700が75/105で各回同じだった。
AB/BA reportは`.local-artifacts/phase87/stage9/abba-gfx{1030,1201}-final.json`。
target専用binary SHA-256はV620 `1a8458197fb442929d9bbd4f2e2e59c4aa0d739d2ac365218031c5dcb430390d`、
R9700 `0ac60a56a47aa4eb9e8e5ea39e974c1c8e6c3e9064cf973f29d033f4bc5027a6`。

MTPなしの通常経路は両GPUで0 warmup＋1 measuredがPASSし、生成token SHA-256が段階7と一致した
（V620 `c9c0b4ee401b11544fe0faaec15d882cb2491c31a460a32a89dce28e437bcc0e`、
R9700 `75d36def8ff45d155373ebb05b885d0e4f0e7197ba9adb2123adfb1fe63b189d`）。
reportは`.local-artifacts/phase87/stage9/mtp-off-gfx{1030,1201}-final.json`。
今回の選択artifactは追跡lockと実payloadを照合し、vocab SHA-256
`24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0`、
manifest SHA-256 `3c6fc65c10eee31d86ccd7a2e54e58eaa9282dfea68d7861fe171bad28329ab4`、
required special IDs 33件が一致した。runtimeは任意の自己整合したモデル側artifactを読み込めるが、
この採否は上記の固定候補だけに適用する。

targetの全語彙headとp/q受理規則を変えず、draft提案分布だけを変える変更としてN1に分類する。
この採用は同一process AB/BAと実M1に基づく。初期探索で設定した打切り線V620 +1.9%／R9700 +1.6%も超えた。

### 参考: R9700の512-token出力（2026-09-24、今回のみ）

ユーザー依頼により、8192入力／512出力、MTP幅2、固定seed、0 warmup＋1 measuredで全語彙Aと縮小版Bを各1回実行した。
既定の128出力を変えず、診断binaryだけに8192/512行と全語彙指定を一時追加し、測定後にsourceから戻した。
この1回ずつの比較は速度採用判定に使わない。

| draft head | 生成token | 提案token | 受理token | 受理／提案 |
| --- | ---: | ---: | ---: | ---: |
| 全語彙A | 512 | 423 | 300 | 70.92% |
| 縮小版B | 512 | 423 | 300 | 70.92% |

両runとも同じ512-token列（SHA-256 `a7cd2e1b45bad82c92789950e4dc1b6420c651d133646bc503f011d4bc079a7e`）、
HIP-only、fallbackなし、cleanup zeroでPASS。通常128出力のR9700 AB/BAでは両版とも75/105（71.43%）だった。
参考runは`.local-artifacts/phase87/stage9/long512-gfx1201-{full,reduced}.json`、診断binary SHA-256は
`5fb7b119db18d32ed32e7428c4c8a6b5b61fa2c1e6b30f5ae333068fdb754814`。

計画: [Phase 87単一要求NVFP4計画](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
