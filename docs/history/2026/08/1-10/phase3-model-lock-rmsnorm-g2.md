# Phase 3 Stage A model lock・RMSNorm・G2履歴

## 2026-08-04

- Phase 3 Stage Aの完了境界を、Qwen/Qwen3.5-4Bの完全model lock、最初のpublic semantic opであるRMSNorm、semantic G1、real-weight model sliceのG2、短いRMSNorm P0 smokeまでに限定した。
- full model生成、attention、MLP、KV/state、tokenizer/chat template実行、prefill/decode、G3、性能最適化、P1は後続計画へ分離した。
- RMSNorm baselineをBF16 weight / BF16 activation、FP32 accumulationとし、exact shapeとepsilonは固定model configから取得する方針にした。
- 固定llama.cpp/vLLM reader調査により、Qwen3.5 HF RMSNorm weightは実効scaleを`1 + raw_weight`とするoffset-one variantであることを確認し、raw BF16 weightを事前変換せずdescriptorとoracleで明示する方針にした。
- model-free private G1をsemantic opへ流用せず、public Rust/C ABI/native HIP経路を設ける方針にした。
- G2はread-only model cacheから実行時にreal weightを抽出し、raw model/sliceはGit管理せず、source lock fingerprint、tensor、recipe、hashだけを記録する方針にした。
- private diagnostic G1 reportへ数値結果を継ぎ足さず、semantic RMSNorm G1とG2に専用schema/runner/aggregateを作り、既存private G1は同一candidateの前提evidenceとして再実行する方針にした。
- model lock fingerprint用のRFC 8785 JCSを、既存の通常JSON key sortで代用しないことを明記した。
- G2/P0の非実行schema/runner contractだけを構築するbootstrap candidateはH0〜H2とhost negative testを必須とし、GPU/runtime behaviorも変えるcandidateには影響範囲に応じてH3/G0/private diagnostic G1を追加する。初回enablement candidateから同一SHAのsemantic G1/G2/P0を省略しない方針にした。
- canonical `gfx1030`/`gfx1201`の同一immutable candidate H0〜H3、G0、private diagnostic G1、semantic RMSNorm G1、G2、P0、oracle、fallbackなし、実行後healthを完了条件にした。
- CI正本とmain planがpublic HIP runtime/kernelにP0を要求するため、性能最適化は対象外のまま、RMSNorm kernel latencyと`B-1/B/B+1`だけを測る短いP0とversioned `review_required` dispositionを完了条件へ追加した。
- `Qwen/Qwen3.5-4B`はrequested revision `main`を2026-08-04に完全SHA `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`へ解決し、Phase 3の採用revisionとして固定した。multimodal top-level configと全shardをlockするが、Phase 3実行対象はtext RMSNormだけとし、vision tensorは既知の未消費集合として扱う。
- checkout外cacheへ全13入力file、合計9,342,905,899 bytesをimmutable URLから取得し、全fileの実bytes SHA-256を計算した。2 weight shardの実hashは公式LFS SHA-256と一致した。G2 tensor `model.language_model.layers.0.input_layernorm.weight`は第2shardのabsolute byte range `[94432, 99552)`、BF16 `[2560]`として確定した。
- model lock validatorは全cache fileのhash済みFDをsemantic validation終了まで保持し、config、index、safetensors headerを同じFDから読む方式へ修正した。semantic read後のroot/path identityも再検査し、同size path差し替えと同一inode改変をfail closedにした。
- safetensors fixtureの未被覆末尾byteを除去して111 bytesへ修正し、fixture fingerprintを`sha256:4c0dddf51b7568e3cd1863c3ae214ec4beddd2f7b8dd1d62707567515ffa0bdf`へ更新した。tensor spanはdata bufferを先頭から末尾まで連続して完全被覆することを要求し、gap、overlap、trailing byte、u64 overflowを拒否する。
- model lockの独立再監査はblocking/high/medium 0件でPASSした。恒久host contract 15件、公式safetensors parser、実Qwen cache全13 fileのcontent-only hash、両shard738 tensorのindex/header/classification、locked slice `[94432, 99552)`を再検証した。trusted read-only modeは現在の`root_mode=0700`かつwritable mountを期待どおり拒否し、content identityと物理immutabilityの証拠を分離した。
- H3 required昇格の20回・7日観測は引き続き非blocking follow-upとし、Phase 3の開始・完了条件に含めなかった。
- baseline execute設計を、leading dimension flatten、prepared plan再利用・同一plan in-flight 1件、generic completion再利用、nonfinite payloadのIEEE伝播、`N <= 4096`、wave32・256 threads、additive execute/dispatch ABI、専用RMSNorm H3 artifactとして固定した。
- BF16出力の初期acceptance budgetを`tolerance_id=rmsnorm-bf16-f32-output-v1`、`atol=0.0078125`、`rtol=0.015625`としてGPU結果の前に固定し、finite値の複合誤差比較とNaN/Inf classification比較、同一candidateでの事後拡大禁止を決定した。
- generated-token停止policyをmodel lockへ追加し、Qwen lock fingerprintを`sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`へ更新した。停止policy導入前の`sha256:89ba8a6b2e1b7c0324090ddf15ce0e673ff4c3dc242c4127690d490056d8efd1`は過去candidateのidentityとして保持し、現行runtime/evidenceへ混在させない。
- この計画をPhase 3全体の完了点にはせず、full model、CLI text生成、G3までを含む[Phase 3全体計画](../../../../plans/active/2026/08/1-10/phase3-qwen35-4b-bf16.md)のStage Aへ位置付けた。

## 2026-08-07

- public HIP runtime、baseline RMSNorm、専用H3 compile-only、controller-owned semantic G1のhost実装を進め、review9のfresh-process authority、candidate identity、compiler transcript、resource containment、raw frame/numerics/artifact binding指摘を修復した。
- controllerとworkerを固定Pythonの`-I -S`で起動し、controller起動前の`sitecustomize`介入を禁止した。production schema検証から未固定のinstalled `jsonschema` authorityを除き、review済みGit object byteから読む4 schemaだけをstdlib-onlyの閉じたvalidatorで検証するようにした。host toolingは従来どおり固定dependencyの`jsonschema`を使い、両経路を分離した。
- 現行dirty worktreeでsemantic G1 38件、H0 131件、H1 151件、H2 35件とlocal fail-closed aggregateをPASSさせた。この結果は`local-development`であり、immutable/GPU evidenceまたは独立review PASSを意味しない。
- Stage Aの作業単位ごとの進捗表を計画へ追加し、fresh独立review後もG2/P0のschema、runner、専用binary、canonical 2 GPU evidence、aggregateが未実装であることを明示した。

[対応する計画](../../../../plans/active/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
