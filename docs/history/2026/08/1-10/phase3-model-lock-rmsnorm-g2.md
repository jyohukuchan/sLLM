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
- この計画をPhase 3全体の完了点にはせず、full model、CLI text生成、G3までを含む[Phase 3全体計画](../../../../plans/archive/2026/08/1-10/phase3-qwen35-4b-bf16.md)のStage Aへ位置付けた。

## 2026-08-07

- public HIP runtime、baseline RMSNorm、専用H3 compile-only、controller-owned semantic G1のhost実装を進め、review9のfresh-process authority、candidate identity、compiler transcript、resource containment、raw frame/numerics/artifact binding指摘を修復した。
- controllerとworkerを固定Pythonの`-I -S`で起動し、controller起動前の`sitecustomize`介入を禁止した。production schema検証から未固定のinstalled `jsonschema` authorityを除き、review済みGit object byteから読む4 schemaだけをstdlib-onlyの閉じたvalidatorで検証するようにした。host toolingは従来どおり固定dependencyの`jsonschema`を使い、両経路を分離した。
- 現行dirty worktreeでsemantic G1 38件、H0 131件、H1 151件、H2 35件とlocal fail-closed aggregateをPASSさせた。この結果は`local-development`であり、immutable/GPU evidenceまたは独立review PASSを意味しない。
- Stage Aの作業単位ごとの進捗表を計画へ追加し、fresh独立review後もG2/P0のschema、runner、専用binary、canonical 2 GPU evidence、aggregateが未実装であることを明示した。

## 2026-08-09

- trusted solo-development期間に限定して中断A0のcustom capsuleをStage Aの前提から外し、既存direct runner、固定container、timeout/resource上限、process cleanup、candidate/artifact identity、前後GPU healthを維持する最小baselineを完成させた。元worktreeの変更済み`.gitignore`と未追跡`execution_capsule.py`、`process_containment.py`はcandidateへ含めず、読取・実行・削除も行わなかった。
- A5 runtime candidateをcommit `ac2baa3a0734d0894353ba180259d979da5a831e`、tree `4e43a9c42c9aa2dfa6a6d438610fa54c4e482d10`へ固定した。`986c8b86`以降の5-file差分はP0 builderへ900秒timeout、combined 4 MiB output上限、private session/process group、TERM・2秒grace・KILL、bounded reap、同一group消滅確認、独立resource closeを追加し、artifact schema/validatorへlimitsを結合した。
- P0 cleanupのfocused matrixはsystem Pythonとrequired CPython 3.12.10で各31件をPASSした。timeout、exact output bound、EOF後残留member、TERM/KILL/close各失敗、SIGKILL失敗時の主error保持と残留member診断、KILL成功後のgroup消滅、pidfd非依存を回帰化し、focused独立再reviewはaccepted scopeのhigh/medium 0件で`PASS`した。
- fresh host evidenceはH0 305/305、H1 151/151、H2 35/35で`PASS`した。固定ROCm containerのbase H3とRMSNorm H3はcanonical `gfx1030`/`gfx1201`をcompile-onlyで`PASS`し、同じidentityのpre-GPU G0、private G1、sealed-controller semantic G1、G2、P0、post-GPU G0も全てcanonical順で`PASS`した。
- G2はread-only固定cache全13 fileを再hashし、`model.language_model.layers.0.input_layernorm.weight`の5120-byte BF16 slice SHA-256 `8104f6b0c777fd9bc60925f81a7179cfb7bf9621b4abf26a4d0f98b6e9a9bfe9`を使用した。両targetで各6 case・6 HIP dispatch、fallbackなし、health OK、process cleanを記録し、raw model/sliceは保存・uploadせずpathもreportへ記録しなかった。
- P0は両targetで各5 case・130 HIP dispatch、fallbackなし、health OK、process cleanを記録した。wall medianは約1.06 msで255/256/257境界に病的な不連続を認めず、`review_required` dispositionとして受理した。threshold承認、最適化済み、他engineより高速、performance hard gate確立の主張は行わなかった。
- review 9の最初のread-only transportはbubblewrap `RTM_NEWADDR`で全command実行前に停止したため判定へ使用せず、同一非変更scopeのfresh unrestricted transport fallbackを実行した。fallback reviewerはfull 5-file差分、固定SHA/tree、host件数、全aggregateのrow/order/state、57 sidecar、G2/P0 validator、P0 cleanup、focused 15 test、`git diff --check`を独立確認し、23:16 JSTにhigh/medium/low 0件の`PASS`を確定した。
- 手書きで再構成したlocal A5 commandは、container内`/workspace`、target別build root、numeric workflow run ID、短いUNIX socket root、canonical JSON末尾改行、builder-owned outputという現行contractとの差異を各gateでfail-closedに拒否された。最終evidenceは全て現行contractへ合わせてfresh取得したが、同じ試行錯誤を機能追加ごとに繰り返さないため、次のGPU evidence refresh前にworkflow/controllerからcommandを導出するtracked orchestrationまたはdry-run preflightを2〜4時間の独立作業単位で整備する。
- 以上によりPhase 3 Stage Aを完了し、`ac2baa3a`をpublic RMSNorm/model provenanceのrollback境界とした。Phase 3全体は未完了であり、次はA5運用負債を解消し、その後にStage BのRust model I/O・text frontendへ進む。full model生成、attention/MLP/KV/state、CLI、G3、performance最適化はStage Aの完了主張に含めない。

[対応する計画](../../../../plans/archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
