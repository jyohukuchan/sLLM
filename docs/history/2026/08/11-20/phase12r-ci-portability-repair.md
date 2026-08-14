# Phase 12R CI portability repair履歴

## 2026-08-15: Phase割当と計画作成

- ユーザー指示により、CI不整合の修正へ`Phase 12R`を割り当てた。
- Phase 12のMI300X実機確認を完了扱いにせず、既存Phase 13〜20を繰り下げないremediation subphaseとした。
- GitHub-hosted CIはtracked repositoryだけで完結するhost portability/compile laneとして維持し、実GPU、model、
  llama.cpp実体比較、性能はtrusted local laneへ分離する方針を固定した。
- candidate `39ffa8eb70063282b623fee714b665ce8de5618a`のH0 3分類、H3 link 2 workflow、self-hosted G1 pendingを
  開始baselineとした。
- H0 portability、H3 link正本化・重複整理、workflow event、local parity entrypoint、integration、closeoutを
  P12R-A0〜A6へ分け、機能codeや広いGPU再検証をscope外とした。

## 2026-08-15: P12R-A0〜A4実装

- clang-format 18でPhase 10〜11由来のtracked C++/HIP差分を整形し、CIとlocalのformat contractを一致させた。
- llama Phase 5のtracked contract loadからignored `reference/llama.cpp` checkout検査を分離した。H0は固定commit/tree、
  source-lock、fixture、conversion metadataを完全一致で検査し、`--verify-reference`だけが実checkoutのmissing、wrong
  commit/tree、dirty状態をfail-closedに検査する。
- Rust dependency closureは現Cargo metadataと照合し、追加済みFP8 evidence binary/example targetだけをversioned expectationへ
  反映した。workspace member、package、edge、feature、Cargo.lockの検査は弱めていない。
- public-runtimeとRMSNorm H3のfinal linkへ`libamdhip64.so`、`libhipblas.so`、`libhipblaslt.so`を順序固定で追加し、
  CMakeのpublic-runtime closureとのdrift検査を追加した。リンク成功後に観測されたVMM、hipBLAS/hipBLASLt、FP8/decode、
  wave64 kernel、causal-attention stub symbolもexact allowlistへ同期した。
- semantic RMSNorm G1とPhase 7 lifecycleから通常push/schedule/release triggerを外し、self-hosted jobを含むworkflowへ
  automatic triggerを再導入する変更をvalidatorで拒否した。RMSNorm H3はpublic-runtimeと同じlink closureを使い、固有の
  wave32/wave64 registration/artifact contractを持つmanual regressionとした。
- `ci/tools/run_local_verification.py`を追加し、`ci/matrix/host-v1.json`とsuite registryからH0/H1/H2を解決した。dirty runは
  `local-development, immutable=false`、clean runはstrict identityとし、GPU/model/network属性の混入を拒否する。

## 2026-08-15: P12R-A5 integrationとreview

- local wrapper run `local-1786727252`はH0 `510/510`、H1 collected `412` / selected `379`、H2 collected `44` /
  selected `36`をPASSし、host aggregateもPASSした。matrix manifest SHA-256は
  `0570221d760d4e2db89e2674ebc2d5e8fe5a9b4d6f0e0188ded455a70d08ab0b`である。
- clean integration snapshot `aa2a27b2f6c8b93538a780bdafd3d2795ed6fa78` / tree
  `544fa37fbfacecf24be50ee6c02399a3adcc7aab`でcore H3 `gfx1030`/`gfx1201`とaggregateをPASSした。aggregate
  SHA-256は`8578422ef29f95cccea58bfa8e2476a4036641de4f454f6bb89cfe000eedeac7`である。
- public-runtime H3はbuild inputが同じclean snapshot `ee93664c41247008c46b378e165748c9823692c8` / tree
  `ce3ed611a3c3828a586653a1ee8d195b563f4e3d`で両targetとaggregateをPASSし、aggregate SHA-256は
  `6707170a902306053a7d11e208af0fe5a5fb08113418452f3cfb98ba68c92bc7`である。
- RMSNorm H3はsnapshot `aa2a27b2f6c8b93538a780bdafd3d2795ed6fa78`で両targetとaggregateをPASSし、aggregate
  SHA-256は`f146a5f2dbc804f0a0e548aecccfe5aadb6a09fb823b7df591731c8b844f10f1`である。
- 以上はROCm 7.14.0 pinned containerによるcompile/link/code-object inspectionであり、GPU execution、数値、model、性能の
  PASSではない。Phase 12RではMI300X/V620/R9700の新しいruntime runを行っていない。
- integration reviewはportability、dependency closure、link completeness、trigger境界を一回確認した。llama conversion
  identityを部分比較へ弱めた指摘1件を完全一致へ戻し、llama focused 48件で再確認した。

## 2026-08-15: P12R-A6 closeout

- testing文書、CI・テスト方針、main plan、forward queueをmanual self-hosted/local GPU境界へ同期し、本planをarchiveした。
- Phase単位のcommit/push境界を適用し、Phase 13へgreenなhost portability baselineとregistry-driven local entrypointを渡す。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase12r-ci-portability-repair.md)
