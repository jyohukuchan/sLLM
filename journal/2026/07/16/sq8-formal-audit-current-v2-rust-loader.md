# SQ8 formal audit current-v2 Rust loader hardening

## 前回の要点

正式なcurrent-v2監査レシートは、topologyとtestsに新しい必須項目を追加していた。一方、Rust loaderは移行期v2の型だけを保持していたため、正式な認可済みserved-modelを`promotion.authorization_audit typed schema is invalid`として拒否していた。

## 今回の変更点

- `authorization_audit.topology`と`tests`を、legacy、migrated-v2、current-v2のexact field-setを持つstrict unionへ変更した。全variantでunknown fieldを拒否し、current-v2では旧`worker_live_identity`を契約上許可しない。
- current-v2 topologyのhistorical direct authorization count、lineage predecessor count、predecessor entries SHAを型付きで検証した。監査レシート、lineage reference/document、current GO、immediate v2 predecessorのcountとdigestを相互照合する。
- current-v2 testsの`lineage_v2_successor`、`lineage_old_v2_authorization_rejection`、`served_model_cpu_validation`を必須化した。GPU/service系の実行結果がtrueなら引き続き拒否する。
- unknown、missing、type、count、digest、predecessor、test true、旧・現行variant混在を拒否する専用負試験を追加した。現行正式`b1f2a3a88ea24d65298129c065e77fede46711975ded40ea3a0a802634d6db43`、現行非正式`484ac20f4a9828152c895cd6064371c1851b34dece64a996bf445c431a29d21e`、旧正式`a4d541a8c44edd73e505f223b15cf92933b4e0bf2a257e8e9d08dbad94192542`、移行期v2`31ba7f6483a5baf7d84f8b45a5d86d02c2c22d72d229ca74cfe593192e98ccdd`をfresh release rlibへリンクしたCPU検証器で受理した。
- postmortem証跡は、actual failure receipt `01c6aa1a44f90612f94e3ff285aad4f52734b5aa662f4b4e2e438fe13b74dd63`、maintenance receipt `7af938d5c0cfad98cbcb5d75acadddd4f619f416c2d9e2d790c78c974d1a1ea7`、actual SHA256SUMS `bccffe5b455e2bf5c464edf2ae3272b23c7df8cd85ec2ae660cd666bc169f6a1`、formal audit receipt `08044245855b9bc2d59902fc1b803f47eef35e7ed3932e07ac174c2e74457d60`で固定した。
- `CARGO_BUILD_JOBS=1`でserved-model試験15件、worker試験18件、workspace checkを通した。full lib suiteは748件pass、0件fail、isolated HIP試験1件ignoreで完走した。fresh release worker SHAは`4dcf1bd3164d0a83aec4ded51c199876d407e22a325fff9d7015df7648c9e050`、rlib SHAは`f092100f641e2ddd7571d743769a22509878c9795e761e002382dc5c6f4213c2`、CPU検証器SHAは`6a83df01de986d6604665f6f16e4526c0a3061a82ca00aa2564a2024c82b4f58`である。
- full package testは既存の`ullm-aq4-p2-full-model.rs:3130`で`PromotionContract`初期化に`authorization_audit`、`authorization_lineage`、`readiness`がないためコンパイル停止した。担当外の既存不整合として変更していない。

## 次の行動

このRust loader変更をfresh independent auditへ渡す。今回の作業ではGPU、service、sudo、actual executionを行っていない。
