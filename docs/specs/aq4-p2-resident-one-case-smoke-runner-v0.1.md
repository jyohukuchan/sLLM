# AQ4 P2 resident one-case smoke runner v0.1

## 前回の要点

通常の resident batch runner は representative full-model target 84件をexactに要求し、各caseを2 warmup + 10 measuredで実行する。任意subsetや最大件数指定は持たない。

## 今回の変更点

`--one-case-smoke` は `ullm.aq4_p2_resident_smoke_binding_bundle.v3` 専用の明示modeである。`case-binding.json`、`fixture-index.json`、`identity.json`、`preflight.json`、`policy.json` は同じbundle rootの固定memberでなければならない。case-bindingはtarget caseをexactly 1件だけ含み、bundle、case-binding、fixture index、identityのcase ID、case SHA-256、case-binding SHA-256が一致しなければならない。

one-case smokeも2 warmup + 10 measuredを固定し、`smoke_only=true`、`promotion_eligible=false`である。0件、2件以上、別caseへの差し替えを拒否する。通常modeの84件境界と出力schemaは変更しない。

one-case dry-runはplannerだけを通さない。bundle v3の`fake-ready.json`を実runnerのready identity validatorへ入力し、driver binary、protocol、runtime device、served worker/package/guard bindingのfake handshakeを実行する。plan artifactはbundle/fake-ready hashと`driver_fake_handshake=passed`を保持する。これはvalidate-only evidenceであり、model load、GPU、service操作を証明しない。

## 次の行動

sanctioned実行では同じbundle rootとdetached resident driverを使用する。one-case smokeの成功を84件または完全matrixのpromotion evidenceへ転用しない。
