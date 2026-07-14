# AQ4 P2 resident one-case smoke runner v0.1

## 前回の要点

通常の resident batch runner は representative full-model target 84件をexactに要求し、各caseを2 warmup + 10 measuredで実行する。任意subsetや最大件数指定は持たない。

## 今回の変更点

`--one-case-smoke` は `ullm.aq4_p2_resident_smoke_binding_bundle.v3` を `ullm.aq4_p2_resident_smoke_bundle_root.v4` 境界で読む明示modeである。`--bundle-root` は必須で、`case-binding.json`、`fixture-index.json`、`identity.json`、`preflight.json`、`policy.json` の各引数は同rootの固定memberに一致しなければならない。rootは791a20c形式の19 memberをexactに持ち、各memberのrole、相対path、SHA-256、regular-file type、nlink=1、mode、`SHA256SUMS`完全coverageを検証する。

runnerは特定bundle全体のSHA-256をhardcodeしない。case-bindingはtrusted case ID、runtime-bound case SHA-256、official case SHA-256をexactに1件だけ含む。fixture index/fixture SHA、identity self/file SHA、preflight/policy、served/package/worker/guard binding、prepared dry-run/evidence、fake-ready session/driver identityを相互照合する。これにより欠落member、handcrafted partial root、別case rebound、identity/fake-readyの単純swapを拒否する。

one-case smokeも2 warmup + 10 measuredを固定し、`smoke_only=true`、`promotion_eligible=false`である。0件、2件以上、別caseへの差し替えを拒否する。通常modeの84件境界と出力schemaは変更しない。

one-case dry-runはplannerだけを通さない。`fake-ready.json`はrunner本体から直接JSON readせず、分離したchild processをexactly 1回起動してstdout handshakeとして実runnerのready identity validatorへ入力する。plan artifactはroot member inventory、bundle/fake-ready hash、session/driver identity、`fake_driver_subprocess_count=1`、`driver_fake_handshake=passed`を保持する。normative one-case modeは`--trusted-validator`と`--trusted-validator-sha256`を必須とする。validatorのraw CLI pathはresolve前にabsoluteでなければならず、全ancestorとleafのsymlinkを拒否し、single-link regular fileを`O_NOFOLLOW`でopenする。期待SHAとpinned file identity/hashを実行前後で照合し、source/stdout/report SHAをplanへbindする。validator省略、SHA差し替え、symlink経由は許可しない。これはvalidate-only evidenceであり、model load、GPU、service操作を証明しない。

## 次の行動

信頼連鎖は自己参照cycleを避けて3段に分離する。Rはこのgeneric runner validation commit、BはR blobと実dry-runをpinする更新bundle、LはB manifest SHAとR blob/validator SHAをpinしてvalidator→runnerの順に起動する最後のimmutable launcherである。この変更はRだけを完成させるため、B/L更新が次に必要である。sanctioned実行ではLが選んだbundle rootとdetached resident driverを使用し、one-case smokeの成功を84件または完全matrixのpromotion evidenceへ転用しない。
