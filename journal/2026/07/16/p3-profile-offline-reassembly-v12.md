# P3 profile offline reassembly v12

## 前回の要点

- maintenance final authorityはcommit `9ff2b8861f6d91935679db3bdf1b4af37bc6a543`、tree `ac63aa5dadb213ba4f1f43ef6ff3b2a9c8157e2a`、blob `1d7c99815a4ee3cbc5f8ef4f5ab438d752338dcd`、raw SHA-256 `fb2fc515570b1889c21bd170434159845048971339a084a85027707f79665345`である。
- capture parser authorityはcommit `418e507214b2a4c0352ac8867bf9689b81948ca4`、schema v2であり、actual-v14はcommit `a2fe1ebac5d631919ca9082e17fda2126759a385`の6 roots・35 filesとして固定されている。
- ready-v17 authorityはcommit `19f7d390b97b1e8f0daa72e1007267c27ab4061b`である。offline-v12の2出力rootとactual future 4 pathsは生成前にすべて未作成だった。

## 今回の変更点

- canonical maintenance generatorを1回だけ実行し、次の2 rootsを生成した。
  - `p3/aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v12`
  - `p2/resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v12`
- capture rootは40 files、evidence rootは2 filesである。rootsと`measured-runs/`はmode `0555`、全filesはmode `0444`・nlink 1である。`SHA256SUMS`は全payloadを重複なしで覆う。
- capture artifactはschema `ullm.aq4_p3_diagnostic_rocprof_capture.v2`、status `complete_diagnostic`である。raw kernel 12,263 rowsの隣接order inversionは207、12 marker groupsは各928 rowsであり、per-group inversionは`35,10,8,19,23,16,7,20,17,26,3,15`である。全groupでrow count、dispatch/correlation ID set・multiset、duration sumのpre/post保存が成立した。
- derived measured kernel splitは10 runs×928 rows、HIP API splitは10 runs×2,318 rows、memory-copy splitは10個のheader-only traceである。raw traceは並べ替えず、capture v2 validatorがrawからprovenanceとsplitを再計算した。generic direction adapterや手動sortは使用していない。
- raw 7 filesはactual-v14 capture-v10とbyte-equalである。主要raw SHA-256はkernel `817f97cd97a09c6a2affa6fbc079ea2cac6a9069ae1019b18c8f35b98c0b27dc`、HIP API `0f6db01548159fe82c831ed67b617820e610b2df3a5bdfdd98eea865d6980470`、memory copy `e91bc82d7819509b5bbc841c6a037dccd74cc40abfa90d6e340b09a10e80eda2`、marker `4f848455354da857fb0bb89976639a03655784c086ec0311120b8139922e5280`である。
- capture artifact raw SHA-256は`f07b520a46b1b0e641e10c3a22179aa4c499a337a3328505e49955f9be333f4a`、self SHA-256は`d7d55ce12bfa3f91b0945922ed73aff900dff436d76a5646f9f78bd481ec8fcc`である。capture root `SHA256SUMS`は`555b8a1711b02cb51dbb5d1b3bc5ee1f2c4a28feabe858e7f13d78b85a99ce73`である。
- offline evidence raw SHA-256は`731a8b81c9553280412f8ae3d45028bf71f5c64cc5439d5296461fcd2a99d991`、self SHA-256は`0a4b1f421f807cb9e2fe857d05e9ef198c4a957608966c0df9bb26f1bde9a657`、evidence root `SHA256SUMS`は`6ff1a790de7098301c0735b647fbe12fd8288309adfd89abb39087d702f14fb9`である。
- execution countersはoffline assemble 1以外すべて0である。workload、rocprof、GPU、service、operator、actual、model loadは実行していない。actual-v14 sealは生成前後で同一だった。actual future runtime-v11、execute-evidence-v11、capture-v11、maintenance-v12は未作成のままである。

## 次の行動

- 2 rootsとこのjournalだけをartifact authority commitとして固定し、commit後にcanonical validator、mode/nlink、`SHA256SUMS`、archiveとGit objectのbyte一致を再検証する。
- offline-v12 authorityをready/operator cascadeへ渡す。normal actual-v11は別途live gateと明示的実行許可が揃うまで実行しない。
