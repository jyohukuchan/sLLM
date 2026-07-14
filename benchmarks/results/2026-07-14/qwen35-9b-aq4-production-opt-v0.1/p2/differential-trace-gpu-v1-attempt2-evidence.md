# P2 AQ4 GPU 中間差分トレース（attempt2）証跡

このファイルは、2026-07-15 JST に完了した attempt2 の read-only 証跡をまとめる。attempt2 の出力・ログ・marker は run-started marker があるため保持しており、attempt1 の artifact は変更していない。

## 実行 identity

| 項目 | 値 |
|---|---|
| gate script | `p2/differential-trace-gpu-v1-input/run-gpu-gate-attempt2.sh` |
| gate script SHA256 | `1d6b22df73cf3646d5048ce6785f3caae29a030f87f8bfe944a42047e4adbd07` |
| build commit | `28ec343ac59e6d22e710035d7874df9fbd8f890f` |
| candidate binary SHA256 | `356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7` |
| active manifest SHA256 | `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44` |
| package manifest SHA256 | `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad` |
| worker SHA256 | `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d` |
| runtime | HIP device index 1 / AMD Radeon Graphics / gfx1201 |
| candidate output | `p2/differential-trace-gpu-v1-attempt2/` |

The candidate invocation used the corrected package argument:

```text
ullm-aq4-differential-trace \
  /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package \
  tests/fixtures/qwen35-aq4-p2-oracle/cases.json \
  p2/differential-trace-gpu-v1-input/replay.json \
  p2/differential-trace-gpu-v1-attempt2 1 --enable-intermediate-trace
```

The package argument is the directory containing the regular, non-symlink `manifest.json`. This matches the trace source's `package_dir.join("manifest.json")` behavior. The script preflight also required the package argument to differ from the product root, and verified the expected manifest SHA.

## Artifact hashes and filesystem identity

| artifact | SHA256 | bytes | owner | nlink | dev:ino |
|---|---|---:|---:|---:|---|
| attempt2 gate log | `592dddca795324f0744e67370594d05a07d7449022a3cd94e677d33337f4c8d3` | 7353 | root:root | 1 | 10302:10507330 |
| attempt2 monitor log | `c3a806818f394a9e1ccd81071936d0ab69f7cf6d6266643478b66b84d223c135` | 3070 | root:root | 1 | 10302:10512889 |
| attempt2 run log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | root:root | 1 | 10302:10512893 |
| output `SHA256SUMS` | `bd8da05bcb360b6b621c0c7f6c645444c5b4afb871e921f27f54c17533bfd966` | 239 | homelab1:homelab1 | 1 | 10302:10512909 |
| output `manifest.json` | `dd02cb0d6a802090c26ed35b986daa3120970015afe3e55160e6b8d7b72b1284` | 4264 | homelab1:homelab1 | 1 | 10302:10512896 |
| output `payload.jsonl` | `bcc54f8e6968412244f642e24889483ab43dee751ab198f8d5eb0365b1dced00` | 30253 | homelab1:homelab1 | 1 | 10302:10512895 |
| output `runtime.json` | `241347906a25a8dc7f1c3257680f0a06fe05297c8976baffa1f55097f5b5a3c2` | 85 | homelab1:homelab1 | 1 | 10302:10512908 |
| detached candidate | `356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7` | 2928200 | root:root | 1 | 10302:10509260 |
| service-stopped marker | `46ef3e4505384f7f4873b6b4f0c32d205580325d9901319283ab105dd5f683bb` | 369 | root:root | 1 | 10302:10512888 |
| run-started marker | `3a0c32170a7952e8f3ae3c8889a23c2fc49f0fb7f8836b75868709cbb49db972` | 37 | root:root | 1 | 10302:10512890 |
| observer-sample marker | `4d547a53f64ace63203a264860bd53421f845ecc9a333775c81540fb9898b340` | 20 | root:root | 1 | 10302:10512892 |

`observer-failed.marker` は存在しない。monitor には 6 samples（00:09:06--00:09:12 JST）があり、candidate 実行中は card2 が最大 40% GPU use、最大 7,439,523,840 B VRAM を使用した。card0/card1 はほぼ idle だった。

## Gate 結果と verifier cwd バグ

candidate 自体は manifest、payload、runtime を生成し、Python の identity/input binding 検査も通過した。gate の最終 rc=1 は、`SHA256SUMS` に相対名（`manifest.json`、`payload.jsonl`、`runtime.json`）が記録されているのに、gate が output directory 外の cwd で次を実行したことによる。

```text
sha256sum -c "$OUTPUT/SHA256SUMS"
```

gate log 末尾は `sha256sum: manifest.json: No such file or directory`（3 件）を記録する。これは output 内容の破損ではない。次の read-only 検証では 3/3 OK だった。

```text
(cd p2/differential-trace-gpu-v1-attempt2 && sha256sum -c SHA256SUMS)
manifest.json: OK
payload.jsonl: OK
runtime.json: OK
```

attempt2 は既に一度だけ authorized production run を実施したため、この verifier バグを理由に再実行していない。

## Trace / analyzer 結果

- output manifest: 3 rows、各 row 35 stages、`decoder_layers=32`、hidden coordinates `[0,1,1024,2048,4095]`、logit coordinates `[0..31]`。
- input binding: cases SHA `15fed90d...4836f2`、replay SHA `1ee0b922...4747dc`。source-v2 と candidate の row context hashes は一致。
- analyzer command（read-only、rc=0）:

  ```text
  /usr/bin/python3.12 tools/trace-qwen35-aq4-differential.py analyze \
    --source-trace p2/source-differential-trace-v2 \
    --path-trace p2/differential-trace-gpu-v1-attempt2 \
    --output /tmp/p2-attempt2-analysis
  ```

- 全 3 行の first mismatch は `decoder_layer:0`、missing stages は空。
- embedding は max abs 0、cosine 1.0。
- stage aggregate peak は lm_head max abs 8.347782、relative L2 0.540729、cosine 0.998058。final norm は max abs 1.084648。
- greedy は source/candidate が `[220/41330, 16/16, 15/15]`。intermediate trace 自体には top-k 配列がないため、source oracle と、attempt2 と同じ package SHA に結び付いた既存 candidate path oracle で top-10 set overlap を確認すると row0 は 1/10、row1/row2 は 10/10。

## Service restore

gate cleanup 後に service は `active/running`、`NRestarts=0`、MainPID `2952687`、lock owner `2952687` に復帰し、healthz は `{"status":"ok"}`、models は `ullm-qwen3.5-9b-aq4` を返した。active/package/worker SHA は pre-run 値から変わっていない。runtime directory と lock は元の owner/mode/nlink に復元された。

## attempt1 との分離

attempt1 の gate/monitor/run log SHA は script preflight に固定してあり、attempt2 artifact と別名で保持している。attempt1 の失敗は product root を candidate の package argv に渡したことによる package manifest 不在で、attempt2 は `PACKAGE/package` に修正した。attempt1 の raw evidence は書き換えていない。
