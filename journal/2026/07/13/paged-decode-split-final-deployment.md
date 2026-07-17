# Paged decode split 最終配備証跡

## 前回の要点

長文 decode の主な律速は、文脈全体を毎 token 走査する canonical paged attention だった。モデル名に依存した実験分岐では将来の plain Qwen3、sigmoid-gated Qwen3.5、任意の GQA 比率、block table geometry へ拡張しにくく、短い文脈での回帰と実験環境の取り残しも避ける必要があった。

## 今回の確定内容

最終 git HEAD は `4be10d007c48db444f6b18fe6ac22e09dc17f168`（`Fix legacy paged geometry preflight`）。関連する主要実装コミットは次のとおり。

- `6096c03` generic split runtime
- `1ccb019` typed registry
- `426c49a` resident integration
- `ffb62a0` production promotion
- `f7fe63b` versioned receipt
- `4be10d0` generic paged geometry compatibility fix

選択条件はモデル名ではなく、paged GQA geometry、runtime feature probe、caller-owned persistent workspace、source tile 128、cache length threshold 256 である。plain/sigmoid-gated の両方、任意の GQA/block table、将来のモデル geometry を typed generic registry で扱う。feature または workspace が使えない場合、あるいは cache length が閾値未満の場合は canonical single へ fail-closed fallback する。workspace は request state が所有し、decode step ごとに再解決・再確保しない。

production の選択値は `source tile=128`、`cache_len >= 256` である。短文脈の crossover を baseline → tile128（tok/s）で示すと、`p16 70.087 → 70.092`、`p64 69.045 → 68.286`、`p128 67.215 → 66.740`、`p160 66.200 → 66.751`、`p224 64.522 → 66.640`、`p256 63.752 → 66.962`、`p512 58.115 → 66.821`、`p1339 約44.7 → 約66.3` となった。閾値未満の短文脈を single に残すことで、長文 decode の改善と短文回帰回避を同時に満たす。

## 配備状態

最終 active manifest は `/etc/ullm/served-models/active.json` で、SHA256 は `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80`。required guards は 30、split guard は exactly once、実験用 environment variable は存在しない。service はユーザー指示どおり inactive のままである。

## 最終 smoke と性能証跡

active manifest を直接使った p1339/g64 smoke は、prefill `129.327636 tok/s`、decode `66.526692 tok/s` だった。64 token はすべて `87328` で、token SHA256 は `9423d3a2cc7b87ae26f58b35c39e96b476273c02b5b7e121abf51d4f7ddcbe9d`。baseline と完全一致した。

production promotion 後の no-env kernel profile は、split partial の平均 `144.068 us`、merge の平均 `5.561 us`、legacy paged attention の平均 `1055.319 us` だった。506 decode calls の attention 合計は `533991.609 us → 75712.702 us` となり、約 85.8% 削減した。

formal evidence は `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/resident-promotion-evidence-paged-decode-split-v1.json`、SHA256 は `11579142039d7b5de1a191657bbcb8ed9edb3b19e38551d5b713c81a798a329a`。source/worker/legacy/manifest の binding は一致し、resident-vs-legacy の 2 cases は token が exact match、`verified=true` である。receipt は `promotion-paged-decode-split-v1.json`、SHA256 は `3e6f343831f267f69d5bf8f1612d9c8a0da575e213a9f3b7ecdcaf29ae49bc47`。旧 `promotion.json` は rollback 用に不変である。

## 検証結果

- runtime-sys: 156 passed（既実施）
- 最終 engine lib: 689 passed / 1 ignored / 0 failed
- profile pytest: 19 passed
- worker tests: 11 passed
- snapshot tests: 2 passed
- active validator: 成功

## 残課題

prefill は約 `129 tok/s` で、ユーザー期待の数千 tok/s にはまだ到達していない。今回の変更は長文 decode bottleneck を対象にしたものであり、prefill の不足は別の profile と最適化で扱う必要がある。次の行動は、prefill の主要 kernel・転送・量子化経路を個別に profile し、改善候補を別証跡として検証することである。
