# AQ4 P3 maintenance v7 cascade

## Authority

- launcher final v7: commit `60461d796ba64a7f0ba28353cb4f263d08d18dab`, tree `f3e461734923222dde178a75dbc50600689b9737`, blob `98105c77f330f794ebb326d2fb19b70f2a21c2bc`, SHA-256 `65b6258cb07a053455c05e65c184a873a3d39c2b2fe1e237970bbd11147dc750`
- capture/parser: commit `e86cf512183574340ddfc6564477395766262092`, tree `4ca4ba52084b15c78a95e5a7c4580e5bd2fd2a07`, blob `124f5e89834fda2ace8a2d8c42e362ec1adce29c`, SHA-256 `ab3d77d4bc77c43c82ac9ee1d993a029266119ca3365f1a285ab03cca9bcf00a`
- runner、validator、B、resident の authority は v6 から変更しない。

## Cascade

- profile ready、dry-run、maintenance evidence、capture output を未使用の v7 経路へ更新した。
- QA attestation は launcher execute 71 件、capture 56 件、resident trust chain 381 件、aggregate 543 件へ再集計した。
- v6 の ready、dry-run、execute、execute evidence、maintenance evidence、capture failure、operator result、actual audit は `SHA256SUMS` がすべて一致した。これらを履歴証跡として読み取り専用で検証する。
- GPU、service、actual は実行していない。

## Verification

- maintenance 全 155 件: 確認待ち
- aggregate QA: 確認待ち
- fresh v7 経路未作成: 確認待ち
