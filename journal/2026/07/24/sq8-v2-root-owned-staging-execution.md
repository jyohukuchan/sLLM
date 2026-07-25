# SQ8_0 v2 root-owned staging execution

Date: 2026-07-24
Recorded retrospectively: 2026-07-26

Status: the sealed SQ8_0 v2 worker release was staged below protected
root-owned ancestry.  This retrospective record does not authorize or record
an SQ8_0/AQ4_0 final activation, and AQ4_0 root operations remain unexecuted.

## 前回の要点

SQ8_0 final worker release
`uLLM-sq8-v2-final-worker-release-3bc9078d` had been sealed from source
commit `3bc9078d1ca5a49aad060d667aac19e2aa53ee86`, but its user-owned
`/home` ancestry could not satisfy the protected runtime-closure policy.
The staging runbook selected a root-owned no-hardlink destination under
`/opt/ullm/releases`.  AQ4_0 had no compliant root-owned closure and was
explicitly deferred to a separate AQ4-to-AQ4 hardening promotion.

## 今回の変更点

2026-07-24 に、sealed SQ8_0 v2 worker release を次の protected
root-owned destination へ staging した。本エントリは 2026-07-26 に事後記録
したものであり、下記の検証値も同日に read-only で再測定した。

```text
/opt/ullm/releases/uLLM-sq8-v2-final-worker-release-3bc9078d
```

再測定では、`/opt`、`/opt/ullm`、`/opt/ullm/releases` はすべて
root:root mode `0755` だった。release directory は root:root mode
`0555`、nlink 2、各 metadata member は root:root mode `0444`、nlink
1、`ullm-sq8-worker` は root:root mode `0555`、nlink 1 だった。

`sha256sum -c SHA256SUMS` は README、build provenance、build receipt、
worker の4 memberすべてで `OK` を返した。実測した member SHA-256 は次の
とおりである。

| Member | SHA-256 |
|---|---|
| `README.md` | `b3e2157a02105d1ff7e8771ee0b51bea76d128f1e4ec5e086a8e369d817cf07b` |
| `SHA256SUMS` | `ded0a829ef8ab67a19883b454621131a63ff036afe1181d931a5e39d1cd548c5` |
| `SEALED.json` | `e01c18593606e173dc154a584feb68d511e224f418ebf2b700aaf377fd171381` |
| `build-provenance.json` | `d4a123210ea9680e115f2af1ea8e2285bf6a5c36a18c5db7b8ec779231a0c19d` |
| `build-receipt.json` | `986708497df09d4d7998f79c0e5fe29a0a69c8c37aa7ed2e28643c16faf69cd3` |
| `ullm-sq8-worker` | `0b9989c26e656123addef15ffbf96b1aadf866a6eca06f02af8cab9bccb18a83` |

`SEALED.json` の全 hash binding も staged bytes と一致した。source
commit は `3bc9078d1ca5a49aad060d667aac19e2aa53ee86`、source tree は
`bd95c4f65168b05f4ed572a7f89e35be23ede975`、worker / build provenance /
build receipt / SHA256SUMS はそれぞれ上表の hash である。

ユーザー側 source release
`/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-v2-final-worker-release-3bc9078d`
との `diff -r` は exit 0、出力なしだった。対応する6 memberはすべて
source と staged で別 inode であり、staging copy が hardlink ではないこと
も確認した。

`/etc/ullm/served-models/active.json` は read-only で SHA-256
`5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` を
観測した。この staging では同ファイル、SQ8_0/AQ4_0 candidate/release、
`ullm-openai.service`、GPU、AQ4 asset を変更していない。AQ4_0 の
copy/chown/staging/final activation を含む root-operation は依然として
未実行である。

## 次の行動

1. staged SQ8_0 release は protected root-owned input として読み取り専用で
   保持し、profile、promotion pair、candidate、campaign をこの記録だけで
   作成・変更しない。
2. AQ4_0 は別途 review された AQ4-to-AQ4 runtime-hardening promotion を
   実施するまで、既存の user-owned closure を変更しない。
3. SQ8_0 / AQ4_0 の final activation は、人間の明示承認を得るまで
   `/etc/ullm/served-models/active.json` の実バイトを差し替えない。
