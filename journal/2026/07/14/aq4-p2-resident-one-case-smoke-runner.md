# AQ4 P2 resident one-case smoke runner

## 前回の要点

prepared bundle v3はone-case planを保持していたが、実runnerは84 target caseを要求するため、そのplanを実行できなかった。

## 今回の変更点

- resident runnerへ明示的な`--one-case-smoke`を追加した。
- bundle v3、case-binding、fixture index、identity間のcase ID/hashを固定した。
- 0件、2件、case swapを拒否し、通常84件modeを維持した。
- dry-runでもbundleのfake-readyを実ready validatorへ通し、validate-only handshakeをartifactへ記録するようにした。
- one-case artifactは常に`smoke_only=true`、`promotion_eligible=false`である。

## 次の行動

CPU fake driverで2+10とcase bindingを確認した後、別途許可されたR9700 one-case smokeだけに使用する。この作業ではGPU/liveを実行しない。
