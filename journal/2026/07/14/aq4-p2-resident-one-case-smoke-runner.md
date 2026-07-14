# AQ4 P2 resident one-case smoke runner

## 前回の要点

prepared bundle v3はone-case planを保持していたが、実runnerは84 target caseを要求するため、そのplanを実行できなかった。

## 今回の変更点

- resident runnerへ明示的な`--one-case-smoke`を追加した。
- bundle v3、case-binding、fixture index、identity間のcase ID/hashを固定した。
- 0件、2件、case swapを拒否し、通常84件modeを維持した。
- dry-runでもbundleのfake-readyを実ready validatorへ通し、validate-only handshakeをartifactへ記録するようにした。
- one-case artifactは常に`smoke_only=true`、`promotion_eligible=false`である。
- independent QA followupでone-case入口を必須`--bundle-root`へ変更した。791a20c形式のexact member/role/path/SHA/type/nlink/mode、`SHA256SUMS`、trusted case ID/bound/official SHA、fixture、identity self/file、preflight/policy、prepared dry-run/evidenceをgenericに相互検証する。
- fake-readyはrunnerから直接loadせずchild processを1回通す。任意のtrusted validatorは同一source blob SHAを前後確認してsubprocess実行し、report SHAまでplanへbindする。
- 特定bundle全体のSHAはrunnerへhardcodeせず、R→B→Lの信頼連鎖で自己参照cycleを避ける。通常84件経路は変更していない。
- normative one-caseでは`--trusted-validator`と期待source SHAを必須にした。validator省略またはvalidator SHA swapはbundle内部が自己整合していても拒否する。
- validatorのraw input pathをresolve前にabsolute検証し、ancestor/leaf symlink、hardlink、実行中のidentity/hash変化を拒否する。

## 次の行動

R=generic runner、B=R blob/実dry-runをpinするbundle、L=B manifest/R/validator SHAをpinするimmutable launcherに分離する。今回のcommitはRのみであり、B/L更新後にsanctioned one-caseを実行する。この作業ではGPU/liveを実行しない。
