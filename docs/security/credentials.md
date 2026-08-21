# Credentials and privileged operations

## 基本方針

- 無人での進行を優先しつつ、secret exposure、権限の継承、credentialの寿命を最小化する。
- privilegeやcredentialの利用はtask scopeを拡張しない。対象確認、影響確認、破壊的操作の安全確認は常に必要とする。
- `.gitignore`は誤commitを減らす補助であり、access controlや漏えい防止の代替ではない。

## sudo

- 特権操作はmain agentだけが、task scope内で対象と効果を限定して`sudo -n`で実行する。
- 現在の専用local hostは、`homelab1`に`NOPASSWD: ALL`を意図的に許可している。無人での進行を優先するための明示的かつ受容済みのrisk trade-offであり、least privilegeなsudoers設定ではない。
- sudo用の平文passwordや`passwords.txt`は不要であり、使用しない。authentication情報をstdin、argv、環境変数、file、pipeでsudoへ渡さない。

## sudo以外のcredential

- secret manager、workload identity、短命のruntime injectionを優先する。short-lived、least-privilege、最小scopeとする。
- non-secretな代替手段がなく、taskに必要で、ユーザーが対象local credential fileの読取りを明示的に許可した場合に限り、main agentだけがその値を読み取れる。
- credentialをprint、log、commitせず、argv、source、shellまたはprojectのhistory、文書、issue、artifactへcopyしない。subagentやuntrusted codeへ渡さない。toolが対応する場合はstdinまたはfile descriptorを使い、環境変数を使う場合も不要なchild processへの継承と保持時間を最小化する。
- local credential fileはignoredかつuntrackedとし、modeを`0600`、可能ならparent directoryを`0700`にする。AIは`passwords.txt`を編集しない。

## sLLM server credential file

- Phase 39のserver key fileは1行につき`user:<token>`または`admin:<token>`とし、空行、未知role、空token、
  whitespace/controlを含むtoken、duplicate tokenを拒否する。最大32 key、1 token 4,096 byte、file全体64 KiBである。
- serverはkeyのSHA-256 digestだけを保持し、requestごとに固定32 byteのconstant-time比較を全entryへ実行する。
  admin keyはuser surfaceも利用できるが、user keyは`/slots`、slot cancel、key reloadを利用できない。
- key fileはregular fileかつnon-symlinkでなければならない。Unixではgroup/other permission bitが一つでもあれば起動・reloadを
  拒否する。`POST /admin/keys/reload`は新fileを全検証してからsnapshotを交換し、失敗時は旧key setを保持する。
- 従来の`--api-key-env NAME`は単一user keyとして維持し、`--api-key-file`とは同時指定できない。どちらも省略したlocal profileは
  user surfaceだけopenで、admin surfaceはcredential不在のため閉じる。
- TLS private keyもregular non-symlink fileかつUnixのgroup/other permissionなしを要求する。certificate/keyはpairで指定し、
  Unixでは`O_NOFOLLOW`で開いた同じdescriptorから各1 MiBを上限に読み、PEM parseをmodel/GPU load前に完了する。
  ready/shutdown logへpathや内容を出力しない。

## Untrusted CIとGPU job

- fork、外部PR、未検証branch等のuntrusted codeへsecretを注入せず、secretを保持するself-hosted runnerで実行しない。
- untrustedなGPU jobはephemeralかつ隔離し、host home、credential file、secret manager、Docker socket、永続runner credentialへアクセスさせない。modelはhost側で検証したread-only mountから提供し、storage credentialをjobへ渡さない。
