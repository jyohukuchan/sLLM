# SQ8 OpenWebUI Stop formal gate

## 前回の要点

- 実ブラウザでStop click、gatewayの`client_disconnect` cancel、`reset_complete=true` releaseまでは確認済みだった。
- 診断runはgateway release後の制御ファイルを渡さずに終了したため、同じchat内の別messageによる回復までを単一の正式証跡として閉じていなかった。

## 今回の変更点

- `tools/run-openwebui-stop-gate.py`を追加した。AF_UNIX datagram observer、`SO_PASSCRED`によるMainPID/UID照合、authoritative journalとのbyte一致、gateway Stop/recovery順序、OpenWebUI browser interim/final証跡、排他的な0600 nonce制御を一つのfail-closed gateに統合した。
- observerとjournalは上限付きで継続排出し、終了時にもキュー済み入力を捨てない。observer close後の固定snapshotに対してtrace数、active数、cancel/release、recoveryを再検証する。
- tokenとbrowser scriptは`O_NOFOLLOW`の単一FDからprivate stagingへ固定し、そのread-only snapshotだけをDockerへbindする。成功公開前にtoken、script、nonceを削除する。
- Dockerはmutable tagを拒否し、`sha256:<64hex>`または`name@sha256:<64hex>`だけを受理する。host network、4 bind mount、host UID/GID、PID上限、no-new-privilegesで一時起動し、正規化したcontent digestをsummaryへ残す。公開summaryはtoken、prompt、URL、chat/message/request/completion IDをcleartextで含まない。
- observer/journal/browser rawとsummaryは不完全名からのrename、および最後のrun directory renameでatomicに公開する。失敗時にはpass artifactを残さない。
- `tests/test_run_openwebui_stop_gate.py`を追加し、strict JSON、lifecycle順序、credential、close時drain、journal cursor/byte一致、control、browser契約、secret、atomic publish、private snapshot、process group cleanupを14件で検査した。

## 検証

- 新規unit 14件: 合格
- 既存OpenWebUI browser static 6件: 合格
- `mypy --strict`: 合格
- Python `py_compile`: 合格
- Node `--check`: 合格
- `git diff --check`: 合格
- 実GPU/browser formal runは後続で実行し、以下の結果で合格した。

### 実formal run

- output: `/home/homelab1/datapool/openwebui-stop-formal-pilot-20260711-180303`
- browser image content ID: `sha256:dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa`
- 最初のrunでOpenWebUI 0.9.4の`chat:active`、回復まで進んだrunで`chat:outlet`を状態通知として観測した。どちらも本文0 byte / `done=false`のみを許可し、本文またはterminal状態を持つ改変testを追加した。
- 最終runはexit 0、`summary.json.passed=true`、browser action 9件、Socket.IO event 27件、gateway request 2件、maximum active 1だった。
- targetは`outcome=cancelled` / `client_disconnect` / `reset_complete=true`、回復は同一chat内の別messageで`outcome=stop` / `reset_complete=true`だった。gateway PID 712139と`NRestarts=0`は不変である。
- Stop click→gateway cancel: 2,953,185 ns、cancel→release: 5,140,666 ns、release→recovery admission: 1,197,734,947 ns、recovery duration: 2,673,119,590 ns。
- observer 11件 / journal 11件は全payload byte一致、journal cursor 11件は全て一意、`_PID=712139`、`_SYSTEMD_UNIT=ullm-openai.service`、boot IDは単一だった。
- screenshotは41,093 bytes、SHA-256 `d68acebb01b7ef9acfeeca0b32064e17366f822bab3753dca3f2fdc2015e236b`。目視でStop control、部分本文、model名を確認し、更新通知の遮蔽はない。
- 成功後はobserver socket、nonce/control、private token/script snapshot、transient containerが残っていない。OpenWebUIはhealthyである。

## 次の行動

1. 成功bundleのbrowser action、observer、journalをrelease collectorの同session recordへ変換する。
2. 独立validatorにStopと同一chat回復の再構築gateを追加する。
3. 5段階cancelの他4件を同じobserver release同期で閉じる。
