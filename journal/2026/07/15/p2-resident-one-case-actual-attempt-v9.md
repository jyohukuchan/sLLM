# P2 resident one-case actual attempt v9

## 前回の要点

- v8はready/model load/prepared preflightを通過したが、driverがruntime `gfx1201`を`RDNA4`へ変換してbound caseの`gfx1201`と比較し、case beginで拒否した。
- v9はsingle-link prepared bundleとrunner v6でarchitecture bindingを整合させ、2 warmup + 10 measuredを完走する契約だった。

## 今回の変更点

- commit `2f52c0e5fc0abd2860c312708eeb9b3db87fca5a`のoperator manifestだけをargv源とした。manifest SHA-256は`18d59f07cc2d40f08fb70d7d2dc1c1331430927187048871abc18188bae7a6a9`、canonical command SHA-256の独立再計算値は`627ba28b5137110ff24a46117c5021f19dc3537f22c794514a7997c0c7b3c5df`でrecordと一致した。
- operator/ready SHA256SUMS、Python/harness/ready artifactのhashと権限、harness latest commit `3e24fb66c7fd5f5604afdc48e972c7291c7eed26`、ready latest commit `8b2fa37d72cc710eeaffc6646799cc6a1b9e4634`、ready semantic bindingを確認した。3 fresh outputは実行直前までABSENTだった。
- preflightはformal health `9/6/6`、全4 endpoint 200、service main PID `1553870`、worker PID `1554296`、`NRestarts=0`、AMD/KFD owner、busy lock、production hashesをPASSした。RAM availableは約87.2GB、disk availableは約2.56TBだった。
- 同一PTYでsudoをprimeし、manifestのcwdと9要素argvを`subprocess.run(..., shell=False)`へ変更せず1回だけ渡した。開始は`1784085765108889339` unix ns、終了は`1784085859233246122` unix ns、elapsedは`94,124,356,783 ns`、return codeは`0`。再試行とprofile実行はしていない。
- lock substrateはdirectory device `26`/inode `761682`、lock device `26`/inode `761684`で作成された。stopped gateは2 pollともstableで、poll SHA-256は`468dcab035f523dd52883afa5a0e74780a8f4683449114202756039e22727619`と`dc7503bbe37f016573228a80f6939594982835e8e612b49ed48cdb6560342daa`。live preflightはAMD/KFD owner `[]`、VRAM used `0`、同じlock inode freeでPASSし、SHA-256は`46d76d111cdab6986d1a58f040c6438f14841fc032e86e36eb1e2563c69bcaa9`だった。
- validatorはexit code `0`。runner PID `1654328`は同じlock inode `761684`を取得し、driver PID/process group `1655055`を起動した。driver identityはSHA-256 `18d8d1a6da74b29a0e1bd38d691827a59a8f47309b994a645c8b989a34900f76`、build commit `eb7bf4513a5bdcc8ea44f111ef42e7fa735a7edf`、protocol v2、HIP/`gfx1201`/R9700/runtime index `1`。resident model loadsは`1`だった。
- one-caseはstatus `ok`で完走した。driver protocolはready `1`、case begin/end `1/1`、warmup `2/2`、measured `10/10`、stdout event count `15`。全12 runはstatus `ok`、requested/resolved/token width `128/128/128`、request width `1`だった。
- 全12 runでreset attempted/complete/failedは`1/1/0`、baseline before/afterは`true/true`、audit coverage complete、physical operation invocations `64`。deterministic digestは全runで同じ`13e6ab56fc76f9b6cd0c69ddcb36a33803d8aa6b344b4392d301e0b8a8f3de48`。OOM、HIP fault、reuse forbiddenは全runで`false`、terminal reasonは`none`だった。
- measured 10 runのelapsedはmin `944.727913 ms`、median `949.3473845 ms`、mean `949.1745433 ms`、max `954.27067 ms`、population standard deviation `2.867731863 ms`。このone-case smokeはpromotion eligibleではなく、正式な性能比較へ単独流用しない。
- driver process artifactはstatus `complete`、exit code `0`、stderr `0` bytes、secret material recorded `false`。shutdown send、reap、process group cleanupはPASSし、children remainingはない。launcher/maintenance safetyは`model_load_executed=true`、`gpu_command_executed=true`だった。
- raw case SHA-256は`22b39c7bf2556d97abf1df5fc0497ab855a865f4eb767579a020dd12d5c339a0`、driver process `c957ea39da53466d0231d4f5849dd308bea5ba706d50048ced765c8c64fb1f2a`、lock owner `e3c48f6d6dfe363a59bcb6a8c8c4f683b1ef3fa232dd52236a0fcfc11ea3c7fc`、summary `c0cb6113f0cfd375c73cc89ebe85e9e8bd22eb5e1d59ad03f29c868a6334ac00`、runner tree `dbc2ed209a9a76c9db42f78fd859cc649b817e9693e165f38b9f43d772f3a884`。
- substrate cleanupとouter restoreはPASSした。post service main PID `1656556`、worker PID `1656718`、`NRestarts=0`、formal health `9/6/6`、全4 endpoint 200、production hashes、AMD/KFD owner、busy lockは正常。systemd再作成後のdirectoryはdevice `26`/inode `761711`、lockはdevice `26`/inode `761721`でactual substrateとは別epochである。
- maintenance evidence SHA-256はlauncher `8a37c5b1638cb797fab0f9f88f73df475b695589c275779e0fdb04a57afce8ce`、marker `c9400db344b0c24230df03779e8d1577951b087a56a33ffaa837ca3c159046c7`、poll 0 `468dcab035f523dd52883afa5a0e74780a8f4683449114202756039e22727619`、poll 1 `dc7503bbe37f016573228a80f6939594982835e8e612b49ed48cdb6560342daa`、SUMS `025f4348ceaa81417dee9cac7b3c78b02bd9e705a25499bcc1fb891957139980`。verificationはPASSした。
- launcher evidence SHA-256はlauncher `b31f5b64df2fb80e0154dbb1277e5f3b0717032edb4c252cb55401d8ddbbcc92`、live preflight `46d76d111cdab6986d1a58f040c6438f14841fc032e86e36eb1e2563c69bcaa9`、runner stdout/stderrはempty SHA、validator stdout `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a`、SUMS `2e01f455c7f1abbbb96249d6e0758151604e0b8cc593680bc4a03138849f8ed7`。verificationはPASSした。

## 次の行動

- v9 single-use outputは再利用・再実行しない。
- この成功結果をone-case resident smokeの通過証拠として採用する。promotion eligibleではないため、性能採用判断には別の測定契約を使う。
- 次はraw/summaryの12 run、reset、deterministic digest、driver process cleanup、live preflight bindingをrelease validatorへ入力し、成功契約を回帰テストとして固定する。
