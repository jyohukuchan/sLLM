# Credentials and privileged operations

## 基本方針

プロジェクトのpassword、token、private key、cookieその他の資格情報を、repository、working tree、文書、log、issue、CI artifact、command-line argumentへ平文で保存しない。

`.gitignore`は誤commitを減らす補助策であり、file access control、暗号化、漏えい防止の代替ではない。

`passwords.txt`と`password.txt`を資格情報の保管場所として使用しない。AI、CI、build、testはこれらの内容を読み取らない。既存の平文資格情報は内容をcopy・表示・再利用せず、所有者が発行元で失効・rotationする。

## 資格情報の供給

優先順位は次のとおりとする。

1. 承認済みsecret managerまたはworkload identity。
2. CIの短命OIDC tokenまたはprotected environment secret。
3. toolが他方式へ対応しない場合に限る、実行時の環境注入、stdin、またはfile descriptor。

- 資格情報をargv、source、project設定file、shell history、log、artifactへ渡さない。
- 環境変数を使用する場合は、不要なchild processへ継承せず、debug outputとerror reportから除外する。
- short-lived、least-privilege、最小scopeを優先する。
- 長期資格情報は発行元のpolicyに従い、少なくともowner、用途、scope、発行元、期限、rotation責任者を値とは別に記録する。
- 漏えいの疑い、誤commit、runner変更、担当者変更、repository権限変更時は直ちに失効・再発行する。

## やむを得ないローカルfile

secret managerまたはOS keychainが利用できず、一時的なlocal fileが不可避な場合もrepository外へ置く。

- file mode: `0600`。
- parent directory mode: `0700`。
- 作成時umask: `077`以上。
- backup、swap、core dump、logへの複製を避ける。
- 用途終了後は所有者が安全な手順で削除する。

これは平文資格情報を常用してよいという例外ではなく、一時保管時の最低基準である。

## CIとuntrusted GPU job

- fork、外部PR、未検証branch、issue由来入力等のuntrusted codeへsecretを注入しない。
- secretを保持するself-hosted runnerでuntrusted codeを実行しない。
- PR由来codeのGPU検証では、secret manager、host home、credential file、Docker socket、永続runner credentialへアクセスできないephemeral isolated runnerを使う。
- modelはhost controllerが事前検証したread-only mountから提供し、PR jobへmodel storage credentialを渡さない。
- trusted jobでもsecretは最小scope、短い期限、protected environment、masked logを必須とする。
- credentialが取得できない場合にproject内の平文fileへfallbackしない。

## sudo

- `sudo`は明示的に必要で、対象と効果を限定できる操作だけに使う。
- non-interactiveに許可済みかを`sudo -n`で確認できる場合だけ自動実行する。
- passwordを`sudo -S`、stdin、argv、環境変数、file、pipeで渡さない。
- `sudo`を資格情報fileの閲覧・copy・権限回避に使用しない。
- authentication promptが必要な場合、AIはpasswordを取得せず停止し、ユーザーが値を共有せず対話的に認証する。

## 現在必要な移行

- project rootの`passwords.txt`と`password.txt`を新規処理で使用しない。
- いずれかのfileに有効な資格情報がある場合、所有者が発行元で失効・rotationする。
- 移行完了までの最低限のaccess制限として、所有者が各fileのmodeを`0600`へ変更する。
- 新しい資格情報はproject外のsecret managerまたはOS keychainへ登録し、値をこの文書やhistoryへ記録しない。
