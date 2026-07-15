# AQ4 resident worker stderr evidence

## 前回の要点

`capture-aq4-resident-executor-record.py` は、worker の stderr を JSON object として読む行だけ保持し、invalid UTF-8・非JSON・JSON非objectを捨てていた。kill、timeout、nonzero、audit欠落時に stderr の全体量や秘密を含まない診断を、呼び出し元へ構造化して伝える契約もなかった。

## 今回の変更点

- `WorkerStderrCollector` を追加し、stderr raw bytes の総バイト数と SHA-256 をストリーミング集計した。
- JSON dict の既存解釈を維持しながら、巨大行・大量行を上限付きで処理した。
- 秘密マーカーを含む行は行全体を固定文字列へ置換し、32KiB以内の UTF-8 preview を生成した。invalid UTF-8 は置換表示とフラグで記録し、raw bytes は保持しない。
- process failure の finally で worker の kill/reap、stderr drain thread の join、VRAM observer の終了を行い、`CaptureError` に versioned stderr summary を添付した。
- main の失敗出力へ `ullm.aq4_resident_capture_error.v1` の JSON envelope を追加した。
- 専用テストで JSON互換、raw SHA/count、invalid UTF-8、長行境界外秘密マーカー、秘密行大量、巨大非JSON行、main envelope を確認した。

## 次の行動

親エージェントが outer runner の opaque capture failure evidence と統合し、関連する全テストを実行する。GPU、service、sudo を使う実機検証は未実施である。
