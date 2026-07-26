# 共有 GPU 窓の無効化記録

この記録は、最初の BH service-stop 窓で得た timing artifact を性能結果から
除外する根拠である。削除せず、後続の再計測と混同しないよう残す。

## 時系列（JST）

- 21:09:11: BH の `before-stop.json` は AQ4_0 worker が R9700 を使用中であることを記録した。
- 21:09:39: BH が `ullm-openai.service` を停止し、`/run/ullm/r9700.lock` の
  non-blocking lock を取得した。45 秒の cooldown 後の BH preflight は 38/39 C
  （edge/hotspot）、worker なしだった。
- 21:11:26: 別セッションの `prefill-tail-fix/run_service_window.sh` がその inactive
  service を inherited window として採用した。そこで実行された baseline oracle は
  21:11:27--21:13:33、candidate oracle は 21:13:33--21:15:16 である（同 runner の
  `service/window-events.tsv` に記録）。
- 21:13:35--21:13:45: BH の `*-final.json` probe timing は上記 candidate oracle と重なる。
- 21:15:17: 同セッションの `run_measurements.py` が開始した。BH の direct full-model
  結果ファイルは 21:16:10 に書かれたため、この禁止対象プロセスと重なる。

したがって、次の **timing 値は全て無効**であり、性能比較・full-model 結論・昇格判断に
用いない。

- `probe/*tile*.json` の `direct_timing` / `split_timing`;
- `full-model/direct.json` の `mean_tokens_per_second = 15.260295805978306`。

probe の shape、有限値、および output diff は、同じ deterministic input に対する
機能 smoke としてのみ保持する。GPU 共有は timing 結果を汚染するが、ここで観測された
non-finite が 0 という事実を「性能値」に変えるものではない。

`flock` は取得済みだったが、この別 runner は同じ advisory lock を待たなかった。従って、
次の BH window ではプロセス sentinel が空であることに加え、外部 runner の終了を確認し、
固定 commit の専用 worktree から実行する。

service は競合 runner が service lifecycle を所有していたため、BH はこの無効化時点で
start/stop を追加発行しなかった。最終 service 復旧は後続の有効 BH window 終了時に別途
記録する。
