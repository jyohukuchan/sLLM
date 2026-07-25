# AQ4_0 runtime hardening Phase 4

## 前回の要点

Phase 1〜3 で、AQ4_0 worker の bit-identical protected closure、最小 product/tokenizer closure、promotion source、および manifest freezer control source は `/opt/ullm/aq4-runtime-hardening-v0.1/` に seal 済みだった。一方で fresh path-bound evidence、receipt、candidate manifest、rollback manifest、reviewed operations、activation plan、credential seal set は未作成であり、read-only pre-plan は意図どおり `ready: false` だった。

## 今回の変更点

- workspace の P3 profile を入力にせず、live `/etc/ullm/served-models/active.json` から candidate profile を機械的に導出した。guard flag は live と順序まで一致する 30 件（unique 30 件）で、P3-only 6 key の交差は 0 件である。profile SHA-256 は `ee3d9d4374b79f03e402027a48c6e32601912f79429013893a023083a497439e`。
- R9700 (`gfx1201`, HIP GPU index `1`, AMD SMI card `2`) だけで fresh resident-versus-legacy evidence を収集した。GPU exclusivity preflight は positive VRAM process 0 件、`raw-p0001-g0004` / `raw-p0008-g0004` はともに exact token match、resident/legacy clean shutdown はともに `true`。evidence SHA-256 は `4a604453abb6c7a672731d2b17d3333e471d6c5239b4fed1f6b338fe19a19adb`、receipt SHA-256 は `99ead62f6d5d6062690d78431dbb888949e100bf8951c55f9ff16c71545f1f24` である。
- fresh evidence の protected absolute paths と detached/clean promotion-source commit/tree を別の immutable path-binding document に記録した（SHA-256 `e1b6158cddfab37b84afc2b85351a109d4530af7c4668adb932e5b94532ebe2b`）。旧 evidence、旧 receipt、旧 manifest hash は流用していない。
- sealed freezer control source で candidate manifest を freeze した。SHA-256 は `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` で、live との差分は `tokenizer.root`、`worker.binary`、`product.root`、`promotion.receipt`、`promotion.receipt_sha256` の5項目だけである。`/home/` 参照はない。
- current live bytes と完全一致する immutable rollback manifest を作成した（SHA-256 `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`、4,459 bytes）。sealed activation control source (`d11085c4…`, tree `c41bf381…`) と root-owned reviewed operations を用いて immutable activation plan を作成した。plan SHA-256 は `72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194`。
- gateway API key と offline-minted OpenWebUI session JWT を credential seal set に含めた。JWT の発行・offline verify は campaign、OpenWebUI login、authorization consumption、activation を呼ばない。raw credential は記録していない。
- plan-bound read-only preflight を独立して再実行し、`ready: true`、`blockers: []`、全 10 check PASS を得た。`production_activation_performed: false` のままであり、activation intent/outcome/recovery/rollback outcome/live proof は未使用である。
- GPU/service window は evidence 成功窓 1 回、早期 preflight abort 後に即復旧した stop/restore cycle 1 回、合計 stop/restore cycle 2 回。成功窓の telemetry は edge 36–38 °C、hotspot 37–40 °C、memory 34–38 °C、gfx 6–2835 MHz、socket power 13–44 W、`THROTTLED` と `UNTHROTTLED` の両方を記録した。pre-stop は 36/37/34 °C・46 MHz・11 W・`THROTTLED`、復旧後は 38/39/38 °C・49 MHz・15 W・`UNTHROTTLED`。throttle status の原因は未確認である。復旧は `ullm-openai.service` active/running、`reset-failed` 不要で完了し、`llama-qwen35-udq4.service` は inactive/disabled、`gdm3` は inactive を維持した。
- `/etc/ullm/served-models/active.json` の bytes、systemd unit content、SQ8 paths、V620 compute、promotion campaign、campaign authorization consumption は変更していない。

## 次の行動

1. 人間の明示承認までは execute を一切呼ばない。approval 時には active/unit/environment/credential とすべての sealed inputs を再検証し、plan-bound read-only preflight が再び `ready: true` であることを確認する。
2. approval 直前に current active bytes が plan の rollback SHA-256 と正確に一致しない場合、この plan を適応・上書きせず停止し、superseding plan を別途レビューする。
3. candidate live proof が成功して初めて Phase 7 の fresh AQ4 campaign/browser evidence/bundle v1 に進む。旧 path-bound artifact は downstream evidence として扱わない。
