# P2 AQ4 GPU 中間差分トレース（attempt3）production 証跡

実行日: 2026-07-15 JST

attempt3 は commit `57d07f2e8d610bf9e8e23778eb71daf1f244c500` に含まれる freeze script（SHA256 `5ca306194c3624e5f8882e98e851dc92d49bde11e93c631922d16d1ca5bbc836`）を一度だけ実行した。`PREFLIGHT_ONLY` と `PREFLIGHT_LOCKED_ONLY` は unset、再試行はしていない。実行後の script は変更していない。

## 実行結果と output

- command: `sudo env -u PREFLIGHT_ONLY -u PREFLIGHT_LOCKED_ONLY run-gpu-gate-attempt3.sh`
- exit code: `0`
- wall time: `11.80 s`
- output root: `p2/differential-trace-gpu-v1-attempt3/`
- output `SHA256SUMS` の output 内検証: `manifest.json: OK`, `payload.jsonl: OK`, `runtime.json: OK`
- payload は attempt2 と byte-identical。runtime も attempt2 と byte-identical。

| artifact | SHA256 | bytes | uid:gid | mode | nlink | dev:ino |
|---|---|---:|---:|---:|---:|---|
| attempt3 gate log | `c3b37dc1097a5c38fc30d522ed68f752ddb7662833b2a826d6fd2948ad59bdc7` | 7161 | 0:0 | 644 | 1 | 10302:10512803 |
| attempt3 monitor log | `512acd2c9dc4883a60e124ca103bfd5466ef1046eecdeeec76525c9ce8fd271d` | 3069 | 0:0 | 644 | 1 | 10302:10512921 |
| attempt3 run log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | 0:0 | 644 | 1 | 10302:10512946 |
| output `SHA256SUMS` | `34ff3648e553d8957f72bc98fca1dd1d190e179bcc7a6f6f8a503c8024fcaac0` | 239 | 1000:1000 | 644 | 1 | 10302:10512951 |
| output `manifest.json` | `5ac7ae6934f8bb955f7194f7edab8e20647e313870f8117533f1422d9bb2c4c6` | 4264 | 1000:1000 | 644 | 1 | 10302:10512949 |
| output `payload.jsonl` | `bcc54f8e6968412244f642e24889483ab43dee751ab198f8d5eb0365b1dced00` | 30253 | 1000:1000 | 644 | 1 | 10302:10512948 |
| output `runtime.json` | `241347906a25a8dc7f1c3257680f0a06fe05297c8976baffa1f55097f5b5a3c2` | 85 | 1000:1000 | 644 | 1 | 10302:10512950 |
| detached candidate attempt3 | `356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7` | 2928200 | 0:0 | 755 | 1 | 10302:10512861 |
| service-stopped marker | `22dd3a82d34b662fc1b2f15b9c42c9603ac80641afca4ea386d87eb8fdca8422` | 369 | 0:0 | 600 | 1 | 10302:10512862 |
| observer-sample marker | `de40c5cb7e649d748886ef10b3fbc1c0a24bf78c4bd493d1aa159797404da8fd` | 20 | 0:0 | 644 | 1 | 10302:10512945 |
| run-started marker | `5d26a0038021868fdefe59656324d4b4e3428cfa9515da8ab8adc6011468993a` | 37 | 0:0 | 644 | 1 | 10302:10512944 |

`observer-failed.marker` は生成されていない。attempt3 の runtime directory と lock は cleanup 後に元の systemd 所有物へ戻った。

## Trace contract

- rows: `3`
- stage counts: `35, 35, 35`（embedding、decoder 0--31、final_norm、lm_head）
- greedy: `[41330, 16, 15]`
- hidden coordinates: `[0, 1, 1024, 2048, 4095]`
- logit coordinates: `[0..31]`
- first mismatch（source-v2 比較、read-only analyzer rc=0）: 3 行とも `decoder_layer:0`
- first mismatch max abs: `0.005409110337495804`, `0.004887105897068977`, `0.0024267658591270447`
- missing stages: 全行 `[]`

manifest identity は build commit `28ec343ac59e6d22e710035d7874df9fbd8f890f`、active SHA `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、package SHA `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`、HIP device index 1 / AMD Radeon Graphics / gfx1201 と一致した。cases SHA は `15fed90dd2e16a5b68d4498c8632257d80ac94c56ed614696b0884c65f4836f2`、replay SHA は `1ee0b9228e1bc3a0ae9175e5693bf3770f9b89e872349554562dbd4b6b4747dc`。

## Observer と service restore

observer は 6 samples（00:41:43--00:41:49 JST）を取得した。

| card | GPU use | VRAM used | power |
|---|---:|---:|---:|
| card0 | 0--0% | 21,401,600--21,458,944 B | 7--13 W |
| card1 | 0--0% | 21,401,600--21,458,944 B | 7--10 W |
| card2 / gfx1201 | 0--41% | 87,384,064--7,439,523,840 B | 7--36 W |

post-run の service は `active/running`、`NRestarts=0`、MainPID `3090367`、`/run/ullm/r9700.lock` owner `3090367`。healthz は `{"status":"ok"}`、models は `ullm-qwen3.5-9b-aq4` を返した。runtime directory は mode 750・uid/gid 1000・nlink 2、lock は regular empty file mode 600・uid/gid 1000・nlink 1・size 0。

## attempt1/attempt2 immutability

attempt1 の gate/monitor/run SHA はそれぞれ `f6fa8e3c...bebb80cc`、`d482724f...b274dd14`、`507ecb23...eec0eb2` で unchanged。attempt1 の observer/run marker、service marker、detached binary SHA も既存証跡値（`0a8cd6ab...dd98a73`、`5cb3c0a7...be0b6`、`dab36bd0...4b12d`、`356d131f...e3bd0b7`）から変わっていない。

attempt2 の gate/monitor/run SHA は `592dddca...37f4c8d3`、`c3a80681...223c135`、`e3b0c442...b7852b855`。attempt2 output の `SHA256SUMS`/manifest/payload/runtime SHA は `bd8da05b...3bfd966`、`dd02cb0d...72b1284`、`bcc54f8e...1dced00`、`24134790...f5b5a3c2` で unchanged。attempt2 markers と detached binary も unchanged。attempt3 は専用 `attempt3-*` paths のみに残り、attempt1/2 paths を上書きしていない。

## 未実施

- attempt3 の再試行
- attempt3 の追加 GPU/service 操作
- script または committed freeze artifact の変更
