# `SQ8_0` R9700 attention optimization U — PMC, Flash2, paged split

## 前回の要点

- R9700 (`gfx1201`, PCI `0000:47:00.0`) の実経路では、paged decode
  attention が 51.05% である一方、Flash2 M=128 prefill attention が
  75.63% を占めていた。
- decode は既に default で wave-shuffle (`ds_bpermute`) を使っており、
  `ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE` は値ではなく存在だけで fallback
  を選ぶ。Flash2 は 64-token full tile ごとに source-level rendezvous が
  661 回あった。
- PMC の `FETCH_SIZE` / `VALUInsts` はゼロだったが、counter 名か
  gfx1201 定義か、権限/driver 側かは未分離だった。

## 今回の変更点

- `SQ8_0` 専用の HIPRTC standalone prototype を追加した。legacy、QK-only、
  QK+max、QK+max+sum staged wave32 の各 symbol は分離され、normal Flash2
  body の既定選択は残した。
  - static full-staged は LDS 1296 B / VGPR 27 / SGPR 48 / wave32 / spill 0、
    legacy は LDS 1296 B / VGPR 21 / SGPR 46 / spill 0 だった。
  - short、63→68 tail、synthetic 896→1024、adversarial の standalone
    differential は非有限値なし。synthetic standalone timing は
    13.317192 ms → 12.876236 ms (1.03425x) だった。ただしこれは serving
    throughput ではない。
- PMC は deterministic load+FMA probe で raw primitive まで切り分けた。
  - gfx1201 の SDK counter 定義は存在するが、`SQ_INSTS_VALU` と
    GL2C 32/64/128B request はすべてゼロ、`SQ_WAVES` だけ非ゼロだった。
  - selected Flash2 の 160 launches も `FETCH_SIZE=0`、`VALUInsts=0`、
    `Wavefronts=40960`/launch だった。derived 名の問題ではない。
  - exact root cause は未確認である。start-limit budget を消費した後に
    root-only retry は開かず、physical HBM efficiency と最終的な
    memory-bound / compute-bound 判定は未確認のままにした。
- canonical artifact + vLLM-source `raw-p0512` fixture の full-model
  differential を行った。
  - normal baseline は 4×M=128 unit を 1.167487403 s
    (`438.548629` input tok/s) で実行した。
  - staged の final hidden は max abs `0.7760314941` / relative L2
    `0.0145683599`、logits は max abs `0.2401080132` / relative L2
    `0.0084836396`。frozen SQ8 gate (`2e-5`, `1e-5`, cosine `0.999999`)
    を明確に失敗した。生成 token は双方 66 でも quality gate の代替には
    ならない。
  - staged run の serving timing は一時的な service restart overlap により
    無効化した。数値 NO-GO がすでに十分なので、production Flash2 symbol
    は置換していない。
- existing explicit paged split API のみを使う M=1/C=1036 probe を追加した。
  direct legacy dispatch は変更していない。
  - direct は 0.643241770 ms、tile 128 / 256 / 512 はそれぞれ
    0.228016370 / 0.227932360 / 0.383530140 ms。
  - differential max abs は各 tile で `1.34110e-7` / `1.26660e-7` /
    `1.34110e-7`、non-finite は 0。tile 256 は direct より 2.822x の
    isolated attention-call time だった。
  - 40 head の partial workgroup 供給は direct 320 waves (15.625%) に対し、
    tile 128/256/512 で 2880/1600/960 waves
    (140.625%/78.125%/46.875%)。これが 128/256 優位、512 後退と整合するが、
    full-model end-to-end claim ではない。
- `uint4` load/lane re-layout は着手しなかった。raw physical counter と
  lane/physical traffic validation が未成立である。

## 非干渉と運用記録

- 実行は R9700 のみで、prototype / runner は `gfx1201` を検証してから起動した。
  V620 (`gfx1030`) は使用していない。
- `llama-qwen35-udq4.service` は inactive/disabled、`gdm3.service` は
  inactive を事前確認した。R9700 pre/post telemetry は unthrottled で、
  in-run peak temperature/clock/power は未確認である。
- primary stop は 05:05:32+09:00、scripted restore は 05:07:48+09:00。
  tool lifecycle を誤読して 05:06:24 に manual start、05:06:51 に
  compensating stop を一度行ったため、staged serving timing は採用しない。
  path-error retry は GPU kernel を launch せず即時復旧した。最終的に
  `ullm-openai.service` は active/running、`NRestarts=0` だった。
- `/etc/ullm/served-models/active.json`、systemd unit 内容、campaign、
  authorization、candidate/release、`/opt/ullm`、external ABI、direct legacy
  dispatch、remote repository は変更していない。

## 次の行動

1. Flash2 は current body を維持する。新たな candidate は full-model vector
   gate を先に通すよう、softmax reduction association を設計し直す必要がある。
2. PMC root cause を再開する場合は、service start-limit interval を越えた別の
   approved window で root/non-root matched raw probe を一回ずつ取る。それまでは
   physical memory-bound claim をしない。
3. tile 256 の split route を serving に統合する場合は、direct legacy route を
   保存した explicit selector と clean full-model M=1 timing/differential を別途
   gate する。
4. `uint4` は physical PMC と lane mapping が検証できるまで未判定のままにする。
