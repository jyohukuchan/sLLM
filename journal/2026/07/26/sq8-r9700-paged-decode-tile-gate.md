# SQ8_0 R9700 paged-decode source-tile full-model gate

## 前回の要点

R9700 gfx1201 の M=1 full-model timing では paged decode source-tile 128 が
direct より速かった。ただし API-level F32 differential だけでは default
dispatch を変更できない。Flash2 staged wave32 が standalone では速くても
full-model hidden/logits を壊した前例があるため、tile 128/256 も実際の decode
feedback を direct とベクトル比較する必要があった。

## 今回の変更点

- normal serving の lean な direct dispatch は維持し、test-only oracle capture
  で実際の M=1 decode feedback の final hidden state と logits を採取できる
  harness を追加した。外部 ABI、legacy direct dispatch、SQ8_1 は変更していない。
- 数値基準を測定前に固定した。全値有限、greedy token exact、max abs <= 2e-5、
  relative L2 <= 1e-5、cosine >= 0.999999 を全比較対に要求し、欠損・hash/
  geometry mismatch・early EOS も不合格にした。criteria SHA-256 は
  645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978 である。
- prompt length 127/128 と 511/512 を用い、decode cache length 128--131 と
  512--515 を横断した。各 request で三つの decode feedback step、hidden と
  logits を比較した。
- GPU/service 窓は一回だけ使用した。R9700 gfx1201 だけを選択し、V620 は
  選択していない。llama-qwen35-udq4.service は inactive/disabled、gdm3 は
  inactive を preflight で確認してから ullm-openai.service を停止した。
  06:34:54+09:00 開始、06:39:36+09:00 restore で、最終 service は
  active/running、NRestarts=0 だった。

## 結果

tile 128 と tile 256 はどちらも token exact かつ有限値だったが、full-model
vector gate は不合格だった。

| route | pass pairs | fail pairs | worst max abs | worst relative L2 | min cosine | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tile 128 | 4 / 24 | 20 / 24 | 2.317678451538086 | 0.08369554694605848 | 0.9965189313620728 | FAIL |
| tile 256 | 12 / 24 | 12 / 24 | 1.9435234069824219 | 0.03318822738718883 | 0.9996737107487421 | FAIL |

tile 128 は一部の 128-boundary 最初の feedback の後に、tile 256 は 128-group
では一致する一方で 512-group で発散した。発散の source-level root cause は
**未確認**である。source-tile geometry/tail との相関はあるが、根因の証明では
ない。token が同じであることは vector gate の代わりにならない。

従って tile 128 の default 化は行わない。direct は既定のままとし、既存の
test-only opt-in は調査用に残す。tile 64/96 の追加探索も、数値 gate が既に
不合格であり追加の service 窓を要するため実施しなかった。

## THROTTLED の整理

240 秒の AMD SMI polling では THROTTLED 119、UNTHROTTLED 121 だった。
sampled range は socket 8--260 W、gfx 2--3427 MHz、edge 36--57 C、hotspot
37--76 C、memory 34--60 C。PPT0 は 300 W、slow edge/hotspot は 110 C、
VRAM は 108 C と報告された。

GPU metrics v1.3 の raw throttle status は hotspot thermal を示す bit を含み、
dependent status の一件には PPT0 bit も含まれた。TDC bit は観測されなかった。
ただし AMD SMI の reason/violation は N/A で、二つの raw field は別読み
（atomic pair ではない）、1 秒 sampling では瞬間ピークも見逃し得る。温度/
電力の sampled value も limit 未満だった。ゆえに「持続的な物理 throttle の
原因」は **未確認** とする。過去の低温 THROTTLED 表示も遡及して一意には決め
られない。

この状態は timing の絶対値・順位を条件付きにするため、将来の性能測定では
atomic metrics capture、cool-down/all-clear gate、reason bit 発生時の
discard/repeat を必要とする。一方、frequency/power throttle は通常 timing に
影響するもので、source-tile 相関を持つ full-model vector divergence の説明には
ならない。今回の数値 NO-GO は有効である。power cap/profile の恒久変更は行って
いない。

## 次の行動

1. source-tile split body の tail/source-boundary accumulation を、direct と
   lane-by-lane に比較して根因を特定する。修正候補は同じ frozen gate を再度
   full-model で通す。
2. timing を再開する場合だけ、atomic telemetry と all-clear guard を harness
   に組み込み、THROTTLED を含む series は performance claim から除外する。
3. default を変更するのは、全 real-prompt vector pair が固定 gate を通過して
   からにする。active manifest、promotion、authorization は別承認なしに扱わない。
