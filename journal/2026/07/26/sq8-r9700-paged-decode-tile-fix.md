# SQ8_0 paged-decode source-tile 数値発散の診断と封じ込め

## 前回の要点

前回の frozen full-model gate は、tile128 / tile256 が token exact でも
hidden/logits を大きく発散させるため NO-GO とした。direct は既定のまま、
split は `ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE` による調査専用
opt-in のままだった。

## 診断結果

- g0001（prompt 128、cache length 129）の直後に、未修正 direct と未修正
  tile128 の論理KV prefixを40 layer全てで採取した。各 layer の K と V は
  132,096 F32 要素で、全 10,567,680 F32 要素（K+V+40 layer）は hash、
  bit mismatch count、max abs、relative L2 とも一致した。
  - よって、最初のdecode後のKV書込み位置/書込み値の乖離は今回の発散原因では
    ない。g0002の大差は同一のKV stateを読んだ当該decode中に発生する。
- standalone F32 API sweep は `split_count > 1` でのみ差分が始まった。
  tile128 は C=128（1 split）で 0、C=129（2 splits）で
  `1.4901e-8`、tail のない C=256（2 splits）でも `1.63913e-7` だった。
  tile256 は C=256（1 split）で 0、C=257（2 splits）で `7.451e-9`、
  C=512（2 splits）で `9.6858e-8` だった。
  - tail は必要条件ではない。C=256/512は16-token pageと該当tileの双方に
    整列しているため、page/tail相互作用も必要条件ではない。
  - 全てのpage-indexing defectを独立のfault injectionで否定したわけではなく、
    その広い否定は **未確認** である。
- sourceを確認すると、splitは各tileのonline softmax
  `(max, denominator, weighted value)` を別々に計算してからrescale/mergeし、
  direct はsource全体で一つのonline stateを更新する。multi-tileでF32演算の
  associationが異なることが、geometry依存の微小差の直接の原因である。
- SQ8 stackは40 layerで160回のactivation quantizationを実行する。g0001の
  微小差がg0002で大きなhidden/logits差へ変わる観測、同一KV prefix、及び上記
  source構造を合わせると、根因は **multi-tile partial mergeがdirect SQ8
  numerical contractを満たしていないこと** と判定した。
  - 最初にどのactivation quantizerが境界をまたいだかは採取しておらず、
    その一点は **未確認** である。

## 修正

- tile経路だけに封じ込めを入れた。`cache_len <= source_tile` のone-tile時は
  既存split bodyを使い、multi-tile時は既存のdirect paged-decode bodyへ
  fallbackする。
- normal direct dispatch、legacy dispatch、runtime/kernel ABIは変更していない。
  opt-in結果には `direct-fallback-exact-state.v1` を記録し、gate passを
  multi-tile実装のpassと誤認しないようにした。
- test-onlyのlogical KV prefix captureと比較toolを追加した。物理cache全体では
  なく、実際に次decodeが読むlogical written prefixだけをblock table順で採取する。

## 同一基準の再ゲート

criteriaは前回とbyte-identicalで、SHA-256は
`645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978` のまま
である。127/128と511/512 prompt、cache length 128--131 / 512--515、
各request 3 decode feedback stepという同じ条件で再実行した。

| route | token exact | finite | vector pair | max abs | verdict |
| --- | --- | --- | ---: | ---: | --- |
| tile128 | true | true | 24 / 24 pass | 0.0 | PASS |
| tile256 | true | true | 24 / 24 pass | 0.0 | PASS |

これは安全なfallbackを含むcontainment passであり、unsafeなmulti-tile splitを
default化する根拠ではない。directを既定のまま維持した。

## 性能

raw-p0512 のgenerated index 1--7（C=513--519）を同期whole-model M=1で測定した。

| route | mean ms | median ms | speedup vs direct |
| --- | ---: | ---: | ---: |
| direct | 54.277224 | 50.515862 | 1.0000x |
| tile128 | 55.596485 | 50.755068 | 0.9763x |
| tile256 | 54.017278 | 50.316713 | 1.0048x |

以前のtile128 `1.2365x` は失われた。全測定Cがtileより大きく、修正後は
direct fallbackを取るためこれは期待通りである。AMD SMI throttle stateを含む
窓なので、近い1xのtiming値は条件付きであり新しい性能主張にはしない。

## 非干渉とサービス窓

- R9700（GPU 2、`gfx1201`、`0000:47:00.0`）のみを使用した。V620は選択していない。
- preflightで `llama-qwen35-udq4.service` が inactive/disabled、`gdm3` が
  inactiveを確認した。
- 07:09:42+09:00に `ullm-openai.service` を一度停止し、07:19:21+09:00に
  initial startで復旧した。最終状態は active/running、Result=success、
  `NRestarts=0`。reset-failedは不要だった。
- 495 one-second sampleで、socket power 8--326 W、gfx 3--3431 MHz、
  edge/hotspot/memory 37--58 / 38--76 / 34--62 C、AMD SMI
  THROTTLED/UNTHROTTLED は223/272だった。raw throttle reasonの持続的な
  物理原因は **未確認**。これはgeometry依存の数値発散の説明ではない。
- active manifest、campaign、authorization、`/opt/ullm`、systemd unit、
  power cap/profile、remoteは変更していない。

raw evidenceは
`benchmarks/results/2026-07-26/sq8_0-paged-decode-tile-fix/` に保持した。

## 次の行動

multi-tileを再び性能候補にするには、directと同じ数値contractを実現する
exact-state merge（または同等のalgorithm）を作り、同じKV prefix診断とfrozen
full-model gateを通す必要がある。それまではdirect既定を維持する。
