# 軽量昇格方針と汎用昇格 self-test

## 前回の要点

- AQ4_0 の既存 hardening activation は、sealed input、candidate 固有の control route、plan SHA、
  literal confirmation token を組み合わせた再現用の一回限りの経路だった。これは過去の AQ4_0
  昇格を説明する記録として残す必要がある一方、通常の候補を速く昇格する経路には適さない。
- 方針文の一部には、final activation に人間の明示承認を要求する将来形の記述が残っていた。

## 今回の変更点

- これは 2026-07-26 のユーザーによる方針変更である。開発専用マシンでは人間承認を通常昇格の
  要件にせず、実推論の出力文章が明らかに崩壊していないことと、速く確実に戻せることを判断の
  中心にする。過去に承認が行われた事実を記した journal、benchmark、AQ4_0 hardening plan の
  記録は書き換えていない。
- `docs/plans/lightweight-promotion-policy-v0.1.md` と固定 10 prompt suite を追加した。日本語、
  英語、Python/JavaScript、長文要約、多ターン会話、翻訳、構造化推論を実際に生成し、現行と
  候補を並べて保存する。空応答、HTTP failure、反復、制御文字、言語/コード要求の放棄、極端な
  長さの偏りは自動検出する。exact match や cheap metric は記録だけであり、閾値 gate ではない。
- `tools/promote-served-model.py` と `tools/rollback-promoted-served-model.py` を追加した。任意の
  manifest を引数で取り、candidate 固有 confirmation は持たない。active bytes の strict snapshot、
  atomic exchange、append-only ledger、actual gateway response、失敗時の automatic rollback を
  共通実装 `tools/lightweight_promotion.py` に集約した。
- gateway が Docker bridge のみで listen するこの配置では host から直接 probe できなかった。
  `--gateway-container` は任意のローカル container を選べる汎用 transport であり、`direct` も
  指定できる。token と request body は curl の process argument に含めず stdin config で渡す。
- first bridge run の rollback は saved raw bytes を正しく戻したが、直後の `systemctl restart` が
  `start-limit-hit` となり response proof を完了できなかった。この失敗を隠さず evidence に残した。
  tool は `start-limit-hit` を確認した場合だけ `reset-failed` と 1 回の `start` を実行し、無制限
  retry はしないよう修正した。
- 修正後の self-test は同じ AQ4_0 manifest を semantic self-test candidate として
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` から
  `159f4d743b65977bc3602bc613216693bcd7f50812fc3d6338fa97e3cdd73b1c` へ切り替えた。
  baseline/candidate は各 10 本、blocking finding は 0、comparison は `passed: true` である。
  その後 generic rollback を実行し、strict raw-byte check を通して c57a2b6… に戻した。rollback は
  8 回の期限付き probe 後に HTTP 200 の実生成応答を確認した。
- 最終確認では active manifest SHA-256 は
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`、worker SHA-256 は
  `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`、
  `ullm-openai.service` は active/running、`llama-qwen35-udq4.service` は inactive/disabled である。
  self-test 実行前には R9700 prefill measurement、`llama-bench`、`llama-server` が動いていないことを
  確認し、V620 は使っていない。
- service control は本作業全体で `restart` を 4 回発行した（3 回成功、1 回は start-limit で拒否）。
  `start` は、当初 inactive だった service の起動と rate-limit 復旧の 2 回である。最終的に成功した
  generic promotion + rollback 経路そのものは restart 各 1 回、計 2 回であり、systemd の
  `NRestarts` は 0 のままである。
- `/etc/ullm/served-models/candidates/` は読み取りで inventory した。AQ4_0 の既存 manifest は
  5 件あり、4 件は static validator を通り、`qwen35-9b-aq4.json` は validation failure だった。
  ただしどれも BA Phase 1 の bitwise-identical candidate であることを示す manifest/evidence では
  ないため、この inventory だけで次の昇格対象とはしない。

検証証跡は
`benchmarks/results/2026-07-26/lightweight-promotion-selftest-v0.1-README.md`、および同 README が
列挙する attempt directory に保存した。

## 次の行動

1. 新しい AQ4_0、SQ8_0、または別 architecture の candidate が manifest と worker validation を
   満たしたら、固定 prompt suite とこの generic route を使って昇格する。重い FP32 corpus、bitwise
   gate、campaign/authorization は通常昇格の前提にしない。
2. 依頼 BA Phase 1 の bitwise-identical candidate は、登録済み 5 manifest のどれとも確認できて
   いない。candidate 固有の manifest と evidence が揃うまでは、次の昇格対象とは記録しない。
3. 次の service 操作前にも共有 GPU の計測 process と StartLimit 状態を確認する。rate-limit recovery
   が evidence に出た場合は、候補品質とは独立した運用事象として記録する。
