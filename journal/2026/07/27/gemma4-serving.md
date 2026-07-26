# Gemma4 E2B BF16 served-model integration

## 前回の要点

- BL は `Gemma4TextExecutor` を HF 5.12.1 と層ごとに照合し、BO は resident BF16 / KV cache /
  sliding window / shared K/V を R9700 で検証していた。
- ただし Gemma4 E2B を served-model manifest から起動する strict worker、product/package、
  tokenizer/chat-template closure は未実装だった。

## 今回の変更点

- `BF16_0` 専用の `ullm-gemma4-worker`、strict `gemma4_e2b_bf16_rdna4_v1` backend、package
  assembler、receipt writer、manifest generator admission を追加した。Gemma を SQ8/AQ4 generic
  loader に流さず、architecture / format / profile / source hashes / kernel guards を
  fail-closed で一致させる。
- source E2B tokenizer に chat template が無いことを確認した。E2B-it revision の template を
  explicit provenance overlay として使い、`GemmaTokenizer`、render、token range、gateway request
  まで mechanical contract は通った。
- immutable candidate manifest は
  `/opt/ullm/gemma4-e2b-serving-v0.1/manifests/gemma4-e2b-bf16.manifest.json`
  （`e01fa275…c8c9`）へ配置した。active manifest を Gemma に切り替えず、127.0.0.1:18080 の
  isolated gateway で `/readyz`（3.276 s）、`/v1/models`、英日 2 prompt、10-case suite を実行した。
- raw worker の sequential wire test は BL/BO と完全一致した。France は
  `[9079,236761,108,818]`、Once は `[528,496,1902,1298]` であり、各 request は
  `reset_complete=true` だった。
- gateway transport は全 request HTTP 200 / nonempty completion まで到達したが、生成文には prompt
  echo、反復、`<unused56>`、回答を持たない echo があり、actual-text policy により Gemma candidate
  は **promotion 不可**とした。base E2B 用の upstream-supported chat interface 又は
  instruction-tuned checkpoint が次の前提であり、template を推測で差し替えない。
- 隔離中に外部 session が AQ4 active manifest を `d3d9…` へ変更したため、trusted protected
  source から atomic restore した。最終 snapshot は
  `3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7`、service は
  `active/running`。外部 GPU 計測との競合を待ち、start-limit が明示された場合だけ
  `reset-failed` と一回の start を使った。
- この task が明示的に発行した service 操作は `stop` 2 回、`reset-failed` 2 回、`start` 3 回。
  途中で観測した lock-conflict / manifest promotion / additional stop-start は他 session 起点として
  evidence に分けた。Gemma candidate を service に設定又は start したことはない。
- その後の shared `prefill-adaptive-chunk` window も AQ4 を一時停止したが、window 自身が
  07:38 JST に restore した。最終 `aq4-final-external-restore-stability.txt` は 4 / 8 / 16 s
  の全観測で `350710…b3e7` と `active/running` を確認している。

## 次の行動

1. Gemma4 E2B を production active に昇格しない。保存済み candidate は worker / manifest / gateway
   integration の regression target として残す。
2. base E2B に対する upstream chat contract を確認できる場合だけ、その source revision/template を
   新しい immutable package に bind して text-quality suite を再実行する。そうでなければ E2B-it
   のような instruction-tuned checkpoint を別モデルとして導入する。
3. AQ4 service を操作する次の task は、active manifest SHA と R9700 lock を先に確認し、今回の
   external stop/start の記録を踏まえて start-limit を消費しない。
