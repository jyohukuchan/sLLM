# P2 full-model driver

- ullm-aq4-p2-full-modelを追加した。served-model manifestと安全な公開fixtureから一件を選び、同じAQ4 package/session経路で指定Mを実行する。
- 結果はullm.qwen35_aq4_p2.full_model_result.v1のbounded JSONとし、identity digest、timings、requested/resolved/actual widths、operation audit digest、lifecycle/reset、OOM/fallbackを記録する。prompt/token ID/output本文は保存しない。
- 出力は一時ファイルのfsync後にhard-linkでatomic公開し、既存ファイルを拒否する。
- v2はmanifest環境値、benchmark/worker binary role、streaming package tree、immutable failure/OOM、physical fallback、audit/resetをfail-closedにした。P2 raw v2のbinary-role/embedded-link adapter不足はv0.2 specにhandshakeとして記録した。
- CPU検証: cargo fmt --all --check、cargo check -p ullm-engine --bin ullm-aq4-p2-full-model、cargo test -p ullm-engine --bin ullm-aq4-p2-full-model -- --test-threads=1（14件成功）、cargo test -p ullm-engine --lib --no-default-features -- --test-threads=1（707件成功、1件ignored）。GPU/live worker実行は未実施。
