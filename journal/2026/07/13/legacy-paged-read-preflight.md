# Legacy paged-read preflight compatibility

- 原因: legacy incremental cache は短い要求ごとに `block_size/cache_blocks` を縮めるが、canonical M1 paged reader/writer は固定の本番 geometry だけを解決していた。
- 変更: typed backend registry に generic な paged geometry 再束縛を追加し、既存 single operation の ID・feature probe・fail-closed 検証を維持した。split registry は canonical single と分離した。
- 検証: registry 単体テスト、release build、HIP device 1 の layer 3 legacy 相当生成（prefill 2 tokens、生成 2 tokens、`verified=true`）を確認した。
