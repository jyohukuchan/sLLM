# Phase 37以降 性能・機能ロードマップ履歴

## 2026-08-21: 計画作成

- ユーザー指示により、Phase 36で残ったMI300X `gfx942`性能差と、main planに記録済みのllama.cpp比機能差へ
  Phase 37以降を割り当てた。
- Phase 37–38をMI300X性能laneとし、Session Dでdevice timeの`73.95%`を占めたGDN、`25.12%`を占めた
  Full Attention、続くfresh residualの順に扱う。
- Phase 39–48をservice基盤、token selection/grammar、state/cache、基本endpoint、Responses/Anthropic/tool protocol、
  template/CLI UX、adapter/model lifecycle、周辺tool、組込みtool/MCP、WebUIへ依存順に分けた。
- ユーザー方針どおりMI300X実機baseline/performanceはVM再確保までdeferredとし、Phase 37はhost prepだけを進行可能にした。
  Phase 39以降のhost実装はPhase 37/38のGPU完了を開始・merge gateにしない。
- Vulkan、一般INT4/INT8+scale、model/hardware/parallel追加は意図的除外を維持した。組込みtool/MCP実行は新しい
  security boundaryのため、Phase番号は割り当てるが実装開始にはtrust modelのユーザー承認を必要とする。
- focused reviewを反映し、fixed llama.cpp比較をpeer artifactが一致するBF16 weight＋FP16 KV行に限定した。FNUZ FP8は
  sLLM内BF16対照とし、対応peerなしに比率を作らない。resumable transport、`n` choice state、assistant prefill、FIM/infillは
  各一つの所有Phaseを定め、後続Phaseはwire/renderer/UX adapterだけを担当する。
- この時点では計画と文書同期だけで、production source、GPU、VM、external service、commit/pushを変更していない。

## 2026-08-22: V620先行の性能系列へ再編

- ユーザー指示により、直近の性能最適化をV620だけで開始し、固定llama.cppと同等になった後にR9700、MI300Xへ
  順に適用・検証する直列経路へ変更した。
- 既存Phase 46〜48は機能計画へ予約済みのため、性能系列をPhase 49〜51へ割り当てた。旧Phase 37〜38のMI300X計画は
  production sourceやGPU証拠へ着手する前に再編し、Phase 51へ吸収した。完了・棄却した実装としては扱わない。
- 「llama.cpp同等」は要求batchと並行sequenceを1に固定し、input/output token数`17/17`、`32/32`、`1,024/128`、
  `32/256`、`10,001/2`の5行すべてで判定する。単一点のPASSでは次Phaseへ進めない。
- 同等条件は全5行のE2EとTTFT、output 17以上の4行のTPOTについて、sLLM中央値がllama.cpp中央値と両者のMADから
  定義した測定幅を超えて遅くないこととした。各行は3 warmup＋10 measured、同一token列、BF16 weight、FP16 KV、
  greedy、単一GPU、HIP-only、fallbackなし、cleanup 0で測る。
- Phase 49はexact `gfx1030`だけへ性能routeを限定し、全5行PASSまでR9700／MI300Xを扱わない。Phase 50はexact `gfx1201`、
  Phase 51はexact `gfx942`へ順に適用する。後二Phaseは同じ5行で同等達成の有無を報告するが、ユーザー指示にない
  R9700同等化をMI300X開始の追加gateにはしない。
- Phase 46〜48の内容と番号は保持するが、ユーザーが優先順位を変更しない限りPhase 49〜51を先に実行する。
- この再編は計画文書だけの変更であり、production source、GPU、VM、外部service、commit/pushを変更していない。

## 2026-08-22: 100k input・20k outputを長時間matrixへ追加

- ユーザー指示により、通常5行に`100,000/2`と`32/20,000`を加え、V620同等判定とR9700／MI300X適用検証を
  合計7行へ拡張した。前者はtoken ID `23066`を100,000個、後者は既存32-token decode入力とEOS無効化を使う。
- 長時間2行は両engineのcontext長を`131,072`に固定し、要求batch／並行sequence 1を維持する。OOM、timeout、
  途中終了をPASSや行の省略へ読み替えない。
- 通常5行は3 warmup＋10 measuredを候補ごとに使う。長時間2行は費用を考慮して1 warmup＋3 measuredとし、
  Phase開始時の基準値、100k prefill／KVまたは20k decodeへ影響する候補群の採否時、最終候補だけで実行する。
- Phase 49の完了と後続GPU開始には全7行のPASSを必要とする。Phase 50〜51でも最終7行を検証するが、
  R9700／MI300Xでのllama.cpp同等達成そのものを新しい相互gateにはしない。

## 2026-08-23: Phase 49完了条件を3候補の判定へ変更

- ユーザー指示により、Phase 49の全7行llama.cpp同等達成を完了条件とPhase 50／51の開始条件から外した。
- Phase 49の残作業をlong-prefill v2、GQA P32、HIP Graphの採否判定に限定する。各候補は関連するV620実機行、
  数値、fallback、後始末、selectorを確認し、採用または棄却の理由を記録する。
- 3候補の判定後、採用経路を含む通常5行で正しさ・資源と重大な退行がないことを確認してPhase 49を完了する。
  100k inputと20k outputは対応候補の判定材料として維持するが、全7行同等未達を後続GPUへの阻害条件にしない。
- Phase 50／51では同じ7行を引き続き測定し、V620の未達と移植後の残差を隠さず報告する。

## 2026-08-23: Phase 49完了

- 変更後の完了条件どおり3候補を判定した。GQA P32はexact `gfx1030`、decode、GQA4、head dimension 256、FP16 KV、
  KV長4,096以上だけへ限定して既定有効化した。`32/20,000`の1 warmup＋3 measuredでE2E中央値を
  `934.261957`秒から`529.330751`秒へ43.34%、TPOT中央値を`46.639333` msから`26.341916` msへ43.52%短縮した。
  4要求はすべて20,000 tokenを生成し、digest、HIP-only、fallbackなし、メモリピーク、後始末が一致した。
- long-prefill v2はoperatorのM=1,024/4,096/10,001を52.96%/56.36%/58.60%短縮したが、`100,000/2` full-modelの
  単一warmupが約33分を要し、current controlの1 warmup＋3 measured合計約20分を超えたため不採用とした。
  性能判定によるSIGTERM後にGPU process消滅、VRAM/GTT復帰、lock解放を確認し、候補は明示的opt-inへ隔離した。
- HIP Graphは無効時の`17/17`がPASS、有効時は17/17要求でSIGSEGVしたため不採用とした。候補固有API、selector、
  native実装、テストを撤去し、public host、token selector host、`sllm-core` prepared/Qwen、gfx1030 production HIPの
  全29 object build/linkで撤去後の経路を確認した。
- 採用sourceの最終通常5行をexact `gfx1030`、3 warmup＋10 measuredで実行し、5/5 PASSした。E2E中央値／Phase 49開始時比は
  `17/17` 423.961 ms／45.43%短縮、`32/32` 750.651 ms／39.35%短縮、`1,024/128` 4,214.241 ms／28.22%短縮、
  `32/256` 5,779.410 ms／29.41%短縮、`10,001/2` 13,507.666 ms／24.24%短縮だった。全行でHIP-only、fallbackなし、
  出力反復一致、要求後とprocess終了後の資源復帰を確認した。証拠は
  `/home/homelab1/.local/share/sllm-evidence/phase49/final-normal5-current-20260823/phase49-v620-sllm-v1.json`に保存した。
- 固定llama.cppとのE2E差は同じ順で+0.78%、+2.16%、+3.04%、+6.65%、-9.45%だった。current controlの
  `100,000/2`約295.093秒対llama.cpp約194.121秒、P32採用後の`32/20,000`約529.331秒対llama.cpp約428.989秒は未達として残す。
  同一最終binaryの全7行同等は取得・主張せず、緩和後の条件でPhase 49を完了し、Phase 50／51の開始待ちを解除した。
- integration review、Rust selector試験、native host試験、Phase 49 schema/summary試験、gfx1030 production HIP build/linkで
  correctness/security blockerはなかった。最終binary SHA256は
  `abc4f0f4772a71d0f582a15c5323908ed9243dcf2396ce3cc754489a94eeac38`、GGUF SHA256は
  `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、model lock SHA256は
  `425151d06832347a01b946b27336ceffac074eb7f6932af61e8c9821edc1e318`である。

## 2026-08-23: Phase 50をR9700移植・MI300X引継ぎへ詳細化

- Phase 50はR9700 exact `gfx1201`でPhase 49変更を実機採否し、MI300X exact `gfx942`向けのwave64引継ぎを準備する。
  MI300XのGPU実行はPhase 51に維持し、Phase 50ではexact feature compile/linkとhost selector非選択までとした。
- R9700のfresh 7行profileを先に取得し、変更をtarget共通、gfx1201 wave32候補、gfx1030限定、不採用、gfx942再設計へ分類する。
  GQA P32やgfx1030のrocBLAS solution IDを無検証で横展開せず、残差上限の大きいcandidateだけを個別採否する。
- target tuple、通常5行3+10、長時間2行1+3、selector境界、資源、V620 focused regression、停止／再計画条件を
  [Phase 50詳細計画](../../../../plans/archive/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)へ固定した。
- R9700での全7行llama.cpp同等は残差報告とし、Phase 51開始のhard gateへ戻していない。

## 2026-08-24: Phase 50完了

- R9700 exact `gfx1201`、Code Object V6、wave32の最終7行は6/7 PASS、1/7 FAILだった。PASS行はHIP-only、fallbackなし、
  反復一致、cleanup 0を満たした。E2E中央値（sLLM／固定llama.cpp、ms）は、`17/17` `407.915/332.726`、
  `32/32` `759.729/604.069`、`1,024/128` `3,383.627/2,509.156`、`32/256` `5,959.860/4,712.364`、
  `10,001/2` `4,002.834/2,072.476`、`32/20,000` `532,486.026/377,632.768`である。`100,000/2`はlayer 31のKV commit OOMであり、
  未達を省略せず記録した。追跡済み要約は`ci/matrix/phase50-r9700-summary-v1.json`である。
- exact `gfx1201`ではresidual RMSNorm、GDN projection bundle、MLP gate-up-SiLU bundle、GQA4 P32（KV長4,096以上）を採用した。
  target共通意味契約、`gfx1030`限定経路、不採用経路、gfx942 wave64再設計への分類を完了し、llama.cpp同等未達をhard gateにしなかった。
- 共通source変更後のV620 exact `gfx1030`通常5行は5/5 PASSで、Phase 49 closeout比は`-0.21〜+1.16%`だった。
- exact `gfx942` Cargo build、feature compile/link probe、host selector非選択はPASSした。MI300X実機7行、性能採否、
  `project-verified`昇格は未実施でPhase 51へ引き継ぐ。wave64ではwave32前提のlane ownership、block、LDS/register、barrier、
  GQA partitionを直接流用せず再設計する。

## 2026-08-24: 自動prefill capacity tier修正

- Phase 50のR9700 `100,000/2` OOMを分析し、16 GiB超の全GPUを16K候補から評価していた自動selectorを誤りと判断した。
  固定SGLang v0.5.16のtotal GPU capacity別chunk設定を比較参照し、sourceはコピーせずsLLMの既存exact graph memory見積りと
  組み合わせた。ユーザー指定により24 GiB未満は512、24〜35 GiB未満は2K、35〜60 GiB未満は4K、60〜160 GiB未満は8K、
  160 GiB以上は16Kを最大候補とし、各tierでは512までの下位bucketへ見積り結果に応じて落とす。明示指定は従来どおり
  capacity tierを上書きする。
- 32 GiBのV620/R9700は自動2K開始、192 GiBのMI300Xは16K開始となる。24/35/60/160 GiBの両側、非境界prompt、
  zero入力を含むselector testはPASSした。R9700 `100,000/2`のGPU再実行は未実施であり、OOM解消をまだ主張しない。
  Phase 31 summaryとschema/testは当時の16 GiB境界と実測を保存する履歴証拠なので書き換えず、現在のcontractはRust selector testと
  runtime文書を正とする。

## 2026-08-24: 自動2K候補での100k再実行とPhase 52計画

- push済みcommit `159bc526cb26d180161f2cd7abcc22abb7e67e84`のfresh exact `gfx1201` binaryで、同じR9700
  `100,000/2`を明示chunk overrideなしで再実行した。32 GiBのcandidate列は`[2048, 512]`だが、失敗rowには
  実効chunkが保存されていない。
- 約`152.867`秒後、`layer.23.kv_append`のvirtual KV physical commitmentがHIP status 260 OOMとなった。
  HBM peakは旧Phase 50失敗の`26,414,587,904` bytesから`13,160,554,496` bytesへ約50.18%低下したため、
  selector修正のmemory効果とOOM解消を分けて扱う。終了後はprocessなし、HBM／GTTはbaselineへ復帰したが、stderrには
  `execution resource is busy`のcleanup errorも記録された。
- ユーザー指示により、この残件をPhase 52へ割り当てた。実効chunk、provider、per-plane physical commitを失敗時にも保存し、
  VMM grow／copy-on-writeのtransactional rollbackとprofiled abortのpending completion処理を優先して検証する。
  明示512は限定診断であり、最終解決やsilent retryには使わない。
- Phase 51はMI300X wave64移植、Phase 52はR9700 100k OOMを所有し、相互の開始・完了gateにはしない。詳細は
  [Phase 52計画](../../../../plans/archive/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)へ固定した。

## 2026-08-24: Phase 51一時保留とPhase 52完了

- ユーザー指示によりPhase 51を一時保留し、Phase 52を先に実施した。Phase 51はPhase 52完了によって自動再開しない。
- exact gfx1201かつlogical capacity 65,536以上へresident KVを限定採用し、自動2Kの`100,000/2`を4/4、
  従来VMMの`10,001/2`を13/13 PASSした。生成、HIP-only、fallback/cleanup 0、資源復帰を確認した。
- VMM grow/COWのtransactional rollback、profiled abortのbounded drain、selector/KV physical metadataを共通contractへ追加した。
- 詳細identity、全反復、HBM/GTT peakは
  [`phase52-r9700-kv-commit-summary-v1.json`](../../../../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)へ固定した。

[対応する計画](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
