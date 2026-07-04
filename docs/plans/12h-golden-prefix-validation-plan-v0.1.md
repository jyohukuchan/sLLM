# 12h golden prefix validation plan v0.1

## Purpose

前回の12時間計画では、Qwen3.5-9Bの固定token idからHugging Face参照hidden stateを出し、uLLMの実`.ullm.d` package decoder layerを1 layerだけ実行して、layer境界の誤差をMSE、mean_abs_diff、max_abs_diff、cosine similarityで比較できるようにした。

次の12時間では、1 layer単体ではなく、連続するdecoder layer prefixを参照実装と比較する。目的はfull prompt generationへ進む前に、量子化済みpackage weight、paged KV cache、RoPE位置、層間residual受け渡し、backend差分のどこで誤差が増幅するかを、layerごとのmetricとして残すことである。

この計画は、文章生成品質を測るものではない。Rust側にtokenizer、embedding、final RMSNorm、lm_head、samplingをまだ持ち込まず、参照実装から保存した`hidden_before_layer_0`を入力にして、uLLM側で`0..N`または小さい連続windowを順に実行する。

## 前回の要点

- `tools/export-qwen-golden-tensors.py`で固定token idとselected layerの`hidden_before` / `hidden_after`を保存できる。
- `ullm-engine package-layer-golden-smoke`で1 layer分のfixtureを読み、実package weightで実行した出力を参照hidden stateと比較できる。
- CPU fallback `0`、R9700/RDNA4 `2`、V620/RDNA2 `1`で同じfixtureを実行し、CPU/HIP間のruntime差分はほぼ同等であることを確認した。
- 現在の差分はAQ package量子化誤差と参照実装差分を含むため、厳密一致ではなくmetricとして記録する方針が妥当である。
- ただし、1 layer単体では層間residual受け渡しと誤差蓄積をまだ検査できていない。

## 今回の変更点

- golden fixtureを「単発layer」から「連続layer prefixまたは連続window」へ広げる。
- uLLM側に`package-golden-prefix-smoke`相当のCLIを追加し、fixtureの初期hidden stateから連続layerを順に実行する。
- 各layer後に参照`hidden_after_layer_N`と比較し、誤差がどの層で増えたかを機械可読なJSONLまたはJSON summaryとして保存する。
- OOMを避けるため、最初から全layerをresident f32 weightとして保持しない。小さいprefixまたはlayer window単位でload、execute、dropできる経路を優先する。
- CPU/R9700/V620のbackend間差分と、seq8/seq16の入力長差分を同じ結果表で比較する。

## 次の行動

最初の実装対象は、`layer_start=0`、`layer_end=4`、`seq8`、固定token id列のgolden prefix fixtureである。CPUで通してからR9700へ広げる。`0..4`が安定したら`0..8`、`0..12`、`seq16`、V620へ広げる。

## Target Scope

対象:

- model: Qwen3.5-9B
- package: 既存のQwen3.5-9B p4p6 `.ullm.d`
- primary device: CPU fallback `0`、R9700/RDNA4 `2`
- optional device: V620/RDNA2 `1`
- sequence length: `8`を必須、`16`を拡張
- layer range: `0..4`を必須、`0..8`と`0..12`を拡張
- input: 固定token id列。Rust側tokenizer統合はまだ行わない
- output: layerごとのmetric、backendごとのmetric、失敗分類、実行コマンド

対象外:

- 実プロンプト文字列からのtokenization
- embedding tableのRust実装
- full 36 layer常駐実行
- final RMSNorm
- lm_head/logits
- sampling/greedy decode
- real token/s benchmark
- fused AQ kernel
- AQ/fq仕様の更新

## Deliverables

1. Golden prefix fixture exporter
   - 既存の`tools/export-qwen-golden-tensors.py`を拡張するか、互換性を保つ小さい追加オプションを入れる。
   - `--layer-range 0:4`または同等の指定で、`hidden_before_layer_0`と各layer後の`hidden_after_layer_N`を保存する。
   - metadataにfixture schema version、model path、dtype、token ids、position ids、layer range、hidden size、sequence length、export command、Transformers/PyTorch versionを残す。
   - payloadはraw f32 little-endianを継続し、既存fixture loaderとの互換を壊さない。

2. Rust fixture loader拡張
   - 既存の`GoldenTensorFixture`でprefix fixtureを読めるようにする。
   - layerごとの`hidden_after`をstreamingまたは必要時読み込みできるようにし、大きいfixtureを一度に全保持しない。
   - fixture欠損、payload byte数不一致、非連続layer range、position id不一致を明確なエラーにする。

3. `ullm-engine package-golden-prefix-smoke`
   - `.ullm.d` package、golden prefix fixture、device index、chunk bytes、layer rangeを受け取る。
   - 初期inputはfixture内の`hidden_before_layer_start`を使う。
   - 各layerを実package weightで実行し、出力を次layerのinputとして渡す。
   - 各layer完了後に参照`hidden_after_layer_N`と比較する。
   - OOM回避のため、実装方針は次の順で検討する。
     - まず`0..4`を一括loadして通す。
     - GPU memoryが厳しい場合はlayer単位または2 layer window単位でload、execute、dropする。
     - drop後もruntime bufferが解放されない場合は、deviceごとに短いprocess runへ分割するrunner scriptを用意する。

4. Machine-readable report
   - stdoutの人間向け1行出力に加えて、JSONLまたはJSON summaryを`benchmarks/results/YYYY-MM-DD/engine/`へ保存する。
   - 各entryに少なくとも次を含める。
     - package path
     - fixture path
     - device index/backend name
     - layer index
     - sequence length
     - hidden size
     - candidate ids
     - MSE
     - mean_abs_diff
     - max_abs_diff
     - cosine similarity
     - expected/actual/diff preview
     - failure class

5. Validation matrix
   - 必須:
     - CPU `0`, `seq8`, `0..4`
     - R9700/RDNA4 `2`, `seq8`, `0..4`
   - 拡張:
     - CPU `0`, `seq8`, `0..8`
     - R9700/RDNA4 `2`, `seq8`, `0..8`
     - CPU/R9700, `seq16`, `0..4`
     - V620/RDNA2 `1`, `seq8`, `0..4`
   - 時間が残る場合:
     - `0..12`
     - 別token id列
     - layer window `4..8`、`8..12`

6. Documentation and journal
   - `docs/words.txt`に`golden prefix fixture`と`package golden prefix smoke`を追加する。
   - `journal/YYYY/MM/DD/`に実行コマンド、結果、詰まりどころ、残課題を残す。

## 12h Task Breakdown

### T0: Baseline and fixture audit, 0.75h

目的:

- 現在のmain branch、既存1-layer golden smoke、既存fixture形式を確認する。

手順:

1. `git status --short`
2. `cargo fmt --all --check`
3. `cargo check -p ullm-engine`
4. 既存の`package-layer-golden-smoke`をCPUで再実行する。
5. 既存fixtureのmetadataとpayload byte数を確認する。

完了条件:

- 既存1-layer golden smokeがCPUで再現する。
- 失敗した場合は新規実装へ進まず、fixture破損、package path、runtime変更のどれかを切り分ける。

### T1: Prefix fixture schema design, 1.0h

目的:

- 単発fixtureとprefix fixtureを混在させても壊れないmetadata schemaを決める。

手順:

1. 既存`golden-v1`の互換性を確認する。
2. prefix fixtureで必要なmetadata keyを決める。
3. layer entryに`before_file`を毎layer持たせるか、`initial_before_file`と各layerの`after_file`だけにするかを決める。
4. fixture integrity checkのルールを決める。

完了条件:

- 既存fixtureが読み続けられる。
- prefix fixtureのshape、position、layer連続性をRust側で検証できる設計になっている。

### T2: Export prefix fixture, 1.5h

目的:

- Qwen3.5-9Bの`0..4` prefix fixtureを実際に出力する。

手順:

1. exporterに`--layer-range`または複数layer指定のprefix modeを追加する。
2. `torch.inference_mode()`で固定token id列を実行する。
3. forward hookでは保存対象layerのafter hidden stateだけをCPU f32へ退避する。
4. `hidden_before_layer_start`も保存する。
5. seq8の`0..4` fixtureを作成する。
6. 保存後にmetadataとpayload byte数を読み戻す。

完了条件:

- `benchmarks/golden/YYYY-MM-DD/qwen35-9b-prefix0-4-seq8/`相当のfixtureが作れる。
- 余計な全layer activationを保持せず、OOMしない。

### T3: Loader and metric extension, 1.25h

目的:

- Rust側でprefix fixtureを読み、layerごとの参照hidden stateを安全に取り出せるようにする。

手順:

1. `GoldenTensorFixture`にprefix用accessorを追加する。
2. layer rangeの連続性検査を追加する。
3. position idsとshape検査をprefix smokeから共通化する。
4. payloadをlayer単位で読むunit testを追加する。

完了条件:

- 欠損layer、非連続layer、payload byte数不一致で明確に失敗する。
- 既存golden unit testsが壊れない。

### T4: Prefix runtime execution path, 2.5h

目的:

- uLLMの実package decoder layerを連続実行し、各layer後の出力を参照と比較する。

手順:

1. CLI `package-golden-prefix-smoke`を追加する。
2. `Qwen3PackageModelRuntime`を使う一括load pathをまず作る。
3. fixtureの初期hidden stateをruntime inputにする。
4. layerごとに`qwen3_self_attn_prepare_sequence_for_paged_decode_f32`と`qwen3_decoder_layer_sequence_to_host_f32`相当の経路へ接続する。
5. 各layer後のactual hiddenを次layer inputにする。
6. 各layer後に参照hiddenとmetric比較する。
7. GPU memoryが厳しい場合はlayer window load pathへ切り替える。

完了条件:

- CPUで`0..4`がmetric出力まで通る。
- 失敗時にどのlayerで止まったかがstdoutとresult fileに残る。

### T5: Report format and failure classification, 1.0h

目的:

- 結果を後から比較できる形にする。

手順:

1. JSONLまたはJSON summaryのschemaを決める。
2. layerごとのmetricを保存する。
3. failure classを最低限の固定値にする。
   - `ok`
   - `shape_mismatch`
   - `position_mismatch`
   - `package_tensor_mismatch`
   - `runtime_error`
   - `numeric_drift`
   - `possible_quantization_error`
4. stdoutには短いsummaryを残す。

完了条件:

- result fileだけでlayer、device、seq length、差分、失敗分類が読める。
- 既存の人間向けsmoke出力を大きく壊さない。

### T6: CPU/R9700 validation, 1.5h

目的:

- primary targetであるCPUとR9700で同じprefix fixtureを走らせる。

手順:

1. CPU `0`, `seq8`, `0..4`
2. R9700/RDNA4 `2`, `seq8`, `0..4`
3. CPUとR9700のlayerごとのmetric差を比較する。
4. 差分が大きいlayerをjournalへ記録する。

完了条件:

- CPU/R9700の両方でresult fileが残る。
- backend差分なのか、package量子化由来の共通誤差なのかを最低限分類できる。

### T7: Expansion runs, 1.0h

目的:

- 12時間枠を使って、最小成功から一段広げる。

手順:

1. `0..8`をCPUで実行する。
2. 可能なら`0..8`をR9700で実行する。
3. 可能なら`seq16`の`0..4` fixtureをexportしてCPU/R9700で実行する。
4. 可能ならV620/RDNA2で`seq8`、`0..4`を実行する。

完了条件:

- どこまで通ったかがresult fileとjournalに残る。
- 通らなかった場合は、memory、runtime、numeric drift、fixture exportのどれで止まったかを残す。

### T8: Verification and record, 1.0h

目的:

- 変更の品質を確認し、次の実装者が続きを始められる状態にする。

手順:

1. `python3 -m py_compile tools/export-qwen-golden-tensors.py`
2. `cargo fmt --all --check`
3. `cargo check -p ullm-engine`
4. `cargo test -p ullm-engine golden -- --test-threads=1`
5. `cargo test -p ullm-engine -- --test-threads=1`
6. `cargo test --workspace -- --test-threads=1`
7. `git diff --check`
8. journal更新

完了条件:

- 実装、検証コマンド、fixture path、result path、残課題がjournalにまとまっている。

### Buffer: 0.5h

目的:

- ROCm/PyTorch/HIP memory、package load、fixture schema変更、JSON出力調整で詰まった場合の吸収時間にする。

Fallback:

- `0..4`一括loadがOOMする場合は`0..2`へ縮小する。
- GPUだけ詰まる場合はCPU成功とR9700失敗分類までを完了条件にする。
- prefix実行が難しい場合は、まず`package-layer-golden-smoke`を複数fixture batch runnerとして回し、連鎖実行は次タスクに送る。
- JSON summaryが詰まる場合はJSONLを優先し、stdout互換を保つ。

## Why This Should Actually Take 12 Hours

- 1 layer単体とは違い、層間出力を次層入力にするため、shape、position、RoPE、paged cache、residual受け渡しの前提が全layerで揃っている必要がある。
- Qwen3.5-9Bの複数layerをresident f32 weightとして持つとGPU memoryに当たりやすい。load/drop/window実行の判断と検証が必要になる。
- 単発のstdoutではなく、後から比較できるmachine-readable reportを設計する必要がある。
- CPU/R9700/V620、seq8/seq16、`0..4`/`0..8`の組み合わせを回すと、model loadとHIP初期化だけでも時間を使う。
- 数値差分が出たときに、backend差分、量子化誤差、layer接続ミス、position mismatchを分ける必要がある。

## Success Criteria

12時間後に成功とみなす条件:

- prefix fixture作成経路がある。
- uLLMがprefix fixtureを読み、連続layerを実package weightで実行できる。
- CPUで`seq8`、`0..4`のlayerごとmetricが残っている。
- 可能ならR9700でも同じfixtureのlayerごとmetricが残っている。
- result fileから、layerごとの誤差増幅とbackend差分を追える。
- 失敗時に、どのlayerで何の分類に近いかが記録されている。

この条件を満たせば、full prompt generationへ進む前に、少なくとも最初の数layerでuLLM package executionが参照実装からどの程度ずれるかを判断できる。

## Non-goals

12時間内にやらないこと:

- Rust側tokenizer統合
- token embedding実装
- final RMSNorm
- lm_head/logits比較
- sampling/greedy decode
- full 36 layerの常駐実行
- token/s benchmark
- fused dequant実装
- AQ/fq仕様更新

## Next Goal Candidate

`/goal`に渡すなら、目的は次のように絞る。

```text
Qwen3.5-9Bの固定token idから連続decoder layer prefixのgolden fixtureを出力し、uLLMの実package weightで0..4以上のlayerを順に実行して、CPU/R9700でlayerごとのMSE・max_abs_diff・cosine similarityと失敗分類を機械可読なresult fileへ保存できるpackage-golden-prefix-smokeを作成する。
```
