# 12h golden layer validation plan v0.1

## Purpose

推論エンジン土台作りを約24時間進めた時点で、次の12時間は「full prompt generation」へ無理に進まない。まず、実推論前に基盤が崩壊していないことを確認するため、Qwen3.5系の1 layer単位で参照実装と比較できるgolden tensor検証経路を作る。

この計画の目的は、実プロンプト品質評価ではなく、token embeddingや全layer実行をまだ持たない状態でも、実package weight、scheduler、paged KV cache、decoder layer境界の正しさを参照実装に近い入力で検査することである。

## 前回の要点

- AQ/fqの最終仕様は、推論エンジン上の実測なしには固定しない。
- 現在はQwen3.5-9B由来の`.ullm.d` packageを読み、selected decoder layersのself-attention + MLP blockをCPU、R9700/RDNA4、V620/RDNA2でsmokeできる。
- 既存smokeはhost期待値との部品比較として有効だが、token idからのend-to-end generationの正しさはまだ保証していない。
- 実プロンプトへ進む前に、Hugging Face Transformersなどの参照実装から出した中間hidden stateとuLLMの1 layer出力を比較する必要がある。

## 今回の変更点

- 12時間で終わる範囲に絞り、full model generation、tokenizer統合、sampling、速度最適化は対象外にする。
- 既存のpackage model loop smokeを広げるのではなく、golden tensor fixtureを使った1 layer検証を作る。
- uLLM側ではembeddingをまだ実装しない。参照実装から保存した`hidden_before_layer_N`をlayer入力として注入する。
- 出力比較は厳密一致ではなく、AQ package由来の量子化誤差を含むため、MSE、mean_abs_diff、max_abs_diff、cosine similarityを記録する。

## Target Scope

対象:

- model: Qwen3.5-9B
- package: 既存のQwen3.5-9B p4p6 `.ullm.d`
- devices: CPU fallback `0`、R9700/RDNA4 `2`
- optional device: V620/RDNA2 `1`
- sequence length: 8または16
- layer: まず1 layer。時間が余れば`0,3,7,11`の複数layer
- input: tokenizer済みではなく、固定token id列

対象外:

- 実プロンプト文字列からのtokenization
- 全decoder layer連結
- final RMSNorm
- lm_head/logits
- sampling/greedy decode
- real token/s benchmark
- fused AQ kernel
- llama.cppへのquant実装

## Deliverables

1. `tools/export-qwen-golden-tensors.py`
   - Hugging Face Transformersで固定token idを処理する。
   - `hidden_before_layer_N`と`hidden_after_layer_N`を保存する。
   - 保存先は`benchmarks/golden/`配下にする。
   - 大きなtensorを避け、short sequenceとselected layersだけを保存する。

2. `ullm-engine package-layer-golden-smoke`
   - `.ullm.d` package、golden fixture、layer index、device indexを受け取る。
   - golden fixture内の`hidden_before_layer_N`をuLLMのlayer入力として使う。
   - uLLMの1 layer出力と`hidden_after_layer_N`を比較する。
   - MSE、mean_abs_diff、max_abs_diff、cosine similarityを出力する。

3. 検証記録
   - CPUとR9700で同じfixtureを実行した結果を`benchmarks/results/2026-07-04/engine/`または実行日の結果ディレクトリへ保存する。
   - 失敗した場合は、shape mismatch、projection layout mismatch、RoPE position mismatch、quantization errorのどれに近いかを切り分けて記録する。

4. 用語とjournal
   - `docs/words.txt`にgolden tensor fixtureとpackage layer golden smokeを追加する。
   - `journal/YYYY/MM/DD/`に実施内容と結果を残す。

## 12h Task Breakdown

### T0: Baseline check, 0.5h

目的:

- 現在のmain branchが壊れていないことを確認する。

手順:

1. `git status --short`
2. `cargo fmt --all --check`
3. `cargo check -p ullm-engine`
4. 既存の`package-self-attn-mlp-block-model-loop-smoke`をCPUで1回実行する。

完了条件:

- 既存smokeが成功する。
- 失敗した場合は新規作業へ進まず、まず直近変更の影響を切り分ける。

### T1: Golden tensor exporter, 2.0h

目的:

- uLLMにembeddingやfull model loopが無くても、参照実装のlayer入力/出力を固定fixtureとして使えるようにする。

手順:

1. `tools/export-qwen-golden-tensors.py`を追加する。
2. 入力は`--model-dir`、`--token-ids`、`--layers`、`--output`にする。
3. `torch.no_grad()`または`torch.inference_mode()`を使う。
4. CPU/RAM消費を抑えるため、sequence lengthは8または16、保存layerは最小にする。
5. fixtureにはmetadataも保存する。
   - model id
   - dtype
   - token ids
   - layer index
   - hidden size
   - sequence length
   - position ids

完了条件:

- Qwen3.5-9Bで少なくとも1 layer分のfixtureが作れる。
- 保存tensorのshapeとdtypeを読み戻して確認できる。

### T2: Fixture loader and metric utility, 1.5h

目的:

- uLLM側でgolden tensor fixtureを読み、比較指標を一貫して出せるようにする。

手順:

1. Rust側に小さいfixture loaderを追加する。
2. まずは`.npz`より実装が簡単な形式を選ぶ。短期ではJSON metadata + raw f32 little-endian payloadでよい。
3. MSE、mean_abs_diff、max_abs_diff、cosine similarityを計算する共通関数を作る。
4. unit testで小さい固定配列を比較する。

完了条件:

- fixture loaderがshape不一致、payload bytes不一致、layer index不一致をエラーにできる。
- metric utilityのunit testが通る。

### T3: One-layer package golden smoke, 4.0h

目的:

- 実package weightを使って1 layerを実行し、参照実装の`hidden_after_layer_N`と比較する。

手順:

1. CLI `package-layer-golden-smoke`を追加する。
2. 既存のQwen3 package runtime weight loaderを使って指定layerを読み込む。
3. golden fixtureの`hidden_before_layer_N`をresidual/input sequenceとして扱う。
4. 既存のQwen3 decoder layer sequence実行経路へ接続する。
5. uLLM出力と`hidden_after_layer_N`を比較する。
6. CPUで先に通し、R9700で再実行する。

完了条件:

- CPUでshape、実行、metric出力まで通る。
- R9700でも同じfixtureで実行できる。
- 差分が大きい場合でもpanicではなく、metricと診断情報を出す。

### T4: Failure classification, 1.0h

目的:

- 失敗時に「基盤崩壊」なのか「量子化誤差」なのかを分けやすくする。

手順:

1. metric出力に次の情報を入れる。
   - layer index
   - sequence length
   - hidden size
   - device
   - package tensor candidate id
   - max_abs_diff
   - MSE
   - cosine similarity
2. 可能なら先頭数tokenのpreview差分を出す。
3. 形状不一致と数値不一致を別エラーにする。

完了条件:

- 失敗しても原因分類に必要な情報がstdoutまたはresult fileに残る。

### T5: Rerun and record, 1.5h

目的:

- 次の/goalへ引き継げる検証結果を残す。

手順:

1. `cargo fmt --all --check`
2. `cargo check -p ullm-engine`
3. 関連unit tests
4. CPU golden smoke
5. R9700 golden smoke
6. optionalでV620 golden smoke
7. 結果を`benchmarks/results/`へ保存する。
8. journalを更新する。

完了条件:

- 12時間タスクの成果物、実行コマンド、結果、残課題がjournalにまとまっている。

### Buffer: 1.5h

目的:

- Transformers側のmodel loading、ROCm/PyTorch環境、fixture形式の調整で詰まった場合の吸収時間にする。

Fallback:

- HF exporterが詰まる場合は、まず小型tensorのdummy fixtureで`package-layer-golden-smoke`のloader/metric/CLIだけを完成させる。
- R9700実行が詰まる場合は、CPU成功までを完了条件にして、R9700は次タスクへ送る。
- 1 layer比較が想定より重い場合は、layer 0のみ、sequence length 4へ縮小する。

## Success Criteria

12時間後に成功とみなす条件:

- golden tensor fixture作成経路がある。
- uLLMがfixtureを読み、1 layer出力比較を実行できる。
- CPUで少なくとも1 layerの比較結果が残っている。
- 可能ならR9700でも同じ比較結果が残っている。
- 失敗時の原因分類に使えるmetricが出る。

この条件を満たせば、次にfull model generationへ進む前に、layer境界で崩壊しているかどうかを判断できる。

## Non-goals

12時間内にやらないこと:

- 実プロンプトから文章生成すること
- full logits一致を保証すること
- decode-tps/prefill-tpsを測ること
- llama.cppにuLLM quantを実装すること
- AQ/fq仕様を更新すること
- fused dequantを作ること

## Next Goal Candidate

`/goal`に渡すなら、目的は次のように絞る。

```text
Qwen3.5-9Bの固定token idからHugging Face参照hidden stateをgolden tensor fixtureとして出力し、uLLMの実package decoder layerを1 layer単位で実行してCPU/R9700でMSE・max_abs_diff・cosine similarityを比較できるpackage-layer-golden-smokeを作成する。
```

