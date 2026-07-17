# P2 入力・sampling・runtime layout 制御監査

## 前回の要点

- 重み監査では、active AQ4 package の全256量子化テンソルが manifest 指標を再現し、shape/dtype/row-major grouped 配置に異常はなかった。
- source/path oracle の hidden/logit drift と greedy/top-k 不一致は残っている。GPU、サービス、共有 runtime/session/model は今回も変更していない。

## 今回の変更点

- `uLLM-project/tools/audit_aq4_p2_input_controls.py` を追加した。BF16 source oracle、AQ4 path oracle、calibration case、pure-prefill fixture、gateway fixture を読み取り、case/step coverage、prompt hash、context 算術、greedy/top-k tie policy、sampling、vocabulary shape、行優先 AQ4 matvec を照合する。
- `uLLM-project/tests/fixtures/aq4-p2-input-controls/pure-prefill.json` と `gateway-request.json` を追加した。calibration の `[11,12,13]` を共通入力にし、pure prefill の `position_ids=[0,1,2]` と causal lower-triangular mask、gateway の decode positions `[3,4]`、greedy sampling (`temperature=0`, `top_p=1`, `top_k=1`, `seed=0`) を明示した。
- `uLLM-project/tests/test_aq4_p2_input_controls.py` を追加し、compact oracle の未観測フィールドを誤って不一致扱いしないこと、fixture 入力・位置・mask、matvec の再現を固定した。

## 実測結果

### oracle の観測範囲

source/path oracle v2 は bounded payload の仕様上、`context_length`、`context_token_ids_sha256`、`position_ids`、`attention_mask` を行に保存していない。したがって以下は「再構成できた算術」と「実測できたフィールド」を分けた。

- manifest の prompt token count と step から、期待 context は `prompt_tokens + step` と再構成できる。3 rows の算術不一致は 0 件。
- compact oracle 行では context length/hash は未観測 (`observed=false`) であり、これを一致証拠とは扱っていない。
- source/path case/step coverage は 3/3 行で一致した。
- source/path greedy mismatch は 1/3 行（`fixture-prompt-0`, step 0: source `220`, path `41330`）。top-k token 列は 3/3 行で不一致。step 1 と `fixture-prompt-1` の greedy token は一致した。
- source/path の ranking 宣言は `maximum_logit_then_smallest_token_id` と `logit_descending_then_token_id_ascending` で同一。観測 top-k の同値 logit は token ID 昇順で並び、tie policy 自体の異常はない。
- oracle payload の logits shape は `[248320]`、full-vocabulary ranking であり、LM head の vocab slicing/axis は metadata 上一致する。

### calibration/pure-prefill/gateway の入力と layout

- calibration case の実 token IDs `[11,12,13]` と、pure-prefill fixture、gateway fixture の token IDs は完全一致し、canonical token-ID hash も一致した。
- pure-prefill fixture は context length 3、position IDs `[0,1,2]`、causal mask の下三角を明示し、独立 checker が全要素一致を確認した。
- gateway fixture は source-control の generated IDs `[220,16]` と decode positions `[3,4]` を明示した。これは source calibration と一致する fixture であり、path oracle の別 greedy (`41330`) を隠していない。
- oracle manifest の prompt-token hash と calibration case の token IDs は、canonical JSON + 末尾改行の SHA-256 契約で 2/2 case が一致した。入力の取り違えによる source/path 差ではない。
- gateway の context extension は `3 + 2 = 5` と宣言値が一致し、生成 step 数 2 と calibration の `step_count=2`、decode positions `[3,4]` も一致した。
- active served manifest の `temperature=false`, `top_p=false`, `top_k=1` は、gateway schema の無指定時に有効値 `temperature=0`, `top_p=1`, `top_k=1` として正規化される。case/gateway の greedy sampling と一致した。
- position IDs と attention mask は compact source/path oracle では未観測であり、fixture に基づく制御確認だけが有効である。

### AQ4 runtime CPU matvec control

Rust `cpu_aq4_matvec_f32_computes_expected_values` と同じ小型 fixtureを、独立 Python reference (`idx4_low_nibble_first`, row-major, group size 2, no bias, f32 scalar accumulation) で計算した。`indices=[0x21,0x03,0x54]`, `scale_indices=[0,1,0]`, codebook `0..15`, scale `[0.5,2.0]`, tensor scale `10`, input `[0.5,-1.0,2.0]` に対し、出力 `[112.5,30.0]` が完全一致した。Rust CPU test も 1 passed である。

この結果は CPU fallback の transpose/stride/bias/dtype 規約が小型 fixture で壊れていないことを示す。ただし実際の GPU kernel、full-model matvec chain、Qwen3.5 attention/linear-attention 分岐を証明するものではない。

## 残る仮説と次の行動

1. 入力 token IDs、greedy sampling、vocabulary indexing、CPU の基本 matvec layout は、今回の control 範囲では異常が再現しなかった。
2. source/path step0 の greedy/top-k 不一致は依然として実測差であり、AQ4 近似誤差か runtime の full-model 演算経路のどちらかは未確定である。compact oracle だけでは position/mask を否定できない。
3. 次は、GPUを使わない専用候補で path の `context_length`、position IDs、causal mask の要約（各行の許可 key 範囲）を bounded trace に追加し、source-control と同じ fixture/hash chain へ束縛する。続いて承認済みGPU窓で、単一 QKV/O/MLP matvec の runtime 出力と独立 reference を比較する。

## 検証

- `python3 -m unittest tests/test_aq4_p2_input_controls.py -v` — 3 tests passed。
- `python3 -m py_compile tools/audit_aq4_p2_input_controls.py` — passed。
- `python3 tools/audit_aq4_p2_input_controls.py` — exit 0, status `controls_match_except_source_path_greedy_and_topk`。
- `cargo test -p ullm-runtime-sys cpu_aq4_matvec_f32_computes_expected_values -- --test-threads=1` — 1 passed（CPU、GPU/service未使用）。
