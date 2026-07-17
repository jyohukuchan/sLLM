# P2 AQ4 重みの来歴・復号監査

## 前回の要点

- Qwen3.5-9B の source oracle と active AQ4 package の source/path 比較では、hidden/logit の相対 L2 が約 0.545/0.615、greedy/top-k は一致しなかった。
- 正式な P2 policy は unbound template のままで、greedy exact は必須だが数値しきい値は未束縛である。GPU またはサービスは実行していない。

## 今回の変更点

- `uLLM-project/tools/audit_aq4_weight_provenance.py` を追加した。safetensors のヘッダーとデータ範囲を seek で読み、AQ4 index/scale/codebook を 4096 group 単位で復号する。全テンソルをメモリへ保持せず、source SHA、payload SHA、shape/dtype、行優先配置、relative MSE、最大絶対誤差を記録する。
- `uLLM-project/tests/test_aq4_weight_provenance.py` を追加し、passthrough の完全一致、AQ4 low-nibble 復号、shape 反転の転置疑いを合成 fixture で検証した。
- active package (`/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package`) の manifest SHA は active served manifest の宣言値と一致した。manifest の `source_model_dir` と Qwen3.5 source directory も一致した。

代表テンソルの実測値（BF16 source を AQ4 へ復号、相対 MSE）は以下である。

| テンソル | relative MSE |
| --- | ---: |
| layer 3 Q/K/V/O | 0.005315 / 0.003667 / 0.003797 / 0.003640 |
| layer 3 MLP gate/up/down | 0.005190 / 0.005249 / 0.005305 |
| LM head | 0.003627 |

embedding、layer 0 の入力・post-attention norm、final norm は raw payload と source tensor の SHA が一致した。量子化テンソルは測定 MSE/max と manifest の量子化時 metrics がおおむね 1e-13 以下の差で一致し、shape/dtype は全代表で一致、行優先配置で転置疑いはない。実行時の最大 RSS は約 40 MB、GPU/サービスは未使用である。

追加で manifest の全 256 量子化テンソルを同じストリーミング処理で走査した。256/256 が `ok` となり、shape/dtype/row-major の整合性は全件で成立した。manifest `relative_mse` の再現差は最大 `3.04e-8`、manifest `max_abs_error` の差は最大 `3.18e-9` だった。層別には特定層だけの破損はなく、linear-attention 層と full-attention 層の反復的な分布が見える。

- linear-attn 120件: 平均 0.0050626、範囲 0.0037321–0.0057831
- MLP 99件: 平均 0.0052748、範囲 0.0051862–0.0054614
- self-attn 36件: 平均 0.0041510、範囲 0.0036397–0.0053680
- lm_head: 0.00362669

最大の相対 MSE は layer 28 `linear_attn.in_proj_a` (0.0057831)、layer 8 `in_proj_b` (0.0057752)、layer 26 `in_proj_a` (0.0057616) である。これらも manifest 指標と再現し、単一ファイルの壊れ方を示す異常値ではない。

## 疑わしい箇所の順位と根拠

1. **runtime の matvec/attention/linear-attention 経路** — package の index/scale/codebook と export 時 metrics が一致し、passthrough の embedding/norm も完全一致する。一方、source-vs-path の hidden/logit drift と greedy mismatch は残るため、実行時の復号、行スケール適用、行列レイアウト、Qwen3.5 の分岐処理を最優先で確認する。
2. **P1/P2 の入力・プロトコル・sampling の結線** — active worker の manifest binding や source/path oracle の metadata は既知の blocker があり、同一入力・同一 tokenizer・同一 top-1 条件が保証されていない可能性がある。
3. **export/package** — 256 tensors の shape/dtype/row-major と量子化時 metrics の再現性が確認できるため、形式・選択・パッケージ内の復号入力のバグである可能性は低い。ただし、AQ4 の近似誤差そのものが top-1 不一致を説明できるかは、この重み監査だけでは判定できない。

## 次の行動

- P2 policy の correctness thresholds をバインドするまでは promotion を行わない。
- GPU を使える承認済みの次段で、独立 validator の hidden/logit 差分を runtime の復号バッファ直後、matvec 出力、QKV 分割・RoPE・norm 後に分解して再測定する。
- 同一入力の source oracle と path oracle を exact greedy/top-k 条件で再結線し、runtime 起因と入力/プロトコル起因を分離する。

## 検証

- `python3 -m unittest tests/test_aq4_weight_provenance.py -v` — 2 tests passed。
- `python3 -m py_compile tools/audit_aq4_weight_provenance.py` — passed。
- `python3 tools/audit_aq4_weight_provenance.py --chunk-groups 4096 ...` — 12 representative tensors passed; no GPU/service action。
- `python3 tools/audit_aq4_weight_provenance.py --all-quantized --chunk-groups 8192 ...` — 256/256 passed; max manifest relative-MSE reproduction delta `3.04e-8`; no GPU/service action。
