# P2 AQ4 GPU 中間差分トレース（attempt2）分析

分析日: 2026-07-15 (JST)

## 結論

attempt2 は 3 行すべてについて 35 stage（embedding、decoder layer 0--31、final norm、lm head）を生成した。source-v2 との差分は 3 行すべてで `decoder_layer:0` が最初であり、embedding は全固定座標で一致した。したがって、入力・再生列・サンプリング・実行時の package identity を原因とする仮説は低位で、AQ4 の非可逆量子化誤差が最も整合する。さらに、量子化された重みの復号または matvec カーネルの数値実装が、その誤差を過大化していないかは、今回の evidence だけでは切り分けられない。

最終出力では `fixture-prompt-0/step0` の greedy token が source `220` から candidate `41330` に変わった。残り 2 行は `16` と `15` で一致した。中間トレースの schema は top-k リストを出力しないため、top-k の比較は同一 package SHA に結び付いた既存の `source-oracle-v2` / `path-oracle-v2` で補足した。

## 比較対象と固定条件

- source: `p2/source-differential-trace-v2`（CPU BF16、3 行）
- candidate: `p2/differential-trace-gpu-v1-attempt2`（HIP device index 1、3 行）
- candidate package: `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package`
- candidate binary SHA256: `356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7`
- build commit: `28ec343ac59e6d22e710035d7874df9fbd8f890f`
- active manifest SHA256: `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`
- package manifest SHA256: `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`
- device: HIP index 1 / AMD Radeon Graphics / `gfx1201` / 34,208,743,424 B
- 固定 hidden 座標: `[0, 1, 1024, 2048, 4095]`
- 固定 logit 座標: `[0..31]`
- 比較式: 固定サンプル全要素の max absolute、`L2(delta)/L2(source)`、cosine similarity

`tools/trace-qwen35-aq4-differential.py analyze` の read-only 実行は rc=0 で、全行の first mismatch が `decoder_layer:0`、missing stages は空だった。以後の数値は同じ固定座標から再計算した。

## 行別の first mismatch と greedy

| case / step | first mismatch | max abs at first mismatch | relative L2 | cosine | source greedy | candidate greedy | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixture-prompt-0 / 0 | decoder 0 | 0.005409 | 0.211989 | 0.980983 | 220 | 41330 | no |
| fixture-prompt-0 / 1 | decoder 0 | 0.004887 | 0.178452 | 0.990004 | 16 | 16 | yes |
| fixture-prompt-1 / 0 | decoder 0 | 0.002427 | 0.029534 | 0.999567 | 15 | 15 | yes |

`relative L2` と `cosine` は first mismatch の固定 hidden サンプルについての値である。この表では判定上重要な first-stage max abs と greedy を示し、stage 全体の集計は次表に示す。

## stage 別集計

各行の max abs の最大値、relative L2 の最大値、cosine の最小値を示す。`kind` は Qwen3.5 の `config.json` にある `layer_types`（3 層ごとの linear attention と 4 層目の full attention）から付けた。

| stage | kind | max abs | max rel L2 | min cosine |
|---|---|---:|---:|---:|
| embedding | passthrough | 0.000000 | 0.000000 | 1.000000 |
| decoder 0 | linear | 0.005409 | 0.211989 | 0.980983 |
| decoder 1 | linear | 0.008840 | 0.078180 | 0.998007 |
| decoder 2 | linear | 0.040261 | 0.194784 | 0.981958 |
| decoder 3 | full | 0.042466 | 0.220259 | 0.991954 |
| decoder 4 | linear | 0.075667 | 0.221358 | 0.978404 |
| decoder 5 | linear | 0.093476 | 0.196770 | 0.981266 |
| decoder 6 | linear | 0.184408 | 0.293102 | 0.968037 |
| decoder 7 | full | 0.145702 | 0.275743 | 0.974660 |
| decoder 8 | linear | 0.160033 | 0.310219 | 0.962039 |
| decoder 9 | linear | 0.159724 | 0.397278 | 0.923550 |
| decoder 10 | linear | 0.199803 | 0.615198 | 0.817744 |
| decoder 11 | full | 0.173228 | 0.367022 | 0.940421 |
| decoder 12 | linear | 0.158949 | 0.641367 | 0.905703 |
| decoder 13 | linear | 0.161022 | 0.649263 | 0.833460 |
| decoder 14 | linear | 0.174585 | 0.475540 | 0.903738 |
| decoder 15 | full | 0.161362 | 0.422545 | 0.913053 |
| decoder 16 | linear | 0.126969 | 0.458320 | 0.917877 |
| decoder 17 | linear | 0.189945 | 0.265784 | 0.965764 |
| decoder 18 | linear | 0.177223 | 0.214054 | 0.978290 |
| decoder 19 | full | 0.193602 | 0.155446 | 0.987874 |
| decoder 20 | linear | 0.196801 | 0.229895 | 0.973319 |
| decoder 21 | linear | 0.209550 | 0.297482 | 0.954818 |
| decoder 22 | linear | 0.240236 | 0.177435 | 0.984818 |
| decoder 23 | full | 0.237356 | 0.241554 | 0.972416 |
| decoder 24 | linear | 0.267872 | 0.268714 | 0.963587 |
| decoder 25 | linear | 0.344191 | 0.276584 | 0.971211 |
| decoder 26 | linear | 0.464012 | 0.304600 | 0.968113 |
| decoder 27 | full | 0.369539 | 0.280576 | 0.973872 |
| decoder 28 | linear | 0.415981 | 0.380490 | 0.931409 |
| decoder 29 | linear | 0.696200 | 0.232744 | 0.979927 |
| decoder 30 | linear | 0.821305 | 0.302967 | 0.953278 |
| decoder 31 | full | 0.761988 | 0.199674 | 0.985556 |
| final_norm | passthrough weights | 1.084648 | 0.545288 | 0.983324 |
| lm_head | AQ4 | 8.347782 | 0.621299 | 0.939127 |

最大値は `lm_head`（fixture-prompt-1/step0、max abs 8.347782、relative L2 0.540729、cosine 0.998058）で、次いで final norm（1.084648）が大きい。decoder の差分は layer 0 から始まり、後段に向かって一般に増幅する。単一の final norm/export のみを原因とする形ではない。

## 256 重み監査

package manifest の `tensors` は 256 件すべてが AQ4 index/scale/codebook 付きで、group size は g16 が 204 件、g8 が 52 件だった。passthrough は 519 件で、embedding は BF16 raw passthrough 1 件である。language model 32 層は config と一致し、linear 層 24 層は 8 family、full attention 層 8 層は 7 family、加えて MTP 7 件と lm_head 1 件で 256 件になる。

| family | 件数 | max abs error | max relative MSE |
|---|---:|---:|---:|
| linear_attn_a / b / qkv / z / out | 120 (各24) | 0.062870 | 0.005783 |
| attn_q / k / v / o | 36 (各9) | 0.074449 | 0.005368 |
| mlp_up / gate / down | 99 (各33) | 0.036133 | 0.005462 |
| lm_head | 1 | 0.017830 | 0.003627 |

全 256 件の tensor-level `max_abs_error` は最大 0.074449、relative MSE は最大 0.005783（平均 0.005011）だった。embedding は量子化対象外で実測一致し、最初の差分が decoder 0 に出るため、量子化重みが最初の演算段階で差を導入した説明と整合する。lm_head 自体の量子化誤差は小さく、最終 logits の大きな差は主に前段 hidden の伝播後に現れている。

## input / sampling / matvec 監査

- `cases.json` SHA256 は `15fed90d...4836f2`、`replay.json` SHA256 は `1ee0b922...4747dc`。candidate manifest の `input_binding` の expected/actual hash は両方一致した。
- cases は `fixture-prompt-0=[11,12,13]`（2 steps）、`fixture-prompt-1=[21,22]`（1 step）。各 candidate row の context hash は source-v2 と一致し、後続 step は source replay `[220,16]` / `[15]` を固定している。candidate token を次の入力へ回す閉ループではない。
- candidate trace は `SamplingParams::greedy_with_top_k(0, 1)` を使用し、温度・top-p・seed による乱択はない。`greedy_token_id` は各 row の実際の top-1 である。
- attempt2 intermediate payload は top-k 配列を保存しない。source の `source-oracle-v2` と、attempt2 と同じ package SHA (`a790a033...f08e5ad`) に結び付いた candidate path の `path-oracle-v2`（各 top-10）では、row 0 の set overlap が 1/10（top-1 flip）、row 1 と row 2 は 10/10（順位は一部変動）だった。したがって、top-k の完全な attempt2 固有値は未取得だが、candidate path の top-k は 2/3 行で集合一致している。
- gate preflight は 30 個の required HIP kernel environment を exact order で検証し、実行時にも AQ4 matvec、matvec add/pair/triple、BF16 matvec、linear-attention、RMSNorm、Top1 等を要求した。これは fallback を抑止する dispatch guard の証拠であり、guard_set SHA は `0d034d87...ce3f90`。ただし、各 matvec の数値を独立した CPU/dequantized oracle と照合するものではないため、kernel/scale-index の誤りは残る。

## 仮説の順位

1. **AQ4 の非可逆量子化誤差（最有力、期待される fidelity 差）** — 256 件の量子化 tensor、embedding の完全一致、decoder 0 からの差分、後段での増幅、lm_head/final_norm での大きな差が一貫する。
2. **AQ4 復号または matvec/linear-attention kernel の数値実装（次点、未除外の実装バグ）** — divergence が最初の decoder で観測され、30 kernel guard は fallback を除外するが、scale/index 復号と演算結果の独立照合はまだない。
3. **runtime / export / package の取り違え（低位）** — active/package/binary/build SHA、regular-file・非 symlink、device index、入力 hash、35 stage coverage がすべて一致し、attempt1 で判明した package root argv 問題は `PACKAGE/package` に修正済み。残る gate rc=1 は出力ディレクトリ外で `sha256sum -c` した verifier の cwd バグであり、trace 内容の失敗ではない。
4. **sampling または入力再生の不一致（低位）** — fixed replay と top-k=1 greedy を manifest/code で検証済み。row 0 の greedy flip は decoder 0 以降の数値差で説明できる。

## 次の切り分け

GPU を再実行せずに進めるなら、package の各 quantized row を同じ group decode で CPU 側に再構成し、固定 hidden/sample と matvec の一段ずつを照合する。そこで CPU dequantized と GPU が一致すれば「期待される lossiness」、一致しなければ scale/index decode または kernel 実装を上位へ繰り上げる。top-k を中間 trace に残す場合は schema 拡張が必要で、今回の attempt2 artifact は変更しない。
