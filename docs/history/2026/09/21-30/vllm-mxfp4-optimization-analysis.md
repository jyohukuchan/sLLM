# vllm-mxfp4の最適化分析とsLLMへの適用候補

2026-09-21、[参照追加・実測](../11-20/vllm-mxfp4-investigation.md)に続き、
`reference/vllm-mxfp4/`（`GGZ14/vllm-mxfp4` commit `31b9a94a`）が行っている最適化を読み取り専用で分析した。
コードのコピー・移植は行っていない。vLLM系はsLLMの方針上no-copy referenceであり、
このcheckoutではRadiance部分を覆うtop-level LICENSEも確認できないため、設計の参照に限る。

分析は`codex exec -m gpt-5.6-sol`（reasoning max、read-only sandbox、ファイル変更・git操作・build・GPU実行なし）で実施した。
生ログは`.local-artifacts/vllm-mxfp4-analysis/sol-run.log`に保持し、Gitへは追加しない。
以下の「確認済み」は相手source・文書にある事実、「見込み」はsLLMへの推定である。

## 相手が行っている最適化

### 量子化形式とscale

- 本体weightはOCP MXFP4 E2M1、K方向block 32、E8M0 scale（4.25 bit/weight）。
  activationはper-token FP8 E4M3FN＋FP32 scale 1個のW4A8で、OCP MXFP8のblock活性値ではない。
- E8M0が2の冪であることを使い、出力行ごとの基準指数をload時に作り、block指数差をE2M1→E4M3変換表へ畳み込んで
  内側loopからscale乗算を除く。
- MTP drafterはMXFP4ではなくoutput channelごとのFP8 E4M3。MXFP4ではacceptanceが2.5→2.21へ低下し、
  FP8では2.60〜2.80を維持したのが理由。

### NVFP4ではなくMXFP4を使う理由

- kernelがRDNA4の`v_wmma_f32_16x16x16_fp8_fp8`前提で、E2M1をE4M3へ無損失に引き当ててFP8 WMMAへ入れる構成。
  相手の計測ではregister常駐のFP8 WMMA 325 TFLOP/s、FP16 WMMA 160、Triton FP8 43。
- NVFP4のcheckpointはnativeに動かさず、load時にMXFP4へ再量子化して同じW4A8 kernelへ載せる。
  block指数はno-clip規則と1 binade細かい側の二乗誤差比較で選ぶ（既定`mse`、他に`ocp`、`noclip`）。
- 精度は実テンソル実測でNVFP4がbf16比relRMS 0.113（19 dB）、再量子化MXFP4が0.158（16 dB）、
  bf16から直接MXFP4なら0.112。悪化は二重丸めであり形式差ではない、と明記されている。
- FP8本体（12.6 GiB/GPU）に対しMXFP4本体は9.24 GiB/GPUで、差がKV容量になる。
  混合精度checkpointのFP8層もMXFP4へ再量子化するのが既定で、FP8のまま残すと融合が付かず、
  8並列の持続負荷でGPUハング（ドライバリセット）が2回起きたと記録されている。

### epilogue融合（sLLMに最も関係する）

- RMSNorm／residual add＋RMSNorm、SiLU(gate)×up、GDN gated RMSNormを、いずれもper-token FP8量子化まで含めて
  1 producer kernelにする。producerは`(FP8 codes, scale)`を次のlinearへ直接渡し、consumer側で再量子化しない。
- decodeの128箇所で2〜3個のepilogue kernelを1個（約2.4 µs）へ集約。
  activation quantだけでdecode wallの約4%、実処理約2.2 µsに対しdispatch約4.7 µsという計測がある。
- 量子化を不透明なcustom opの内側へ置くとgraph fusionから見えずkernel数が増えたため、
  graphから見えるproducer側へ引き上げている。**graph化とkernel融合は別問題**であることを示す設計判断である。

### そのほか

- GDNの`in_proj_qkvz`と`in_proj_ba`をload後にN方向連結して1 GEMM化。48層で96 launchと48量子化を削減し、
  26.25→25.50 ms/step（約2.9%）。split-K geometryが変わるためbit exactではない。
- 小M decode kernel（M≤64、`TM=ceil(M/16)`、M≤16で深いsplit-K）、weightのWMMA fragment順byte-permuteと
  non-temporal load、LDS 8 byte padding、SGPRへのwave-uniform base address、`sched_barrier`によるVGPR圧低減。
- R4D attention（gfx1201、GQA6、head 256、block 16、BF16/FP8 KV専用）、GDNのWY・scan・outputの3 kernel統合、
  hybrid構成に合わせたKV group size選択（容量+20.7%）、DFlash2 drafterとint2 head、launcher/patch群。

## sLLMへの適用候補

| 優先 | 候補 | 変更箇所 | 効果の見込みと根拠 | リスク・数値分類 |
| --- | --- | --- | --- | --- |
| P0-1 | 活性値量子化を前段producerへ融合 | `residual_rmsnorm`／`rmsnorm`／`elementwise`／GDN gated normに、現行のNVFP4／FP8 code＋scaleを直接出すvariantを追加し、prequantized入力としてprojection pack・matmulへ渡す | sLLMは233 matmulに対し量子化185回/token。dispatch分の粗い上限で約0.8〜1.2 ms/token。相手側でも量子化はdispatch優勢 | FP8 per-rowは比較的容易、NVFP4のblock16＋tensor scaleが難所。bit一致ならN0 |
| P0-2 | gate/upとGDN qkv/zのdual-output bundle | `qwen38_projection_pack_runtime.inc`の2 launchを、multi-pointerまたはload時連結weightの1 kernelへ | 相手のGDN mergeが約2.9%。sLLMは最大104 pairが対象。ただしsLLMは量子化を既に共有しており、同じ削減量は期待しない | VGPR、split-K、N端、二outputのworkspace寿命。加算順維持ならN0 |
| P1 | MTP draft headの低bit化＋厳密rerank | BF16 companionのproposal headのみ。target head・sampling・verifyは不変 | 相手は2002→473 µs/call。ただし短contextで+6.5%、中程度で+0.1% | 受理率低下が主リスク。まずhead寄与をprofileで確認 |
| P2 | prefill向けA-tiled producer-consumer、lowp内側loopの監査 | `native/lowp`のlarge-M provider、fragment order・non-temporal load・LDS padding等 | 単一要求decodeには効かず、Phase 88やTTFT向け | layoutのみならN0 |
| P3 | GDN conv＋recurrentの追加融合 | `linear_attention_kernel.hip.cpp`のdecode state update | 相手の単独効果は0.4%、norm＋quantで0.8%。単独では1%未達の可能性 | state correctnessのリスクが高い |

P0-1とP0-2は独立に採否を決め、効果を加算しない。段階6で`gfx1030`のNVFP4 gate/upとFP8 M1が並列枝になったため、
P0-2はV620では並列枝、R9700では直列枝をそれぞれ対照にする必要がある。

## 適用しない・すべきでないもの

- MXFP4 W4A8形式そのもの。sLLMの決定はMXFP4 W4A6で、本番はNVFP4 block16＋E4M3 block scale＋tensor scaleであり、
  E8M0の指数畳み込みが使えない。
- R4D attentionの移植（gfx1201専用、通常FP8/BF16 KV前提、相手もdecodeはnoise内としている）。
- vLLMのgraph／scheduler patch群。sLLMは既にdevice制御・whole graph・pinned非同期readbackを持つ。
- DFlash2、動的verify幅、int2の**target** verify head（真のtop-kを候補に含む保証が経験則のためN2）。
- lazy GDN state snapshot（相手側でmulti-turnを壊しrevert済み）。
- KV group size探索・KV pin（複数要求の容量向け。Phase 88の参考に留める）。
- ParoQuant／INT4系、TP関連、W4A8の品質値を形式採用の根拠にすること、そしてsourceの直接流用。

## 判断に必要だが分からなかったこと

- ローカル実測の32.00→72.07 tok/sを、W4A8・R4D・DFlash・verify head・runner差へ分解できない。
  速度経路のKLDは未測定で、KLD測定はDFlash・int2 head・graphを無効にしたeager経路である。
- FP8 KVとFP16 SSMを同時に変えているため、KLD差をKV単独・state単独へ帰属できない。
- sLLMの185 quantizerのうち、どのproducer familyまでN0のまま融合できるか。特にNVFP4 tensor scaleのreduction契約。
- dual-output kernelがV620の並列枝やR9700の直列枝に勝つか。
- MTP BF16 companionの実測時間内でlm/draft headが占める割合。
- Radiance部分全体のライセンス。

先行: [参照追加・実測](../11-20/vllm-mxfp4-investigation.md) ／ 関連: [段階6](phase87-stage6.md)
