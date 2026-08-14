# Phase 9 実行エンジン構造最適化履歴

## 2026-08-14: 計画作成とPhase繰り下げ

- ユーザー指示により、Phase 8の残差性能backlogをPhase 8.5ではなく正式なPhase 9とした。
- 旧Phase 9以降を一段繰り下げ、model本体FP8 W8A8をPhase 10、CDNA3移植をPhase 11、MI300X実機確認を
  Phase 12、Gemma 4をPhase 13、Weight NVFP4をPhase 14とした。以降も同様に一段繰り下げた。
- Phase 8 short-oddで残ったllama.cpp比約20.4倍/26.9倍のE2E差と、output tokenあたり約468 submission /
  492 kernelを開始根拠とした。最初の優先順位をdtype非依存graph/segment実行、completion集約、M=1
  GEMV/MMVF、24 layerのQwen GDN、MLP/RMSNorm fusion、prefill provider再評価とした。
- llama.cpp固定commitのHIP Graph、MMVF、GDN、小規模fusionを直接reuse候補とし、Rust service、semantic op、
  vAttention、scheduler、ownership/error契約はsLLM側に維持する。llama.cpp以外はno-copy参考とする。
- 通常iterationはmicro/O0/O1、canonical両GPU・2B/9B・llama.cpp・serviceはintegrationまたは意味変更時に
  限定した。未承認の性能倍率やparityはhard gateにしていない。

## 2026-08-14: 後続Phase 13の挿入

- Phase 9完了後のユーザー指示により、モデル非依存prepared execution制御を新しいPhase 13として挿入した。
- 上記のPhase 9計画時点の番号は履歴として維持するが、現在のGemma 4はPhase 14、Weight NVFP4はPhase 15、
  以降の旧Phase 15〜19もPhase 16〜20へ一段繰り下げられた。

## 2026-08-14: A0 gap accounting

- Phase 8 short-oddを開始baselineに固定した。V620はTTFT/E2E `1.0987/9.6533 s`、prefill/decode
  `15.56/1.88 tok/s`、R9700は`0.6834/8.8907 s`、`25.10/1.95 tok/s`だった。
- R9700の代表rocprofではdecode 1 tokenが約526–528 msなのにGPU kernelは約21 ms、GPU busyは約4%だった。
  約468 submission/tokenの各completionで固定1 ms sleepしていたことが主要因だった。
- fixed sleepを64 immediate poll、64 `yield`、以降25 us sleepへ変更した。timeout/error意味は変えず、
  short-oddはV620 `1.259 s`、R9700 `1.105 s` E2Eまで改善した。変更後profileではdecode wall約58–60 ms、
  GPU約19.2 ms、busy約32–33%となり、次の支配候補をcompletion集約、Matvec、GDNへ更新した。

## 2026-08-14: A1 HIP Graph PoCとbounded reader

- fixed llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`のgraph state、MMVF、GDNだけを
  [bounded reader](../../../../references/phase9-engine-optimization-reader.md)へ固定した。他engineはno-copyとした。
- standalone PoCで、sLLM kernel 1 nodeとhipBLAS+epilogue 2 nodeを各64 replayした。device parameter blockで
  pointer/scalar/generationを更新し、独立oracleとgraph/resource cleanupを両targetでPASSした。

| target | kernel replay+update | hipBLAS mixed | instantiate（kernel/mixed） |
| --- | ---: | ---: | ---: |
| V620 `gfx1030` | 25.921 us | 35.718 us | 9.324 ms / 0.108 ms |
| R9700 `gfx1201` | 14.967 us | 16.262 us | 0.085 ms / 0.110 ms |

- capture自体は成立したが、production pathは8 full-attention layerのtransactional KV appendが明示境界で、
  request-local graph instantiateを導入する利得がこの段階では小さい。Phase 9はprepared semantic planを
  model/request residentに再利用するsame-stream segmentを採用し、全production graph replayは残差backlogへ残した。

## 2026-08-14: A2 completion集約

- semantic、causal attention、linear attention submissionをownerごとsegmentに保持し、full-KV appendまたは
  terminal argmaxだけをblocking completion境界にした。同じstreamの境界eventがterminalになった後、先行opは
  blocking waitせずqueryで成功を確認し、dispatch auditを記録する。
- buffer、workspace、KV/linear state ownerはsegment terminalまで保持する。error/drop/timeout時は既存request
  poisonとtransactional publicationを維持し、未完了stateを公開しない。
- short-oddはV620でE2E `0.963 s`、decode `25.23 tok/s`、R9700で`0.692 s`、`36.61 tok/s`となった。
  submission/kernel audit countは意味を変えていないため7,956/8,364のままだが、固定sleepを伴うper-op host waitは
  廃止された。

## 2026-08-14: A3 MMVF v3

- llama.cpp `mmvf.cu`からpaired BF16 loadとwave reductionをboundedにadaptし、ggml runtime、generic tensor、
  fusion、ID routingを持ち込まない`matmul.bf16_fp32.decode.v3`を追加した。odd/unaligned Kはchecked scalar loadを
  使い、BF16 input/weight、FP32 accumulation、BF16 RNE outputのsemantic contractを維持した。
- 17形状の独立oracleを両targetでPASSした。M=1,K=2560,N=9216はV620 259.282 us、R9700 75.002 usだった。
  V620の旧v2 381.645 us比で約32%改善し、R9700もfull modelでは旧hipBLAS選択より改善したためM=1は両targetで
  v3を選択した。M>1の境界255/256/257、odd K、NaN/Inf classification、cleanupもPASSした。
- direct reuse noticeとsource headerを追加した。import commit/hashは未commit integration candidateのため
  `pending-import-commit`であり、release/distribution前に解消する。

## 2026-08-14: A4 Qwen GDN

- 24 linear-attention layerのprivate FP32 recurrent stateを、V620だけwave-coalesced transposed physical layoutへ
  変更した。R9700では同layoutが退行したため、従来のthread-contiguous rowを維持した。public ABI、数値順序、
  generation/rollback、BF16 outputは変更していない。
- prefill 3/decode 1のreal GPU differentialを両targetでPASSし、最大BF16 ULP差は0だった。short-oddはV620で
  TTFT/E2E `0.303/0.843 s`、decode `30.06 tok/s`、R9700で`0.246/0.682 s`、`37.33 tok/s`となった。
- A4後R9700 profileでは1 tokenのGPU約18.6 ms中、MMVFが約15.5 ms（約81%）、GDN約1.46 ms、attention
  preprocess約0.70 ms、RMSNorm約0.48 msだった。full attentionや追加GDN/MLP fusionは支配要因でないため、
  profile-driven方針に従いPhase 9では一律実装しなかった。

## 2026-08-14: A5 prefill provider

- real M=17 Qwen shapeを比較した。V620 hipBLASは主要projectionが約1.4–2.2 ms、vocabが32.4 msでcustom
  tiled16を上回らなかった。R9700は主要projection約34–51 us、vocab約2.10 msだった。
- provider enumを一般化し、R9700 `M>1`だけmodel contextのhipBLAS handleを再利用する
  `matmul.hipblas.gemm_ex.v2`へ切り替えた。M=1は両targetでMMVF v3、V620 M>1はtiled16を維持する。
- R9700 short-odd prefillは約71から377 tok/sへ改善した。weight repack、library workspace、requestごとのhandle
  生成はなく、Phase 10のFP8 provider registryと同じ境界を再利用できる。

## 2026-08-14: A6 integration結果

最終4B direct engineは1 correctness control、3 warmup、10 measuredで測定した。raw reportはlocal-onlyとし、
数値とdigestは`ci/matrix/phase9-profile-summary-v1.json`を正本とする。

| target | TTFT | E2E | prefill tok/s | decode tok/s | fixed llama.cpp E2E / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | 0.306 s | 0.855 s | 56.91 | 29.69 | 0.473 s / 41.00 |
| R9700 `gfx1201` | 0.051 s | 0.490 s | 377.46 | 37.20 | 0.331 s / 52.27 |

- Phase 8比でE2EはV620約11.3倍、R9700約18.2倍高速化した。固定llama.cppとの差は約1.81倍/1.48倍、
  decode throughputはllama.cppの約72%/71%まで縮小した。resident/peak VRAMは8,411,592,192 /
  8,540,569,292 bytesでPhase 8から増えていない。
- 32/32 surrogateはV620 E2E `1.351 s` / decode `30.19 tok/s`、R9700 `0.936 s` / `36.44 tok/s`。
  2B V620 short-oddは`0.479 s` / `51.64 tok/s`、9B R9700は`0.684 s` / `26.11 tok/s`で、全てHIP-only、
  fallbackなし、request/session cleanup 0だった。
- canonical O2はminimum、short-odd、255/256/257、prefill-long、decode-longを両targetで実行し、14/14
  reportをPASSした。V620 prefill-longはprefill/decode `106.99/23.30 tok/s`、decode-longは`29.22 tok/s`、
  R9700は`790.76/28.33 tok/s`、`36.08 tok/s`だった。全reportでfallbackなし、cleanup 0である。
- R9700 optimized serverでraw OpenAI non-streamとSSEをPASSし、同一text/usage、terminal `[DONE]`、HIP-only、
  fallbackなし、shutdown時model/request/workspace current bytes 0を確認した。Phase 8の両target service evidenceは
  semantic/API契約不変のため引き続き有効である。
- 採用しなかった候補は、R9700のtransposed GDN、V620のM>1 hipBLAS、profile上位でないfull attention追加、
  この段階でのrequest-local production HIP Graph instantiateである。RDNA4 FA3-likeもattentionが支配要因で
  ないため将来taskのままとした。
- 残差の主因はmemory-bound M=1 matvecとhost launch/owner traversalである。production HIP Graphまたは
  native command-list replay、gate/up+SiLU等のMLP fusionはPhase 10をblockしない共通backlogとして残し、
  Weight NVFP4の現在のPhase 15開始前にfresh profileで再評価する。

## Reviewとcloseout

- 累積integration reviewでは数値/target dispatch、completion ownership、state publication、silent fallback、
  cleanup、provenance、historical Phase 8 evidenceの非改変を確認した。新しいcorrectness/security blockerはない。
- Phase 9の受入条件を満たし、次のactive product phaseをPhase 10 model本体FP8 W8A8とする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase9-engine-structural-optimization.md)
