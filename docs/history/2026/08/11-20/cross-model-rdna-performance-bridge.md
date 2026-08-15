# Phase 14→15 Qwen/Gemma共通RDNA性能bridge履歴

## 2026-08-15: B0開始

- Phase 14をcommit `048e02a9`で完了・pushし、localと`origin/main`が一致するclean identityからbridgeを開始した。
- Qwen3.5-2BとGemma 4-12Bのreviewed cache、R9700 exact `gfx1201`、V620 exact `gfx1030`、
  ROCm 7.14.0 `rocprofv3 1.3.2`が利用可能で、測定開始時に対象GPU processがないことを確認した。
- candidateは最大二つ、通常iterationはO0/O1、GPU証拠はfallback/timeout/crash/zero selectionをfail-closedとする。

## 2026-08-15: B0/B1 fresh profileと候補選定

- R9700 exact `gfx1201`でQwen3.5-2B short-oddをwarmup 3回+measured 10回取得した。median decodeは
  `65.427 tok/s`、TTFT `33.123 ms`、E2E `281.621 ms`、requestあたりsubmission/kernelは
  `5,984/6,290`で、生成token列は全回一致、fallbackなし、cleanup 0だった。
- 同じR9700でGemma 4-12Bをfreshに取得した。`3/17`はdecode `13.798 tok/s`、TPOT median
  `72.494 ms`、prefill `2.932 tok/s`、peak `23,867,610,772 byte`、`32/32`はdecode
  `13.427 tok/s`、TPOT median `74.496 ms`、prefill `405.927 tok/s`、peak
  `24,216,250,864 byte`だった。submission/kernelはそれぞれ`17,918/17,918`、
  `33,728/33,728`、token digest一致、fallbackなし、cleanup 0である。
- bounded rocprofではGemmaのdevice timeはdecode BF16 matvec v3が`84.28%`、RMSNorm `5.03%`、
  attention `4.07%`だった。host側はkernel launch 50,988回/331.6 ms、event record
  105,410回/207.1 ms、malloc/free各51,407回/103.6+126.8 msである。Qwenはdecode matvec
  `63.28%`、recurrent GDN `16.71%`、attention preprocess `4.95%`、RMSNorm `2.86%`、
  Argmax `2.72%`で、kernel launch 85,442回/462.8 ms、event record 168,748回/318.2 msだった。
- attentionは代表wall timeの支配要因でないためFA3-likeを除外した。候補を最大二つに固定し、(1) Gemma requestの
  compatible workspaceとprepared semantic再利用、(2) 両model/両GPUが使うM=1 BF16 matvecのstreaming weight loadを
  選んだ。いずれも2026-08-15のbridge内だけを実装範囲・期限とし、model固有graph rewriteや新しいhard gateは作らない。

## 2026-08-15: B2 candidate 1採用

- Gemmaのprefill allocationをcapacity ownerとして、decodeで同名・同backingかつviewが収まるtoken/position/workspaceを
  再bindした。request-owned `PreparedSemanticCache`をprefill/decode間で共有し、descriptor、buffer ID、view、access、
  token countをexact keyにする。position/state依存のcausal attentionとKV appendは`Transient`のままである。
- R9700反復3回のmedianは`3/17`が`14.116 tok/s`、`32/32`が`13.744 tok/s`で、fresh baseline比
  `+2.30%/+2.36%`だった。token digest、dispatch、fallback、cleanupは不変で、peak workspaceは11.5 MB減少した。
- 採用後rocprofではkernel countは不変のままmalloc/freeが各51,407回から4,031回へ`92.2%`減少した。
  V620 exact `gfx1030`のfull Gemma bounded runも収容でき、`3/17` `11.704 tok/s`、`32/32`
  `11.362 tok/s`、fallbackなし、cleanup 0だった。改善方向とownership/audit不変を満たしたため採用した。

## 2026-08-15: B3 candidate 2採用

- M=1 BF16 decode matvecのpaired weight readへnon-temporal loadを適用し、device/logical identityを
  `sllm_matmul_bf16_fp32_decode_v4` / `matmul.bf16_fp32.decode.v4`へ更新した。prefill、odd-K tail、
  wave64 path、dispatch条件は変更していない。
- candidate 1込みのGemma R9700反復3回medianは`3/17` `14.221 tok/s`、`32/32`
  `13.949 tok/s`で、candidate 1比`+0.75%/+1.49%`、fresh baseline比`+3.07%/+3.89%`だった。
  V620もcandidate 1比`+0.55%/+0.31%`で退行せず、token digest、dispatch、fallback、cleanupは不変だった。
- Qwen3.5-2B R9700 measured 10回medianはdecode `66.490 tok/s`、TTFT `33.144 ms`、E2E
  `277.675 ms`で、fresh baseline比decode `+1.62%`、E2E `-1.40%`、TTFTは実質不変だった。
  V620 short-oddもdecode `54.942 tok/s`、TTFT `159.628 ms`、E2E `455.151 ms`、fallbackなし、
  cleanup 0で完走した。
- M=1のK=`1/3/255/256/257/2560`とM/K/N非整列を含む17 operationをR9700 `gfx1201`と
  V620 `gfx1030`のpublic Rust/C/HIP pathで実行し、両targetとも17/17 numerical match、exact v4 symbol、
  fallbackなし、cleanup 0だった。両modelと両GPUの採用条件を満たすためcandidate 2を採用した。

## 2026-08-15: B4 integration reviewとcloseout

- 1回のintegration reviewで、prepared cache key、workspace capacity、KV/attention transient境界、decode v4の
  odd-K/wave64/prefill非変更、manifest source inventoryを確認した。correctness/security blockerはなく、変更findingだった
  H3 symbolとimmutable hash連鎖をpublic runtime/rmsnorm H3 matrixへ同期した。
- focused evidenceはcore 129 unit+20 integration、core/hip clippy、C++ format/static、JSON/schema/workflow、
  Rust dependency closure、markdown link、両GPU matmul 17/17がPASSした。Phase境界のhost evidenceはH0
  `513/513`、H1 `421/421`、H2 `36/36` PASSである。
- R9700/V620のGemma full bounded、Qwen short-odd、matmul oracleはいずれもexact target、fallbackなし、cleanup 0だった。
  raw profiler/model/token列は追跡せず、repository外local artifactと本履歴のbounded summaryだけを残した。
- bridgeの二候補を採用してPhase 15開始baselineへ同期し、追加candidate探索を停止した。残差はM=1 matvecとQwen GDN、
  Gemma RMSNorm、host launch/eventであり、NVFP4または後続の独立profileで再評価する。

[対応する計画](../../../../plans/archive/2026/08/11-20/cross-model-rdna-performance-bridge.md)
