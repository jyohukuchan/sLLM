# Phase 61 OCP MXFP8／MXFP6 weight-activation

## 目的

OCP MX v1.0に従うMXFP8 E4M3 W8A8とMXFP6 E3M2 W6A6を、sLLMの数値型、GGUF recipe、
Qwen dense lowering、公開HIP runtimeへ追加する。

## 固定acceptance

- block sizeは32、scaleはE8M0、丸めはroundTiesToEven、overflowは最大有限値へsaturationとする。
- MXFP8はE4M3 valueを1 byte、MXFP6はE3M2 valueを4 value／3 byteで保持する。Kは32の倍数だけを受理する。
- GGUF標準type番号を発明しない。value／E8M0 scaleをI8 carrierの別tensorとし、versioned recipeで一意に結合する。
- weightは量子化済みresident、activationはBF16から実行時に同じMX形式へ動的量子化し、FP32 accumulate／BF16 RNE outputとする。
- exact `gfx1030`／`gfx1201`のdecode M=1とnon-aligned prefill M=3を独立CPU oracleへ照合する。
- E8M0 NaN scaleはblock全体へNaN伝播し、Infはelement最大有限値へsaturationする。
- 未対応target、scale欠落、K非32倍、runtime failureは別dtypeへfallbackせず拒否する。
- Qwenのexact text-linear setをGGUF recipeから同じCLI/server graphへlowerでき、形式名を監査出力へ残す。

## 対象外

- exact `gfx1201`／`gfx942`のfull-model品質・性能・memory評価とproduction default化。
- `gfx942`、generic target、`gfx1031`–`gfx1036`、`gfx1200`の実行対応。
- OCPが規定しない物理packingを標準仕様として主張すること。

## 実装状況

- [x] OCP E3M2 encode/decode、MXFP8／MXFP6 block quantizer、境界・RNE・saturation oracle。
- [x] `Encoding`、matmul semantic contract、GGUF recipeと厳密なplane長検証。
- [x] C ABI、Rust lowering、dynamic activation quantizer、decode／prefill HIP kernel。
- [x] Qwen dense GGUF load、resident plane連結、graph lowering、CLI/server選択・監査。
- [x] exact `gfx1030`／`gfx1201`の6 caseずつ（M=1／3／17）をHIP-only、fallback 0、cleanup 0でPASS。
- [x] exact `gfx1030`でQwen3.5-4BのBF16／MXFP8／MXFP6 artifactを実行し、20 logit行、perplexity、VRAM、短いprefill／decode速度を比較。
- [x] workspace全体の最終check/testと文書整合確認。

## 実モデルfollow-up

- 固定Qwen3.5-4B BF16 GGUFからMXFP8 E4M3 W8A8とMXFP6 E3M2 W6A6 GGUFを実変換し、exact V620 `gfx1030`で
  BF16 referenceとcandidateを完全直列に常駐させた。KVは明示FP16とし、weight／dynamic activation以外の差を除いた。
- 10 caseのprefill／teacher-forced decode、合計20 rowでは、MXFP8はtop-1 `0.80`、KLD mean／p99
  `0.01490／0.06963`、perplexity相対差`-0.18%`、MXFP6はtop-1 `0.75`、KLD `0.03713／0.14640`、
  perplexity相対差`+2.97%`だった。両形式とも実行はPASSしたが、BF16同等品質とは判定しない。
- model residentはBF16 `8.412 GB`に対しMXFP8 `4.954 GB`（`-41.10%`）、MXFP6 `4.062 GB`（`-51.71%`）。
  17 input／4 output、1 warmup＋3 measuredの中央値はBF16 prefill／decode `284.03／45.68 tok/s`、
  MXFP8 `48.10／20.17 tok/s`、MXFP6 `100.16／20.06 tok/s`だった。現行software encode/decode correctness providerは
  BF16より遅く、VRAM削減と速度のtrade-offが成立していないためdefault候補にしない。

## Evidence境界

両RDNAのmodel-free operator correctnessに加え、full-model claimはexact `gfx1030`の固定Qwen3.5-4B／固定artifact／短caseだけに限定する。
`gfx1201`／`gfx942`、長context、production default、別modelへ一般化しない。R9700のoperator testは`HIP_VISIBLE_DEVICES=2`で
単独可視化し、論理device 0を使った。最新operator report SHA-256はgfx1030
`91a04761b600fee47efcd18ccbd23cc09424be3def7c5d9241e42ed7fb233bea`、gfx1201
`2d701aaf4e8b2e8b410a948514fc9ed64481479458c353995874b46c733dad10`である。release/push時に必要なimmutable candidate identityは、
このdraft実装の完了条件には含めない。

履歴: [Phase 61 history](../../../../../history/2026/08/21-31/phase61-ocp-mxfp-weight-activation.md)
