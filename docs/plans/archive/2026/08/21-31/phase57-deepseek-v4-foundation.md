# Phase 57: DeepSeek V4 Flash architecture foundation

> 状態: 完了
> 作成日: 2026-08-31

## 目的

計画済みのDeepSeek v4 MoE／DFlashに先立ち、現行公式DeepSeek V4 Flashのidentity、typed config、tensor catalog、
圧縮attention、mHC、MoE、混合FP4／FP8、容量判定をsLLMのcontainer-neutral contractへ追加する。
architecture文字列のaccept、CPU emulation、compile-only、model-free operatorをfull-model production対応とは扱わない。

## 固定対象

- semantic／artifact source: `deepseek-ai/DeepSeek-V4-Flash-0731` revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`、MIT license。
- official topology: hidden 4,096、43 main layer（先頭3 layerはtoken-ID hash routing）＋3-stage DSpark、
  64 attention head、1 KV head、head dim 512、256 routed expert、top-6、1 shared expert、expert intermediate 2,048、
  context 1,048,576。
- attention schedule: `compress_ratios`を正とするuncompressed／CSA 4:1／HCA 128:1の混在、sliding window 128、
  YaRN RoPE、compressed index top-k 512、Q/O low-rank projection、mHC residualを含む。
- artifact recipe: routed expertはFP4、その他の大部分はFP8。公式48 shard、72,317 tensor、advertised tensor payload
  166,878,536,440 bytesをreviewed identityへ固定する。
- `DeepSeek-V4-Flash-0731`はpreviewをsupersedeしcheckpoint内にDSparkを含む。一方、現行要件のDFlashをDSparkへ
  無断で置換しない。本Phaseでは共通speculative transactionへの接続点と両者のidentity区別だけを固定し、DFlash／DSpark
  production decodingは後段とする。
- primary operator targetはR9700 exact `gfx1201`、secondary compile／operator targetはV620 exact `gfx1030`とする。
  公式artifactは単一32 GiBへ収まらないため、full-model resident／generationを本PhaseのPASSとして主張しない。

## 受入条件

1. 完全revision、license、support file、48 shardのsize／LFS SHA-256、index、tokenizer、generation configを固定する。
   shard payload全体を取得していない段階ではHub LFS identityとlocal byte hashを同じ証拠として扱わない。
2. official configをtyped contractへ変換し、architecture、層数、圧縮schedule、attention、mHC、MoE、FP4／FP8、
   DSpark fieldのmissing／extra／範囲／積／加算overflow／相互矛盾をresident allocation前にfail-closeする。
3. 72,317 tensorのindex coverage、shard coverage、tensor family count、target／hash／next-token／DSpark区分を固定し、
   duplicate、unknown、missing、path traversal、shard size不一致を拒否する。header range readerで確認したshape／dtype／rangeだけを
   verified payload contractとして記録する。
4. mHC結合、CSA 4:1、HCA 128:1、index top-k 512、stable top-6 MoEとshared expert結合のsemantic contractを、
   外部engine codeをcopyしない独立oracleへ一致させる。token／expert／圧縮境界の前後、tie、skew、非finiteを含める。
5. 影響する既存semantic opと不足opをcontainer-neutral graphへ接続し、host testとexact GPU operator／verified sliceで
   HIP-only、fallbackなし、nonfiniteなし、cleanup 0を確認する。model-free／tiny-random証拠の範囲を明記する。
6. source identityとcanonical GGUF metadata／tensor mapping／dry-run conversion contractを追加する。full source bytesを
   取得しない限り、full GGUF作成、全tensor byte一致、full-model生成一致を主張しない。
7. model library／WebUIはproduction load可能と表示せず、reviewed architectureと必要resident bytes、単一GPU容量超過理由を
   gray表示できるようにする。CLI／APIの通常生成へ未検証backendを接続しない。
8. affected test／clippy／format、1回のintegration review、model lock、GGUF、runtime、provenance、compatibility、main plan、
   historyを実装範囲に合わせて同期する。

## 実装順序

1. official source identity、reader記録、typed config、index／shard contractを実装する。
2. mHC、compressed attention、MoE routingの独立oracleとsemantic contractを追加する。
3. container-neutral graphと既存HIP operatorを接続し、不足opを境界test付きで実装する。
4. GGUF metadata／mapping／dry-runとmodel libraryのfail-closed表示を追加する。
5. exact GPU operator／slice、workspace checks、integration review、docs同期を行う。

## 後段production条件

- full-model production対応は、reviewed artifactが対象GPU topologyへresidentできること、またはmulti-GPU／tensor parallel／
  expert parallel／partial residency等を別計画で明示的にscopeへ入れることを前提とする。
- source／canonical GGUFのfull resident生成、fixed／Unicode／code／stop、prefill／decode、連続要求、cancel／recovery、
  CLI／API／WebUI、metrics、load／unload、clean shutdownをexact GPUでHIP-only、fallbackなし、cleanup 0として確認する。
- DFlashと公式DSparkは別identity／別sampling contractとして実装し、相互の名前や証拠を流用しない。

## 非対象

- 166.9 GBの公式weight payloadを単一32 GiB GPUへ無理に収容すること。
- CPU fallback、全expert dense計算、requestごとのweight展開、未検証community quantizationのproduction採用。
- million-token actual、multimodal、multi-GPU、TP／EP、DFlash／DSpark production decoding。
- vLLM／SGLang／Transformers／公式inference sourceのcopy、adapt、port。

[対応する履歴](../../../../../history/2026/08/21-31/phase57-deepseek-v4-foundation.md)
