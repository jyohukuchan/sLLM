# Phase 58: MiniMax M3 architecture foundation

> 状態: 完了（foundation、2026-08-31）
> 作成日: 2026-08-31

## 目的

計画済みのMiniMax M3について、公式identity、typed config、tensor catalog、Multi-head Sparse Attention（MSA）、
MoE、MTP、multimodal metadata、容量判定をsLLMのcontainer-neutral contractへ追加する。architecture文字列のaccept、
CPU oracle、compile-only、model-free operatorをfull-model production対応とは扱わない。

## 固定対象

- semantic／artifact source: `MiniMaxAI/MiniMax-M3` revision
  `f0e1c1e04d40177e4673a22097036854f536e9c0`、MiniMax Community License。
- official topology: hidden 6,144、60 text layer、64 query head、4 KV head、head dim 128、先頭3 layerはdense、
  layer 3..59は128 routed expert／top-4／1 shared expert、7 MTP module、context 1,048,576。
- MSA: layer 3..59、block size 128、top-16 block、4 index head、index dim 128、current local block強制包含。
  選択はGQA groupごとに独立し、main branchは選択block内のcausal tokenへexact softmaxを適用する。
- vision topologyとspecial tokenはidentity／metadataとして固定するが、multimodal executionは本Phaseでproduction接続しない。
- official BF16 repositoryは59 safetensors shard、23,416 tensor、shard file合計854,176,398,808 bytesである。
  index `metadata.total_size=869,157,697,024`はshard file合計より14,981,298,216 bytes大きく整合しないため、
  silent normalizationを禁止し、admissionは大きい方へfail-closeする。
- primary operator targetはR9700 exact `gfx1201`、secondary compile／operator targetはV620 exact `gfx1030`とする。
  公式BF16／MXFP8／MXFP4／NVFP4はいずれも現行local GPU topologyへ収まらないため、full-model resident／generationを
  本PhaseのPASSとして主張しない。

## 受入条件

1. 完全revision、license、support file、59 shardのsize／LFS SHA-256、index、tokenizer、generation／processor設定を固定する。
   shard payload全体を取得していない段階ではHub LFS identityとlocal byte hashを同じ証拠として扱わない。
2. official configをtyped contractへ変換し、architecture、text／vision、dense／MoE schedule、MSA、RoPE、MTP、special tokenの
   missing／extra／範囲／積／加算overflow／相互矛盾をresident allocation前にfail-closeする。
3. 23,416 tensorのindex coverage、shard coverage、tensor family countを固定し、duplicate、unknown、missing、path traversal、
   shard size不一致を拒否する。公式indexの`total_size`不整合を明示的な状態として報告し、容量不足判定を弱めない。
4. dense layer 0..2とMSA layer 3..59、block 128、stable top-16、current block強制包含、per-group selection、partial current block、
   causal exact attentionを独立FP32 oracleへ固定する。127／128／129境界、tie、skew、非aligned値、非finite、shape、overflowを含める。
5. sigmoid score、selection bias、stable top-4、shared expert、routed scalingを公式実装と照合し、container-neutral MoE oracleへ固定する。
   MTP 7 moduleはidentity／graph contractまでとし、full speculative generationを証拠なしに接続しない。
6. 影響する既存semantic opと不足opをcontainer-neutral graphへ接続し、host testとexact GPU operator／verified sliceで
   HIP-only、fallbackなし、nonfiniteなし、cleanup 0を確認する。model-free／tiny-random証拠の範囲を明記する。
7. source identityとcanonical GGUF metadata／tensor mapping／dry-run conversion contractを追加する。full source bytesを
   取得しない限り、full GGUF作成、全tensor byte一致、full-model生成一致を主張しない。
8. model library／WebUIはreviewed architecture、license、必要resident bytes、manifest不整合、capacity／production loader未対応を
   gray表示できるようにする。CLI／APIの通常生成へ未検証backendを接続しない。
9. affected test／clippy／format、1回のintegration review、model lock、GGUF、runtime、provenance、compatibility、main plan、
   historyを実装範囲に合わせて同期する。

## 実装順序

1. official source identity、reader記録、typed config、index／shard contractを実装する。
2. MSAとMoE routingの独立oracle／semantic contractを追加する。
3. container-neutral graphと既存HIP operatorを接続し、不足opを境界test付きで実装する。
4. GGUF metadata／mapping／dry-runとmodel libraryのfail-closed表示を追加する。
5. exact GPU operator／slice、workspace checks、integration review、docs同期を行う。

## 後段production条件

- full-model production対応は、reviewed artifactが対象GPU topologyへresidentできること、またはmulti-GPU／tensor parallel／
  expert parallel／partial residency等を別計画で明示的にscopeへ入れることを前提とする。
- source／canonical GGUFのfull resident生成、fixed／Unicode／code／stop、prefill／decode、連続要求、cancel／recovery、
  CLI／API／WebUI、metrics、load／unload、clean shutdownをexact GPUでHIP-only、fallbackなし、cleanup 0として確認する。
- Community Licenseの表示／配布条件をmodel artifactへ保持し、engine本体のMIT licenseと混同しない。

## 非対象

- 854 GB超の公式weight payloadを単一32 GiB GPUへ無理に収容すること。
- CPU fallback、全expert dense計算、requestごとのweight展開、未検証community quantizationのproduction採用。
- million-token actual、multimodal、full MTP generation、multi-GPU、TP／EP。
- Transformers／vLLM／SGLang／公式CUDA MSA sourceのcopy、adapt、port。

[対応する履歴](../../../../../history/2026/08/21-31/phase58-minimax-m3-foundation.md)
