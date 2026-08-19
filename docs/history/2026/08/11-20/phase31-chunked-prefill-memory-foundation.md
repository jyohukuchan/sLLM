# Phase 31: chunked prefill・workspace memory基盤 実装履歴

## 結果

Phase 30で10k+ inputがrequired 53,758,880,592 byte、available 34,135,343,104 byteとして実行前に拒否された原因を、
KV cacheそのものではなくprompt全行へ比例するrequest-owned intermediateの個別allocation合計へ局所化した。
Qwen text prefillへresource-based automatic chunk selectorとcompletion-boundary liveness slotを実装し、
10,001-token workspace high-waterを5,278,049,280 byteまで縮小した。canonical V620/R9700の双方で10k+ full-modelを
HIP-only実行し、low-bit KVの通常経路を長contextで直接選択・検証できる状態へ進めた。

## 実装

- backend sessionからdevice total/available memoryを別々に取得し、model resident、全request終了時のKV/GDN state、
  candidate workspace、`max(totalの5%, 1 GiB)` reserveを分離したpreflightを追加した。
- total VRAM 16 GiB以下は512、16 GiB超は16K/8K/4K/2K/512の順にdispatch前にfitを選ぶ。promptがbucketより短い場合は
  actual行数を使い、allocation失敗後のsilent retryは行わない。
- graph tensorのfirst/last useを実際のsubmission completion segmentまで延長し、重ならないlifetimeだけを再利用するslot plannerを
  追加した。HIP wrapperがbuffer handle単位でin-flight leaseを持つため、一つのbufferのoffset aliasではなく複数の再利用slotを使う。
- promptを連続chunkへ分割し、absolute position、full-attention KV、GDN recurrent stateを継続した。中間chunkではLM head、
  Argmax、visible outputを省略し、queue terminal fence後にだけ次chunkへ進む。最終chunkのhidden stateはMTP target向けに維持する。
- Qwen BF16 weight graphとKV encodingを分離し、CLI/serverへ`--kv-cache-encoding fp16|fp8|fp8-static|nvfp4`を追加した。
  省略時はFP16。low-bitと未検証のMTP/multimodal/MoEの組合せはfail-closedする。
- auditへselected chunk/count、total/available/required、model/state/reserve、arena/個別allocation合計、KV encodingを追加した。
- resident作成前のCLIはfull layout requiredを、model作成済みのserver/request ownerはmodel bytesを除くincremental requiredを
  current availableへ比較し、常駐modelの二重計上で長contextを不必要に拒否しないようにした。

## 実装中に見つかった問題

1. arenaを一つのphysical buffer handleへした初期案は、non-overlap offsetでもHIPのbuffer-level busy ownershipと衝突した。
   再利用単位を独立slot bufferへ変更した。
2. nodeのlast consumer直後にslotを再利用すると、非同期kernelがcompletion前に同じbufferを保持してBusyとなった。
   lifetime終端をstate-publicationまたはterminal completion boundaryまで丸めた。
3. 中間chunkからArgmaxを省略するとterminal waitがなく、`ExecutionSegment::flush`がpending submissionをqueryした。
   同一ordered queueの明示terminal fenceを中間chunk末尾へ追加した。
4. generic static FP8 descriptorのscaleが0でHIP createに拒否された。明示static FP8設定だけK/V scale 1.0を設定し、
   default FP16とdynamic FP8 recipeは変更しなかった。
5. Phase 20の旧`phase20-final-*` artifactはrank-5 tensorを含みcurrent loaderが拒否したため、canonical
   `phase20-audit-qwen35-bf16.gguf`と対応lockへidentityを固定した。

## 実機証跡

使用artifactはGGUF SHA-256 `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、
lock SHA-256 `425151d06832347a01b946b27336ceffac074eb7f6932af61e8c9821edc1e318`、ROCm 7.14.0である。
R9700はUUID `GPU-a8e9ddefa2d60f55`、V620は`GPU-08b2ddcbd6e6b36c`を一台だけ可視化し、論理device 0で実行した。

| target | input | KV | chunk | required / available byte | token | 結果 |
| --- | ---: | --- | --- | ---: | --- | --- |
| gfx1201 | 10,001 | FP16 | 10,001 × 1 | 15,779,335,475 / 34,135,343,104 | 1228 | HIP-only PASS |
| gfx1030 | 10,001 | FP16 | 10,001 × 1 | 15,786,046,361 / 34,311,503,872 | 1228 | HIP-only PASS |
| gfx1201 | 16,385 | FP16 | 16,384 + 1 | 19,357,165,875 / 34,135,343,104 | 1228 | HIP-only PASS |
| gfx1201 | 10,001 | dynamic FP8 | 10,001 × 1 | 15,618,039,859 / 34,135,343,104 | 1228, 1228 | decode 1 PASS |
| gfx1030 | 10,001 | dynamic FP8 | 10,001 × 1 | 15,624,750,745 / 34,311,503,872 | 1228, 1228 | decode 1 PASS |
| gfx1201 | 16,385 | dynamic FP8 | 16,384 + 1 | 19,092,909,107 / 34,135,343,104 | 1228, 1228 | decode 1 PASS |
| gfx1201 | 10,001 | static FP8 | 10,001 × 1 | 15,618,039,859 / 34,135,343,104 | 1228, 1228 | decode 1 PASS |
| gfx1201 | 513 | NVFP4 | 513 × 1 | 10,449,157,171 / 34,135,343,104 | 1228, 1228 | spot PASS |

全rowでfallback=false、cleanup failure=0だった。10,001-token request stateはFP16 379,256,832 byteからdynamic FP8
217,961,216 byteへ42.53%減った。この値にはencoding非依存のGDN stateを含むため、KV planeだけの圧縮率ではない。

gfx1201 serverはdynamic FP8 KVで10,013-token chat promptを処理し、non-stream/SSEとも1 token `It`と
terminal usage/`[DONE]`を返した。shutdown auditはrequest/workspace current bytes 0、cleanup failure 0だった。

## 判断と制限

chunked prefill/arenaはN0 scheduling/layout変更として採用した。10,001 tokenはarena化だけで一括収容できるためauto selectorが
無用な分割をせず、実際のmulti-chunkは16,385 tokenで確認した。記録したwall timeは各一回のcapability runであり性能claimにしない。
default KVはFP16のままで、NVFP4は短いspot、static FP8のscale 1.0は明示実験設定である。Paged Attention、native append encode、
continuous batching、prefix sharing、low-bit default昇格は後続判断へ残した。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
[bounded summary](../../../../../ci/matrix/phase31-chunked-prefill-summary-v1.json)
[メイン計画](../../../../plans/main-plan.md)
