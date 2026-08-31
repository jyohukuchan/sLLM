# DeepSeek V4 Phase 57 reader record

## 境界

この記録はDeepSeek V4 Phase 57へ渡すcode表現を含まないsemantic／artifact要点である。公式Python、Transformers、
vLLM、SGLang等のsourceをcopy、adapt、portしない。llama.cppからも本Phaseでは直接reuseせず、固定sourceの概念を
独立cross-checkにだけ用いる。

## 固定source

| source | identity | 用途 |
| --- | --- | --- |
| [DeepSeek V4 Flash 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | primary config、artifact、tokenizer、generation、encoding fixture、MIT license |
| [DeepSeek V4 technical report](https://arxiv.org/abs/2606.19348) | arXiv `2606.19348` | architecture terminologyと圧縮attentionの意味 |
| local llama.cpp | `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` | GGUF naming、shape、layer scheduleの概念cross-checkだけ |

`DeepSeek-V4-Flash-0731`はpreview `DeepSeek-V4-Flash`をsupersedeする。model card、config、weightはMITで、
repositoryのlicense noticeはCopyright © 2023 DeepSeekである。

## Artifact identity

- root architectureは`DeepseekV4ForCausalLM`、`model_type=deepseek_v4`。
- 48 safetensors shard、index 5,602,871 bytes、72,317 tensor。
- index metadataのtensor payloadは166,878,536,440 bytes、shard file合計は166,886,535,336 bytes、
  repository全体は166,898,661,074 bytes。
- Hub LFS identityはremote artifact identityであり、local full-byte SHA-256検証済みという意味にはしない。
- 各shardの先頭8-byte length fieldとJSON headerだけをbounded rangeで取得した。48 prefix合計は7,998,896 bytesで、
  header catalog SHA-256は`6d90aa665f26217f4488809b1fdf87a1459702aa4ec46c8b02b44ce66bd4afcc`である。
  72,317 tensorのdtype、shape、relative offset、absolute range、shard対応とpayload合計をindexへ照合したが、weight payloadは読んでいない。
- root tensor groupは`layers.*` 67,606、`mtp.*` 4,705、その他6。main layerは0..42、DSpark stageは0..2である。
- root configの`num_nextn_predict_layers=1`だけではartifact実体を表せない。公式inference config、index、DSpark metadataは
  3 stageを示すため、loaderは単一fieldからstage数を推測しない。

## Main model topology

- hidden 4,096、vocab 129,280、43 main layer、context 1,048,576、RMSNorm epsilon `1e-6`、SiLU。
- mHCはexpansion 4、Sinkhorn 20 iteration。
- attentionは64 query head、1 KV head、head dim 512、RoPE dim 64、Q/O low-rank 1,024、O group 8。
- layer 0..1はsliding window 128の非圧縮attention。layer 2..42はCSA 4:1とHCA 128:1を交互に使う。
- CSA indexは64 head×128 dim、top-k 512。HCAはindexerを持たない。
- YaRN factor 16、original context 65,536、通常RoPE theta 10,000、圧縮経路theta 160,000。

## MoE topology

- 各main layerは256 routed expert、token当たりstable top-6、shared expert 1、expert intermediate 2,048。
- main layer 0..2はtoken IDからexpert IDを引くhash routing、layer 3以降は`sqrtsoftplus` score routingを使う。
- score routingはnormalization有効、routed scaling factor 1.5。hash routingの証拠をscore routerへ流用しない。
- shared expertとrouted expertの結合順、route tie、非finite、top-k normalizationは独立oracleで固定する。

score routerのunbiased scoreは`sqrt(softplus(logit))`である。layer 3以降のexpert選択だけにselection biasを加え、
選択後のmix weightにはbiasなしscoreを使う。指定時は選択weightを再正規化してからrouted scaleを掛ける。hash layerは
token IDから得るK個のexpert IDを直接使うが、mix weightは同じunbiased scoreから取得する。hash layerにselection biasを
持ち込まず、NextN／DSpark blockをhash routingとして扱わない。

## mHC semantic

hidden streamを`X[hidden, stream, token]`、stream数を4とする。各attention／FFN sublayerは別parameterからpre gate、
post gate、4×4 mixing logitsを作る。

1. pre gateはsigmoid後にepsilonを加え、4 streamをoperator入力1本へcollapseする。
2. post gateはsigmoidの2倍でoperator出力を各destination streamへ配る。
3. mixing logitsはsoftmax、epsilon加算、反復する行／列正規化によりmixing matrixへ変換する。
4. destination streamは、post gate付きoperator出力とmixing matrixで結合した入力streamの和になる。

embeddingは最初に4 streamへ複製し、LM head前にもlearned collapseがある。通常の1本のresidualへ縮退させない。

## 圧縮attention state

cacheはraw sliding K、CSA compressed K、HCA compressed K、Lightning Indexer compressed Kを別planeとして持つ。
completed blockだけを公開し、position `p`、ratio `r`のvisible block数は`floor((p+1)/r)`である。rollback可能なpersistent
compressor stateはF32で、公開KV dtypeと同一とは限らない。

- CSA ratio 4は単純poolingではない。前4-token windowの第1候補と現4-token windowの第2候補を合わせた8候補を
  feature-wise softmaxで圧縮する。先頭の前windowはzero KV／negative-infinity scoreのsynthetic rowを使う。
- HCA ratio 128は非重複128-token blockをfeature-wise softmaxで圧縮する。
- Lightning IndexerはCSA compressed blockだけをtop-k 512へ絞り、raw sliding windowは常に残す。HCAにindexerを使わない。
- 通常Q／KVはNoPE部とRoPE部を区別し、圧縮経路だけ専用RoPE base／scalingを使う。

GGUF `attention.compress_ratios`の許容値は0／4／128だけで、scheduleをruntimeへhard-codeしない。trunk末尾のNextN blockは
ratio 0かつhash範囲外でなければならない。targetとMTP-only artifactを別identityとして扱う。

## Quantization

- semantic activation／base dtypeはBF16。
- non-expert weightはFP8 E4M3、activationはdynamic、weight blockは128×128、scaleはUE8M0。
- routed expert weightはFP4であり、Hub analyzerのpacked `I8`表示をINT8 semantic weightと解釈しない。
- model cardのrecipe名は`FP4 + FP8 Mixed`。GGUF／runtimeはvalue plane、scale plane、logical shapeを別々に固定する。

固定llama.cpp converterはsource FP8をGGUF Q8_0へ再量子化するが、これは配布上の選択でarchitecture semanticsではない。
sLLMは一般的INT8+scaleを原則非対応としているため、その変換を採用せずsource FP8＋E8M0をlosslessに保持する。
MXFP4 expertは32 value当たりE2M1 code 16 bytes＋E8M0 scale 1 byteの17-byte blockとして扱う。

## Canonical GGUF naming cross-check

fixed llama.cppのarchitecture名は`deepseek4`である。sLLMもこのcanonical名を使い、rootのembedding／output norm／head／
head mHC、各blockのattention projection／sink／mHC、ratio別compressor／indexer、router／hash table／selection bias、
routed expert／shared expert、NextN projection／norm／optional shared headを別tensor familyとして表す。
sourceのexpert別planeはexpert軸へlosslessにstackするが、hash table、selection bias、shared expertをrouted expert blobへ混ぜない。

## DSparkとDFlash

- `DeepSeek-V4-Flash`というmodel名の`Flash`とspeculative方式`DFlash`は別概念である。
- 0731 checkpointが内蔵するのはDSparkで、block size 5、noise token 128799、target main layer 40/41/42、
  Markov rank 256、3 stageである。
- sLLM要件にあるDFlashをDSparkへ読み替えない。両者は別artifact identity、別proposal layout、別sampling／accept contractとして
  後段へ渡す。

## Tokenizer／wire encoding

- fast BPE。base vocab 128,000、added token ID 128000..129279の1,280件、最終vocab 129,280。
- BOS 0、EOS 1、padはEOS。自動BOS／EOS追加は無効、tokenizer max lengthは1,048,576。
- Jinja chat templateはない。固定repositoryの`encoding/`文書と4 input／output fixtureをwire semantic sourceとし、
  Python実装を移植せずfixtureから独立renderer／parserを作る。
- generation defaultはsampling有効、temperature 1.0、top-p 1.0、BOS 0、EOS 1。

## Phase 57の証拠範囲

- local R9700の34,208,743,424 bytesに対しweight payloadだけで約4.88倍であり、KV／workspace前に単一GPU容量を超える。
- Phase 57はidentity、typed config、catalog、checked capacity、semantic oracle、model-free／verified slice GPU、GGUF dry-runを
  foundation証拠とする。full resident、通常generation、CLI／API／WebUI production対応を主張しない。
- full-model production proofはmulti-GPU等の明示scopeか、reviewed single-device artifactが存在する後段へ残す。

[対応する計画](../plans/archive/2026/08/21-31/phase57-deepseek-v4-foundation.md)
