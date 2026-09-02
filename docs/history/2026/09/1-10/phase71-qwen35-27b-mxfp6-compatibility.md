# Phase 71: Qwen3.5-27B MXFP6 compatibility and bounded-VRAM benchmark

## 結論

2026-09-02にQwen3.5-27Bをreviewed modelへ追加し、Phase 70のmodel非依存MXFP6 E3M2 W6A6経路で
canonical V620 exact `gfx1030`とRadeon AI PRO R9700 exact `gfx1201`の実modelを実行した。
512入力は両GPUで3 warmup＋10 measured、2,048入力はchunkを1,024へ抑えて1 warmup＋3 measuredを完走した。
全PASS行はHIP-only、fallbackなし、正常cleanupで、32 GiB classの単一GPUへ収まった。

27B専用matmul kernelやmodel名selectorは追加していない。互換性対応は24 query heads／4 KV heads／GQA比6、
linear-attention value heads 48、hidden 5,120、intermediate 17,408を既存汎用経路へ加えたものである。
2,048-token単一chunkはgfx1201でworkspace OOMとなったためPASSに数えず、確認済みの安全条件はchunk 1,024とする。

## Sourceとartifact

- source: `Qwen/Qwen3.5-27B`
- immutable revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`
- model lock: 23 file、fingerprint
  `sha256:a4a0a6192babfdb7b1fc3ac75cc340e96df87fe2b0e629cc1510085bfeced97f`
- reviewed inventory: 1,199 indexed tensor、851 loadable text weight、348 known-unconsumed tensor。
- reviewed shape: hidden 5,120、intermediate 17,408、64 layers、q24／kv4／head dim 256、linear qk16／value48／
  head dim 128、linear attention 48 layers、full attention 16 layers、untied embeddings。
- load plan digest: `sha256:6dad09edb6866241f32ac52f07a7e1cc494111177fa3d30645eb6187d5d7f05f`。
- derived MXFP6 GGUF: 25,909,762,816 byte、1,695 tensor、SHA-256
  `3b7151e5c601f3efee524e4998e403b800699fbf6e9097918f983e3c72876ddd`。
- derived lock fingerprint:
  `sha256:d1142468252af487d52ebf72a29a4bb62487a635c174e709bebd73b0c337a82c`。
- metadata／tensor catalog SHA-256:
  `1bdf89db9b872d5d517323012eaa88e77bd9de6897a1d2d2b88b0b84fa010522` /
  `8df1dd979d8c729b7d01afc7a3fcbc43fd46272c8facd1c8ef0617b72373e355`。

変換bundleはrepository外のdraft local artifactである。converter metadataは開始時HEAD
`b1bd42d5d3b4349054c35511f386546e2316b971`を記録し、run manifestは実行binary SHA-256
`a00e2385fac627db4726baf7cb1a48b30bb5b53d97754ee7f00e60c63a2b477c`へ結合するが、dirty working sourceからの
draft evidenceでありrelease provenanceや再配布可能性を主張しない。

## 実装

- reviewed spec、builtin lock、CLI direct benchmarkへ`Qwen/Qwen3.5-27B`／`--model-size 27B`を追加した。
- Qwenのsigmoid-mul、attention preprocess、causal attention、runtime bindingで24 headsとGQA比6を受理した。
  20／28 heads等の未reviewed GQA構成は引き続き拒否する。
- linear-attention stateとdescriptorでvalue heads 48を受理し、conv `[3,10240]`、recurrent
  `[48,128,128]`、output `[M,6144]`をhost/Rust testへ固定した。
- 公開RMSNorm／embedding／matmulの汎用上限を5,120／5,120／17,408へ更新した。kernel本体は既にgrid／loopで
  これらの値を処理可能であり、古いreviewed-shape gateだけを広げた。
- exact gfx1201のGQA4、N64／N128 optimized selectorやexact gfx1030の既存scoped selectorは広げていない。
  27Bの未一致shapeは既存baseline／generic providerへfail closedに戻る。

## 検証

- 11 BF16 shardを含む23 fileのsize／SHA-256とlock fingerprintを完全検証した。
- `model-lock-v1`のgeneration config有無とpathの整合、presentなpathのlocked file集合への所属をschema／validator／host testで検証した。
- external-cache testは2B／9B／27Bを通し、required load planと1／3／17／255／256／257 token graphをPASSした。
- converter dry-run、最終conversion、`verify-model`をPASSし、53,792,013,824 destination byteと851 loadable entryを確認した。
- Rustのreviewed family、27B CLI identity、sigmoid head境界、GQA比境界、linear-attention layout、HIP wrapper testをPASSした。
- native public runtime host testは24-head sigmoidと48-value-head linear stateを含めPASSした。
- target別release binaryはROCm 7.14.0、HIP 7.14.60850、AMD clang 23、Code Object V6、wave32で生成した。
  gfx1030／gfx1201 CLI SHA-256は
  `3df29f1af4feab36c23a79f4bb040947c2edbab0400813b9dcb046fd0e108732` /
  `957aa9ae037dfd60701a364e1faeca58d040bd87f295a52a33ba21a306a9e13b`である。

## ベンチマーク

共通条件はdirect pretokenized lane、token ID `23066`の反復、MXFP6 E3M2 W6A6、FP32 accumulation、BF16 output、
明示FP16 KV、最大4 output、greedy、ignore EOS、single requestである。512入力はchunk 512／3+10、2,048入力は
chunk 1,024／1+3とした。

| target | input / chunk | warmup + measured | prefill中央値 | MAD | resident | allocator peak | 結果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V620 exact `gfx1030` | 512 / 512 | 3 + 10 | 34.298907 tok/s | 0.157267 | 24,115,002,880 B | 24,777,018,880 B | PASS |
| R9700 exact `gfx1201` | 512 / 512 | 3 + 10 | 81.746517 tok/s | 0.065546 | 24,115,002,880 B | 24,777,018,880 B | PASS |
| V620 exact `gfx1030` | 2,048 / 1,024 | 1 + 3 | 33.448016 tok/s | 0.148480 | 24,115,002,880 B | 25,351,937,536 B | PASS |
| R9700 exact `gfx1201` | 2,048 / 1,024 | 1 + 3 | 77.409011 tok/s | 0.018784 | 24,115,002,880 B | 25,351,937,536 B | PASS |

512行は各13 requestでsubmission 48,464、kernel dispatch 78,000、segment／boundary 884、2,048行は各4 requestで
submission 18,632、kernel dispatch 30,280、segment／boundary 340だった。全測定sampleの生成tokenは
`[23066,23066,23066,23066]`で一致し、model loadは各process一回、request内再load 0、fallback 0、
retryable cleanup 0、durable quarantine 0だった。

gfx1030の2,048行を実行中に取得した外部VRAM snapshotは`26,990,432,256 / 34,342,961,152` byteで、
process終了後はbaselineへ復帰した。allocator peakはprovider外／driver側allocationを完全には表さないため、両値を区別する。

## OOM境界と解釈

gfx1201の2,048入力／chunk 2,048はplacement required `28,010,885,427` byteとして選択されたが、layer 56
`mlp_down_matmul`のprepared workspace `hipMalloc`がstatus 260 OOMとなった。workspace arenaは1,897,021,440 byte、
diagnostic上のseparate-allocation集計は28,851,033,344 byteだった。この失敗はGPU PASSに数えず、同じ入力をchunk 1,024へ
下げた行だけを収容可能な証拠とする。placementと一時workspaceの差は後続resource estimator改善候補だが、Phase 71の
互換性受入条件や27B supportを妨げない。

27Bは4B/9Bよりhidden、intermediate、layer数が大きく、intermediate `N=17408`はPhase 70のgfx1201既定上限16,384外である。
したがって今回の速度をparameter比だけで説明したり、27B専用最適化の成果と解釈したりしない。既存共通MXFP6経路で
機能・capacityを実証した値であり、27B wide MLPのprovider拡張は別の最適化判断単位とする。

[保存済み計画](../../../../plans/archive/2026/09/1-10/phase71-qwen35-27b-mxfp6-compatibility.md) /
[全体計画](../../../../plans/main-plan.md) /
[追跡要約](../../../../../ci/matrix/phase71-qwen35-27b-mxfp6-compatibility-v1.json)
