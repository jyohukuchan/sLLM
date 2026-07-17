# SQ8 P8-B variable-length serving session

日付: 2026-07-10

## 前回の要点

P8-Aで製品artifact/package、serving/worker/OpenAI契約、raw prompt fixture、実vLLM oracle 6 prompts/21 runsを固定した。P8-BではHTTPやrequest batchingより先に、R9700上で使い回せるactive1/waiting0の可変長sessionを成立させる。

## 今回の変更点

- 4096 context、block size 16、256 blocks/layer、40 layersのF32 paged KVを起動時に確保する`Qwen3Sq8ServingSession`を追加した。KVは1層33,554,432 bytes、合計1,342,177,280 bytesである。
- raw token promptをposition 0からM=1で連続実行し、最後prompt tokenのみheadを実行し、その後は生成tokenをM=1 decodeへfeedbackする。
- P7の`run_paged_decode_optimized_synchronized`は`OutputReady`前提を維持した。P8は専用active flagとposition cursorを持つ別入口に分離した。
- `start`はrequest、resident baseline、HIP guards、workspace、KVをscheduler mutation前に検証する。設定不備は`InvalidConfiguration`でReadyを維持し、active中の不正操作とruntime invariant破れは`Failed`へ移行する。
- serving headをP7監査用の4 readback経路から分け、RMSNorm、BF16 matvec、logits 1 readback、finite/hash検証だけのlean経路にした。
- additiveな`RuntimeBuffer::zero`を追加し、CPUは`memset`、HIPは`hipMemsetAsync`を使う。scheduler release後、KV、stack/decode workspace、resident hidden、embedding、headを同一streamにzero enqueueし、1回のsynchronize成功後だけmetadataをbaselineへcommitする。
- 生成tokenは最終Acquire cancel checkより後にだけscheduler/active countersへcommitする。実際のworker stdout flushとcancel storeの厳密な線形化は、event writerを持つP8-Cで固定する。

## 検証

- `cargo test -p ullm-engine --lib`: 249 passed。
- `cargo test -p ullm-runtime-sys --lib`: 141 passed。
- gfx1201 feature付きserving example check/release build: passed。
- R9700同一model load内でprompt `[1]`, G=8は`[25,330,16,13,15,13,15,756]`、prompt `[1..8]`, G=8は`[353,10,4999,1725,15,16,17,18]`となり、checked-in vLLM oracleと両方8/8一致した。
- 上記2 requestの各resetは約3.2 ms。その後のdecode中cancelは追加tokenを返さず、cancel reset後もallocator、40層cache、sessionがReady baselineへ戻った。
- P7 M=8/G=8を現HEADで再実行し、token 8/8一致、final KV length 15、2240 projections、1280 activation quantizationsを`validate-sq8-generation-result.py --contract-only`が再計算して合格した。
- 独立レビュでP7/P8入口分離、single-sync reset、lean headにcommit blockerなし。
- 通常servingのlogits-only readbackを変えず、明示指定時だけfinal hidden/logitsを各1回readbackするoracle診断入口を追加した。validatorはproducerの判定を信用せず、raw payload hash、全F32値、relL2、cosine、top-10、top-1を再計算する。
- validatorはsource oracleの`metadata.json`、`payload-manifest.json`、`SHA256SUMS`を固定hashへ結び付け、各promptで使う3 payloadのsize/hashを再検証する。captureはresult/manifest親内かつsource外に限定し、source別名およびP7/P8同一inodeを拒否する。記録済みtop-1 token/logitもraw logitsからの再計算値へ照合する。
- R9700でprompt 1/8/32/128のfinal hidden/logitsをvLLM source oracleへ照合した。最悪relL2は0.181061、最低cosineは0.994686、top-1は4/4一致、top-10 overlapは最低9で、P8-Aの凍結gateを全件通過した。
- prompt 8についてP7 M=8とP8 sequential M=1をraw tensorで直接比較し、hidden relL2 0.036843 / cosine 0.999352、logits relL2 0.032693 / cosine 0.999498、top-1一致、top-10 overlap 10で、より厳しい直接比較gateを通過した。
- prompt 32/128、G=64はいずれもR9700で完走してresetできた。source greedy列とはそれぞれ生成37 token目、54 token目前後から分岐したが、同一contextのfinal prompt hidden/logit gateは通る。SQ8誤差をgreedy feedbackが増幅した挙動と考えられるが、後段decodeの不具合を完全には除外していない。
- P8-Bのsource oracle acceptanceは凍結済みfinal prompt hidden/logitsの数値gateであり、G=64は完走/cache/reset gateである。長生成のsource逐語一致を合格条件として扱わず、P7 prompt 8/G=8の固定token列一致は別途維持する。
- serving exampleを任意個のascending raw prompt長へ広げ、各requestのterminal sequence、全40層cache長、last position/block、scheduler/allocator、reset後40層zeroを記録する。result JSONはcreate-new + fsyncで保存する。
- R9700同一loadでprompt 15/16/17と255/256/257、G=1を連続実行した。全40層cache長は各prompt長と一致し、last position/blockは14/0、15/0、16/1および254/15、255/15、256/16となった。各resetは約3.17msである。
- 上記sessionを再利用し、prompt 15の8 token処理時点でprefill cancelした。cancel前は40層cache長8、generated 0、active 1で、次のadvanceはprogress/tokenを返さずCancellationObservedとなり、約3.16msでReady baselineへ戻った。
- prefill/decode cancelの双方で、CancellationObserved直後にreset前snapshotを取得する。観測前後のactive ID、prompt/generated counter、全40層cache、scheduler、allocator statsが完全一致しなければproducer自体を失敗させ、両snapshotをJSONにも保存する。
- decode cancel専用runではprompt 8/G=8固定列`[353,10,4999,1725,15,16,17,18]`を完走・reset後、別requestのfirst token 353でcancelした。観測前後はgenerated 1、全40層cache長8、active 1のまま不変で、約3.18msでresetした。
- prompt 1/8/32/128を同一loadでG=8実行し、全4 requestがvLLM source列と8/8一致、合計32/32 token一致した。全40層cache長は`prompt+7`、各resetは約3.2msである。
- 同じ4 promptをG=64実行し、全件64 token完走、全40層cache長`prompt+63`、reset baselineを確認した。source列との共通prefixはprompt 1/8/32/128で64/28/36/52 tokenである。G=64はSQ8 feedback増幅を含む完走/cache gateとして扱い、first tokenのsource一致は全件必須にした。
- generation matrix validatorは固定source oracleの管理hashとG8/G64 payload hashへbindし、G=8全列完全一致、G=64全件完走/first-token一致、cache/position/reset、入力evidence hashを再計算する。4改ざんtestsを含む関連Python 153 testsが合格した。
- prompt 4095 / G=1はOOMなしで完走した。reserved/terminal sequenceは4096、全40層cache長4095、last position 4094、last block 255、reset約3.25msである。逐次M=1 prefillは369.55秒であり、correctness oracleとしては成立するが製品TTFTにはP8-B2のM=8 chunk化が必須である。
- 独立matrix validatorはrunnerの`passed`を信用せず、device/artifact/package、ascending prompt、全cache vector、off-by-one位置/block、prefill/decode cancel前後、exact context、reset baseline、3入力evidence hashを再計算する。合成改ざん7 tests、関連Python 149 tests、engine 251 testsが合格した。

## 次の行動

1. EOS first-output/decode中EOSとmax-tokenのterminal判定を、実行state遷移から分離した純粋なcommit contract testで固定する。
2. context 4097、invalid token、active中の不正startがmutation前に拒否される既存testをP8-B acceptance表へ結び付け、evidence/checksumを閉じる。
3. P8-B完了後、P8-B2で同一request内の固定M=8 prefill chunkを追加し、4095 tokenのhard TTFT gateを満たす。

## P8-B2 M=8 cached-prefix chunk着手

### 前回の要点

P8-Bのall-M1 sessionは正しさと4096 contextを満たしたが、prompt 4095 / G=1に369.55秒かかり、製品TTFTには単一request内のprefill chunk化が必要だった。

### 今回の変更点

- `8b859d7`でidentity block table上のcached-prefix FlashAttentionと、schedulerの幅単位の原子的prefill commitを追加した。
- `80ef86c`でP7の`Prefill/Decode` reportを変更せず、serving専用M=8 chunk reportと40層stack経路を追加した。
- `4b06865`でM=8 stackとM=1 decode workspaceを同居させ、8行embedding buffer、row 7のlean head、M=1 tail/decodeをsessionに統合した。weight/KVの複製やrequest batchingは追加していない。
- `e76c331`でall-M1 v2を維持しつつ、M=8時だけv3 schemaと各prefill unitの幅・40層cacheを出すrunnerを追加した。
- `ce0cb8a`でraw captureをchunk/all-M1/sourceへ直接比較する独立validatorと、cache改ざん・inode alias・payload改ざんの拒否testを追加した。
- engine library testは258件合格、validator改ざんtestは4件合格。`clippy -D warnings`は既存の`ullm-runtime-sys/build.rs:86`のneedless borrow 1件で停止した。
- R9700初期smokeでprompt 8/9/16/17をそれぞれ1/2/2/3 prefill callで完走し、first tokenは`353/10/17/18`でall-M1と全一致した。全unitの40層cache長とreset baselineも合格した。
- chunk対all-M1の最悪relL2は0.047343390、最低cosineは0.998961757、top-10 overlapは最低9。prompt 8のchunk対vLLMはhidden relL2 0.046074 / cosine 0.998981、logits relL2 0.042204 / cosine 0.999110、top-1一致、top-10 overlap 9だった。

### 次の行動

1. prompt 32/128/512のchunk/all-M1/source oracleを同じvalidatorで閉じる。
2. prompt 4095のoracleと3584+512 deep boundaryを実行する。
3. warmup 2 / repeat 5のTTFTとprompt 32 / G=64 decode性能runnerを固定し、hard gateを判定する。

## P8-B2 prompt 32-4095 correctness oracle完了

### 前回の要点

M=8 cached-prefix chunkをsessionへ統合し、prompt 8/9/16/17でall-M1およびvLLM sourceとの初期比較を通した。P8-B2 acceptanceには長いpromptでの同じ比較と、4096境界・性能gateが残っていた。

### 今回の変更点

- clean build `28cd88e`でprompt 32/128/512をchunk、all-M1、vLLM sourceへ比較した。chunk対all-M1の最悪relL2は`0.055494862`、最低cosineは`0.998492050`で、top-1は全件一致した。
- clean build `55562d9`でprompt 4095 / G=1を実行した。M=8経路は511個のM=8 chunkと7個のM=1 tail、合計518 prefill callで78.043268秒、all-M1は4095 callで369.181784秒だった。
- 両経路はtoken 291を生成し、全40層cache長4095、最終position 4094、logical block 255へ到達し、約3.6msのreset後にReady baselineへ戻った。
- 独立validatorのprompt 4095比較は、chunk対all-M1でhidden relL2 `0.011411250` / cosine `0.999950021`、logits relL2 `0.008940925` / cosine `0.999987181`だった。
- chunk対vLLM sourceはhidden relL2 `0.019835477` / cosine `0.999888264`、logits relL2 `0.020959889` / cosine `0.999974552`だった。両比較ともtop-1一致、top-10 overlap 10である。
- producerのcommit、clean worktree、binary SHAを一致必須にし、repoへコピーした相対capture pathから再検証した結果も合格した。

### 次の行動

1. 通常requestへ影響しない明示的なtest-only `ignore_eos`を追加し、prompt 3584 + 512生成で全decode位置と最終KV長4095を検証する。
2. resident model、warmup 2 / repeat 5、各sample完全resetのTTFT runnerを実装し、32/128/512/2048/3584のhard gateを判定する。
3. prompt 32 / G=64でdecode p50とp95 inter-token latencyを測定し、P8-B2を閉じる。

## P8-B2 4096-token deep boundary完了

### 前回の要点

M=8 chunkのprompt 8/32/128/512/4095 correctness oracleは完了した。P8-B2の正しさに残っていたのは、EOSで短縮せずprompt 3584 + generation 512を実際に完走する4096-token境界だった。

### 今回の変更点

- `9ef7fce`で通常の`greedy`とは分離したtest-only ignore-EOS constructorと、固定3584+512以外を拒否するdeep-boundary runnerを追加した。通常requestのEOS停止contractは維持した。
- runnerはgenerated index 0..511ごとにtoken、cache length/write position、全40層cache、status、scheduler active/waiting、allocator、terminal reasonを記録する。GPU tensorを保持せず、CPU上の小さいtraceだけを保持する。
- `5084396`で独立validatorと14改ざんtestsを追加した。producerの`passed`を使わず、448個のM=8 prefill unit、447 progress、511 decodeを含む959 call、全512 step、terminal/reset、外部commit/binary anchorを再計算する。
- clean build `5084396` / binary `58d1af40...f6f13`をR9700単独で実行した。model loadは24.088511秒、resident requestは136.763141秒、resetは3.174msだった。
- 3584 promptと512 actual generated tokenで総sequence 4096へ到達した。最終KV長4095、write position 4094、logical block 255、全40層一致、scheduler active1/waiting0である。
- reset後はReady、active0/waiting0、allocator0、全40層cache zeroへ戻った。実行前後ともR9700を使う他のKFD/AMD SMI processはなかった。
- repo内へコピーした1,091,101-byte resultを同じ独立validatorへ再入力し、build identityを含む全条件が合格した。

### 次の行動

1. timerを`session.start`直前からfirst host tokenまでとするresident TTFT runnerを実装する。
2. prompt 32/128/512/2048/3584をwarmup 2 / measured 5で測り、各sample後のabort/reset、allocator、全40層KV、VRAMを検証する。
3. prompt 32 / G=64のdecode p50 throughputとp95 inter-token latencyを測定し、P8-B2を完了判定する。

## P8-B2 正式性能gate初回結果

### 前回の要点

M=8 cached-prefix chunkの正しさ、4095-token oracle、3584+512の厳密な4096境界は完了した。残件はresident TTFTとdecode性能の正式判定だった。

### 今回の変更点

- clean runner `08bdcecdbfad78827131b8b2d390122e4e19457a`、binary SHA-256 `ee109068...6c71b`をR9700単独で実行した。model loadは24.379317秒で、全42 requestを同一load内で処理した。
- prompt 32/128/512/2048/3584を各warmup 2、measured 5で測定した。TTFT p50/p95は順に`0.144360/0.144457`、`0.602628/0.603478`、`3.035701/3.037958`、`23.481711/23.503208`、`61.023836/61.025951`秒だった。
- 32/128/512/2048は固定gateに合格した。3584だけがp50 50秒を11.023836秒、p95 60秒を1.025951秒超えて不合格だった。
- prompt 32 / G=64 decodeはp50 27.779928 token/s、全315 ITLのp95 36.896757msで、15 token/s以上・100ms以下の両gateに合格した。
- 各TTFT sampleはfirst token後にcancelを観測し、decode sampleは64 tokenのlength terminalへ到達した。全sampleがactive0/waiting0、40層KV zero、allocator全解放へ戻った。
- 全44 VRAM captureでAMD SMIとKFDがbyte単位で一致し、worker以外の使用processはなかった。初期/最終resident VRAMは18,183,073,792 / 18,183,774,208 bytesだった。
- 独立validatorはraw schema、時刻順序、token/EOS、terminal/reset、GPU metrics、VRAM、build identityを受理し、3584の2 threshold failureだけを記録した。raw result SHA-256は`71a89668...22b03f`である。
- prompt長の実測は`T(N) ~= 0.0195 + 0.004036N + 3.623e-6N^2`秒に近い。これは診断上の推定であり、M=8 stack反復の線形成分と長prefix attentionの二次成分が支配していると考える。

### 次の行動

1. P8-Cは開始せず、既にCK primitiveの全projectionで測定済みのM=32/M=128をserving専用の単一request chunk候補として追加する。batch、queue、threshold変更は行わない。
2. 31/32/33、127/128/129、4095境界とprompt 32/128/512 oracleを先に通し、正しい候補だけを3584 prompt、warmup 2 / measured 3で限定比較する。
3. 既存50/60秒gateを通る速い幅を選び、4095 oracle、3584+512 deep boundary、正式2+5性能matrixを再実行する。両候補が外れた場合だけ、3584 requestを1回profileして支配kernelに限定した変更を行う。

## P8-B2 M=128候補選択

### 前回の要点

M=8正式runは3584-token TTFTだけが61.024秒で不合格だった。実測分解ではM=8 stack反復の線形成分が約14.47秒あり、既に全projectionが測定済みのM=128を最初の候補とした。

### 今回の変更点

- serving専用cached-prefix chunkをM=8/32/128へ一般化し、default M=8とP7監査経路を維持した。M=32/M=128は新しいv4 schemaと固有implementation IDを持つ。
- 31/32/33、127/128/129、255/256/257、4095のplanner、chunk report、head最終行、resident/reset geometryを追加testで固定した。engine 262 testsとgfx1201 example checkが合格した。
- R9700でM=128 prompt 128/512を実行し、tokenは既存M=8と同じ115/66、final hidden/logitsは全F32要素でbitwise一致した。requestは0.325525/1.015741秒だった。
- vLLM sourceに対するprompt 128/512のhidden/logit gateも合格し、top-1一致、top-10 overlap 9/10だった。
- prompt 3584は28個のM=128 chunkで31.310532秒、token 1、全40層cache長3584、position 3583、block 223、reset完了だった。M=8正式p50より48.7%速く、50秒gateへ18.69秒の余裕がある。
- probe中にvalidator fileの変更が存在したためworktree cleanはfalseであり、選択判断には使うがrelease evidenceには使わない。
- 十分な余裕が得られたため、M=32測定、sequence KV-write、F32 KV変更、attention改修は停止した。formal runが外れた場合だけ再検討する。

### 次の行動

1. clean同一binaryでM=128/all-M1のprompt 32/128/512/4095 oracleを作り、独立validatorへ通す。
2. 3584+512 deep boundaryをM=128で再実行し、全decode位置と最終KV 4095を検証する。
3. M=128の正式TTFT/decode 2 warmup + 5 measured matrixを実行し、全gate合格後だけP8-Cへ進む。

## P8-B2 M=128 clean correctness oracle完了

### 前回の要点

M=8の3584-token正式gate不合格に対し、M=128の診断runは31.310532秒で十分な余裕を示した。ただしworktree cleanではなかったため、選択判断とrelease evidenceを分離した。

### 今回の変更点

- clean commit `72008b91d3e2ada892208803b1891a5af466c5f2`、release binary SHA-256 `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`を固定し、同一R9700でM=128とall-M1を別process実行した。
- prompt 32/128/512/4095のM=128 request時間は`1.131062 / 0.176792 / 1.005426 / 56.753855`秒、all-M1は`1.160059 / 3.979503 / 18.786001 / 369.124277`秒だった。
- prompt 4095は31個のM=128と127個のM=1 tail、合計158 callでtoken 291を生成した。全40層KV長4095、position 4094、block 255に到達し、reset後はactive0/waiting0、allocator0、全cache zeroである。
- 独立validatorはM=128対all-M1で最悪relL2 `0.055494862`、最低cosine `0.998492050`、全4 promptでtop-1一致、top-10 overlap 10を確認した。prompt 32だけがbitwise一致で、他は定義済みの数値gate合格である。
- M=128対vLLM sourceも全4 promptでtop-1一致、top-10 overlap最佉9、最悪relL2 `0.065402638`、最低cosine `0.997865524`で合格した。
- result/capture/validationを`benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/m128-p32-p4095-clean-72008b9/`へ保存し、repo相対pathからの再検証も合格した。

### 次の行動

1. clean M=128でprompt 3584 + 512 generated tokenの4096境界を実行し、全decode位置と最終KV 4095を独立検証する。
2. 変更していないformal TTFT/decode 2 warmup + 5 measured matrixをM=128で実行する。
3. boundaryと全formal gate合格後にP8-B2を完了し、P8-Cのresident worker実装へ進む。

## P8-B2 M=128 clean deep boundary完了

### 前回の要点

M=128のclean correctness oracleはprompt 32/128/512/4095で合格した。P8-B2の正しさに残っていたのは、prompt 3584 + 512実生成tokenの4096-token境界である。

### 今回の変更点

- clean commit `3bb1ef206e05aafc47bde82f105eea0bd8278443`、binary SHA-256 `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`をR9700単独で実行した。
- test-only ignore-EOS条件でprompt 3584と512実生成tokenを完走した。M=128 prefillは28 call、M=1 decodeは511 call、合計539 callである。
- 全512 generated stepにtoken、cache write位置、全40層cache、scheduler active1/waiting0を記録した。最終KV長は4095、position 4094、block 255である。
- model loadは23.497561秒、resident requestは107.083953秒、resetは3.267msだった。reset後はReady、active0/waiting0、allocator0、全40層cache zeroに戻った。
- 独立validatorはproducerの`passed`を信用せず、prefill/decode/terminal/reset/build identityの全条件を再計算して合格した。raw result SHA-256は`885bbd1a84fdd18c81829bc87f0e558d46f1267180263c5adf865a55cb07235e`である。
- evidenceは`benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/deep-boundary-p3584-g512-m128-clean-3bb1ef2/`へ保存し、repo内resultの再検証も合格した。

### 次の行動

1. M=128で変更なしのformal TTFT/decode 2 warmup + 5 measured matrixを実行する。
2. 全sampleのtiming、terminal/reset、VRAM、GPU隔離、clean build identityを独立validatorで検査する。
3. 全gate合格でP8-B2を完了し、P8-Cのresident worker実装へ進む。

## P8-B2 M=128 formal performance合格

### 前回の要点

M=128のclean correctnessと3584+512 deep boundaryは合格した。P8-B2の残件は、変更していないformal TTFT/decode matrixと全sampleのreset/VRAM/隔離証拠であった。

### 今回の変更点

- clean commit `c271e010f18e6683dc53834188c45287434a34ef`、binary SHA-256 `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`でM=128 formal runを実行した。model loadは27.693288秒である。
- prompt 32/128/512/2048/3584のTTFT p50/p95は`0.958687/0.960489`、`0.150361/0.150400`、`0.995855/1.216792`、`10.817689/10.825768`、`31.286809/31.291056`秒で、すべて固定gateに合格した。
- prompt 32はM=128未満なので32個のM=1 callとなり、M=8より遅いが`2.5/3.0`秒gate内である。最低限の製品機能に必要ないhybrid tail最適化は追加しない。
- prompt 32 / G=64 decodeはp50 27.757735 token/s、ITL p95 36.881658msで15 token/s以上・100ms以下の両gateに合格した。
- 全42 requestはcancelまたはlength terminalへ進み、active0/waiting0、allocator0、全40層cache zeroへresetした。全44 VRAM captureはAMD SMI/KFDで一致し、worker以外の利用processはなかった。
- 独立validatorはraw v2構造、timer、greedy sampling、terminal/reset、GPU隔離、VRAM、build identityを検査し、`passed=true`、gate errorなしと判定した。raw SHA-256は`cb6119c9d6be9cbc8c7f55dcf2968be0b543c2e50bff602c046fb908201577e3`である。
- evidenceは`benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/performance-m128-clean-c271e01/`へ保存し、repo内resultの再検証も合格した。P8-B2は完了である。

### 次の行動

1. P8-Cで固定RNGと明示的なtemperature/top-k/top-p順序を持つdeterministic CPU samplingを実装する。
2. worker境界のcross-thread cancellationと、ack後にtokenを返さないcontractを固定する。
3. active1/waiting0、concurrent requestはbusy応答とするresident worker protocolを実装する。batchとqueueは追加しない。
