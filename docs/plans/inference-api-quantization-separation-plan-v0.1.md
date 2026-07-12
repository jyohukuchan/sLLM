# Inference/API and Quantization Separation Plan v0.1

## 1. 前回の要点

uLLMはQwen3-14B SQ8をOpenWebUI製品経路として実装した後、Qwen3.5-9B AQ4を
互換workerで接続した。この結果、OpenAI互換API、SSE、cancel、timings、OpenWebUIは
AQ4でも動作したが、次の設計上の混在が明確になった。

- worker protocolとworker runtimeが`Sq8*`型へ依存する。
- worker profileがモデル情報を環境変数から都度読み、実ロード結果を証明しない。
- gateway、tokenizer、worker、OpenWebUIへ同じモデル情報が分散する。
- AQ4生成本体はCLI private実装で、要求ごとに子processを起動してpackageを再読込する。
- 同一base URLのモデル切替後もOpenWebUIの旧uLLMモデル行がactiveのまま残る。

## 2. 目的

API接続、worker制御、モデル構造、量子化形式、GPU kernel選択を明確に分離する。
新しい量子化形式を追加するとき、OpenAI API、SSE、OpenWebUI、cancel、timingsを
量子化ごとに再実装しない状態を完成条件とする。

最終的な依存方向は次の通りとする。

```text
OpenWebUI
  -> OpenAI gateway
  -> ullm.worker protocol
  -> generic worker driver
  -> InferenceSession
  -> model runtime
  -> quantization backend
  -> GPU kernel dispatch
```

上位層は下位層の具体的な量子化名を知らない。下位層はHTTP、SSE、OpenWebUIを知らない。

## 3. 完成要件

### 3.1 共通API

- `InferenceRequest`、`SamplingParams`、`CancellationToken`、`FinishReason`、
  `ReleaseSummary`、`GenerationTimings`を量子化非依存型として提供する。
- `InferenceSession`はstart、prefill progress、prepare、publish、commit、cancel、resetを表現する。
- prepare済みtokenは二重commitできず、publish失敗時にsampling/scheduler状態を進めない。
- wire protocolは移行中も`ullm.worker.v1`互換を維持する。v2追加時はmanifest digestをreadyへ含める。

### 3.2 backend

- SQ8は`Qwen3Sq8InferenceSession`として共通session interfaceを実装する。
- AQ4は子processを使わず、Qwen3.5 package、weight、KV/recurrent stateをworker内に常駐させる。
- AQ4は生成tokenを完了後一括ではなく逐次publishする。
- request終了・cancel・publish失敗後に全request stateをresetし、連続要求で再利用できる。
- backend capabilityでsampling、batch、prefill方式、context上限を表現する。

### 3.3 served-model manifest

- 1モデルにつき1個の`ullm.served_model.v1`を単一の正とする。
- manifestはpublic metadata、generation contract、tokenizer identity、format/backend、
  worker command、artifact/package identity、検証receiptを含む。
- bind、port、API key、GPU lock、GPU visibility、active manifest pathはslot運用設定へ残す。
- gateway、worker、OpenWebUIは同じmanifestを読む。
- 個別モデル環境変数とmanifest modeの混在をfail-closedで拒否する。
- ready eventはmanifest SHA-256と、実際にロード・検証したbinary/artifact/package identityを返す。

### 3.4 配備

- SQ8とAQ4をactive manifestの切替だけで配備できる。
- activationはcandidate検証、停止、atomic rename、起動、ready、最小生成、OpenWebUI同期、
  browser smokeを行う。
- 任意の段階の失敗で旧manifestとOpenWebUI DBへrollbackできる。
- 同一base URLで現在のuLLMモデルだけをactiveにし、旧管理モデルを選択不能にする。

## 4. manifest v1の構造

```json
{
  "schema_version": "ullm.served_model.v1",
  "public": {
    "id": "ullm-qwen3.5-9b-aq4",
    "name": "uLLM Qwen3.5 9B AQ4",
    "description": "Qwen3.5 9B served locally by uLLM AQ4_0.",
    "upstream_id": "Qwen/Qwen3.5-9B",
    "revision": "...",
    "context_length": 4096
  },
  "generation": {
    "max_completion_tokens": 512,
    "vocab_size": 248320,
    "eos_token_ids": [248044, 248046],
    "sampling": {
      "top_k": 1,
      "temperature": false,
      "top_p": false
    }
  },
  "format": {
    "format_id": "AQ4_0",
    "implementation_id": "qwen35_aq4_rdna4_v1"
  },
  "tokenizer": {
    "root": "/absolute/path",
    "transformers_version": "5.12.1",
    "class": "Qwen2Tokenizer",
    "chat_template_sha256": "...",
    "files": {"tokenizer.json": "..."},
    "template_options": {
      "add_generation_prompt": true,
      "enable_thinking": false
    }
  },
  "worker": {
    "protocol": "ullm.worker.v1",
    "binary": "/absolute/path/ullm-aq4-worker",
    "binary_sha256": "...",
    "arguments": ["--served-model-manifest", "{manifest}"],
    "required_environment": ["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL"],
    "identity": {
      "device": "gfx1201",
      "execution_profile": "rdna4_aq4_resident"
    }
  },
  "product": {
    "root": "/absolute/product/root",
    "artifact": {"path": "artifact", "manifest_sha256": "..."},
    "package": {"path": "package", "manifest_sha256": "..."}
  },
  "promotion": {
    "source_commit": "...",
    "receipt": "/absolute/path/to/receipt.json"
  }
}
```

実装時にはexact-key schemaを固定する。上記は説明用であり、仕様書のJSON schemaを正とする。

## 5. 実装段階

### P0: 基準固定と計画

- 本計画をcommitする。
- 現在のSQ8/AQ4 HTTP、SSE、cancel、timings、OpenWebUI証拠を回帰基準として記録する。
- `.rocprofv3/`など既存の無関係な作業物には触れない。

Gate:

- plan reviewで完成要件、移行順、rollback、検証項目が揃っている。

### P1: 共通推論API型

- `crates/ullm-engine/src/inference_api.rs`を追加する。
- SQ8 servingから共通request、sampling、cancel、finish/release型を移す。
- 旧`Sq8*`名は一時alias/re-exportとして保持する。
- protocolからSQ8 model/vocab定数への依存を除去し、起動時の`WorkerProfile`を明示的に渡す。

Gate:

- worker protocolのJSON golden、重複key、4MiB、busy、cancel、flush testが全件成功する。
- SQ8 workerのwire bytesとfailure orderingが変わらない。

### P2: 共通worker driverとInferenceSession

- `worker_protocol.rs`、`worker_runtime.rs`、`worker_driver.rs`を量子化非依存moduleとして追加する。
- `sq8_worker_protocol`と`sq8_worker_runtime`は移行aliasへ降格する。
- `sq8_worker_backend.rs`内のtransaction driverを共通`InferenceSession` driverへ移す。
- prepare -> publish -> commitとcancel/resetの順序を型とtestで固定する。

Gate:

- scripted sessionでsuccess、EOS、length、prepare後cancel、publish失敗、二重commit拒否、resetを検証する。
- SQ8 worker acceptanceと既存OpenWebUI API contractが成功する。

### P3: served-model manifest仕様とloader

- `docs/specs/served-model-manifest-v0.1.md`を追加する。
- gateway側にbounded/exact JSON loaderを追加する。
- 重複/未知/missing key、型、サイズ、SHA、path escape、symlink、world-writableを拒否する。
- EOS < vocab、max completion <= context、worker/tokenizer/product identityの相互制約を検証する。
- SQ8/AQ4 fixtureと実product manifestを生成する。

Gate:

- malformed/fault-injection testがfail-closedで成功する。
- SQ8/AQ4 manifestが同一loaderとvalidatorを通る。

### P4: gateway/tokenizer/OpenWebUIのmanifest移行

- `GatewaySettings`をslot運用設定と`ServedModel`へ分ける。
- gatewayのmodel/env hardcode、`TOKENIZER_PROFILES`、固定worker argvをmanifestへ移す。
- `configure.py`がmanifestからID/name/description/contextを読む。
- 同一base URLの旧uLLM管理モデルをinactiveにする。
- v1移行中はmanifest modeとlegacy env modeを排他的に扱う。

Gate:

- SQ8/AQ4 parameterized gateway testでmodels、通常応答、SSE、context、EOS、length、timingsが成功する。
- manifestとlegacy envを同時指定すると起動拒否する。
- OpenWebUI DB testで現在モデルだけがactiveになる。

### P5: SQ8 session adapter

- `Qwen3Sq8ServingSession`を`Qwen3Sq8InferenceSession`で包み、共通interfaceへ接続する。
- runtime context、stream、sessionのdrop順を維持する。
- SQ8 worker backendにはartifact load、device選択、kernel guardだけを残す。

Gate:

- 移行前後で固定promptのtoken列、finish reason、timings境界が一致する。
- SQ8 release/worker/OpenWebUI gateが回帰しない。

### P6: AQ4 resident runtime抽出

次の順でCLI private実装をlibraryへ移す。

1. `PackageAq4ResidentMatvec`とshared weight registryを`aq4_package_runtime.rs`へ移す。
2. self-attention/linear-attention resident layerを`qwen35_aq4_layer_runtime.rs`へ移す。
3. embedding、final norm、AQ4 lm_headをlibraryへ移す。
4. layer列挙とdevice-to-device 1-token stepを`qwen35_aq4_session.rs`へ移す。
5. Ready/Prefilling/PreparedToken/Decoding/Terminal/Failed state machineを実装する。
6. CLIを新sessionを駆動して従来reportを作る利用者へ変更する。
7. AQ4 workerから子process起動を削除する。

Gate:

- 移動前後のprompt suiteでtoken列が完全一致する。
- 連続2要求、EOS、length、cancel、publish失敗、reset後再利用が成功する。
- 2要求目以降にmodel reload、子process、継続的VRAM増加がない。
- token eventが生成中に逐次到着する。

### P7: activation/rollback

- `tools/activate-served-model.py`を追加する。
- candidate preflight、atomic active manifest更新、service ready、最小生成、OpenWebUI reconcile、browser smokeを実装する。
- 任意の失敗点で旧manifestとDB backupへrollbackする。
- systemdの`ExecStartPre`でもmanifestを検証する。

Gate:

- SQ8 -> AQ4 -> SQ8がmanifest pathだけで成功する。
- manifest破損、binary hash不一致、worker failure、ready timeout、DB failureのfault injectionでrollbackする。

### P8: 完成監査

- API/gateway/OpenWebUI sourceに量子化名による分岐が残っていないことを確認する。
- 新しい仮想format fixtureを追加し、API側変更なしでmanifest/schemaまで受理できることをtestする。
- SQ8/AQ4の全回帰、資源、cancel、timings、OpenWebUI gateを実機で通す。
- legacy model env、AQ4 CLI compatibility child backend、一時aliasを削除する。

## 6. 非目標

- 最初のmanifest v1で複数モデルを1processへ同時常駐させない。
- request batchingをこの分離の成功条件に含めない。
- AQ4 resident化だけを理由にtemperature/top-pを対応済みとは扱わない。
- 既存SQ8 release evidenceを汎用manifestへ無理に変換しない。receiptとして参照する。

## 7. リスクと対策

- 大規模renameでfailure-path testが無意味に壊れるため、型抽出、alias、利用側移行、alias削除を分ける。
- wire event順序を維持し、内部module名変更とprotocol version変更を同じcommitで行わない。
- AQ4 state reset漏れを防ぐため、KV、conv history、recurrent state、sampling drawを個別に検証する。
- manifestが環境変数の別表になることを防ぎ、worker readyを実ロード結果とmanifest digestへ結び付ける。
- OOMを避けるため、AQ4 weight移動中は既存resident bufferをcloneせず所有権を移し、検証時の並列GPU processを1にする。

## 8. commit単位

1. `Document inference and quantization separation plan`
2. `Extract generic inference API types`
3. `Add generic worker driver`
4. `Define served model manifest`
5. `Load gateway model contract from manifest`
6. `Reconcile OpenWebUI model from manifest`
7. `Adapt SQ8 session to generic driver`
8. `Extract AQ4 resident package runtime`
9. `Add resident Qwen3.5 AQ4 session`
10. `Add atomic served-model activation`
11. `Remove legacy quantization-specific API wiring`

## 9. 次の行動

P1の共通推論API型抽出と、P3のmanifest仕様/厳密loaderを独立して開始する。
両者が揃った時点で、P2 worker driverとP4 gateway移行を接続する。
