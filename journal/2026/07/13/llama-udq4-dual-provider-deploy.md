# llama.cpp UD-Q4_K_XL 二重プロバイダ配備

## 前回の要点

- OpenWebUIには既存の外部プロバイダとuLLMプロバイダがあり、uLLMは
  `http://172.20.0.1:8000/v1`を使っている。
- Qwen3.5 9B UD-Q4_K_XLは既存GGUFと既存`llama-server`バイナリを使う。

## 今回の変更点

- 初回の3 GPU可視起動では`libamdhip64`のGPFが発生した。`HIP_VISIBLE_DEVICES=1`
  で物理R9700だけを可視化すると、プロセス内のデバイスは`ROCm0`として固定できた。
- GPU isolation後は、uLLM (`172.20.0.1:8000`) を同時常駐させたまま llama.cpp
  (`172.20.0.1:8001`) の起動と短文生成に成功した。短文の実測値は
  prefill `131.94`、decode `74.23` token/sだった。
- 実行条件はcontext 4096、parallel 1、fit off、text-only、`--no-mmproj`である。
  このモデルではmmprojは不要であり、mmprojファイルは読み込まない。
- 2つの常駐workerは同じR9700を使うため、同時リクエストはGPU競合を起こす。
  比較は交互に実行し、uLLMの`/run/ullm/r9700.lock`はllama.cppと共有しない。
- OpenWebUIの設定テストに、既存index 0/1とモデル行を保持したままllama providerを
  index 2へ追加し、再実行でURL・key・モデルが重複しない回帰ケースを追加した。

## 配備

- `deploy/README.md`に、別API keyの生成、root管理ファイルの安全なinstall、systemdの
  daemon reload、firewall適用、enable-now、Docker network内の認証付きhealth/models確認、
  OpenWebUI停止・read-only mount・legacy configure引数・再起動手順を記録した。
- 実機のDB、`/etc`、systemdサービス、nftablesはこの作業では変更していない。配備は親
  エージェントが後で実施する。

## 次の行動

親エージェントがREADMEの手順で別keyを配置し、両endpointの認証付き`/v1/models`と
OpenWebUI healthを確認した後、2つのモデルを交互に比較する。

## 最終配備証跡

- 配備対象コミットは `c85e5cd`（llama.cpp systemd、nftables、firewall）と
  `b7c4e96`（OpenWebUI設定テスト、README、journal）である。source testは合計
  19 passed、`systemd-analyze verify`、root権限でのnft `--check`と実適用が成功した。
- `/etc/ullm/llama-qwen35-udq4.env`、別API key（内容は記録しない）、llama unit、
  firewall unitをinstallし、旧nftables/unitのbackupを保持した。llama、uLLM、firewallは
  すべてactiveで、`NRestarts=0`だった。観測時PIDはllama `3395514`、uLLM `3263234`である。
- 2 worker常駐時のR9700 total usedは `13,377,409,024` bytesだった。コンテナnetwork内の
  healthと認証付きmodels確認は成功し、hostから`172.20.0.1:8001`への接続は
  timeout（curl code `000`）となり、firewallのbridge限定を確認した。
- OpenWebUIのbackupは
  `/data/backups/webui-before-ullm-20260713T050608Z-1783919168461802516.db`である。
  provider index 2へ`http://172.20.0.1:8001/v1`を追加し、既存index 0/1と既存uLLM
  model rowのhashは不変だった。OpenWebUIはhealthyで、URLは
  `http://192.168.0.66:3000`である。
- 最終llama chatの`1+1...`はcontent `2`、`finish_reason=stop`、prefill
  `150.42` token/s、decode `131.61` token/sだった。生成は2 tokenだけなので、これらを
  性能代表値として扱わない。model aliasは`llama-qwen3.5-9b-ud-q4`と完全一致した。
- 配備runtimeは`HIP_VISIBLE_DEVICES=1`（プロセス内`ROCm0`）、`--fit off`、context
  4096、parallel 1、text-only、`--no-mmproj`で固定した。3 GPU可視の`ROCm1`構成で
  `libamdhip64` GPFが発生するため、その構成は拒否する。同時生成は行わず、比較は交互に
  実行する。
