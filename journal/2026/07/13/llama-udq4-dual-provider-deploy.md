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
