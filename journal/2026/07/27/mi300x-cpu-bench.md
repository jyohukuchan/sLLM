## 前回の要点

MI300X rental は GPU 側の CR 作業と共有される可能性があり、CPU 計測はまず
競合と guest が実際に公開している ISA/NUMA を確認してから行う必要があった。

## 今回の変更点

- 13 vCPU の Sapphire Rapids guest で `/proc/cpuinfo` flags、`lscpu`、
  `numactl --hardware`、sudo `dmidecode -t memory` を採取した。AVX-512
  BF16/FP16 を含む ISA は見えるが、AMX の三フラグは guest に公開されていない。
- node 0 に CPU/memory を固定し、512 MiB/vector、3 warmups、7 repeats の
  STREAM 相当 read/copy/triad を 1/4/8/13 threads で計測した。13-thread
  median は 56.351 / 42.922 / 49.347 GB/s であり、これは VM から見た値で
  socket の DDR5 peak ではない。
- 同じ独立 warmup/median timer の CPU 用ソースを追加して AVX-512 FP32/BF16
  を計測した。13-thread achieved throughput は 2.464 / 2.439 TFLOPS。
- SGLang は未導入で、system pip は PEP 668、venv は `python3.12-venv` 欠落で
  止まった。モデル未ダウンロードのまま具体的な失敗ログを保存し、llama.cpp
  代替は rental 時間を延ばすため起動しなかった。

## 次の行動

- CPU inference が必要なら、短時間の別 window で `python3.12-venv` を導入し、
  SGLang の CPU backend を model download 前に検証する。SGLang が不可なら
  llama.cpp を明示的な代替として 1--3B GGUF で prefill/decode を計測する。
- 物理 DDR5 channel/MT/s は QEMU DMI が unknown であり、VM 内の値から推測しない。
  物理 host の信頼できる管理面情報が得られたときだけ理論帯域を算出する。
