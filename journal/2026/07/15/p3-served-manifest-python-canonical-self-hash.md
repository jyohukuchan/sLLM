# P3 served-manifest Python canonical self-hash

## 前回の要点

- sealed FD-mapのproducerとrunnerは、Pythonの`json.dumps(sort_keys=True,separators=(",",":"),ensure_ascii=True)`で自己ハッシュを計算する。
- Rust driverは`serde_json::to_vec`を使っていたため、非ASCIIのlogical pathやresolved pathをUTF-8のまま出力し、Pythonの`\uXXXX`表現とハッシュが一致しなかった。

## 今回の変更点

- Rust driverにPython互換のcanonical JSON serializerを追加した。
- object keyをUnicode順に並べ、空白なしのcompact JSONを生成する。
- printable ASCII以外をlowercaseの`\uXXXX`へ変換し、補助平面文字はUTF-16のsurrogate pairとして出力する。
- quote、backslash、slash、制御文字、DEL、BMP、補助平面、結合文字についてPython生成値を固定したcross-language golden testsを追加した。
- Unicodeのkeyとlogical pathを含むFD-mapについて、Python生成の自己ハッシュとRustの計算結果が一致するtestを追加した。

## 検証

- `cargo test -p ullm-engine --bin ullm-aq4-p2-resident-driver -- --test-threads=1`: 22 passed
- 担当Rustファイルの`rustfmt --check`と`git diff --check`を実施する。
- actual、GPU workload、service操作は行っていない。

## 次の行動

- buildレーンでは既存AQ4 served-model fixtureを使い、実Rust executableへdriver/map/manifest FDsをmulti-hop継承する。
- guard environmentを未設定にしてdevice query前で停止すれば、production用test hookを追加せずにFD-map受理とlogical argv維持を検証できる。
