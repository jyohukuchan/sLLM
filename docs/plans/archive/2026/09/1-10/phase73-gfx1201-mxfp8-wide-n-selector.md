# Phase 73: gfx1201 MXFP8 wide-N selector

状態: `完了（2026-09-02）`

## 目的と範囲

ユーザー指示により、exact `gfx1201`のOCP MXFP8 E4M3 W8A8 production WMMA経路に残るN上限を
`N<=16384`から`N<=32768`へ緩和する。model名、M/K下限、64／128列alignment、target、format、decode、
量子化recipe、accumulation、outputは変更しない。詳細なGPU性能・数値再検証は要求どおり省略する。

## 結果

- ID31／34／36／37が共有するkernel selectorとprepared providerのN上限を32,768へ同期した。
- host contractでN=17,408、32,000、32,768がID37、N=32,769とalignment済み上限外N=32,832がrow8 fallbackとなることを確認した。
- exact gfx1201向けprovider testを再ビルド・実行し、provider contract、既存codec、fallbackがPASSした。
- 新しいN範囲のoperator oracle、full-model品質、性能値は未収集であり、追加のGPU実証済み範囲とは扱わない。

[全体計画](../../../../main-plan.md) /
[Phase 73履歴](../../../../../history/2026/09/1-10/phase73-gfx1201-mxfp8-wide-n-selector.md) /
[Phase 73追跡要約](../../../../../../ci/matrix/phase73-gfx1201-mxfp8-wide-n-selector-v1.json)
