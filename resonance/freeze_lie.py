# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
base = r'D:\☆　◆personify primitive◆　☆\◆★●マンデープロンプト\★  元となったアートVer6とVer7\FCA日本語版英語版最新バージョン管理\FCA-Experiments-git\resonance'
t = open(base + '\\DESIGN_嘘の伝播_v02.md', encoding='utf-8').read()

def rep(old, new):
    global t
    assert t.count(old) == 1, 'MISS:' + old[:50]
    t = t.replace(old, new)

rep('設計書 v02（第1R監査対応・A→B→C三世代）', '設計書 v02（事前登録・凍結版）')
rep('状態：第1R監査（3AI・重大5件）対応済みドラフト。第2ラウンドは3AI全員による修正検証。重大ゼロなら凍結。最大3R規程継承。',
    '状態：**事前登録として凍結済み（2026年7月16日・全2ラウンド監査・3AI全員が凍結同意）。実行後の変更はUT境界の例外を除き行わない。**')
rep('promoted=true・promotion_epを記録。',
    'promoted=true・promotion_epを記録（**同一キーの複数回衝突時はpromotion_ep＝初回衝突epで固定**・第2R・ChatGPT推奨採用）。')
rep('p_subj＝C受信直後のpm強度のセル別合計を総和で正規化（25セル）。',
    'p_subj＝C受信直後のpm強度のセル別合計を総和で正規化（25セル・**セルID＝行優先（row-major）**・第2R・ChatGPT推奨採用）。')
rep('R_chain(T_tell=150, E=1) は R_chain(T_tell=1000, E=1) より有意に高い（同上）。',
    'R_chain(T_tell=150, E=1) は R_chain(T_tell=1000, E=1) より有意に高い（同上。**両者は同一Bラン（E=1）のep150／ep1000スナップショットをそれぞれ別のCへ語らせたもの**＝第2R・ChatGPT推奨で明句化）。')

sec9 = '''

## 9. 第2ラウンド監査記録（2026年7月16日・3AI・名乗り確認済み）と凍結宣言

- **Google AIスタジオ：重大ゼロ・「凍結に同意する」**。冒頭で「**書記官のλ検算に基づく裁定は完全に正しい（私の第1R提案は誤りであった）**」と自らのM昇格要求の撤回を明言。§2の開示を「認識論的誠実さの究極の実践」、P2（時間の門）を「状態の比較（恒真）からプロセスの比較（測定）への転換」として承認。L1退路なしを確認。
- **Gemini：重大ゼロ・「凍結に同意します」**。T_tell=150の窓を数理検証——最良条件（λ=0.97・I₀=1.0）で0.218生存、最悪条件（λ=0.94・I₀=0.5）で0.0225消滅＝床効果・天井効果なしの適正な淘汰圧と確認。スナップショット2時点方式・対応あり設計への変更も承認。
- **ChatGPT：重大ゼロ・「凍結に同意します」**。第1R指摘8件を全件「対応済み」判定。推奨3件（セルID行優先／promotion_ep初回固定／P2の明句化）→本凍結版に反映済み。
- 監査サマリ：第1R＝重大5件（全採用・うちAIスタジオのM昇格要求は書記官のλ検算により修正棄却）→第2R＝3AI全員重大ゼロ・凍結同意。
- **凍結宣言**：本設計書v02を事前登録として凍結する（2026年7月16日）。実行・判定・公開は本書の記載のみに従う。
'''
t = t + sec9
open(base + '\\DESIGN_嘘の伝播_v02_凍結.md', 'w', encoding='utf-8').write(t)
print('凍結版保存 行数:', t.count(chr(10)) + 1)
