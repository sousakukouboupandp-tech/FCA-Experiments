# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
base = r'D:\☆　◆personify primitive◆　☆\◆★●マンデープロンプト\★  元となったアートVer6とVer7\FCA日本語版英語版最新バージョン管理\FCA-Experiments-git\resonance'
t = open(base + '\\DESIGN_遅延丸呑み害曲線_v02.md', encoding='utf-8').read()

old = '状態：第1ラウンド監査（3AI）対応済み。重大1件（Gemini・床効果）は裁定のうえ修正反映済み。**第2ラウンドはGeminiのみ（修正の検証）とし、重大ゼロなら凍結する。最大3Rの規程を継承。**'
new = '状態：**事前登録として凍結済み（2026年7月15日・全2ラウンドの監査完了）。実行後の変更はバグ例外条項を除き行わない。**'
assert t.count(old) == 1
t = t.replace(old, new)

sec8 = '''

## 8. 第2ラウンド監査記録（Gemini単独・修正検証）と凍結宣言

- Geminiは第1Rの自らの重大指摘（床効果）に対するv02の裁定——(a)数え上げ天井の非対称は実測基線（微小）により数学的に棄却、(b)成熟方策の不攪乱性は交絡ではなく測定対象の一部——を「完全に正しい」「論理的に無謬」と検証・承認した。
- excess_normの頑健性登録＝許容。注入量非対称の棄却（A_cache全件転送・τ非依存）＝「完全に正しい棄却。私の誤読だった」と確認。
- **判定：重大指摘ゼロ。「本設計書v02の凍結に同意します」（Gemini・2026年7月15日）。**
- 監査サマリ：第1R＝3AI（ChatGPT重大0・AIスタジオ重大0凍結同意・Gemini重大1）→v02反映。第2R＝Gemini重大0凍結同意。3AI全員の同意が揃った。
- **凍結宣言**：本設計書を事前登録として凍結する（2026年7月15日）。実行・判定・公開は本書の記載のみに従う。第3ラウンドは行わない。
'''
t = t + sec8
open(base + '\\DESIGN_遅延丸呑み害曲線_v02_凍結.md', 'w', encoding='utf-8').write(t)
print('凍結版保存完了 行数:', t.count(chr(10)) + 1)
