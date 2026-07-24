# -*- coding: utf-8 -*-
"""本番結果の集計（保存済みJSONから再分析・データは一切変更しない）"""
import sys, json, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resonance'))
from resonance_v07_zerolr import paired_wilcoxon_pratt, holm_2

rows = json.load(open("recon_full_results_30seed.json", encoding="utf-8"))
ok = [r for r in rows if not r.get("skip")]
print("有効シード %d / %d" % (len(ok), len(rows)))

pairs = []
for r in ok:
    sc, qo = r["results"]["SCAR-only"], r["results"]["Q-only"]
    if sc["valid_pair"] and qo["valid_pair"] and None not in (
            sc["N_hist"], qo["N_hist"], sc["D_behav"], qo["D_behav"]):
        pairs.append((sc, qo))
print("有効ペア %d" % len(pairs))

a1 = [s["N_hist"] for s, q in pairs]; b1 = [q["N_hist"] for s, q in pairs]
a2 = [q["D_behav"] for s, q in pairs]; b2 = [s["D_behav"] for s, q in pairs]
r1 = paired_wilcoxon_pratt(a1, b1)
r2 = paired_wilcoxon_pratt(a2, b2)
ph = holm_2(r1["p"], r2["p"])
print("\n【H1 語りの座】SCAR-only vs Q-only （予測: SCAR > Q）")
print("  中央値 SCAR=%.4f  Q-only=%.4f  差=%.4f" % (
    np.median(a1), np.median(b1), np.median(a1)-np.median(b1)))
print("  p=%.6f  holm=%.6f  rank-biserial=%.3f" % (r1["p"], ph[0], r1["rank_biserial"]))
print("  → %s" % ("支持" if (ph[0] < 0.05 and np.median(a1) > np.median(b1)) else "不支持"))
print("\n【H2 行動の座】Q-only vs SCAR-only （予測: Q < SCAR）")
print("  中央値 Q-only=%.4f  SCAR=%.4f" % (np.median(a2), np.median(b2)))
print("  p=%.6f  holm=%.6f  rank-biserial=%.3f" % (r2["p"], ph[1], r2["rank_biserial"]))
print("  → %s" % ("支持" if (ph[1] < 0.05 and np.median(a2) < np.median(b2)) else "不支持"))

# --- G1（装置成立ゲート・凍結値0.5948） ---
fu = [r["results"]["FULL"]["N_hist"] for r in ok if r["results"]["FULL"]["N_hist"] is not None]
no = [r["results"]["NONE"]["N_hist"] for r in ok if r["results"]["NONE"]["N_hist"] is not None]
g = paired_wilcoxon_pratt(fu, no)
print("\n【G1 装置成立】FULL vs NONE（凍結閾値 0.5948）")
print("  FULL中央=%.4f NONE中央=%.4f p=%.6f rb=%.3f 絶対値基準=%s" % (
    np.median(fu), np.median(no), g["p"], g["rank_biserial"],
    "満たす" if np.median(fu) > 0.5948 else "満たさない"))

# --- 分離分析（事前登録済み・記述） ---
print("\n【分離分析】L=喪失チャンネル / P=幻肢痛チャンネル")
for cond in ["FULL", "Q-only", "SCAR-only", "NONE"]:
    v = [r["results"][cond] for r in ok]
    def m(k):
        xs = [x[k] for x in v if x[k] is not None]
        return np.median(xs) if xs else float('nan')
    print("  %-10s N_hist=%.4f  L=%.4f  P=%.4f  D_behav=%.4f  死亡%d/%d 生存中央%.0f" % (
        cond, m("N_hist"), m("N_hist_L"), m("N_hist_P"), m("D_behav"),
        sum(1 for x in v if x["life"] < 1000), len(v), m("life")))
print("  CONT       死亡%d/%d 生存中央%.0f" % (
    sum(1 for r in ok if r["cont_death"] < 1000), len(ok),
    np.median([r["cont_death"] for r in ok])))
