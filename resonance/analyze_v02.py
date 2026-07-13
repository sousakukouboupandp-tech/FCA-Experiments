# -*- coding: utf-8 -*-
"""v02結果の有意性判定(Welch t検定・シード別データ使用)"""
import json, os
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
import sys
fname = sys.argv[1] if len(sys.argv) > 1 else 'resonance_v02_full_results_15seed.json'
d = json.load(open(os.path.join(BASE, fname), encoding='utf-8'))
print(f"[{fname}]")

def welch_t(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    t = (a.mean() - b.mean()) / np.sqrt(va/na + vb/nb)
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    return t, df

def report(world, c1, c2, key="coll_total"):
    a = d[world][c1]["per_seed"][key]
    b = d[world][c2]["per_seed"][key]
    t, df = welch_t(a, b)
    m1, m2 = np.mean(a), np.mean(b)
    # 粗いp値目安: |t|>2.05でp<0.05(df~28), |t|>2.76でp<0.01
    sig = "**p<0.01**" if abs(t) > 2.76 else ("*p<0.05*" if abs(t) > 2.05 else "n.s.")
    print(f"{world:7s} {c1}({m1:.1f}) vs {c2}({m2:.1f}): t={t:+.2f} df={df:.0f} {sig}")

print("=== 衝突数の有意性判定 ===")
report("D0", "FullCopy", "Isolated")
report("D0", "ResoExp", "Isolated")
report("Dsmall", "FullCopy", "Isolated")
report("Dsmall", "ResoExp", "Isolated")
report("Dbig", "FullCopy", "Isolated")
report("Dbig", "ResoExp", "Isolated")
print()
print("=== top側衝突(Dbig=本物の危険への突入) ===")
report("Dbig", "FullCopy", "Isolated", "coll_at_top")
report("Dbig", "ResoExp", "Isolated", "coll_at_top")
