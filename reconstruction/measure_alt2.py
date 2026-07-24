# -*- coding: utf-8 -*-
"""裁定⑧較正やり直し: 探索ノイズを除いた「代替経路の確立」実測。
方針の是正: 前回はε探索による偶発衝突を混入させ「代替路なし」と誤読した。
この世界には常時安全な迂回路(gap_top, y=6)が実在する。
方策が危険路を選ばなくなった時点＝greedy行動での衝突が止まった時点を測る。
※λ_VITALを一時的に0.97に戻して素の学習動態を測る（較正の循環回避）"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import recon_v01 as R
R.LAMBDA_VITAL = R.LAMBDA_OWN   # 素の動態を測るため一時的に無効化

spans, n_greedy, n_expl = [], 0, 0
for seed in R.SEEDS_PRELIM:
    A = R.ReconAgent(seed); env = R.ResonanceEnv(R.PATROL_B_D0)
    pmap = R.make_part_seq(seed); ev, vis = [], {}
    R.run_life(A, env, pmap, 0, 0, R.T_TOTAL, ev, vis)
    g = [e for e in ev if not e["explored"]]
    n_greedy += len(g); n_expl += len(ev) - len(g)
    by = {}
    for e in g: by.setdefault(e["pk"], []).append(e["ep"])
    for k, eps in by.items():
        if len(eps) >= 2: spans.append(max(eps) - min(eps))
print("全衝突: greedy=%d explore=%d" % (n_greedy, n_expl))
s = np.array(spans)
print("greedy衝突が続いた期間: n=%d 中央=%.0f 90pct=%.0f 最大=%d" % (
    len(s), np.median(s), np.percentile(s, 90), s.max()))
for T in [int(np.median(s)), int(np.percentile(s, 90))]:
    if T > 0:
        print("T=%d 跨ぐλ=%.5f" % (T, R.TABOO_TH ** (1.0/T)))
