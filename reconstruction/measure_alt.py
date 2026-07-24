# -*- coding: utf-8 -*-
"""裁定⑧較正: 代替経路が確立するまでの日数を実測（予備シード200-209のみ使用）
操作的定義: あるセルで初めて衝突してから、そのセルで最後に衝突するまでのエピソード数
= その道を諦めて別の道が立つまでにかかった時間。
急所の記憶はこの期間を確実に跨ぐ必要がある（探索者を生かしておくため）。"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import recon_v01 as R

spans = []
for seed in R.SEEDS_PRELIM:
    A = R.ReconAgent(seed); env = R.ResonanceEnv(R.PATROL_B_D0)
    pm_map = R.make_part_seq(seed); ev, vis = [], {}
    R.run_life(A, env, pm_map, 0, 0, R.T_TOTAL, ev, vis)
    by_key = {}
    for e in ev:
        by_key.setdefault(e["pk"], []).append(e["ep"])
    for k, eps in by_key.items():
        if len(eps) >= 2:
            eps = sorted(eps)
            # 修正: ε探索による偶発的迷い込みを除くため、衝突の9割が済むまでを
            # 「その道を実質使わなくなるまで＝代替経路の確立」とみなす
            idx = max(0, int(np.ceil(0.9 * len(eps))) - 1)
            spans.append(eps[idx] - eps[0])
spans = np.array(spans)
print("経路キー数=%d 中央値=%.0f 90pct=%.0f 95pct=%.0f 最大=%d" % (
    len(spans), np.median(spans), np.percentile(spans, 90),
    np.percentile(spans, 95), spans.max()))
for T in [int(np.percentile(spans, 90)), int(np.percentile(spans, 95))]:
    lam = R.TABOO_TH ** (1.0 / T)
    print("T=%d を跨ぐ最低λ = %.5f （強度1.0がTabu閾値0.3を割るまでT ep）" % (T, lam))
