# -*- coding: utf-8 -*-
"""
認識と傷の分離実験 凍結前ブラインド診断（η水準較正）
規律: 測定区間[τ+1,1000]の衝突は一切実行・参照しない。τ=150時点のS分布から
      n_written期待値のみをシミュレートする（第1R監査承認のブラインド手順）。
選定規則（診断実行前に固定・2026年7月18日）:
  低用量 = {0.25,0.30,0.35,0.40,0.50}のうち中央値n_written>=3となる最小η
  中用量 = 低用量とη=1.0の中間に最も近い候補。該当なしなら0.50。以後変更終了。
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from resonance_v07_zerolr import Agent, ResonanceEnv, PATROL_B_D0, run_eps, learn_A, sim_entry, ETA_RES

TAU = 150
SEEDS = list(range(60, 90))
GRID = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]
THRESH = 0.05

rows = {e: [] for e in GRID}
for seed in SEEDS:
    A_pm = learn_A(seed)
    B = Agent(seed + 1000)
    st = {"coll_flags": [], "succ": []}
    run_eps(B, ResonanceEnv(PATROL_B_D0), TAU, st, learn=True)
    own = list(B.archive.keys())
    base = []
    for k, inten in A_pm.items():
        S = max([sim_entry(k, kb) for kb in own], default=0.0)
        base.append(inten * ETA_RES * S)
    for e in GRID:
        rows[e].append(sum(1 for b in base if e * b > THRESH))

print("η別 n_written（30シード・τ=150時点・ブラインド）")
for e in GRID:
    a = np.array(rows[e])
    print("η=%.2f: 中央値%d IQR[%d-%d] min%d max%d ゼロ本数%d/30" % (
        e, np.median(a), np.percentile(a,25), np.percentile(a,75), a.min(), a.max(),
        int((a==0).sum())))
