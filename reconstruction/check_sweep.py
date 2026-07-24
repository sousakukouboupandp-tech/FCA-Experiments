# -*- coding: utf-8 -*-
"""較正チェック2: 致死量・耐久の水準スイープ（1000ep完走時のCONT死亡率）"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import recon_v01 as R

def sweep(lethal, durab, seeds):
    for p in R.VITALS: R.LIMIT[p] = lethal
    for p in R.LIMBS: R.LIMIT[p] = durab
    deaths, death_eps, broken_n, coll = 0, [], 0, []
    for seed in seeds:
        A = R.ReconAgent(seed)
        env = R.ResonanceEnv(R.PATROL_B_D0)
        seq = R.make_part_seq(seed)
        ev, vis = [], {}
        pidx, d, c, s = R.run_life(A, env, seq, 0, 0, R.T_TOTAL, ev, vis)
        if A.body.dead:
            deaths += 1; death_eps.append(d)
        broken_n += len(A.body.broken)
        coll.append(len(ev))
    print("致死量%.1f 耐久%.1f: 死亡%d/%d (死亡ep=%s) 破壊部位計%d hit中央=%.0f" % (
        lethal, durab, deaths, len(seeds), death_eps, broken_n,
        sorted(coll)[len(coll)//2]))

SEEDS = list(range(100, 110))
for lethal, durab in [(25.0, 12.0), (35.0, 15.0), (50.0, 20.0)]:
    sweep(lethal, durab, SEEDS)
