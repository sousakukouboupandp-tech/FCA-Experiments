# -*- coding: utf-8 -*-
"""較正チェック1: 前半生の死亡率と限界値の相場観（予備実験(e)の前哨）"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import recon_v01 as R

print("=== 前半500ep: 生死・衝突・部位蓄積 ===")
first_half_deaths = 0
for seed in range(100, 110):
    A = R.ReconAgent(seed)
    env = R.ResonanceEnv(R.PATROL_B_D0)
    seq = R.make_part_seq(seed)
    ev, vis = [], {}
    pidx, d, c, s = R.run_life(A, env, seq, 0, 0, 500, ev, vis)
    if A.body.dead: first_half_deaths += 1
    dmg = {k: round(v, 1) for k, v in A.body.dmg.items()}
    print("seed%d: 死=%s(%s, ep%s) 衝突ep=%d hit=%d 蓄積=%s" % (
        seed, A.body.dead, A.body.death_part, d, c, len(ev), dmg))
print("前半死亡: %d/10" % first_half_deaths)
