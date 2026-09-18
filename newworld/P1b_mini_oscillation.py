# -*- coding: utf-8 -*-
"""P1改：縮小世界で体力方向の振動が再現するかを見る。
★縮小世界なら Q を全部持っているので、差の大きさが直接測れる★
縮小B（H=40、狩りが必須）で、t を何点か取り、体力ごとの Q を並べる。
"""
import sys
import numpy as np
import mini_world as m
import mini_solver_ref as ref

ATOL, RTOL = ref.ATOL, ref.RTOL
EPS_NEAR = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ORDER = [m.ACT_GRASS, m.ACT_SMALL, m.ACT_LARGE]


def main():
    m.HORIZON = 40          # 縮小B
    sys.setrecursionlimit(1000000)
    print("P1改：縮小B（H=40）で体力方向の振動が再現するか")
    print(f"  体力上限 {m.E_MAX}／消耗 {m.UPKEEP}＋{m.WOUND_COST}W／草 {m.GRASS_GAIN}")
    print(f"  ★草の正味 {m.GRASS_GAIN - m.UPKEEP}/歩／消耗の周期 {m.UPKEEP}★")
    print()
    obj = ref.Objective("D0")
    value, qvalue, V, Q = ref.solve(obj)

    for t in (0, 10, 20):
        print(f"【t={t}, W=0, N_S=3, N_L=2】")
        print("   体力 |     Q草 |   Q小物 |   Q大物 | 選択 | 1位-2位 |    g_rel | tie?")
        prev = None
        n_sw = 0
        gaps = []
        for e in range(1, m.E_MAX + 1):
            qs = {a: qvalue(e, 0, 3, 2, t, a)
                  for a in m.legal_actions(e, 0, 3, 2)}
            sel = ref.selected(qs)
            v = sorted(qs.values(), reverse=True)
            gap = v[0] - v[1]
            denom = max(1.0, abs(v[0]), abs(v[1]))
            g_rel = gap / denom
            thr = ATOL + RTOL * denom
            gaps.append((e, gap, g_rel, gap <= thr))
            if sel != prev:
                n_sw += 1
                if n_sw <= 14:
                    print(f" {e:6d} | {qs.get(m.ACT_GRASS, float('nan')):7.4f}"
                          f" | {qs.get(m.ACT_SMALL, float('nan')):7.4f}"
                          f" | {qs.get(m.ACT_LARGE, float('nan')):7.4f}"
                          f" | {sel:5s} | {gap:7.1e} | {g_rel:8.1e}"
                          f" | {'★tie★' if gap <= thr else ''}")
                prev = sel
        print(f"  ★切り替わりの回数 {n_sw}★")
        tie_n = sum(1 for g in gaps if g[3])
        near_n = sum(1 for g in gaps if g[2] <= 1e-3)
        print(f"  numerical tie: {tie_n}/{len(gaps)}"
              f"  ／ g_rel<=1e-3: {near_n}/{len(gaps)}")
        # 切り替わりの周期を見る
        sws = []
        prev = None
        for e in range(1, m.E_MAX + 1):
            qs = {a: qvalue(e, 0, 3, 2, t, a) for a in m.legal_actions(e, 0, 3, 2)}
            s = ref.selected(qs)
            if s != prev:
                sws.append(e)
                prev = s
        if len(sws) > 2:
            d = np.diff(sws)
            print(f"  切り替わりの間隔: 最小{d.min()} 最大{d.max()} 中央{int(np.median(d))}")
            print(f"    （★消耗 {m.UPKEEP} と一致するか★）")
        print()


if __name__ == "__main__":
    main()
