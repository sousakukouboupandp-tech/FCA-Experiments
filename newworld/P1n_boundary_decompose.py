# -*- coding: utf-8 -*-
"""P1n：帯の境界（6/7 と 11/12）を Q の分解で調べる。
★検算で「草k回でブロック維持」は k=1 しか当たらないと分かっている★
  草1回(-6) の境界 6/7   ← 観測と一致
  草2回(-12) の境界 12/13 ← 観測は 11/12（1ずれる）
  草3回(-18) の境界 18/19 ← 観測は r=16（2ずれる）

やること
  Q = Σ p (1 + V_1[s'])  を「この1歩で得る分」と「次の価値の期待」に分ける
  境界の前後（r=5,6,7,8 と r=10,11,12,13）で、何が不連続に変わるかを見る
  ★説明が付かなければ「付かない」と書く★
"""
import os, json
import numpy as np
import world1g as w
from world1g import State, DECISION
import dense_index as dx
from probe_kernel_size import branches_with_prob

D, S = "kernel_full", "solve_D0"
ACTS = [(w.ACT_GRASS, "草"), (w.ACT_SMALL, "小物"), (w.ACT_LARGE, "大物")]


def main():
    print("P1n：帯の境界を Q の分解で調べる")
    print("  Q = Σ p (1 + V_1[s'])  ＝ 1 + Σ p·V_1[s']（生存した枝のみ）")
    print("  ★t=0, W=0, N_S=12, N_L=3★")
    print()
    V1 = np.load(os.path.join(S, "V1.npy"))

    def decompose(e, a):
        """1歩ぶんを分解する。戻り値：即時の寄与／将来の期待／死ぬ確率"""
        nt, tm = branches_with_prob(DECISION, e, 0, 12, 3, a)
        p_alive = sum(p for _, p in nt)
        p_die = sum(p for p, _, _ in tm)
        fut = sum(p * V1[j] for j, p in nt)
        return 1.0, fut, p_die, p_alive

    def row(e):
        out = {}
        for a, nm in ACTS:
            imm, fut, pd, pa = decompose(e, a)
            out[nm] = dict(Q=imm + fut, fut=fut, p_die=pd)
        return out

    for label, rs, blocks in (
            ("【境界 6/7】", (4, 5, 6, 7, 8, 9), (200, 400, 600, 1000)),
            ("【境界 11/12】", (9, 10, 11, 12, 13, 14), (200, 400, 600, 1000))):
        print(label)
        for base in blocks:
            print(f"  ブロック E={base+1}〜{base+20}")
            print("     r |     E |      Q草 |    Q小物 |  草の将来 | 小物の将来 |"
                  " 草の死亡率 | 小物の死亡率 | 選択")
            for r in rs:
                e = base + r
                d = row(e)
                sel = max(d, key=lambda k: d[k]["Q"])
                mark = "★" if r in (6, 7, 11, 12) else "  "
                print(f"  {mark}{r:2d} | {e:5d} | {d['草']['Q']:8.3f} |"
                      f" {d['小物']['Q']:8.3f} | {d['草']['fut']:9.3f} |"
                      f" {d['小物']['fut']:10.3f} | {d['草']['p_die']:10.3e} |"
                      f" {d['小物']['p_die']:12.3e} | {sel}")
            print()

    # ---- 草の行き先が境界で変わるか
    print("【草を食べた行き先の V_1（草は E → E−6）】")
    print("  ★境界の前後で、行き先のブロックが変わるかを見る★")
    print("     r |     E | 行き先 E | 行き先のブロック | V_1(行き先)")
    for base in (600,):
        for r in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            e = base + r
            e2 = e - 6
            i2 = dx.decision_index_of(e2, 0, 12, 3)
            blk_from = (e - 1) // 20
            blk_to = (e2 - 1) // 20
            mark = "★" if blk_from != blk_to else "  "
            print(f"  {mark}{r:2d} | {e:5d} | {e2:8d} | {blk_from}→{blk_to}"
                  f"{'  ★ブロック落ち★' if blk_from != blk_to else '        （同じ）'}"
                  f" | {V1[i2]:.4f}")
    print()

    # ---- 小物の行き先（探索は E→E−20、捕獲で +600）
    print("【小物を選んだ行き先の V_1】")
    print("  探索が外れると E−20（同じ residue、1ブロック下）")
    print("  探索が当たると chase_1 へ（E−20、MODE が変わる）")
    print("     r |     E | 外れ E | V_1(外れ) | 当たり(chase_1) | V_1(当たり)")
    for base in (600,):
        for r in (5, 6, 7, 11, 12, 13):
            e = base + r
            nt, tm = branches_with_prob(DECISION, e, 0, 12, 3, w.ACT_SMALL)
            miss = hit = None
            for j, p in nt:
                m, ee, ww, ns, nl = dx.from_index(j)
                if m == w.SEARCH_SMALL:
                    miss = (ee, V1[j])
                elif m == w.CHASE_1:
                    hit = (ee, V1[j])
            print(f"    {r:2d} | {e:5d} | {miss[0]:6d} | {miss[1]:9.4f} |"
                  f" {hit[0]:15d} | {hit[1]:.4f}")


if __name__ == "__main__":
    main()
