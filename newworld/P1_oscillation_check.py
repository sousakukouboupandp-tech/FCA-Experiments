# -*- coding: utf-8 -*-
"""P1：体力方向の振動が「実質的な差」か「numerical tie 近傍」かを判定する。
（ChatGPT 返答113の順序）
★V0.npy から Q を再計算する（方策の bit だけでは Q が分からない）★
t=500 の各体力について Q_grass / Q_small / Q_large を出し、
  ・1位と2位の差（絶対と g_rel）
  ・numerical tie の閾値との比較
  ・ε_near の7点（凍結値）での near-tie 分類
を並べる。
"""
import os, sys, json
import numpy as np
import world1g as w
from world1g import State, DECISION
import dense_index as dx
from probe_kernel_size import branches_with_prob
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"
ATOL, RTOL = 1e-9, 1e-12
EPS_NEAR = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # ★凍結値★
T = 500


def main():
    print("P1：体力方向の振動が実質的な差か numerical tie 近傍かを判定する")
    print(f"  t={T}, W=0, N_S=12, N_L=3 の断面")
    print()
    # V_{t+1} が必要。solve_D0 は V_0 だけ保存している。
    # → t=500 の Q を出すには V_501 が必要なので、ここでは
    #   ★V0.npy（t=0 の価値）を使い、t=0 の断面で判定する★
    V0 = np.load(os.path.join(S, "V0.npy"))
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(1000, dx.STRIDE_M))
    print("※solve_D0 は V_0 のみ保存。t=500 の Q には V_501 が必要なので、")
    print("　ここでは★t=0 の断面★で判定する（振動は t=0 でも出ているか確認する）")
    print()

    # t=0 の方策を見る（振動があるか）
    print("【t=0 の体力ごとの行動（W=0, N_S=12, N_L=3）】切り替わりだけ")
    prev, n_sw = None, 0
    sw = []
    for e in range(1, 2001):
        a = int(pol[0][dx.decision_index_of(e, 0, 12, 3)])
        if a != prev:
            sw.append((e, a))
            n_sw += 1
            prev = a
    print(f"  切り替わりの回数: {n_sw}")
    for (e, a) in sw[:12]:
        print(f"    E={e:5d} → {['（無効）','草','小物','大物'][a]}")
    if len(sw) > 12:
        print(f"    …（残り {len(sw)-12} 回）")
    print()

    # t=0 の Q を V_1 から再計算したいが V_1 がない。
    # ★代わりに「t=999 の Q」を出す（V_1000 は参照しないので V なしで計算できる）★
    print("【t=999 の Q（V_1000 を参照しないので V なしで計算できる）】")
    print("   体力 |      Q草 |    Q小物 |    Q大物 | 1位-2位 |    g_rel | tie閾値 | near-tie?")
    rows = []
    for e in list(range(1, 60, 4)) + [500, 501, 512, 521, 1000, 1001, 1012, 1021, 2000]:
        qs = {}
        for a, nm in ((w.ACT_GRASS, "草"), (w.ACT_SMALL, "小物"), (w.ACT_LARGE, "大物")):
            nt, tm = branches_with_prob(DECISION, e, 0, 12, 3, a)
            # t=999：非終端も天寿。報酬は「その歩を生きたら+1」
            qs[nm] = sum(p for _, p in nt) + sum(p for p, _, _ in tm)
        v = sorted(qs.values(), reverse=True)
        gap = v[0] - v[1]
        denom = max(1.0, abs(v[0]), abs(v[1]))
        g_rel = gap / denom
        thr = ATOL + RTOL * denom
        near = [f"{eps:.0e}" for eps in EPS_NEAR if g_rel <= eps]
        rows.append((e, qs, gap, g_rel, thr, near))
        print(f" {e:6d} | {qs['草']:8.5f} | {qs['小物']:8.5f} | {qs['大物']:8.5f}"
              f" | {gap:7.1e} | {g_rel:8.1e} | {thr:7.1e}"
              f" | {'|'.join(near) if near else '（なし）'}")
    print()
    print("【判定】")
    n_tie = sum(1 for r in rows if r[2] <= r[4])
    print(f"  numerical tie の閾値以下: {n_tie} / {len(rows)}")
    n_near = sum(1 for r in rows if r[3] <= 1e-3)
    print(f"  g_rel <= 1e-3（★1歩未満の差に相当★）: {n_near} / {len(rows)}")
    del pol


if __name__ == "__main__":
    main()
