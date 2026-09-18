# -*- coding: utf-8 -*-
"""structural-invalid の内訳を実測で確かめる（ChatGPT 返答94の指摘）。
E=0 の非終端状態が本当に1つも到達していないかを、固定点のマスクで確認する。
"""
import numpy as np
import dense_index as dx
import world1g as w
from world1g import State, Draws, transition, DECISION, E_MAX

print("=== 検証1：E=0 の非終端状態は存在しうるか（ルールから）===")
print("world1g.transition: e_next = min(E_MAX, e - cost + intake)")
print("                    e_next <= 0 なら DEAD_STARVE（終端）")
s = transition(State(e=20, w=0, mode=w.SEARCH_SMALL), None, Draws(find=False))
print(f"  E=20 で消耗20 → E={s.e}, dead={s.dead}")
print("  → E=0 になった歩は必ず終端。非終端の frontier には入らない")
print()

print("=== 検証2：structural-invalid の数（ルールから）===")
per_e0 = 11 * 12 * 4          # 1 MODE の E=0
nl0 = 2001 * 11 * 12          # 1 MODE の N_L=0
e0_nl_pos = 11 * 12 * 3       # N_L>=1 かつ E=0
print(f"  各 MODE の E=0: {per_e0:,}")
print(f"  search_large / combat の N_L=0: {nl0:,}")
print(f"  そのうち N_L>=1 かつ E=0 を足す: {e0_nl_pos:,}")
print()
print("  MODE ごとの structural-invalid")
inv = {}
for m in dx.MODES:
    if m in (w.SEARCH_LARGE, w.COMBAT):
        inv[m] = nl0 + e0_nl_pos
    else:
        inv[m] = per_e0
    print(f"    {m:14s} {inv[m]:>8,}")
tot_inv = sum(inv.values())
print(f"  合計 {tot_inv:,}")
print()
print(f"  structural-valid な primitive 直積: {dx.TOTAL:,} - {tot_inv:,} = {dx.TOTAL - tot_inv:,}")
print()

print("=== 検証3：固定点のマスクで実測 ===")
print("（t=119 まで走らせる必要があるので、ここでは E=0 の到達数だけを別に確認する）")
# E=0 の index を列挙して、reach_variantA の successors が E=0 を返しうるか調べる
from reach_variantA import successors_as_indices
hit_e0 = 0
checked = 0
rng = np.random.default_rng(20260918)
# 適当な状態から successors を取り、E=0 が含まれるかを見る
for _ in range(2000):
    m = dx.MODES[int(rng.integers(len(dx.MODES)))]
    e = int(rng.integers(1, 201))       # 低い体力を狙う
    wn = int(rng.integers(0, 11))
    ns = int(rng.integers(1, 13))
    nl = int(rng.integers(0, 4))
    if m in (w.SEARCH_LARGE, w.COMBAT) and nl == 0:
        continue
    i = dx.to_index(m, e, wn, ns, nl)
    acts = [None]
    if m == DECISION:
        s = State(e=e, w=wn, n_small=ns, n_large=nl, t=0, mode=DECISION)
        acts = list(w.legal_actions(s))
    for a in acts:
        for j in successors_as_indices(i, a):
            checked += 1
            if dx.from_index(j)[1] == 0:
                hit_e0 += 1
print(f"  低い体力の状態から successors を {checked:,} 件調べた")
print(f"  そのうち E=0 だったもの: {hit_e0:,}")
print("  → 0 なら、E=0 は successors に現れない（終端として除かれている）")
