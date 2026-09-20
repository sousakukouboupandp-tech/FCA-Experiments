# -*- coding: utf-8 -*-
"""P1f：mod20 位相構造が「どの種類の switch」に宿っているかを切り分ける。
★物理原因には進まない。観測した構造を分解するだけ★
★新しい統計検定は持ち込まない（凍結していないため）★

対象ごとに E mod 20 の分布を出す
  strict → strict ／ tie-mediated
  g_rel > ε（凍結した7点）
  optimal_set が変わった境界
各カテゴリで ★件数・一様なら何件・余り1と12で何%★ を併記する
（件数の少ないカテゴリは偶然と区別できないため）
"""
import os, sys, json
from collections import Counter
import numpy as np
import dense_index as dx
from P1d_policy_audit import selected_common

D, S = "kernel_full", "solve_D0"
EPS = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # ★凍結値★
NAMES = {0: "-", 1: "草", 2: "小物", 3: "大物"}


def show(label, residues):
    """余りの分布を、件数・期待値・余り1と12の割合つきで出す"""
    n = len(residues)
    if n == 0:
        print(f"  {label:34s} 件数 0 → ★判定不能★")
        return
    c = Counter(residues)
    r1 = c.get(1, 0)
    r12 = c.get(12, 0)
    exp = n / 20
    top = c.most_common(3)
    print(f"  {label:34s} 件数 {n:4d}  一様なら各 {exp:5.1f} 件")
    print(f"    余り1 {r1:4d} ／ 余り12 {r12:4d} ／ "
          f"★2つで {(r1+r12)/n*100:5.1f}%★")
    print(f"    上位3つ: {[(int(k), int(v)) for k, v in top]}")
    print(f"    全分布: {dict(sorted((int(k), int(v)) for k, v in c.items()))}")
    if n < 30:
        print(f"    ★件数が少ないので、偶然との区別はつかない★")


def main():
    print("P1f：mod20 位相構造がどの種類の switch に宿っているか")
    print("★物理原因には進まない。新しい統計検定も持ち込まない★")
    print()
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob, t_prob = A["succ"], A["prob"], A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    V1 = np.load(os.path.join(S, "V1.npy"))
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    acc = np.bincount(owner, weights=prob * (1.0 + V1[succ]), minlength=n_pairs)
    acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    dec = np.nonzero(state_idx < dx.STRIDE_M)[0]
    Q = np.full((dx.STRIDE_M, 4), np.nan, dtype=np.float64)
    Q[state_idx[dec], act_id[dec]] = acc[dec]

    rec = []
    for e in range(1, 2001):
        i = dx.decision_index_of(e, 0, 12, 3)
        q3 = Q[i, 1:]
        best = np.nanmax(q3)
        sel, bits = selected_common(q3, best)
        srt = np.sort(np.nan_to_num(q3, nan=-np.inf))[::-1]
        gap = srt[0] - srt[1]
        g_rel = gap / max(1.0, abs(srt[0]), abs(srt[1]))
        rec.append(dict(e=e, sel=sel, bits=bits, g_rel=g_rel,
                        n_opt=bin(bits).count("1")))

    sws = [(rec[k - 1], rec[k]) for k in range(1, len(rec))
           if rec[k - 1]["sel"] != rec[k]["sel"]]
    strict = [(a, b) for (a, b) in sws if a["n_opt"] == 1 and b["n_opt"] == 1]
    tie = [(a, b) for (a, b) in sws if not (a["n_opt"] == 1 and b["n_opt"] == 1)]
    optchg = [(rec[k - 1], rec[k]) for k in range(1, len(rec))
              if rec[k - 1]["bits"] != rec[k]["bits"]]

    print("【基準：canonical の全 switch】")
    show("全 switch（canonical 170）", [b["e"] % 20 for (a, b) in sws])
    print()
    print("【種類で分けた場合】")
    show("① strict → strict", [b["e"] % 20 for (a, b) in strict])
    print()
    show("② tie-mediated", [b["e"] % 20 for (a, b) in tie])
    print()
    print("【g_rel の閾値で絞った場合（両側とも > ε）】")
    for eps in EPS:
        sub = [b["e"] % 20 for (a, b) in sws
               if a["g_rel"] > eps and b["g_rel"] > eps]
        show(f"g_rel > {eps:.0e}", sub)
        print()
    print("【optimal_set 自体が変わった境界】")
    show("optimal_set の変化", [b["e"] % 20 for (a, b) in optchg])
    print()
    print("【参考：全 E=2..2000 の余りの分布（比較のための基準）】")
    show("全ての境界（1999件）", [e % 20 for e in range(2, 2001)])


if __name__ == "__main__":
    main()
