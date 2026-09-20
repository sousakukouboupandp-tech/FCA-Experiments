# -*- coding: utf-8 -*-
"""P1l：位相レーンの地図（★記述に固定。原因判定には使わない★）
E=1..2000 の canonical policy を20位相に並べ替え、
すでに観測した modular structure の空間配置を可視化する。

出すもの（★これだけ。増やさない★）
  ・canonical selected action の 100×20 地図
  ・optimal_set の 100×20 地図
  ・各 residue ごとの 草/小物/大物 の件数
  ・各 residue 内で、block 方向に action が何回変わるか
  ・CSV に生データ

行：1-20, 21-40, …（block）／列：E mod 20 = 1,2,…,19,0 の順
（余り1が各ブロックの先頭に並ぶので、死亡境界との対応が読みやすい）

★P1l の絵を見て「12がなぜ特殊か」を説明し始めない★
★何か見えたら「P1l で新たに観測したパターン」として保存し、
　原因仮説は次の工程の前に凍結する★
"""
import os, sys, csv
from collections import Counter
import numpy as np
import dense_index as dx

S = "solve_D0"
HORIZON = 1000
COLS = list(range(1, 20)) + [0]        # ★1,2,…,19,0 の順★
MARK = {0: ".", 1: "G", 2: "s", 3: "L"}   # 草=G 小物=s 大物=L
BITS = {0: ".", 1: "G", 2: "s", 3: "a", 4: "L", 5: "b", 6: "c", 7: "*"}
#       -      草     小     草小    大     草大    小大    三つ


def main():
    t, wnd, ns, nl = 0, 0, 12, 3
    print("P1l：位相レーンの地図（★記述のみ。原因判定には使わない★）")
    print(f"  t={t}, W={wnd}, N_S={ns}, N_L={nl}")
    print("  行：E のブロック（1-20, 21-40, …）／列：E mod 20（1,2,…,19,0 の順）")
    print("  記号：G=草  s=小物  L=大物")
    print()
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))

    sel = np.zeros(2001, dtype=np.uint8)
    bit = np.zeros(2001, dtype=np.uint8)
    for e in range(1, 2001):
        i = dx.decision_index_of(e, wnd, ns, nl)
        sel[e] = pol[t][i]
        bit[e] = opt[t][i]

    def emap(block, col):
        """block は 0..99、col は E mod 20 の値。E を返す"""
        return block * 20 + (col if col != 0 else 20)

    # ---- selected の地図
    print("【selected action の地図】")
    print("      " + "".join(f"{c:>2d}" for c in COLS))
    for b in range(100):
        row = "".join(" " + MARK[int(sel[emap(b, c)])] for c in COLS)
        lo, hi = b * 20 + 1, b * 20 + 20
        if b < 12 or b % 10 == 0 or b >= 97:
            print(f"{lo:4d}- {row}")
        elif b == 12:
            print("  …（途中は同じ形が続く。全体は CSV に出す）")
    print()

    # ---- optimal_set の地図（同点がどこにあるか）
    print("【optimal_set の地図】記号：G=草のみ s=小物のみ a=草+小物 L=大物のみ b=草+大物 c=小物+大物 *=三つ")
    print("      " + "".join(f"{c:>2d}" for c in COLS))
    for b in range(100):
        row = "".join(" " + BITS[int(bit[emap(b, c)])] for c in COLS)
        lo = b * 20 + 1
        if b < 12 or b % 20 == 0 or b >= 98:
            print(f"{lo:4d}- {row}")
        elif b == 12:
            print("  …")
    print()

    # ---- residue ごとの集計
    print("【residue ごとの集計】（各 residue に100個の E がある）")
    print("  余り |    草 |  小物 |  大物 | block方向の変化回数 | 同点を含む数")
    for c in COLS:
        es = [emap(b, c) for b in range(100)]
        s = [int(sel[e]) for e in es]
        cnt = Counter(s)
        chg = sum(1 for k in range(1, 100) if s[k] != s[k - 1])
        n_tie = sum(1 for e in es if bin(int(bit[e])).count("1") > 1)
        star = " ★" if c in (1, 12) else "  "
        print(f"  {c:4d}{star}| {cnt.get(1,0):5d} | {cnt.get(2,0):5d} | {cnt.get(3,0):5d}"
              f" | {chg:19d} | {n_tie:12d}")
    print()

    # ---- CSV
    out = os.path.join(S, "P1l_phase_map.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wc = csv.writer(f)
        wc.writerow(["block", "E_lo", "E_hi"] + [f"r{c}" for c in COLS]
                    + [f"opt_r{c}" for c in COLS])
        for b in range(100):
            wc.writerow([b, b * 20 + 1, b * 20 + 20]
                        + [int(sel[emap(b, c)]) for c in COLS]
                        + [int(bit[emap(b, c)]) for c in COLS])
    print(f"CSV: {out}")
    del pol, opt


if __name__ == "__main__":
    main()
