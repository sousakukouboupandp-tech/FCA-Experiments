# -*- coding: utf-8 -*-
"""P1g：行動の組み合わせ × 位相（同じ断面。★構造の分解のみ。原因解釈はしない★）
P1h：★別の断面での再現（本命）★
      HYPOTHESIS_P1（7131f21）で★見る前に固定した★断面だけを測る
        P1h : W=0, N_S=12, N_L=3、t = 100, 250, 500, 750, 900
        P1i : t=0, N_S=12, N_L=3、W = 1
        P1j : t=0, W=0、(N_S,N_L) = (1,3), (12,1), (1,1)
      測るもの（★これだけ。増やさない★）
        switch 総数／使われた余り／{1,7,12,16,17} の割合／{1,12} の割合／間隔の分布

★policy.dat と optimal_set.dat を読むだけ。再計算しない★
★統計検定は持ち込まない。件数30未満には「偶然と区別できない」と明記★
"""
import os, sys, json
from collections import Counter
import numpy as np
import dense_index as dx

S = "solve_D0"
HORIZON = 1000
PHASES5 = {1, 7, 12, 16, 17}
PHASES2 = {1, 12}
NAMES = {0: "-", 1: "草", 2: "小物", 3: "大物"}


def bits_str(b):
    return "".join(n for k, n in ((1, "草"), (2, "小"), (4, "大")) if b & k) or "-"


def scan(pol, opt, t, wnd, ns, nl):
    """1断面の switch を拾う。policy.dat をそのまま読む（canonical）"""
    sels, bits = [], []
    for e in range(1, 2001):
        i = dx.decision_index_of(e, wnd, ns, nl)
        sels.append(int(pol[t][i]))
        bits.append(int(opt[t][i]))
    sw = []
    for k in range(1, 2000):
        if sels[k] != sels[k - 1]:
            e_right = k + 1
            sw.append(dict(e=e_right, a=sels[k - 1], b=sels[k],
                           ba=bits[k - 1], bb=bits[k],
                           strict=(bin(bits[k-1]).count("1") == 1
                                   and bin(bits[k]).count("1") == 1)))
    return sw


def report(label, sw):
    n = len(sw)
    if n == 0:
        print(f"  {label:28s} switch 0 件 ★判定不能★")
        return
    res = [s["e"] % 20 for s in sw]
    c = Counter(res)
    in5 = sum(1 for r in res if r in PHASES5)
    in2 = sum(1 for r in res if r in PHASES2)
    pos = [s["e"] for s in sw]
    d = np.diff(pos) if len(pos) > 1 else np.array([])
    dc = Counter(d.tolist()) if len(d) else Counter()
    n911 = dc.get(9, 0) + dc.get(11, 0)
    print(f"  {label:28s} switch {n:4d} 件")
    print(f"    使われた余り: {sorted(c)}")
    print(f"    ★{{1,7,12,16,17}} に入る: {in5}/{n} = {in5/n*100:5.1f}%★")
    print(f"    ★{{1,12}} に入る:       {in2}/{n} = {in2/n*100:5.1f}%★")
    print(f"    余りの分布: {dict(sorted((int(k), int(v)) for k, v in c.items()))}")
    if len(d):
        print(f"    間隔 9 or 11: {n911}/{len(d)} = {n911/len(d)*100:5.1f}%"
              f"  ／ 分布 {dict(sorted((int(k), int(v)) for k, v in dc.items()))}")
    if n < 30:
        print("    ★件数30未満。偶然と区別できない★")


def main():
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))

    # ---- P1g：行動の組み合わせ × 位相（t=0 の断面）
    print("【P1g】行動の組み合わせ × 位相（t=0, W=0, N_S=12, N_L=3）")
    print("★構造の分解のみ。原因解釈はしない★")
    sw0 = scan(pol, opt, 0, 0, 12, 3)
    print(f"  switch 合計 {len(sw0)}")
    print()
    pairs = {}
    for s in sw0:
        key = f"{NAMES[s['a']]}→{NAMES[s['b']]}"
        pairs.setdefault(key, []).append(s)
    for key in sorted(pairs, key=lambda k: -len(pairs[k])):
        lst = pairs[key]
        res = Counter(s["e"] % 20 for s in lst)
        st = [s for s in lst if s["strict"]]
        res_st = Counter(s["e"] % 20 for s in st)
        print(f"  {key:10s} {len(lst):4d}件  余り {dict(sorted((int(a),int(b)) for a,b in res.items()))}")
        print(f"             うち strict {len(st):3d}件  余り "
              f"{dict(sorted((int(a),int(b)) for a,b in res_st.items()))}")
    print()

    # ---- P1h：別の時刻
    print("【P1h】★別の時刻での再現（見る前に固定した断面）★")
    print("  W=0, N_S=12, N_L=3 固定")
    report("t=0（基準）", sw0)
    for t in (100, 250, 500, 750, 900):
        report(f"t={t}", scan(pol, opt, t, 0, 12, 3))
    print()

    # ---- P1i：傷を入れる
    print("【P1i】傷を入れた断面（消耗が 20→25 になる）")
    print("  t=0, N_S=12, N_L=3 固定")
    report("W=0（基準）", sw0)
    report("W=1", scan(pol, opt, 0, 1, 12, 3))
    print()

    # ---- P1j：資源を変える
    print("【P1j】資源を変えた断面")
    print("  t=0, W=0 固定")
    report("(N_S,N_L)=(12,3) 基準", sw0)
    for (ns, nl) in ((1, 3), (12, 1), (1, 1)):
        report(f"(N_S,N_L)=({ns},{nl})", scan(pol, opt, 0, 0, ns, nl))
    del pol, opt


if __name__ == "__main__":
    main()
