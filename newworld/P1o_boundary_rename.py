# -*- coding: utf-8 -*-
"""P1o：帯の境界を正しい名称で再記述して、D0 診断を閉じる。
★新しい原因探索はしない。3種類を分けて機械的に確定するだけ★

分ける3種類（ChatGPT / AIスタジオ 両方の指示）
  ① Q 構造の境界：r=6/7（草の遷移先が20ブロックを跨ぐ／跨がない）
       → P1n で Q_grass が跳ぶことを機構まで確認済み
       ★これは必ずしも方策の切替点ではない★
  ② 方策の境界：selected action が実際に変わる場所
  ③ numerical tie 帯：Q_grass ≈ Q_small が連続して成立する領域
"""
import os, json
import numpy as np
import world1g as w
from world1g import DECISION
import dense_index as dx
from probe_kernel_size import branches_with_prob

D, S = "kernel_full", "solve_D0"
COLS = list(range(1, 20)) + [0]
MARK = {0: ".", 1: "G", 2: "s", 3: "L"}
ATOL, RTOL = 1e-9, 1e-12


def rle(seq):
    out = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(a, b) for a, b in out]


def main():
    print("P1o：帯の境界を3種類に分けて再記述する（★D0 診断を閉じる★）")
    print()
    V1 = np.load(os.path.join(S, "V1.npy"))
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(1000, dx.STRIDE_M))

    # ---- ① Q 構造の境界（草の遷移先がブロックを跨ぐか）
    print("【① Q 構造の境界：r=6/7】★P1n で機構確認済み★")
    print("  草は E → E−6。r≤6 なら一つ下の20ブロック、r≥7 なら同じブロック")
    print("  → V_1 が変わり、Q_grass が不連続に跳ぶ")
    print("  ★ただし、これは必ずしも selected policy の切替点ではない★")
    print()

    # ---- ② 方策の境界（6種類の帯を正しい名前で）
    print("【② 方策の境界：6種類の帯を再記述】")
    rows = []
    for b in range(100):
        seq = [int(pol[0][dx.decision_index_of(
            b * 20 + (c if c != 0 else 20), 0, 12, 3)]) for c in COLS]
        rows.append(seq)
    types = {}
    for b, seq in enumerate(rows):
        key = tuple((MARK[v], n) for v, n in rle(seq))
        types.setdefault(key, []).append(b)
    for key in sorted(types, key=lambda k: -len(types[k])):
        blocks = types[key]
        pat = " ".join(f"{m}×{n}" for m, n in key)
        # 境界の residue を計算
        pos, bounds = 0, []
        for (m, n) in key[:-1]:
            pos += n
            bounds.append((COLS[pos - 1], COLS[pos]))
        e_lo, e_hi = blocks[0] * 20 + 1, blocks[-1] * 20 + 20
        print(f"  {len(blocks):3d}行  {pat}   E={e_lo}〜{e_hi}")
        if not bounds:
            print("        policy boundary: なし（帯が1つ）")
        elif len(bounds) == 1:
            print(f"        ★policy boundary = r{bounds[0][0]}/r{bounds[0][1]}★")
        else:
            print(f"        ★main policy boundary = r{bounds[0][0]}/r{bounds[0][1]}★")
            for (a, c) in bounds[1:]:
                print(f"        island boundary     = r{a}/r{c}")
    print()

    # ---- ③ numerical tie 帯
    print("【③ numerical tie 帯：Q_grass ≈ Q_small が連続する領域】")
    print("  各20ブロックで、tie の状態がいくつあるか（草と小物が同点）")
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(1000, dx.STRIDE_M))
    print("  ブロック |     E の範囲 | tie の数/20 | 帯の型")
    prev_desc = None
    for b in range(100):
        n_tie = sum(1 for c in COLS
                    if bin(int(opt[0][dx.decision_index_of(
                        b * 20 + (c if c != 0 else 20), 0, 12, 3)])).count("1") > 1)
        seq = rows[b]
        desc = " ".join(f"{MARK[v]}×{n}" for v, n in rle(seq))
        if b < 10 or b % 10 == 0 or desc != prev_desc:
            print(f"  {b:8d} | {b*20+1:5d}〜{b*20+20:5d} | {n_tie:11d} | {desc}")
        prev_desc = desc
    print()

    # ---- まとめ
    print("【D0 診断の結論】")
    print("  ★確認できたこと★")
    print("   ・W=0 では草が E を6減らすため、E mod 20 の r=6/7 に")
    print("     遷移先ブロックの離散境界が生じる")
    print("   ・P1n の Bellman 分解で、この境界で Q_grass が跳び、")
    print("     その原因が次状態の価値 V_1 の変化であることを確認した")
    print("   ・★この Q の不連続点は、必ずしも selected policy の切替点ではない★")
    print()
    print("  ★未解明として残すもの★")
    print("   ・なぜ主な policy 境界が r11/r12 なのか")
    print("   ・なぜ r16 の島が E=621〜840 付近だけに出るのか")
    print("   ・tie 帯がなぜ広く形成されるのか")
    del pol, opt


if __name__ == "__main__":
    main()
