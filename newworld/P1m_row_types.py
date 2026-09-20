# -*- coding: utf-8 -*-
"""P1m：100行を機械的に型分類する（★記述のみ。原因説明はしない★）
★先に「A/B/C のはず」と決めない。run-length で全数列挙する★

出すもの
  ・各20ブロックの行を run-length 符号化
  ・存在する型の全数列挙（どの型が何行あるか）
  ・各型について：開始E の範囲／左側の行動／帯の終端 residue／島の有無／行内の switch 数
  ・★r=16 の小物島の有無を明示的に数える★（ChatGPT の指摘。前回の報告から落ちていた）
"""
import os, csv
from collections import Counter, defaultdict
import numpy as np
import dense_index as dx

S = "solve_D0"
COLS = list(range(1, 20)) + [0]
MARK = {0: ".", 1: "G", 2: "s", 3: "L"}


def rle(seq):
    """run-length 符号化。[(値, 長さ), ...]"""
    out = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(a, b) for a, b in out]


def main():
    print("P1m：100行の型分類（★記述のみ。原因説明はしない★）")
    print("  行：E のブロック（1-20, 21-40, …）／列：E mod 20（1,2,…,19,0）")
    print()
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(1000, dx.STRIDE_M))
    rows = []
    for b in range(100):
        seq = []
        for c in COLS:
            e = b * 20 + (c if c != 0 else 20)
            seq.append(int(pol[0][dx.decision_index_of(e, 0, 12, 3)]))
        rows.append(seq)

    # ---- 型の全数列挙（run-length の形をそのまま型とする）
    types = defaultdict(list)
    for b, seq in enumerate(rows):
        key = tuple((MARK[v], n) for v, n in rle(seq))
        types[key].append(b)

    print(f"【存在する型：{len(types)} 種類】")
    print()
    for key in sorted(types, key=lambda k: -len(types[k])):
        blocks = types[key]
        pat = " ".join(f"{m}×{n}" for m, n in key)
        e_lo = blocks[0] * 20 + 1
        e_hi = blocks[-1] * 20 + 20
        n_sw = len(key) - 1
        print(f"  {len(blocks):3d}行  {pat}")
        print(f"        E の範囲 {e_lo}〜{e_hi}（block {blocks[0]}〜{blocks[-1]}）"
              f"  行内の switch {n_sw}")
        # 帯の境界がどの residue にあるか
        pos = 0
        bounds = []
        for (m, n) in key[:-1]:
            pos += n
            bounds.append(COLS[pos - 1])       # 帯の最後の residue
        if bounds:
            print(f"        帯の終端 residue: {bounds}")
        # 連続したブロックかどうか
        if blocks == list(range(blocks[0], blocks[-1] + 1)):
            print(f"        ★連続したブロック★")
        else:
            gaps = [blocks[i+1] - blocks[i] for i in range(len(blocks)-1)]
            if len(set(gaps)) == 1:
                print(f"        ★等間隔（{gaps[0]} ブロックおき）★ 先頭5つ {blocks[:5]}")
            else:
                print(f"        飛び飛び。先頭10 {blocks[:10]}")
        print()

    # ---- r=16 の小物島（ChatGPT の指摘）
    print("【r=16 の小物島】★前回の報告から落ちていた項目★")
    isl = []
    for b, seq in enumerate(rows):
        i16 = COLS.index(16)
        if seq[i16] == 2:      # 小物
            left = seq[i16 - 1] if i16 > 0 else None
            right = seq[i16 + 1] if i16 + 1 < 20 else None
            isl.append((b, b * 20 + 16, MARK.get(left), MARK.get(right)))
    print(f"  r=16 が小物の行: {len(isl)} 行")
    for (b, e, l, r) in isl:
        print(f"    block {b:3d}（E={e:5d}）  左 {l} / r16=s / 右 {r}"
              f"{'  ★島（両隣が草）★' if l == 'G' and r == 'G' else ''}")
    print()

    # ---- 行内の switch 数の分布
    print("【行内の switch 数の分布】")
    c = Counter(len(rle(seq)) - 1 for seq in rows)
    print(f"  {dict(sorted(c.items()))}")
    tot = sum(k * v for k, v in c.items())
    print(f"  行内の switch の合計 {tot}")
    print(f"  ★1次元の canonical switch は170。差 {170 - tot} が"
          f"「行をまたぐ切り替わり」★")
    del pol


if __name__ == "__main__":
    main()
