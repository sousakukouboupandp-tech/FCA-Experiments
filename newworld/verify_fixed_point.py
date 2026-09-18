# -*- coding: utf-8 -*-
"""primitive frontier の固定点を厳密に確認する（ChatGPT 返答90の指示）。
decision の digest だけでは不十分。★primitive マスク全体の完全一致★を見る。
F_{t+1} = T(F_t) = F_t が一度成立すれば、時間一様な t<999 の区間では
帰納的に同じ集合が続く（1000歩目の終端規則だけは別）。
"""
import os, sys, time, hashlib, itertools
from array import array
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
from reach_variantA import successors_as_indices
import psutil

_proc = psutil.Process(os.getpid())


def digest_mask(mask):
    """マスク全体の決定論的ダイジェスト（bool 配列のバイト列を SHA-256）"""
    return hashlib.sha256(mask.tobytes()).hexdigest()[:16]


def advance(cur, t, cache):
    nxt = np.zeros(dx.TOTAL, dtype=bool)
    for i in np.nonzero(cur)[0]:
        ii = int(i)
        mode = dx.MODES[ii // dx.STRIDE_M]
        acts = [None]
        if mode == DECISION:
            _, e, wnd, ns, nl = dx.from_index(ii)
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
            acts = list(w.legal_actions(s))
        for a in acts:
            key = (ii, a)
            got = cache.get(key)
            if got is None:
                got = array("I")
                got.extend(sorted(successors_as_indices(ii, a)))
                cache[key] = got
            for j in got:
                nxt[j] = True
    return nxt


def main(start_t=118, n_check=4):
    print("primitive frontier の固定点を厳密に確認する")
    print(f"t=0 から {start_t} まで進め、そこから {n_check} 層ぶん")
    print("★primitive マスク全体を直接比較する（decision の digest だけではない）★")
    print()
    cache = {}
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    t0 = time.perf_counter()
    for t in range(start_t):
        cur = advance(cur, t, cache)
    print(f"t={start_t} に到達（{time.perf_counter()-t0:.1f}秒、"
          f"RSS {_proc.memory_info().rss/1024**3:.2f}GB）")
    print()
    prev = cur
    prev_dg = digest_mask(prev)
    print(f"t={start_t:4d}  |F_t|={int(prev.sum()):>9,}  primitiveマスクdigest {prev_dg}")
    for k in range(1, n_check + 1):
        t = start_t + k - 1
        nxt = advance(prev, t, cache)
        dg = digest_mask(nxt)
        same = np.array_equal(prev, nxt)
        print(f"t={t+1:4d}  |F_t|={int(nxt.sum()):>9,}  primitiveマスクdigest {dg}"
              f"  前の層と完全一致: {'★YES★' if same else 'NO'}")
        prev, prev_dg = nxt, dg
    print()
    print("到達可能な primitive 状態:", f"{int(prev.sum()):,}",
          f"/ {dx.TOTAL:,} = {prev.sum()/dx.TOTAL*100:.2f}%")
    dec = int(prev[:dx.STRIDE_M].sum())
    print("到達可能な decision 状態:", f"{dec:,}",
          f"/ {dx.STRIDE_M:,} = {dec/dx.STRIDE_M*100:.2f}%")
    print("到達不能な primitive 状態:", f"{dx.TOTAL - int(prev.sum()):,}")
    print("到達不能な decision 状態:", f"{dx.STRIDE_M - dec:,}")
    # MODE ごとの到達可能数
    print()
    print("MODE ごとの到達可能数")
    for mi, m in enumerate(dx.MODES):
        seg = prev[mi*dx.STRIDE_M:(mi+1)*dx.STRIDE_M]
        print(f"  {m:14s} {int(seg.sum()):>9,} / {dx.STRIDE_M:,}"
              f" = {seg.sum()/dx.STRIDE_M*100:6.2f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 118,
         int(sys.argv[2]) if len(sys.argv) > 2 else 4)
