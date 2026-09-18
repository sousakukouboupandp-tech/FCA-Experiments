# -*- coding: utf-8 -*-
"""メモリの内訳を確定させる診断（ChatGPT 返答82の指示）。
  A. 展開終了 / B. eviction後 / C. gc後 / D. snapshot 前後
  + Σ len(cache_value) と line74 の件数を比較
  + State の retain 元を確認
  + successor 数の分布
遷移規則は変えない。
"""
import os, sys, time, itertools, gc, tracemalloc
from collections import Counter
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
KEEP = 1


def rss():
    return _proc.memory_info().rss / 1024**2


def tmc():
    return tracemalloc.get_traced_memory()[0] / 1024**2


def sum_succ(cache):
    return sum(len(v[0]) for v in cache.values())


def main(max_t=20):
    print("メモリの内訳を確定させる診断")
    print(f"範囲 t=0..{max_t}（1層キャッシュ）")
    print()
    tracemalloc.start(10)
    cache = {}
    succ_hist = Counter()
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    snapA = snapB = None
    for t in range(max_t + 1):
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        for i in np.nonzero(cur)[0]:
            mode, e, wnd, ns, nl = dx.from_index(int(i))
            st = (mode, e, wnd, ns, nl)
            acts = [None]
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                acts = list(w.legal_actions(s))
            for a in acts:
                key = (st, a)
                hit = cache.get(key)
                if hit is not None:
                    outs = hit[0]
                else:
                    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
                    rm = w.start_mode(a) if mode == DECISION else mode
                    heals = range(0, wnd + 1)
                    recs = (False, True) if nl in (1, 2) else (False,)
                    if rm in (SEARCH_SMALL, w.SEARCH_LARGE):
                        f_, c_, cb_ = (False, True), (False,), (None,)
                    elif rm == CHASE_4:
                        f_, c_, cb_ = (False,), (False, True), (None,)
                    elif rm == COMBAT:
                        f_, c_, cb_ = (False,), (False,), ("kill", "acute", "wound", "miss")
                    else:
                        f_, c_, cb_ = (False,), (False,), (None,)
                    o = set()
                    for h, rc, fd, ct, cb in itertools.product(heals, recs, f_, c_, cb_):
                        nx = transition(base, a, Draws(heal=h, recover=rc, find=fd,
                                                       catch=ct, combat=cb))
                        if nx.alive:
                            o.add((nx.mode, nx.e, min(nx.w, dx.K_WOUND),
                                   nx.n_small, nx.n_large))
                    outs = tuple(sorted(o))
                    cache[key] = (outs, t)
                    if t == max_t:
                        succ_hist[len(outs)] += 1
                for oo in outs:
                    nxt[dx.to_index(*oo)] = True

        if t == max_t:
            # A. 展開終了
            print("【A 展開終了】")
            r0, c0 = rss(), tmc()
            print(f"  RSS {r0:9.1f}MB  tm現在 {c0:8.1f}MB  entries {len(cache):,}"
                  f"  Σsuccessors {sum_succ(cache):,}")
            # D. snapshot 自身の影響
            print("【D snapshot 自身の影響】")
            print(f"  撮る前 RSS {rss():9.1f}MB")
            snapA = tracemalloc.take_snapshot()
            print(f"  撮った後 RSS {rss():9.1f}MB")
            # B. eviction
            for k in [k for k, v in cache.items() if v[1] <= t - KEEP]:
                del cache[k]
            print("【B eviction後】")
            print(f"  RSS {rss():9.1f}MB  tm現在 {tmc():8.1f}MB  entries {len(cache):,}"
                  f"  Σsuccessors {sum_succ(cache):,}")
            # C. gc
            gc.collect()
            print("【C gc後】")
            print(f"  RSS {rss():9.1f}MB  tm現在 {tmc():8.1f}MB")
            snapB = tracemalloc.take_snapshot()
        cur = nxt

    print()
    print("=== A→B の差分（eviction で何が消えたか）===")
    for stat in snapB.compare_to(snapA, "lineno")[:8]:
        nm = str(stat).split("\\")[-1]
        print(f"  {stat.size_diff/1024**2:+9.2f}MB  {stat.count_diff:+10,}件  {nm}")
    print()
    print("=== successor 数の分布（最終層で新規に作った entry）===")
    tot = sum(succ_hist.values())
    for n in sorted(succ_hist):
        print(f"  successor {n:3d}個: {succ_hist[n]:>8,} ({succ_hist[n]/max(1,tot)*100:5.1f}%)")
    if tot:
        print(f"  中央値あたり: {sorted(succ_hist.elements())[tot//2]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
