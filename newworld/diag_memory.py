# -*- coding: utf-8 -*-
"""メモリの内訳を tracemalloc で測る（1層キャッシュ）。
ChatGPT 返答80の指示：RSS / tracemalloc current / peak / entries を
  層の開始直後・展開終了直後・clear 後・gc 後 の4点で記録する。
遷移規則は変えない。
"""
import os, sys, time, itertools, gc, tracemalloc
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
KEEP = 1          # 1層キャッシュ


def rss():
    return _proc.memory_info().rss / 1024**2      # MB


def tm():
    c, p = tracemalloc.get_traced_memory()
    return c / 1024**2, p / 1024**2               # MB


def main(max_t=20):
    print("メモリの内訳（1層キャッシュ、tracemalloc）")
    print(f"範囲 t=0..{max_t}")
    print()
    tracemalloc.start(10)
    cache = {}
    n_succ = 0
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    snap_before = None
    for t in range(max_t + 1):
        show = (t in (5, 10, 15, max_t))
        if show:
            c, p = tm()
            print(f"t={t:3d} 層の開始  RSS {rss():8.1f}MB  tm現在 {c:8.1f}MB  tm最大 {p:8.1f}MB")
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
                    n_succ += len(outs)
                for oo in outs:
                    nxt[dx.to_index(*oo)] = True
        if show:
            c, p = tm()
            print(f"        展開終了  RSS {rss():8.1f}MB  tm現在 {c:8.1f}MB  tm最大 {p:8.1f}MB"
                  f"  entries {len(cache):,}")
            snap_before = tracemalloc.take_snapshot()
        # 1層より古いものを捨てる
        for k in [k for k, v in cache.items() if v[1] <= t - KEEP]:
            del cache[k]
        if show:
            c, p = tm()
            print(f"        clear後   RSS {rss():8.1f}MB  tm現在 {c:8.1f}MB"
                  f"  entries {len(cache):,}")
            gc.collect()
            c, p = tm()
            print(f"        gc後      RSS {rss():8.1f}MB  tm現在 {c:8.1f}MB", flush=True)
        cur = nxt

    print()
    print("累計の successor 生成数:", f"{n_succ:,}")
    print()
    print("=== tracemalloc 上位10行（最終層の展開終了時点） ===")
    for stat in snap_before.statistics("lineno")[:10]:
        print(f"  {stat.size/1024**2:8.2f}MB  {stat.count:>9,}件  {stat}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
