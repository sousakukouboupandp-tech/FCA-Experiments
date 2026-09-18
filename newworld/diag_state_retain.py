# -*- coding: utf-8 -*-
"""State オブジェクトの retain 元を特定する。
gc で生きている State を数え、参照元の型を見る。推測しない。
"""
import gc, sys, itertools
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx


def count_states():
    return sum(1 for o in gc.get_objects() if isinstance(o, State))


def build(max_t):
    cache = {}
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
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
                for oo in outs:
                    nxt[dx.to_index(*oo)] = True
        cur = nxt
    return cache


print("State の retain 元を特定する")
print()
cache = build(14)
gc.collect()
n1 = count_states()
print(f"展開後、生きている State の数: {n1:,}")
print(f"cache の entries: {len(cache):,}")
print()

# 生きている State を数個拾って、参照元の型を見る
states = [o for o in gc.get_objects() if isinstance(o, State)]
print("参照元の型（先頭3個の State について）")
for s in states[:3]:
    refs = gc.get_referrers(s)
    kinds = {}
    for r in refs:
        kinds[type(r).__name__] = kinds.get(type(r).__name__, 0) + 1
    print(f"  {s}")
    print(f"    参照元: {kinds}")
    for r in refs[:2]:
        if isinstance(r, (list, tuple)):
            print(f"    → {type(r).__name__} の中身の型:",
                  {type(x).__name__ for x in r[:8]})
        elif isinstance(r, dict):
            print(f"    → dict のキーの型:", {type(k).__name__ for k in list(r)[:5]})
print()

# cache を消したら State が減るか
cache.clear()
gc.collect()
n2 = count_states()
print(f"cache.clear() 後の State の数: {n2:,}  （{n1:,} から {n1-n2:,} 減）")
