# -*- coding: utf-8 -*-
"""step_outcomes のキャッシュ hit/miss を測る（外す前の診断）。
遷移規則は一切変えない。計器を足すだけ。
"""
import os, sys, time, itertools
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
STAT = dict(hits=0, misses=0)
_cache = {}


def step_outcomes_counted(x, action=None):
    """reach_probe4.step_outcomes と同じ。hit/miss だけ数える"""
    key = (x, action)
    hit = _cache.get(key)
    if hit is not None:
        STAT["hits"] += 1
        return hit
    STAT["misses"] += 1
    mode, e, wnd, ns, nl = x
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    run_mode = w.start_mode(action) if mode == DECISION else mode
    heals = range(0, wnd + 1)
    recs = (False, True) if nl in (1, 2) else (False,)
    if run_mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        finds, catches, combats = (False, True), (False,), (None,)
    elif run_mode == CHASE_4:
        finds, catches, combats = (False,), (False, True), (None,)
    elif run_mode == COMBAT:
        finds, catches, combats = (False,), (False,), ("kill", "acute", "wound", "miss")
    else:
        finds, catches, combats = (False,), (False,), (None,)
    out = set()
    n_raw = 0
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        n_raw += 1
        nxt = transition(base, action, Draws(heal=h, recover=rc, find=fd,
                                             catch=ct, combat=cb))
        if not nxt.alive:
            continue
        out.add((nxt.mode, nxt.e, min(nxt.w, dx.K_WOUND), nxt.n_small, nxt.n_large))
    res = (tuple(sorted(out)), n_raw)
    _cache[key] = res
    return res


def main(max_t=30):
    print("step_outcomes のキャッシュ hit/miss（遷移規則は変更なし）")
    print(f"範囲 t=0..{max_t}")
    print()
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    t0 = time.perf_counter()
    for t in range(max_t + 1):
        b = dict(STAT)
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        for i in np.nonzero(cur)[0]:
            st = dx.from_index(int(i))
            mode, e, wnd, ns, nl = st
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    outs, _ = step_outcomes_counted((mode, e, wnd, ns, nl), a)
                    for o in outs:
                        nxt[dx.to_index(*o)] = True
            else:
                outs, _ = step_outcomes_counted((mode, e, wnd, ns, nl), None)
                for o in outs:
                    nxt[dx.to_index(*o)] = True
        h = STAT["hits"] - b["hits"]
        m = STAT["misses"] - b["misses"]
        if t < 6 or t % 5 == 0 or t == max_t:
            rate = h / max(1, h + m) * 100
            print(f"t={t:4d} hit {h:>9,} miss {m:>9,} hit率 {rate:5.1f}%"
                  f"  cache件数 {len(_cache):>9,}"
                  f"  RSS {_proc.memory_info().rss/1024**3:.2f}GB"
                  f"  {time.perf_counter()-t0:.1f}秒", flush=True)
        cur = nxt
    print()
    tot_h, tot_m = STAT["hits"], STAT["misses"]
    print(f"累計 hit {tot_h:,} / miss {tot_m:,} / hit率 {tot_h/max(1,tot_h+tot_m)*100:.2f}%")
    print(f"cache 件数 {len(_cache):,}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
