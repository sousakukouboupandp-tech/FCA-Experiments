# -*- coding: utf-8 -*-
"""hit の由来を分解する診断。
(1) 同じ層の中の hit か、過去の層からの hit か
(2) 何層前の entry を使っているか（hit の時間的距離）
遷移規則は変えない。計器だけ。
"""
import os, sys, time, itertools
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
_cache = {}          # key -> (値, 作られた層 t)
STAT = dict(hit_same=0, hit_past=0, miss=0)
DIST = {}            # 何層前か -> 件数


def step_outcomes_diag(x, action, t):
    key = (x, action)
    hit = _cache.get(key)
    if hit is not None:
        val, born = hit
        if born == t:
            STAT["hit_same"] += 1
        else:
            STAT["hit_past"] += 1
            d = t - born
            DIST[d] = DIST.get(d, 0) + 1
        return val
    STAT["miss"] += 1
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
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        nxt = transition(base, action, Draws(heal=h, recover=rc, find=fd,
                                             catch=ct, combat=cb))
        if not nxt.alive:
            continue
        out.add((nxt.mode, nxt.e, min(nxt.w, dx.K_WOUND), nxt.n_small, nxt.n_large))
    res = tuple(sorted(out))
    _cache[key] = (res, t)
    return res


def main(max_t=25):
    print("hit の由来を分解する（同じ層か、過去の層か。過去なら何層前か）")
    print(f"範囲 t=0..{max_t}")
    print()
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    t0 = time.perf_counter()
    for t in range(max_t + 1):
        b = dict(STAT)
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        for i in np.nonzero(cur)[0]:
            mode, e, wnd, ns, nl = dx.from_index(int(i))
            st = (mode, e, wnd, ns, nl)
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    for o in step_outcomes_diag(st, a, t):
                        nxt[dx.to_index(*o)] = True
            else:
                for o in step_outcomes_diag(st, None, t):
                    nxt[dx.to_index(*o)] = True
        hs = STAT["hit_same"] - b["hit_same"]
        hp = STAT["hit_past"] - b["hit_past"]
        ms = STAT["miss"] - b["miss"]
        tot = hs + hp + ms
        if t < 6 or t % 5 == 0 or t == max_t:
            print(f"t={t:4d} 同層hit {hs:>9,} ({hs/max(1,tot)*100:5.1f}%)"
                  f"  過去hit {hp:>9,} ({hp/max(1,tot)*100:5.1f}%)"
                  f"  miss {ms:>8,}  RSS {_proc.memory_info().rss/1024**3:.2f}GB"
                  f"  {time.perf_counter()-t0:.1f}秒", flush=True)
        cur = nxt
    print()
    tot = sum(STAT.values())
    print(f"累計: 同層hit {STAT['hit_same']:,} ({STAT['hit_same']/tot*100:.2f}%)"
          f" / 過去hit {STAT['hit_past']:,} ({STAT['hit_past']/tot*100:.2f}%)"
          f" / miss {STAT['miss']:,} ({STAT['miss']/tot*100:.2f}%)")
    print()
    print("過去hit の時間的距離（何層前の entry を使ったか）")
    tot_p = max(1, sum(DIST.values()))
    acc = 0
    for d in sorted(DIST):
        acc += DIST[d]
        if d <= 10 or d % 5 == 0:
            print(f"  {d:3d}層前: {DIST[d]:>9,} ({DIST[d]/tot_p*100:5.2f}%)"
                  f"  累積 {acc/tot_p*100:6.2f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
