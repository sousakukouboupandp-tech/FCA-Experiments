# -*- coding: utf-8 -*-
"""reach_probe2 のプロファイル。exactness は変えない。
測るもの：関数別の累積時間・呼び出し回数、生成した枝の数、一意な次状態の数、
          重複率、キャッシュの当たり外れ、行動別の展開時間。
出力：t=0..13 の各層の統計と、cProfile の上位。
"""
import cProfile, pstats, io, time, sys, os
import itertools
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import psutil

K_WOUND = 10
TAU_MAX = 100
_proc = psutil.Process(os.getpid())

# 計器
STAT = dict(branches=0, step_calls=0, cache_hit=0, cache_miss=0,
            macro_calls=0, landed_raw=0)
ACT_TIME = {}


def cap_w(x):
    return min(x, K_WOUND)


_cache = {}


def step_support(mode, e, wnd, ns, nl):
    STAT["step_calls"] += 1
    key = (mode, e, wnd, ns, nl)
    hit = _cache.get(key)
    if hit is not None:
        STAT["cache_hit"] += 1
        return hit
    STAT["cache_miss"] += 1
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    heals = range(0, wnd + 1)
    recs = (False, True) if nl in (1, 2) else (False,)
    if mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        finds, catches, combats = (False, True), (False,), (None,)
    elif mode == CHASE_4:
        finds, catches, combats = (False,), (False, True), (None,)
    elif mode == COMBAT:
        finds, catches, combats = (False,), (False,), ("kill", "acute", "wound", "miss")
    else:
        finds, catches, combats = (False,), (False,), (None,)
    cont, back = set(), set()
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        STAT["branches"] += 1
        nxt = transition(base, None, Draws(heal=h, recover=rc, find=fd,
                                           catch=ct, combat=cb))
        if not nxt.alive:
            continue
        rec = (nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large)
        if nxt.mode == DECISION:
            back.add(rec)
        else:
            cont.add((nxt.mode,) + rec)
    res = (frozenset(cont), frozenset(back))
    _cache[key] = res
    return res


def macro_support(e, wnd, ns, nl, action, t_room):
    STAT["macro_calls"] += 1
    out = {}
    cur = {(w.start_mode(action), e, cap_w(wnd), ns, nl)}
    for tau in range(1, min(TAU_MAX, t_room) + 1):
        if not cur:
            break
        nxt = set()
        landed = set()
        for (mode, ee, ww, nn, ll) in cur:
            cont, back = step_support(mode, ee, ww, nn, ll)
            nxt |= cont
            landed |= back
        if landed:
            out[tau] = landed
            STAT["landed_raw"] += len(landed)
        cur = nxt
    return out


def run_layers(max_t):
    BUF = TAU_MAX + 1
    buf = [set() for _ in range(BUF)]
    buf[0].add((E_MAX, 0, 12, 3))
    rows = []
    t0 = time.perf_counter()
    for t in range(max_t + 1):
        cur = buf[t % BUF]
        before = dict(STAT)
        ts = time.perf_counter()
        per_act = {}
        for (e, wnd, ns, nl) in cur:
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
            for a in w.legal_actions(s):
                ta = time.perf_counter()
                for tau, landed in macro_support(e, wnd, ns, nl, a, HORIZON - t).items():
                    tgt = t + tau
                    if tgt < HORIZON:
                        buf[tgt % BUF] |= landed
                per_act[a] = per_act.get(a, 0.0) + time.perf_counter() - ta
        dt = time.perf_counter() - ts
        rows.append((t, len(cur), dt,
                     STAT["branches"] - before["branches"],
                     STAT["landed_raw"] - before["landed_raw"],
                     STAT["step_calls"] - before["step_calls"],
                     STAT["cache_hit"] - before["cache_hit"],
                     dict(per_act)))
        buf[t % BUF] = set()
    return rows, time.perf_counter() - t0


def main(max_t=11):
    print("reach_probe2 のプロファイル（exactness は変えない）")
    print("世界 386bf76 ／ solver仕様 a66e9e4 ／ K =", K_WOUND, "／ τ_max =", TAU_MAX)
    print(f"測る範囲: t=0..{max_t}")
    print()

    pr = cProfile.Profile()
    pr.enable()
    rows, total = run_layers(max_t)
    pr.disable()

    print("  t | |S_t| |    層の秒 |     生成枝 |   着地(重複込) | step呼出 | キャッシュ当たり | 重複率")
    for (t, ns, dt, br, landed, calls, hits, per_act) in rows:
        dup = (landed / max(1, ns)) if ns else 0
        rate = (hits / max(1, calls)) * 100
        print(f"{t:3d} | {ns:5,} | {dt:9.2f} | {br:10,} | {landed:14,} |"
              f" {calls:8,} | {rate:14.1f}% | {dup:8.1f}")
    print()
    print("行動別の展開時間（最後の層）")
    for a, sec in sorted(rows[-1][7].items(), key=lambda x: -x[1]):
        print(f"  {a:6s} {sec:8.2f} 秒")
    print()
    print("累計:", {k: f"{v:,}" for k, v in STAT.items()})
    print("キャッシュ件数:", f"{len(_cache):,}")
    print("生成枝 ÷ 着地(重複込) =",
          round(STAT["branches"] / max(1, STAT["landed_raw"]), 2))
    print("全体の秒:", round(total, 1), " RSS %.2fGB" % (_proc.memory_info().rss / 1024**3))
    print()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
    print(s.getvalue())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
