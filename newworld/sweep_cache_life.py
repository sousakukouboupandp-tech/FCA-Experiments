# -*- coding: utf-8 -*-
"""キャッシュの保持層数をスイープして、時間とメモリの関係を測る。
遷移規則も表現も変えない。★キャッシュの寿命だけを変える★
保持層数 N: 「N層前までの entry を残し、それより古いものを捨てる」
"""
import os, sys, time, itertools, hashlib
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())


def build_step(cache, stat):
    def step(x, action, t):
        key = (x, action)
        hit = cache.get(key)
        if hit is not None:
            stat["hit"] += 1
            return hit[0]
        stat["miss"] += 1
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
        cache[key] = (res, t)
        return res
    return step


def run(max_t, keep_layers):
    """keep_layers: None なら無制限。整数なら「その層数より古い entry を捨てる」"""
    cache, stat = {}, dict(hit=0, miss=0)
    step = build_step(cache, stat)
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    digests = []
    peak = 0.0
    t0 = time.perf_counter()
    for t in range(max_t + 1):
        idx_dec = np.nonzero(cur[:dx.STRIDE_M])[0]
        h = hashlib.sha256()
        for i in idx_dec:
            _, e, wnd, ns, nl = dx.from_index(int(i))
            h.update(("%d,%d,%d,%d;" % (e, wnd, ns, nl)).encode("ascii"))
        digests.append((len(idx_dec), h.hexdigest()[:16]))
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        for i in np.nonzero(cur)[0]:
            mode, e, wnd, ns, nl = dx.from_index(int(i))
            st = (mode, e, wnd, ns, nl)
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    for o in step(st, a, t):
                        nxt[dx.to_index(*o)] = True
            else:
                for o in step(st, None, t):
                    nxt[dx.to_index(*o)] = True
        if keep_layers is not None:
            cutoff = t - keep_layers
            if cutoff >= 0:
                for k in [k for k, v in cache.items() if v[1] <= cutoff]:
                    del cache[k]
        peak = max(peak, _proc.memory_info().rss / 1024**3)
        cur = nxt
    sec = time.perf_counter() - t0
    tot = stat["hit"] + stat["miss"]
    return dict(keep=keep_layers, sec=sec, peak=peak, entries=len(cache),
                hit=stat["hit"], miss=stat["miss"],
                rate=stat["hit"]/max(1, tot)*100, digests=digests)


def main(max_t=25):
    print("キャッシュの保持層数スイープ（遷移規則・表現は不変。寿命だけ変える）")
    print(f"範囲 t=0..{max_t}")
    print()
    base = None
    print(" 保持層数 |    秒 | ピークRSS | 最終entries |   hit率 | digest一致")
    for keep in (1, 2, 4, 8, 16, None):
        r = run(max_t, keep)
        if base is None:
            base = r["digests"]
            same = "（基準）"
        else:
            same = "OK" if r["digests"] == base else "★不一致★"
        label = "無制限" if keep is None else f"{keep:>2d}層"
        print(f" {label:>8s} | {r['sec']:5.1f} | {r['peak']:8.2f}GB |"
              f" {r['entries']:>11,} | {r['rate']:6.2f}% | {same}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
