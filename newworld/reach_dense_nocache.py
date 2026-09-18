# -*- coding: utf-8 -*-
"""no-cache 版の診断。遷移規則は一切変えず、memoization だけ外す。
ChatGPT 返答76：メモリと再計算時間の交換比を測る。
"""
import os, sys, time, itertools
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0


def step_outcomes_nocache(x, action=None):
    """reach_probe4.step_outcomes と同じ規則。★キャッシュを持たない★"""
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
    out = []
    n_raw = 0
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        n_raw += 1
        nxt = transition(base, action, Draws(heal=h, recover=rc, find=fd,
                                             catch=ct, combat=cb))
        if not nxt.alive:
            continue
        out.append((nxt.mode, nxt.e, min(nxt.w, dx.K_WOUND), nxt.n_small, nxt.n_large))
    return out, n_raw


def run(max_t, quiet=False):
    import hashlib
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    rows = []
    t0 = time.perf_counter()
    stop = None
    for t in range(max_t + 1):
        idx_dec = np.nonzero(cur[:dx.STRIDE_M])[0]
        h = hashlib.sha256()
        for i in idx_dec:
            _, e, wnd, ns, nl = dx.from_index(int(i))
            h.update(("%d,%d,%d,%d;" % (e, wnd, ns, nl)).encode("ascii"))
        dg = h.hexdigest()[:16]
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        raw = 0
        for i in np.nonzero(cur)[0]:
            mode, e, wnd, ns, nl = dx.from_index(int(i))
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    outs, n_raw = step_outcomes_nocache((mode, e, wnd, ns, nl), a)
                    raw += n_raw
                    for o in outs:
                        nxt[dx.to_index(*o)] = True
            else:
                outs, n_raw = step_outcomes_nocache((mode, e, wnd, ns, nl), None)
                raw += n_raw
                for o in outs:
                    nxt[dx.to_index(*o)] = True
        el = time.perf_counter() - t0
        rss = _proc.memory_info().rss / 1024**3
        rows.append(dict(t=t, frontier=int(cur.sum()), dec=len(idx_dec),
                         digest=dg, raw=raw, uniq=int(nxt.sum()),
                         sec=el, rss=rss))
        if not quiet and (t < 15 or t % 5 == 0 or t == max_t):
            r = rows[-1]
            print(f"t={t:4d} |F_t|={r['frontier']:>9,} |S_t|={r['dec']:>8,}"
                  f" {dg} 枝{raw:>10,} 次{r['uniq']:>9,}"
                  f" RSS {rss:.2f}GB {el:.1f}秒", flush=True)
        if t + 1 >= HORIZON:
            stop = "horizon に到達"; break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた（t={t}）"; break
        if rss > LIMIT_RSS_GB:
            stop = f"RSS が {LIMIT_RSS_GB}GB を超えた（t={t}）"; break
        cur = nxt
    return rows, stop, time.perf_counter() - t0


def main(max_t):
    print("no-cache 版（遷移規則は変更なし。memoization だけ外した）")
    print(f"範囲 t=0..{max_t}")
    print()
    rows, stop, total = run(max_t)
    print()
    print("停止理由:", stop if stop else f"t={max_t} まで完了")
    print("全体の秒: %.2f  RSS %.2fGB" % (total, _proc.memory_info().rss/1024**3))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
