# -*- coding: utf-8 -*-
"""Variant A：正本の transition を呼び、返った State を即座に index へ符号化する。
★世界の物理は world1g.py のまま。表現の変更のみ。★

・reachability_support_cache：index -> unique successor indices
  ★確率を潰してよいのは reachability だけ。solver の遷移表に流用しない★
・unique 化は cache 格納前に行う（現在の semantics を維持）
・値の表現を3種類で比較できるようにする（tuple / array('I') / packed bytes）
"""
import os, sys, time, itertools, hashlib
from array import array
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0

# ★successor index は 32bit unsigned で保存する。環境依存にしない★
assert array("I").itemsize == 4, "array('I') が4バイトでない環境です"
assert dx.TOTAL - 1 < 2**32, "index が 32bit に収まりません"


def _draw_sets(run_mode, wnd, nl):
    heals = range(0, wnd + 1)
    recs = (False, True) if nl in (1, 2) else (False,)
    if run_mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        return heals, recs, (False, True), (False,), (None,)
    if run_mode == CHASE_4:
        return heals, recs, (False,), (False, True), (None,)
    if run_mode == COMBAT:
        return heals, recs, (False,), (False,), ("kill", "acute", "wound", "miss")
    return heals, recs, (False,), (False,), (None,)


def successors_as_indices(idx, action):
    """★正本の transition を呼び、返った State を即座に index へ符号化する★
    戻り値：unique な successor index の並び（重複は潰す）"""
    mode, e, wnd, ns, nl = dx.from_index(idx)
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    run_mode = w.start_mode(action) if mode == DECISION else mode
    heals, recs, finds, catches, combats = _draw_sets(run_mode, wnd, nl)
    out = set()
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        nx = transition(base, action, Draws(heal=h, recover=rc, find=fd,
                                            catch=ct, combat=cb))
        if nx.alive:
            # ★State を保持せず、その場で index に変える★
            out.add(dx.to_index(nx.mode, nx.e, nx.w, nx.n_small, nx.n_large))
    return out


def pack(vals, kind):
    if kind == "tuple":
        return tuple(sorted(vals))
    if kind == "array":
        a = array("I")
        a.extend(sorted(vals))
        return a
    if kind == "bytes":
        a = array("I")
        a.extend(sorted(vals))
        return a.tobytes()
    raise ValueError(kind)


def unpack(v, kind):
    if kind == "bytes":
        a = array("I")
        a.frombytes(v)
        return a
    return v


def run(max_t, kind="array", keep_layers=None, quiet=True):
    """kind: 'tuple' / 'array' / 'bytes'
    keep_layers: None なら無制限。整数ならその層数より古い entry を捨てる"""
    reachability_support_cache = {}      # ★用途を限定した名前★
    stat = dict(hit=0, miss=0)
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    rows = []
    peak = 0.0
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
                got = reachability_support_cache.get(key)
                if got is not None:
                    stat["hit"] += 1
                    succ = unpack(got[0], kind)
                else:
                    stat["miss"] += 1
                    succ_set = successors_as_indices(ii, a)
                    packed = pack(succ_set, kind)
                    reachability_support_cache[key] = (packed, t)
                    succ = unpack(packed, kind)
                for j in succ:
                    nxt[j] = True
        if keep_layers is not None:
            cutoff = t - keep_layers
            if cutoff >= 0:
                for k in [k for k, v in reachability_support_cache.items()
                          if v[1] <= cutoff]:
                    del reachability_support_cache[k]
        el = time.perf_counter() - t0
        r = _proc.memory_info().rss / 1024**3
        peak = max(peak, r)
        rows.append(dict(t=t, frontier=int(cur.sum()), dec=len(idx_dec),
                         digest=dg, uniq=int(nxt.sum()), sec=el, rss=r))
        if not quiet and (t < 15 or t % 5 == 0 or t == max_t):
            print(f"t={t:4d} |F_t|={rows[-1]['frontier']:>9,} |S_t|={len(idx_dec):>8,}"
                  f" {dg} 次{rows[-1]['uniq']:>9,} RSS {r:.2f}GB {el:.1f}秒", flush=True)
        if t + 1 >= HORIZON:
            stop = "horizon に到達"; break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた（t={t}）"; break
        if r > LIMIT_RSS_GB:
            stop = f"RSS が {LIMIT_RSS_GB}GB を超えた（t={t}）"; break
        cur = nxt
    tot = stat["hit"] + stat["miss"]
    return dict(rows=rows, stop=stop, sec=time.perf_counter() - t0, peak=peak,
                entries=len(reachability_support_cache),
                hit_rate=stat["hit"] / max(1, tot) * 100)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "array"
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    keep = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "inf" else None
    print(f"Variant A（表現：{kind}、保持層数：{keep if keep else '無制限'}）")
    r = run(mt, kind, keep, quiet=False)
    print()
    print("停止:", r["stop"] or f"t={mt} まで完了")
    print("秒 %.1f / ピークRSS %.2fGB / entries %s / hit率 %.2f%%"
          % (r["sec"], r["peak"], f"{r['entries']:,}", r["hit_rate"]))
