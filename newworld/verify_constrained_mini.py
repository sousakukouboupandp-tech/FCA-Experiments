# -*- coding: utf-8 -*-
"""constrained_passes.core を縮小世界で検証する。
比較相手：mini_world.branches を直接たどる素朴な再帰（★dense 表・bincount を使わない別経路★）
全時刻・全 decision 状態で8種類の値（P_max/P_min/P_eval, V_DS/T_max/T_min/T_eval/T_maxp）を比べる。
"""
import sys
from functools import lru_cache
import numpy as np
import mini_world as m
import mini_solver_dense as den
from constrained_passes import core

ATOL, RTOL = 1e-9, 1e-12
BIT = {m.ACT_GRASS: 1, m.ACT_SMALL: 2, m.ACT_LARGE: 4}
CODE = {None: 0, m.ACT_GRASS: 1, m.ACT_SMALL: 2, m.ACT_LARGE: 3}
DECODE = {1: m.ACT_GRASS, 2: m.ACT_SMALL, 3: m.ACT_LARGE}


def build_K():
    nonterm, term = den.build_tables()
    keys = sorted(nonterm, key=lambda k: (k[0], CODE[k[1]]))
    owner, succ, prob, t_owner, t_prob, sidx, aid = [], [], [], [], [], [], []
    for pid, k in enumerate(keys):
        i, a = k
        nx_i, nx_p = nonterm[k]
        owner += [pid] * len(nx_i); succ += list(nx_i); prob += list(nx_p)
        for (p, c, r) in term[k]:
            t_owner.append(pid); t_prob.append(p)
        sidx.append(i); aid.append(CODE[a])
    return dict(owner=np.array(owner, np.int64), succ=np.array(succ, np.int64),
                prob=np.array(prob), t_owner=np.array(t_owner, np.int64),
                t_prob=np.array(t_prob), state_idx=np.array(sidx, np.int64),
                act_id=np.array(aid, np.uint8), n_pairs=len(keys),
                total=den.TOTAL, n_dec=den.STRIDE_M)


def mini_solve(K, H, obj):
    """solve_D0/solve_DS と同じ規則で bits と sel を作る"""
    owner, succ, prob = K["owner"], K["succ"], K["prob"]
    n, n_dec, sidx = K["n_pairs"], K["n_dec"], K["state_idx"]
    alive = np.bincount(owner, weights=prob, minlength=n)
    term = np.bincount(K["t_owner"], weights=K["t_prob"], minlength=n)
    isd = sidx < n_dec; dp = np.nonzero(isd)[0]; ds = sidx[dp]; da = K["act_id"][dp].astype(np.int64)
    V = np.zeros(K["total"]); bits_t, sel_t = {}, {}
    for t in range(H - 1, -1, -1):
        if obj == "D0":
            acc = alive + term if t == H - 1 else \
                np.bincount(owner, weights=prob * V[succ], minlength=n) + alive + term
        else:
            acc = alive.copy() if t == H - 1 else np.bincount(owner, weights=prob * V[succ], minlength=n)
        Vc = np.zeros(K["total"]); Vc[sidx[~isd]] = acc[~isd]
        best = np.full(n_dec, -np.inf); np.maximum.at(best, ds, acc[dp])
        Vc[:n_dec] = np.where(np.isfinite(best), best, 0.0)
        thr = ATOL + RTOL * np.maximum(1.0, np.abs(best[ds]))
        tie = acc[dp] >= best[ds] - thr
        bits = np.zeros(n_dec, np.uint8)
        np.bitwise_or.at(bits, ds[tie], (1 << (da[tie] - 1)).astype(np.uint8))
        sel = np.where(bits & 1, 1, np.where(bits & 2, 2, np.where(bits & 4, 3, 0))).astype(np.uint8)
        bits_t[t], sel_t[t] = bits, sel; V = Vc
    return bits_t, sel_t


def reference(H, bits_t, sel_t, kind, agg):
    """素朴な再帰。kind: 'P'（天寿）/'T'（寿命）。agg: max/min/eval/vmax/maxp"""
    vds = None
    if agg == "maxp":
        vds = reference(H, bits_t, sel_t, "P", "vmax")

    @lru_cache(maxsize=None)
    def val(t, st):
        mode, e, wn, ns, nl = st
        if mode == m.DECISION:
            i = den.idx(den.DEC, e, wn, ns, nl)
            legal = m.legal_actions(e, wn, ns, nl)
            if agg == "vmax":
                cands = legal
            elif agg == "eval":
                cands = [DECODE[int(sel_t[t][i])]]
            elif agg == "maxp":
                if vds(t, st) == 0.0:
                    cands = [a for a in legal if bits_t[t][i] & BIT[a]]
                else:
                    cands = [DECODE[int(sel_t[t][i])]]
            else:
                cands = [a for a in legal if bits_t[t][i] & BIT[a]]
            vals = [q(t, st, a) for a in cands]
            return min(vals) if agg == "min" else max(vals)
        return q(t, st, None)

    def q(t, st, a):
        s = 0.0
        for p, nxt, cause, raw in m.branches(*st, t, a):
            if kind == "P":
                s += p * ((1.0 if cause == m.TENJU else 0.0) if nxt is None else val(t + 1, nxt))
            else:
                s += p * (1.0 + (0.0 if nxt is None else val(t + 1, nxt)))
        return s
    return val


def main(which):
    sys.setrecursionlimit(1000000)
    if which == "B":
        m.HORIZON = 40
    H = m.HORIZON
    print(f"縮小{which}（H={H}）で constrained_passes.core を検証する")
    K = build_K()
    init = den.idx(den.DEC, m.E_MAX, 0, m.SMALL_INIT, m.LARGE_MAX)
    states = [(e, wn, ns, nl) for e in range(1, m.E_MAX + 1) for wn in range(m.K_WOUND + 1)
              for ns in range(m.SMALL_FLOOR, m.SMALL_INIT + 1) for nl in range(m.LARGE_MAX + 1)]
    worst = 0.0; n = 0
    for side, obj, kind, pairs in (
            ("D0", "D0", "P", [("P_max", "max"), ("P_min", "min"), ("P_eval", "eval")]),
            ("DS", "DS", "T", [("V_DS", "vmax"), ("T_max", "max"), ("T_min", "min"),
                               ("T_eval", "eval"), ("T_maxp", "maxp")])):
        bits_t, sel_t = mini_solve(K, H, obj)
        res, hist = core(K, H, lambda t: bits_t[t], lambda t: sel_t[t], side, init,
                         keep_all=True, log=lambda s: None)
        for name, agg in pairs:
            kd = "P" if name == "V_DS" else kind
            ref = reference(H, bits_t, sel_t, kd, agg)
            mx = 0.0
            for t in range(H):
                for (e, wn, ns, nl) in states:
                    r = ref(t, (m.DECISION, e, wn, ns, nl))
                    d = abs(r - hist[name][t][den.idx(den.DEC, e, wn, ns, nl)])
                    mx = max(mx, d / max(1.0, abs(r))); n += 1
            worst = max(worst, mx)
            print(f"  {side} {name:7s} 初期値 {res[name]:.9f}  最大相対差 {mx:.3e}")
    print(f"比較した値 {n:,} 件  ★最大相対差 {worst:.3e}★  {'★一致★' if worst < 1e-9 else '★不一致★'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "B")
