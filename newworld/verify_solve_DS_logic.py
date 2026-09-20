# -*- coding: utf-8 -*-
"""solve_DS.py のロジックを縮小世界に当てて、参照 solver と一致するか確認する。
★実寸を回す前の関門。solve_D0 のときと同じ手順★
"""
import sys
import numpy as np
import mini_world as m
import mini_solver_ref as ref
import mini_solver_dense as den

ATOL, RTOL = 1e-9, 1e-12


def solve_DS_like(horizon):
    """★solve_DS.py と同じ式で縮小世界を解く★"""
    nonterm, term = den.build_tables()
    keys = sorted(nonterm)
    pair_id = {k: i for i, k in enumerate(keys)}
    n_pairs = len(keys)
    owner, succ, prob = [], [], []
    st_idx, act = [], []
    for k in keys:
        i, a = k
        pid = pair_id[k]
        nx_i, nx_p = nonterm[k]
        for j, p in zip(nx_i, nx_p):
            owner.append(pid); succ.append(j); prob.append(p)
        st_idx.append(i)
        act.append(0 if a is None else {m.ACT_GRASS: 1, m.ACT_SMALL: 2,
                                        m.ACT_LARGE: 3}[a])
    owner = np.array(owner, dtype=np.int64); succ = np.array(succ, dtype=np.int64)
    prob = np.array(prob, dtype=np.float64)
    st_idx = np.array(st_idx, dtype=np.int64); act = np.array(act, dtype=np.int64)
    is_dec = act > 0
    dec_pairs = np.nonzero(is_dec)[0]
    dec_state = st_idx[dec_pairs]
    dec_act = act[dec_pairs]

    V_next = np.zeros(den.TOTAL, dtype=np.float64)
    Vs, sels = {}, {}
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # ★生き残った枝が1（天寿）、終端は0★
            acc = np.bincount(owner, weights=prob, minlength=n_pairs)
        else:
            acc = np.bincount(owner, weights=prob * V_next[succ], minlength=n_pairs)
        V_cur = np.zeros(den.TOTAL, dtype=np.float64)
        V_cur[st_idx[~is_dec]] = acc[~is_dec]
        best = np.full(den.STRIDE_M, -np.inf, dtype=np.float64)
        np.maximum.at(best, dec_state, acc[dec_pairs])
        V_cur[:den.STRIDE_M] = np.where(np.isfinite(best), best, 0.0)
        thr = ATOL + RTOL * np.maximum(1.0, np.abs(best[dec_state]))
        tie = acc[dec_pairs] >= best[dec_state] - thr
        bits = np.zeros(den.STRIDE_M, dtype=np.uint8)
        np.bitwise_or.at(bits, dec_state[tie], (1 << (dec_act[tie] - 1)).astype(np.uint8))
        sel = np.where(bits & 1, 1, np.where(bits & 2, 2, np.where(bits & 4, 3, 0)))
        Vs[t] = V_cur.copy(); sels[t] = sel.copy()
        V_next = V_cur
    return Vs, sels


def main(which="A"):
    if which == "B":
        m.HORIZON = 40
    H = m.HORIZON
    print(f"solve_DS のロジックを縮小{which}（H={H}）で検証する")
    print()
    sys.setrecursionlimit(1000000)
    obj = ref.Objective("DS")          # ★参照 solver の D_S-exact★
    value, qvalue, V, Q = ref.solve(obj)
    Vs, sels = solve_DS_like(H)

    states = [(e, wnd, ns, nl)
              for e in range(1, m.E_MAX + 1)
              for wnd in range(m.K_WOUND + 1)
              for ns in range(m.SMALL_FLOOR, m.SMALL_INIT + 1)
              for nl in range(m.LARGE_MAX + 1)]
    mx, arg, n, sel_ng = 0.0, None, 0, 0
    ORDER = {m.ACT_GRASS: 1, m.ACT_SMALL: 2, m.ACT_LARGE: 3}
    for t in range(H):
        for (e, wnd, ns, nl) in states:
            vr = value(m.DECISION, e, wnd, ns, nl, t)
            i = den.idx(den.DEC, e, wnd, ns, nl)
            d = abs(vr - Vs[t][i])
            if d > mx:
                mx, arg = d, (t, e, wnd, ns, nl)
            n += 1
            qs = {a: qvalue(e, wnd, ns, nl, t, a)
                  for a in m.legal_actions(e, wnd, ns, nl)}
            if ORDER[ref.selected(qs)] != int(sels[t][i]):
                sel_ng += 1
    print(f"比較した (状態, 時刻): {n:,}")
    print(f"V の最大の差: {mx:.3e}  argmax {arg}")
    print(f"★selected の不一致: {sel_ng}★")
    init = (m.E_MAX, 0, m.SMALL_INIT, m.LARGE_MAX)
    i0 = den.idx(den.DEC, *init)
    print()
    print(f"初期状態の V（天寿到達確率）: 参照 {value(m.DECISION, *init, 0):.6f}"
          f" / solve_DS式 {Vs[0][i0]:.6f}")
    print("★一致★" if mx < 1e-9 and sel_ng == 0 else "★不一致★")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
