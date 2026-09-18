# -*- coding: utf-8 -*-
"""縮小世界の最適化 dense solver。
★変えるのは：状態表現／枝の格納形式／Bellman の計算方法★
★変えないのは：世界の物理式（mini_world.branches をそのまま使って事前計算する）★

事前計算する遷移表（CSR 風）
  非終端の枝 : (次状態 index, 確率)
  終端の枝   : (確率, 終端の理由, 終端の生の体力)
目的関数は参照 solver と同じ Objective を使う（評価の層は共通）。
"""
import sys, time, hashlib
import numpy as np
import mini_world as m
import mini_solver_ref as ref

DEC, SS, C1, C2, C3, C4, SL, CB = range(8)
MODE_ID = {m.DECISION: DEC, m.SEARCH_SMALL: SS, m.CHASE_1: C1, m.CHASE_2: C2,
           m.CHASE_3: C3, m.CHASE_4: C4, m.SEARCH_LARGE: SL, m.COMBAT: CB}
MODES = [m.DECISION, m.SEARCH_SMALL, m.CHASE_1, m.CHASE_2, m.CHASE_3,
         m.CHASE_4, m.SEARCH_LARGE, m.COMBAT]
ACTS = [m.ACT_GRASS, m.ACT_SMALL, m.ACT_LARGE]

N_E = m.E_MAX + 1
N_W = m.K_WOUND + 1
N_S = m.SMALL_INIT - m.SMALL_FLOOR + 1
N_L = m.LARGE_MAX + 1
STRIDE_M = N_E * N_W * N_S * N_L
TOTAL = 8 * STRIDE_M


def idx(mode_id, e, wnd, ns, nl):
    return (((mode_id * N_E + e) * N_W + wnd) * N_S
            + (ns - m.SMALL_FLOOR)) * N_L + nl


def unidx(i):
    nl = i % N_L; i //= N_L
    ns = i % N_S + m.SMALL_FLOOR; i //= N_S
    wnd = i % N_W; i //= N_W
    e = i % N_E; i //= N_E
    return MODES[i], e, wnd, ns, nl


def build_tables():
    """★mini_world.branches をそのまま呼んで事前計算する（物理は再実装しない）★
    decision は行動ごと、強制 MODE は行動なしで1本。
    時刻に依存する部分（horizon）は後で扱うので、ここでは t を渡さず
    「天寿以外の枝」を作り、天寿の判定は Bellman 側で行う。
    """
    # 枝は t に依存しない（天寿の判定だけが t 依存）。t=0 で作る
    nonterm = {}     # key -> (next_idx 配列, prob 配列)
    term = {}        # key -> [(prob, cause, raw_e), ...]
    for mi, mode in enumerate(MODES):
        for e in range(1, m.E_MAX + 1):
            for wnd in range(N_W):
                for ns in range(m.SMALL_FLOOR, m.SMALL_INIT + 1):
                    for nl in range(N_L):
                        if mode in (m.SEARCH_LARGE, m.COMBAT) and nl == 0:
                            continue
                        acts = [None]
                        if mode == m.DECISION:
                            acts = m.legal_actions(e, wnd, ns, nl)
                        for a in acts:
                            # ★t=0 で枝を作る（天寿は起きない時刻）★
                            br = m.branches(mode, e, wnd, ns, nl, 0, a)
                            nx_i, nx_p, tm = [], [], []
                            for p, nxt, cause, raw_e in br:
                                if nxt is None:
                                    tm.append((p, cause, raw_e))
                                else:
                                    nx_i.append(idx(MODE_ID[nxt[0]], nxt[1],
                                                    nxt[2], nxt[3], nxt[4]))
                                    nx_p.append(p)
                            key = (idx(mi, e, wnd, ns, nl), a)
                            nonterm[key] = (np.array(nx_i, dtype=np.int64),
                                            np.array(nx_p, dtype=np.float64))
                            term[key] = tm
    return nonterm, term


def solve(obj, nonterm, term, horizon):
    """後ろ向き帰納。V は2層だけ持つ"""
    V_next = np.zeros(TOTAL, dtype=np.float64)
    Q_tables = {}
    for t in range(horizon - 1, -1, -1):
        V_cur = np.zeros(TOTAL, dtype=np.float64)
        qt = {}
        for key, (nx_i, nx_p) in nonterm.items():
            i, a = key
            mode, e, wnd, ns, nl = unidx(i)
            tot = 0.0
            # 非終端の枝：t+1 が horizon なら天寿として扱う
            for j, p in zip(nx_i, nx_p):
                _, e2, _, _, _ = unidx(int(j))
                if t + 1 >= horizon:
                    br = (p, None, m.TENJU, e2)       # 天寿
                    r, g, _ = obj.step_value(e, br)
                    tot += p * r
                else:
                    br = (p, unidx(int(j)), None, e2)
                    r, g, _ = obj.step_value(e, br)
                    tot += p * (r + g * V_next[j])
            # 終端の枝
            for (p, cause, raw_e) in term[key]:
                br = (p, None, cause, raw_e)
                r, g, _ = obj.step_value(e, br)
                tot += p * r
            if mode == m.DECISION:
                qt.setdefault((e, wnd, ns, nl), {})[a] = tot
            else:
                V_cur[i] = tot
        # decision の V は Q の最大
        for (e, wnd, ns, nl), qs in qt.items():
            V_cur[idx(DEC, e, wnd, ns, nl)] = max(qs.values())
        Q_tables[t] = qt
        V_next = V_cur
    return Q_tables


def main(which="A"):
    if which == "B":
        m.HORIZON = 40
    print(f"最適化 dense solver（縮小{which}、H={m.HORIZON}）")
    print(f"index の総数 {TOTAL:,}")
    t0 = time.perf_counter()
    nonterm, term = build_tables()
    print(f"遷移表の事前計算: {len(nonterm):,} 件、{time.perf_counter()-t0:.1f}秒")
    print()
    for name, obj in [("D0", ref.Objective("D0")),
                      ("DS", ref.Objective("DS")),
                      ("D1a_Eset200_q2_g099", ref.Objective("D1a", e_set=200, q=2.0, gamma=0.99)),
                      ("D1b_Eset200_q2_g099", ref.Objective("D1b", e_set=200, q=2.0, gamma=0.99)),
                      ("D1a_Eset140_q2_g099", ref.Objective("D1a", e_set=140, q=2.0, gamma=0.99)),
                      ("D1b_Eset140_q2_g099", ref.Objective("D1b", e_set=140, q=2.0, gamma=0.99))]:
        ts = time.perf_counter()
        Q = solve(obj, nonterm, term, m.HORIZON)
        init = (m.E_MAX, 0, m.SMALL_INIT, m.LARGE_MAX)
        qs = Q[0][init]
        print(f"{name:24s} 秒 {time.perf_counter()-ts:6.1f}  初期状態: "
              + "  ".join(f"{a}={qs[a]:12.4f}" for a in sorted(qs))
              + f"  → {ref.selected(qs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
