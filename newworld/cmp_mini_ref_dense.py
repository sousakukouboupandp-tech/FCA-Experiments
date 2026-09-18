# -*- coding: utf-8 -*-
"""参照 solver と最適化 dense solver の全 Q 比較（ChatGPT 返答100の指示）。
保存するもの：
  max_abs_diff / max_rel_diff / argmax（目的・時刻・状態・行動）
  numerical tie の不一致件数
  optimal_set の不一致件数     ★ゼロを要求★
  selected_for_execution の不一致件数  ★ゼロを要求★
※Q の SHA 一致は合格条件にしない（演算順序が違えば bitwise 一致しない）
"""
import sys, time
import mini_world as m
import mini_solver_ref as ref
import mini_solver_dense as den

ATOL, RTOL = ref.ATOL, ref.RTOL


def compare(which="A"):
    if which == "B":
        m.HORIZON = 40
    H = m.HORIZON
    print(f"参照 vs 最適化 dense の全 Q 比較（縮小{which}、H={H}）")
    print()
    states = [(e, wnd, ns, nl)
              for e in range(1, m.E_MAX + 1)
              for wnd in range(m.K_WOUND + 1)
              for ns in range(m.SMALL_FLOOR, m.SMALL_INIT + 1)
              for nl in range(m.LARGE_MAX + 1)]
    print(f"decision 状態 {len(states):,} × 時刻 {H} = {len(states)*H:,}")
    print()
    nonterm, term = den.build_tables()

    objs = [("D0", ref.Objective("D0")),
            ("DS", ref.Objective("DS")),
            ("D1a_Eset200", ref.Objective("D1a", e_set=200, q=2.0, gamma=0.99)),
            ("D1b_Eset200", ref.Objective("D1b", e_set=200, q=2.0, gamma=0.99)),
            ("D1a_Eset140", ref.Objective("D1a", e_set=140, q=2.0, gamma=0.99)),
            ("D1b_Eset140", ref.Objective("D1b", e_set=140, q=2.0, gamma=0.99))]

    all_ok = True
    print("目的         | 比較数 | max_abs_diff | max_rel_diff | argmax(t,state,act) | tie不一致 | set不一致 | 選択不一致")
    for name, obj in objs:
        sys.setrecursionlimit(1000000)
        value, qvalue, V, Q = ref.solve(obj)
        Qd = den.solve(obj, nonterm, term, H)
        n = 0
        mx_abs, mx_rel = 0.0, 0.0
        arg_abs = arg_rel = None
        tie_ng = set_ng = sel_ng = 0
        for t in range(H):
            for st in states:
                e, wnd, ns, nl = st
                qs_r = {a: qvalue(e, wnd, ns, nl, t, a)
                        for a in m.legal_actions(e, wnd, ns, nl)}
                qs_d = Qd[t][st]
                for a in qs_r:
                    n += 1
                    d = abs(qs_r[a] - qs_d[a])
                    rel = d / max(1.0, abs(qs_r[a]), abs(qs_d[a]))
                    if d > mx_abs:
                        mx_abs, arg_abs = d, (t, st, a)
                    if rel > mx_rel:
                        mx_rel, arg_rel = rel, (t, st, a)
                    if d > ATOL + RTOL * max(1.0, abs(qs_r[a]), abs(qs_d[a])):
                        tie_ng += 1
                if ref.optimal_set(qs_r) != ref.optimal_set(qs_d):
                    set_ng += 1
                if ref.selected(qs_r) != ref.selected(qs_d):
                    sel_ng += 1
        ok = (set_ng == 0 and sel_ng == 0)
        all_ok = all_ok and ok
        print(f"{name:12s} | {n:6,} | {mx_abs:12.3e} | {mx_rel:12.3e} |"
              f" {str(arg_abs):19s} | {tie_ng:9,} | {set_ng:9,} | {sel_ng:10,}"
              f"  {'OK' if ok else '★NG★'}", flush=True)
    print()
    print("★optimal_set と selected_for_execution が全目的でゼロ不一致★"
          if all_ok else "★不一致あり★")


if __name__ == "__main__":
    compare(sys.argv[1] if len(sys.argv) > 1 else "A")
