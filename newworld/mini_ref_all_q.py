# -*- coding: utf-8 -*-
"""縮小A / 縮小B の参照 solver の全 Q を保存する。
縮小A：mini_world のまま（H=20）。草だけで生き切れる
縮小B：★A と同じ物理で H=40 だけ変更★（草だけでは生き切れないことが数学だけで確定）
  草の正味 -6/歩、E_MAX=200 → 34歩目に飢え死。H=40 > 34 なので狩りが必須

★数値を見ながらの調整はしない。H を変えるだけ★
保存するもの：全時刻・全 decision 状態・全行動の Q、optimal_set、selected
"""
import sys, json, hashlib, time
import mini_world as m
import mini_solver_ref as ref


def enumerate_decision_states():
    """縮小世界の decision 状態を全部列挙する（structural-valid のみ）"""
    out = []
    for e in range(1, m.E_MAX + 1):          # ★E=0 は非終端では存在しない★
        for wnd in range(0, m.K_WOUND + 1):
            for ns in range(m.SMALL_FLOOR, m.SMALL_INIT + 1):
                for nl in range(0, m.LARGE_MAX + 1):
                    out.append((e, wnd, ns, nl))
    return out


def solve_all(obj, states, horizon):
    """全時刻・全 decision 状態・全行動の Q を出す"""
    value, qvalue, V, Q = ref.solve(obj)
    table = {}
    for t in range(horizon):
        for (e, wnd, ns, nl) in states:
            qs = {}
            for a in m.legal_actions(e, wnd, ns, nl):
                qs[a] = qvalue(e, wnd, ns, nl, t, a)
            table[(e, wnd, ns, nl, t)] = qs
    return table


def digest_table(table):
    """Q 表の決定論的ダイジェスト（キーを正順に並べ、値を固定書式で）"""
    h = hashlib.sha256()
    for k in sorted(table):
        h.update(("%d,%d,%d,%d,%d:" % k).encode())
        for a in sorted(table[k]):
            h.update(("%s=%.12e;" % (a, table[k][a])).encode())
    return h.hexdigest()[:16]


OBJS = [
    ("D0", ref.Objective("D0")),
    ("DS", ref.Objective("DS")),
    ("D1a_Eset200_q2_g099", ref.Objective("D1a", e_set=200, q=2.0, gamma=0.99)),
    ("D1b_Eset200_q2_g099", ref.Objective("D1b", e_set=200, q=2.0, gamma=0.99)),
    ("D1a_Eset140_q2_g099", ref.Objective("D1a", e_set=140, q=2.0, gamma=0.99)),
    ("D1b_Eset140_q2_g099", ref.Objective("D1b", e_set=140, q=2.0, gamma=0.99)),
]


def main(which="A"):
    sys.setrecursionlimit(1000000)
    if which == "B":
        m.HORIZON = 40                      # ★H だけ変更★
    print(f"縮小{which} の参照 solver（H={m.HORIZON}）")
    print("草だけで生き切れるか：草の正味 -6/歩、E_MAX=200 → 34歩目に飢え死")
    print(f"  → H={m.HORIZON} なら {'生き切れる（狩り不要）' if m.HORIZON <= 33 else '★生き切れない（狩りが必須）★'}")
    states = enumerate_decision_states()
    print(f"decision 状態（structural-valid）: {len(states):,}")
    print(f"時刻 × 状態: {len(states) * m.HORIZON:,}")
    print()
    for name, obj in OBJS:
        t0 = time.perf_counter()
        # 目的関数ごとにキャッシュを分ける（V が目的に依存するため）
        table = solve_all(obj, states, m.HORIZON)
        dg = digest_table(table)
        init = (m.E_MAX, 0, m.SMALL_INIT, m.LARGE_INIT, 0)
        qs = table[init]
        print(f"{name:24s} digest {dg}  秒 {time.perf_counter()-t0:6.1f}")
        print(f"    初期状態: " + "  ".join(f"{a}={qs[a]:12.4f}" for a in sorted(qs))
              + f"  → {ref.selected(qs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
