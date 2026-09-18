# -*- coding: utf-8 -*-
"""縮小世界の参照 solver（目的関数の層）。正しさ優先・速度は無視。
★世界の枝（mini_world.branches）は共通。目的関数だけが評価を変える★

D0        ：期待寿命（その歩を生きたら +1。死んだ歩も1歩として数える）
D_S-exact ：天寿に到達する確率
D1a       ：r = D(E_t) − D(E_{t+1})。終端は生の体力を使う（0に切らない）
D1b       ：不利な死（飢え死・急所死）だけ r = D(E_t) − D(0)。天寿は生の体力
"""
import sys
from functools import lru_cache
import mini_world as m

ATOL, RTOL = 1e-9, 1e-12


def drive(e, e_set, q):
    return abs(e_set - e) ** q


class Objective:
    """目的関数の層。同じ枝を別々に評価する"""
    def __init__(self, kind, e_set=None, q=None, gamma=1.0):
        self.kind, self.e_set, self.q, self.gamma = kind, e_set, q, gamma

    def step_value(self, e_from, br):
        """枝1本の (即時報酬, 割引率, 終端か) を返す"""
        p, nxt, cause, raw_e = br
        g = self.gamma
        if self.kind == "D0":
            return 1.0, 1.0, nxt is None           # その歩を生きたら +1
        if self.kind == "DS":
            # 天寿に到達したら 1、それ以外の終端は 0
            return (1.0 if cause == m.TENJU else 0.0), 1.0, nxt is None
        if self.kind == "D1a":
            e_to = raw_e if nxt is None else nxt[1]
            r = drive(e_from, self.e_set, self.q) - drive(e_to, self.e_set, self.q)
            return r, g, nxt is None
        if self.kind == "D1b":
            if nxt is None and cause in (m.STARVE, m.ACUTE):
                e_to = 0                            # ★不利な死だけ 0 として評価★
            else:
                e_to = raw_e if nxt is None else nxt[1]
            r = drive(e_from, self.e_set, self.q) - drive(e_to, self.e_set, self.q)
            return r, g, nxt is None
        raise ValueError(self.kind)


def solve(obj):
    """後ろ向きに解く。戻り値: V（辞書）, Q（decision 状態のみ）"""
    V, Q = {}, {}

    def value(mode, e, wnd, ns, nl, t):
        key = (mode, e, wnd, ns, nl, t)
        if key in V:
            return V[key]
        if mode == m.DECISION:
            qs = {}
            for a in m.legal_actions(e, wnd, ns, nl):
                qs[a] = qvalue(e, wnd, ns, nl, t, a)
            Q[(e, wnd, ns, nl, t)] = qs
            best = max(qs.values()) if obj.kind != "MIN" else min(qs.values())
            V[key] = best
            return best
        # 強制 MODE：行動の選択はない
        tot = 0.0
        for br in m.branches(mode, e, wnd, ns, nl, t, None):
            p, nxt, cause, raw_e = br
            r, g, term = obj.step_value(e, br)
            if term:
                tot += p * r
            else:
                tot += p * (r + g * value(nxt[0], nxt[1], nxt[2], nxt[3], nxt[4], t + 1))
        V[key] = tot
        return tot

    def qvalue(e, wnd, ns, nl, t, a):
        tot = 0.0
        for br in m.branches(m.DECISION, e, wnd, ns, nl, t, a):
            p, nxt, cause, raw_e = br
            r, g, term = obj.step_value(e, br)
            if term:
                tot += p * r
            else:
                tot += p * (r + g * value(nxt[0], nxt[1], nxt[2], nxt[3], nxt[4], t + 1))
        return tot

    return value, qvalue, V, Q


def optimal_set(qs):
    """numerical tie（凍結済み）で同点をまとめる"""
    best = max(qs.values())
    return sorted(k for k, v in qs.items()
                  if abs(v - best) <= ATOL + RTOL * max(1.0, abs(v), abs(best)))


def selected(qs):
    """固定順序：草 → 小物 → 大物"""
    order = [m.ACT_GRASS, m.ACT_SMALL, m.ACT_LARGE]
    s = optimal_set(qs)
    for a in order:
        if a in s:
            return a
    raise AssertionError


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    print("縮小世界の参照 solver（正しさ優先）")
    print(f"寿命 {m.HORIZON}／体力上限 {m.E_MAX}／小物 {m.SMALL_INIT}（下限{m.SMALL_FLOOR}）"
          f"／大物 {m.LARGE_INIT}／傷の上限 {m.K_WOUND}")
    print()
    init = (m.E_MAX, 0, m.SMALL_INIT, m.LARGE_INIT, 0)
    for obj in (Objective("D0"), Objective("DS"),
                Objective("D1a", e_set=200, q=2.0, gamma=0.99),
                Objective("D1b", e_set=200, q=2.0, gamma=0.99),
                Objective("D1a", e_set=140, q=2.0, gamma=0.99),
                Objective("D1b", e_set=140, q=2.0, gamma=0.99)):
        value, qvalue, V, Q = solve(obj)
        v0 = value(m.DECISION, *init)
        qs = Q[init]
        label = obj.kind
        if obj.e_set:
            label += f"(E_set={obj.e_set},q={obj.q},γ={obj.gamma})"
        print(f"{label:32s} V0={v0:14.6f}  " +
              "  ".join(f"{a}={qs[a]:12.4f}" for a in sorted(qs)) +
              f"  → {selected(qs)}  同点{optimal_set(qs)}")
