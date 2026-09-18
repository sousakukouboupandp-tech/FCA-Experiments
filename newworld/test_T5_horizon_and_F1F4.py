# -*- coding: utf-8 -*-
"""T5：実寸 solver の horizon 境界を検証する（ChatGPT 返答108の1）。
★kernel は時刻に依存しないので、天寿の判定は solver の責務★
  t=998 : 通常の非終端は V_999 へ進む
  t=999 : 通常なら非終端になる枝を ★天寿★ にする
  t=999 : 飢え死・急所死はそのまま
  天寿では D1b の0置換を発動しない
  天寿の終端の体力は、1000歩目を実行した後の生の値
  V_1000 を参照しない

F1〜F4：実寸での終端の生の体力の回帰
  正本の枝 = 保存した kernel のデータ = 目的関数が使った値
  まで一本につなげて確認する
"""
import os, sys, json
import numpy as np
import world1g as w
from world1g import State, Draws, transition, DECISION, E_MAX, HORIZON
import dense_index as dx
from probe_kernel_size import branches_with_prob
from gateB_kernel_vs_canonical import pair_index_map, ACT_ID, CAUSE_ID

D = "kernel_full"
ok, ng = 0, []


def check(name, cond, detail=""):
    global ok
    if cond:
        ok += 1
        print(f"  OK   {name}  {detail}")
    else:
        ng.append(name)
        print(f"  ★NG★ {name}  {detail}")


def drive(e, e_set, q):
    return abs(e_set - e) ** q


# ---------------------------------------------------------------- solver の層
def layer_terminal_kind(t, cause_id):
    """solver が t で枝をどう扱うか。
    cause_id: 0=非終端, 1=飢え死, 2=急所死
    戻り値: 'next'（次の層へ）／'starve'／'acute'／'tenju'
    """
    if cause_id == 1:
        return "starve"
    if cause_id == 2:
        return "acute"
    # 非終端の枝
    if t + 1 >= HORIZON:
        return "tenju"          # ★1000歩目を生き抜いたら天寿★
    return "next"


def reward_D1a(e_from, e_to, e_set, q):
    return drive(e_from, e_set, q) - drive(e_to, e_set, q)


def reward_D1b(e_from, e_to, kind, e_set, q):
    if kind in ("starve", "acute"):        # ★不利な死だけ0として評価★
        e_to = 0
    return drive(e_from, e_set, q) - drive(e_to, e_set, q)


def main():
    print("T5：実寸 solver の horizon 境界")
    print(f"  HORIZON = {HORIZON}")
    print()
    E_SET, Q = 1400, 2.0

    # --- T5-1：t=998 と t=999 で非終端の枝の扱いが変わる
    print("T5-1 非終端の枝の扱い（草を食べる歩）")
    nt, tm = branches_with_prob(DECISION, 1000, 0, 12, 3, w.ACT_GRASS)
    check("草の枝に非終端がある", len(nt) > 0, f"非終端 {len(nt)} 本、終端 {len(tm)} 本")
    k998 = layer_terminal_kind(998, 0)
    k999 = layer_terminal_kind(999, 0)
    check("t=998 の非終端は次の層へ", k998 == "next", f"{k998}")
    check("★t=999 の非終端は天寿になる★", k999 == "tenju", f"{k999}")

    # --- T5-2：t=999 でも飢え死・急所死はそのまま
    print()
    print("T5-2 t=999 での不利な死")
    check("t=999 の飢え死は飢え死のまま", layer_terminal_kind(999, 1) == "starve")
    check("t=999 の急所死は急所死のまま", layer_terminal_kind(999, 2) == "acute")

    # --- T5-3：天寿では D1b の置換が発動しない
    print()
    print("T5-3 天寿での D1b")
    e_from, e_to = 1000, 994          # 草を食べた1歩（-20+14）
    r_a = reward_D1a(e_from, e_to, E_SET, Q)
    r_b = reward_D1b(e_from, e_to, "tenju", E_SET, Q)
    check("★天寿では D1a と D1b が一致★", abs(r_a - r_b) < 1e-12,
          f"D1a={r_a:.1f}, D1b={r_b:.1f}")
    r_b_star = reward_D1b(e_from, e_to, "starve", E_SET, Q)
    check("飢え死なら D1b は0として評価（対照）", abs(r_b_star - r_a) > 1.0,
          f"D1b(飢え死)={r_b_star:.1f}")

    # --- T5-4：天寿の終端の体力は1000歩目の後の生の値
    print()
    print("T5-4 天寿の終端の体力")
    s = State(e=1000, w=0, n_small=12, n_large=3, t=HORIZON - 1, mode=w.GRASS)
    nx = transition(s, None, Draws())
    check("1000歩目を生き抜くと dead=tenju", nx.dead == "tenju", f"{nx.dead}")
    check("★天寿の体力は1歩実行した後の値★", nx.e == 994, f"E={nx.e}（1000-20+14）")

    # --- T5-5：V_1000 を参照しない（solver の層の作り）
    print()
    print("T5-5 V_1000 を参照しない")
    check("t=999 の非終端が 'tenju' に分類されるので次の層を見ない",
          layer_terminal_kind(HORIZON - 1, 0) == "tenju")

    # --- F1〜F4：実寸での終端の生の体力の回帰
    print()
    print("F1〜F4 実寸での終端の生の体力（正本 = kernel = 目的関数）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    keys, order = pair_index_map(A["state_idx"], A["act_id"])

    def kernel_terminals(mode, e, wnd, ns, nl, a):
        i = dx.to_index(mode, e, wnd, ns, nl)
        k = i * 4 + ACT_ID[a]
        pos = np.searchsorted(keys, k)
        pid = int(order[pos])
        lo, hi = int(A["t_offs"][pid]), int(A["t_offs"][pid + 1])
        return [(float(A["t_prob"][x]), int(A["t_cause"][x]), int(A["t_rawe"][x]))
                for x in range(lo, hi)]

    cases = [
        ("F1 飢え死ちょうど0", w.SEARCH_SMALL, 20, 0, 12, 3, None, 1, 0),
        ("F2 飢え死の行き過ぎ", w.SEARCH_SMALL, 2, 0, 12, 3, None, 1, -18),
        ("F3 急所死で体力が正", w.COMBAT, 2000, 0, 12, 3, None, 2, 1980),
        ("F4 傷つきの飢え死", w.SEARCH_SMALL, 20, 10, 12, 3, None, 1, -50),
    ]
    for (name, mode, e, wnd, ns, nl, a, want_cause, want_e) in cases:
        # 正本
        nt_ref, tm_ref = branches_with_prob(mode, e, wnd, ns, nl, a)
        ref = sorted({(c, raw) for _, c, raw in
                      [(p, CAUSE_ID[c], raw) for p, c, raw in tm_ref]})
        # kernel
        ker = sorted({(c, raw) for _, c, raw in kernel_terminals(mode, e, wnd, ns, nl, a)})
        check(f"{name}：正本と kernel が一致", ref == ker, f"{ker}")
        check(f"{name}：期待した (理由,体力) がある", (want_cause, want_e) in ker,
              f"期待 ({want_cause},{want_e})")
        # 目的関数が使う値
        r_a = reward_D1a(e, want_e, E_SET, Q)
        kind = {1: "starve", 2: "acute"}[want_cause]
        r_b = reward_D1b(e, want_e, kind, E_SET, Q)
        print(f"       → D1a={r_a:+.1f}（生の {want_e} を使う）"
              f" / D1b={r_b:+.1f}（0 として評価）")

    print()
    print(f"T5 + F1〜F4: 合格 {ok} 件 / NG {len(ng)} 件")
    for n in ng:
        print("  NG:", n)


if __name__ == "__main__":
    main()
