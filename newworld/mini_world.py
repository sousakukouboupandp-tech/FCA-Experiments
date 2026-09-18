# -*- coding: utf-8 -*-
"""縮小世界の参照 solver（正しさ優先・速度は無視）。
ChatGPT 返答96の分離を守る：
  ★世界は「何が起こったか」だけを返す★
  ★目的関数は「その出来事をどう評価するか」を決める★
  → 目的関数ごとに遷移の表現を変えない

枝（branch）の形：
  (probability, next_state or None, terminal_cause, terminal_raw_e)
  next_state が None なら終端。terminal_raw_e は終端の生の体力（負もありうる）

縮小世界（検証用。世界の物理の変更ではない）
  寿命 20歩／体力の上限 200／消耗 20 + 5W／草 14／小物 60／大物 180
  小物 3（下限1）／大物 2／傷の上限 3
  ※比を World_1-G と合わせる（体力上限 2000→200 なので摂取も1/10）
"""
import itertools
from dataclasses import dataclass

# --- 縮小世界の定数 -------------------------------------------------------
E_MAX = 200
UPKEEP = 20
WOUND_COST = 5
GRASS_GAIN = 14
SMALL_GAIN = 60
LARGE_GAIN = 180
HORIZON = 20
SMALL_INIT, SMALL_FLOOR = 3, 1
LARGE_INIT, LARGE_MAX = 2, 2
K_WOUND = 3
SMALL_SEARCH_K = 6.0          # 平均 6/N_S 歩
CHASE_STEPS = 4
CATCH_P = 0.7
LARGE_BASE_SEARCH = 4.0
LARGE_SEARCH_EXP = 1.5
LARGE_RECOVER_P = 1.0 / 25.0
HEAL_P = 1.0 / 15.0
COMBAT_KILL, COMBAT_ACUTE, COMBAT_WOUND, COMBAT_MISS = 0.25, 0.04, 0.36, 0.35

DECISION = "decision"
GRASS = "grass"
SEARCH_SMALL, CHASE_1, CHASE_2, CHASE_3, CHASE_4 = (
    "search_small", "chase_1", "chase_2", "chase_3", "chase_4")
SEARCH_LARGE, COMBAT = "search_large", "combat"
CHASE_NEXT = {CHASE_1: CHASE_2, CHASE_2: CHASE_3, CHASE_3: CHASE_4}
ACT_GRASS, ACT_SMALL, ACT_LARGE = "grass", "small", "large"

STARVE, ACUTE, TENJU = "starve", "acute", "tenju"


def small_find_p(ns):
    return ns / SMALL_SEARCH_K


def large_find_p(nl):
    if nl <= 0:
        return 0.0
    return 1.0 / (LARGE_BASE_SEARCH * (LARGE_MAX / nl) ** LARGE_SEARCH_EXP)


def legal_actions(e, wnd, ns, nl):
    acts = [ACT_GRASS, ACT_SMALL]
    if nl >= 1:
        acts.append(ACT_LARGE)
    return acts


def start_mode(a):
    return {ACT_GRASS: GRASS, ACT_SMALL: SEARCH_SMALL,
            ACT_LARGE: SEARCH_LARGE}[a]


def binom(n, k, p):
    from math import comb
    return comb(n, k) * p**k * (1 - p)**(n - k)


def branches(mode, e, wnd, ns, nl, t, action=None):
    """★世界の層★ 1歩の枝を全部返す。確率つき。
    戻り値: [(prob, next or None, cause, terminal_raw_e), ...]
    next = (mode, e, w, ns, nl)
    """
    run_mode = start_mode(action) if mode == DECISION else mode
    cost = UPKEEP + WOUND_COST * wnd
    out = []

    # 背景：傷の治癒（二項）と大物の回復
    heal_dist = [(binom(wnd, k, HEAL_P), k) for k in range(wnd + 1)] if wnd else [(1.0, 0)]
    rec_dist = ([(LARGE_RECOVER_P, 1), (1 - LARGE_RECOVER_P, 0)]
                if nl in (1,) or (nl == 2 and LARGE_MAX > 2) else [(1.0, 0)])
    # 縮小世界では大物の上限が2なので、回復は nl==1 のときだけ
    if nl != 1:
        rec_dist = [(1.0, 0)]

    # 行動側の分岐
    if run_mode == GRASS:
        act_br = [(1.0, dict(intake=GRASS_GAIN, nxt=DECISION))]
    elif run_mode == SEARCH_SMALL:
        p = small_find_p(ns)
        act_br = [(p, dict(intake=0, nxt=CHASE_1)),
                  (1 - p, dict(intake=0, nxt=SEARCH_SMALL))]
    elif run_mode in (CHASE_1, CHASE_2, CHASE_3):
        act_br = [(1.0, dict(intake=0, nxt=CHASE_NEXT[run_mode]))]
    elif run_mode == CHASE_4:
        act_br = [(CATCH_P, dict(intake=SMALL_GAIN, nxt=DECISION, ns_dec=True)),
                  (1 - CATCH_P, dict(intake=0, nxt=DECISION))]
    elif run_mode == SEARCH_LARGE:
        p = large_find_p(nl)
        act_br = [(p, dict(intake=0, nxt=COMBAT)),
                  (1 - p, dict(intake=0, nxt=SEARCH_LARGE))]
    elif run_mode == COMBAT:
        act_br = [(COMBAT_KILL, dict(intake=LARGE_GAIN, nxt=DECISION, kill=True)),
                  (COMBAT_ACUTE, dict(intake=0, nxt=None, acute=True)),
                  (COMBAT_WOUND, dict(intake=0, nxt=COMBAT, wound=True)),
                  (COMBAT_MISS, dict(intake=0, nxt=COMBAT))]
    else:
        raise ValueError(run_mode)

    for ph, heal in heal_dist:
        for pr, rec in rec_dist:
            for pa, d in act_br:
                p = ph * pr * pa
                if p <= 0.0:
                    continue
                w2 = wnd - heal + (1 if d.get("wound") else 0)
                w2 = min(w2, K_WOUND)
                ns2 = max(SMALL_FLOOR, ns - 1) if d.get("ns_dec") else ns
                kill = 1 if d.get("kill") else 0
                if kill and nl == 1:
                    nl2 = 0                      # 絶滅優先
                else:
                    nl2 = min(LARGE_MAX, nl + rec - kill)
                e2 = min(E_MAX, e - cost + d["intake"])
                t2 = t + 1
                if d.get("acute"):
                    out.append((p, None, ACUTE, e2))
                elif e2 <= 0:
                    out.append((p, None, STARVE, e2))
                elif t2 >= HORIZON:
                    out.append((p, None, TENJU, e2))
                else:
                    out.append((p, (d["nxt"], e2, w2, ns2, nl2), None, e2))
    return out
