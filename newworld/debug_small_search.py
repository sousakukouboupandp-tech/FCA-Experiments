# -*- coding: utf-8 -*-
"""小物の探索歩数の不一致（A-1 実測1.4998 / 理論2.0）を乱数なしで切り分ける。
遭遇の歩を強制し、探索歩数と decision→decision の合計歩数を両方見る。
A-1 の集計と同じ書き方（s.mode == SEARCH_SMALL and d.find のとき s_next.t - t0 を足す）も
並べて表示し、どこで食い違うかを1歩ずつ出す。
"""
import world1g as w
from world1g import State, Draws, transition, DECISION, SEARCH_SMALL, CHASE_4, ACT_SMALL

E = w.E_MAX

for hit_step in (1, 2, 3, 4):
    s = State(e=E, w=0, n_small=12, mode=DECISION)
    a = ACT_SMALL
    t0 = s.t
    search_by_a1 = None      # A-1 の集計方法
    steps_log = []
    step_i = 0
    while True:
        step_i += 1
        find = (step_i == hit_step)   # 探索の歩なら遭遇させる（1歩目は mode が decision）
        catch = (s.mode == CHASE_4)
        d = Draws(find=find, catch=catch)
        s_next = transition(s, a, d)
        if s.mode == SEARCH_SMALL and d.find:
            search_by_a1 = s_next.t - t0
        steps_log.append("%d歩目 %s find=%s → t=%d mode=%s"
                         % (step_i, s.mode, find, s_next.t, s_next.mode))
        a = None
        s = State(e=E, w=0, n_small=12, n_large=s_next.n_large,
                  t=s_next.t, mode=s_next.mode)
        if s.mode == DECISION:
            break
    print("── %d歩目で遭遇 ──" % hit_step)
    for line in steps_log:
        print("   ", line)
    print("    A-1 の集計による探索歩数 =", search_by_a1,
          " 期待 =", hit_step,
          " 合計歩数 =", s.t - t0, " 期待 =", hit_step + 4)
    print()
