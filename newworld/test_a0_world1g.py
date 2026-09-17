# -*- coding: utf-8 -*-
"""A-0：乱数を使わないテスト（決定的テストと構造テスト）
設計書 v02（386bf76）§5-A-1／A-2 と、ChatGPT 返答32 の追加4本。
抽選結果を直接渡して、同時に起きる出来事を強制的に作る。
"""
import world1g as w
from world1g import (State, Draws, transition, legal_actions,
                     DECISION, GRASS, SEARCH_SMALL, CHASE_1, CHASE_2, CHASE_3,
                     CHASE_4, SEARCH_LARGE, COMBAT,
                     ACT_GRASS, ACT_SMALL, ACT_LARGE,
                     ALIVE, DEAD_STARVE, DEAD_ACUTE, DEAD_TENJU, E_MAX)

ok = 0
ng = []


def check(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        ng.append(name)


def expect_error(name, fn):
    try:
        fn()
    except Exception:
        check(name, True)
        return
    check(name, False)


# 1. 戦闘の4結果の確率（コードの値そのもの）
check("戦闘4結果の和が1", abs(sum(w.COMBAT_P) - 1.0) < 1e-12)
check("戦闘 kill=0.25", w.COMBAT_KILL == 0.25)
check("戦闘 acute=0.04", abs(w.COMBAT_ACUTE - 0.04) < 1e-12)
check("戦闘 wound=0.36", abs(w.COMBAT_WOUND - 0.36) < 1e-12)
check("戦闘 miss=0.35", abs(w.COMBAT_MISS - 0.35) < 1e-12)

# 2. 草だけ：333歩目は生存、334歩目に飢え死（傷0、初期2000）
s = State()
for i in range(333):
    s = transition(s, ACT_GRASS if s.mode == DECISION else None, Draws())
check("草だけ333歩目は生存", s.alive and s.e == 2000 - 6 * 333)
s334 = transition(s, ACT_GRASS, Draws())
check("草だけ334歩目に飢え死", s334.dead == DEAD_STARVE and s334.e == 2000 - 6 * 334)

# 3. 絶滅優先：N_L=1 で仕留めと回復が同じ歩に当たる
s = State(n_large=1, mode=COMBAT, e=1000)
s2 = transition(s, None, Draws(recover=True, combat="kill"))
check("N_L=1 で仕留め＋回復なら絶滅優先", s2.n_large == 0)
check("仕留めた歩に摂取1800が入る", s2.e == min(E_MAX, 1000 - 20 + 1800))

# 4. N_L=2 で仕留めと回復が同じ歩 → 2 のまま
s = State(n_large=2, mode=COMBAT, e=1000)
check("N_L=2 で仕留め＋回復なら2のまま",
      transition(s, None, Draws(recover=True, combat="kill")).n_large == 2)

# 5. N_L=0 は吸収状態。大物探索は選べない
s = State(n_large=0)
check("N_L=0 で大物探索は選べない", ACT_LARGE not in legal_actions(s))
expect_error("N_L=0 で大物探索を強制すると拒否", lambda: transition(s, ACT_LARGE, Draws()))
s0 = transition(s, ACT_GRASS, Draws())
check("N_L=0 なら次も0", s0.n_large == 0)
expect_error("N_L=0 で回復の抽選を渡すと拒否",
             lambda: transition(State(n_large=0, mode=GRASS), None, Draws(recover=True)))
expect_error("N_L=3 で回復の抽選を渡すと拒否",
             lambda: transition(State(n_large=3, mode=GRASS), None, Draws(recover=True)))

# 6. 新しい傷は、その歩の治癒判定に入らない（返答32の追加2本目）
s = State(w=1, mode=COMBAT, e=1000)
s2 = transition(s, None, Draws(heal=1, combat="wound"))
check("既存傷が治り同じ歩に新しい傷 → W=1", s2.w == 1)
check("その歩の消耗は歩の始めの傷の数で決まる", s2.e == 1000 - (20 + 5 * 1))
expect_error("歩の始めの傷より多く治すと拒否",
             lambda: transition(State(w=1, mode=GRASS), None, Draws(heal=2)))

# 7. 摂取を足してから飢え死判定（傷0、体力20で追跡4歩目に捕獲成功）
s = State(e=20, w=0, mode=CHASE_4)
s2 = transition(s, None, Draws(catch=True))
check("体力20で捕獲成功なら生存", s2.alive and s2.e == 20 - 20 + 600)
check("捕獲成功で小物が1減る", s2.n_small == 11)
s3 = transition(State(e=20, w=0, mode=CHASE_4), None, Draws(catch=False))
check("体力20で捕獲失敗なら飢え死", s3.dead == DEAD_STARVE)
check("捕獲失敗では小物が減らない", s3.n_small == 12)

# 8. 強制行動の途中で飢え死（返答32の追加3本目）
s = State(e=20, w=0, mode=SEARCH_SMALL)
s2 = transition(s, None, Draws(find=False))
check("探索の途中で飢え死（E=20、遭遇なし）", s2.dead == DEAD_STARVE and s2.e == 0)

# 9. 上限クリップ（返答32の追加4本目）
s = State(e=1900, w=0, mode=CHASE_4)
check("上限を超えた栄養は捨てる",
      transition(s, None, Draws(catch=True)).e == E_MAX)

# 10. 小物の下限：N_S=1 で捕獲成功しても1のまま
s = State(n_small=1, mode=CHASE_4, e=1000)
check("N_S=1 で捕獲成功しても1のまま",
      transition(s, None, Draws(catch=True)).n_small == 1)

# 11. 急所死は体力に関係なく終端
s = State(e=E_MAX, mode=COMBAT)
check("急所死は体力満タンでも終端",
      transition(s, None, Draws(combat="acute")).dead == DEAD_ACUTE)

# 12. 天寿：1000歩目を生き抜けば天寿。1000歩目に死んだら天寿ではない
s = State(t=999, e=1000, mode=GRASS)
check("1000歩目を生き抜けば天寿", transition(s, None, Draws()).dead == DEAD_TENJU)
s = State(t=999, e=1000, mode=COMBAT)
check("1000歩目に急所死なら天寿ではない",
      transition(s, None, Draws(combat="acute")).dead == DEAD_ACUTE)
s = State(t=999, e=20, w=0, mode=SEARCH_SMALL)   # 草は摂取14が入るので飢え死にならない
check("1000歩目に飢え死なら天寿ではない",
      transition(s, None, Draws(find=False)).dead == DEAD_STARVE)

# 13. decision 境界以外では行動を選べない／時間を使わない
expect_error("探索中に行動を渡すと拒否",
             lambda: transition(State(mode=SEARCH_SMALL), ACT_GRASS, Draws()))
expect_error("decision で行動を渡さないと拒否",
             lambda: transition(State(), None, Draws()))
s = State()
s2 = transition(s, ACT_SMALL, Draws(find=False))
check("decision は時間を使わない（探索の1歩目が同じ歩）", s2.t == 1 and s2.mode == SEARCH_SMALL)
check("探索の1歩でも消耗する", s2.e == 2000 - 20)

# 14. 追跡の連鎖と歩数（探索の遭遇歩を含めて 1+4 歩で decision に戻る）
s = State()
s = transition(s, ACT_SMALL, Draws(find=True))
check("遭遇した歩で chase_1 へ", s.mode == CHASE_1 and s.t == 1)
for m in (CHASE_2, CHASE_3, CHASE_4):
    s = transition(s, None, Draws())
    check("追跡の連鎖 %s" % m, s.mode == m)
s = transition(s, None, Draws(catch=True))
check("追跡4歩目のあと decision へ（合計5歩）", s.mode == DECISION and s.t == 5)

# 15. 大物：遭遇した歩で戦闘へ。仕留めたら decision へ
s = State()
s = transition(s, ACT_LARGE, Draws(find=True))
check("大物に遭遇した歩で戦闘へ", s.mode == COMBAT and s.t == 1)
s = transition(s, None, Draws(combat="kill"))
check("仕留めたら decision へ、頭数が1減る", s.mode == DECISION and s.n_large == 2)

# 16. 終端状態からは遷移しない
expect_error("終端から遷移しない",
             lambda: transition(State(dead=DEAD_STARVE), None, Draws()))

# 17. 純関数性（ChatGPT 返答34の追加）
s = State(e=1234, w=2, n_small=7, n_large=2, t=42, mode=COMBAT)
d = Draws(heal=1, recover=True, combat="wound")
n1 = transition(s, None, d)
n2 = transition(s, None, d)
check("同じ入力なら同じ出力", n1 == n2)
check("入力の状態を書き換えない",
      s == State(e=1234, w=2, n_small=7, n_large=2, t=42, mode=COMBAT))

# 18. 同じ seed の別インスタンスから同じ抽選列が出る（再現手順の確認）
import numpy as np
r1 = np.random.default_rng(20260918)
r2 = np.random.default_rng(20260918)
s1 = State()
seq1, seq2 = [], []
for _ in range(50):
    a1 = ACT_LARGE if s1.at_decision else None
    d1 = w.sample_draws(s1, a1, r1)
    d2 = w.sample_draws(s1, a1, r2)
    seq1.append(d1)
    seq2.append(d2)
    s1 = transition(s1, a1, d1)
    if not s1.alive:
        s1 = State()
check("同じ seed なら同じ抽選列", seq1 == seq2)

# 19. 回帰テスト（★初回 Monte Carlo の結果を見た後に追加した post-diagnostic regression test★）
#     事前登録した A-0 ではない。小物探索の1歩目の遭遇が集計から落ちていた件の再発防止。
for hit, exp_search, exp_total in ((1, 1, 5), (2, 2, 6), (3, 3, 7), (4, 4, 8)):
    s = State(e=E_MAX, w=0, n_small=12, mode=DECISION)
    a = ACT_SMALL
    t0, got_search, i = s.t, None, 0
    while True:
        i += 1
        run_mode = w.start_mode(a) if s.mode == DECISION else s.mode
        d = Draws(find=(i == hit), catch=(run_mode == CHASE_4))
        nxt = transition(s, a, d)
        if run_mode == SEARCH_SMALL and d.find:
            got_search = nxt.t - t0
        a = None
        s = State(e=E_MAX, w=0, n_small=12, n_large=nxt.n_large, t=nxt.t, mode=nxt.mode)
        if s.mode == DECISION:
            break
    check("回帰:%d歩目で遭遇 → 探索%d歩・合計%d歩" % (hit, exp_search, exp_total),
          got_search == exp_search and s.t - t0 == exp_total)

print("A-0 合格 %d 件 / NG %d 件" % (ok, len(ng)))
for name in ng:
    print("  NG:", name)
